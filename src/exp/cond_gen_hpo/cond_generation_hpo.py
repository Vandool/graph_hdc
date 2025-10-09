import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from pprint import pprint
from typing import Any

import normflows as nf
import optuna
import pandas as pd
import torch
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm

from src.encoding.configs_and_constants import QM9_SMILES_HRR_1600_CONFIG, ZINC_SMILES_HRR_7744_CONFIG
from src.encoding.oracles import Oracle, SimpleVoterOracle
from src.generation.evaluator import GenerationEvaluator, rdkit_logp
from src.generation.generation import OracleGenerator
from src.generation.logp_regressor import LogPRegressor
from src.utils import registery
from src.utils.chem import draw_mol
from src.utils.utils import GLOBAL_ARTEFACTS_PATH, GLOBAL_MODEL_PATH, find_files, pick_device

# keep it modest to avoid oversubscription; tune if needed
num = max(1, min(8, os.cpu_count() or 1))
torch.set_num_threads(num)
torch.set_num_interop_threads(max(1, min(2, num)))  # coordination threads

# (optional but often helps BLAS backends)
os.environ.setdefault("OMP_NUM_THREADS", str(num))
os.environ.setdefault("MKL_NUM_THREADS", str(num))

seed = 42
seed_everything(seed)
device = pick_device()
# device = torch.device("cpu")
EVALUATOR = None

logp_stats = {
    "qm9": {
        "max": 3,
        "mean": 0.30487121410781287,
        "median": 0.27810001373291016,
        "min": -5,
        "std": 0.9661976604136703,
    },
    "zinc": {"max": 8, "mean": 2.457799800788871, "median": 2.60617995262146, "min": -6, "std": 1.4334213538628746},
}


def get_model_type(path: Path | str) -> registery.ModelType:
    res: registery.ModelType = "MLP"
    if "bah" in str(path):
        res = "BAH"
    elif "gin-c" in str(path):
        res = "GIN-C"
    elif "gin-f" in str(path):
        res = "GIN-F"
    elif "nvp" in str(path):
        res = "NVP"
    return res


def make_lambda_cosine_decay(*, steps: int, lam_hi: float = 1e-2, lam_lo: float = 1e-4):
    steps = max(1, int(steps))

    def sched(s: int) -> float:
        # s ∈ [0, steps-1]
        t = min(max(s / (steps - 1 if steps > 1 else 1), 0.0), 1.0)
        # starts at lam_hi, ends at lam_lo
        return lam_lo + 0.5 * (lam_hi - lam_lo) * (1.0 + math.cos(math.pi * t))

    return sched


def make_lambda_linear_decay(*, steps: int, lam_hi: float = 1e-2, lam_lo: float = 1e-4):
    steps = max(1, int(steps))

    def sched(s: int) -> float:
        t = min(max(s / (steps - 1 if steps > 1 else 1), 0.0), 1.0)
        return lam_hi + t * (lam_lo - lam_hi)

    return sched


def make_lambda_two_phase(
    *,
    steps: int,
    lam_hi: float = 1e-2,
    lam_lo: float = 1e-4,
    warm_frac: float = 0.2,
):
    steps = max(1, int(steps))
    warm = int(max(0.0, min(1.0, warm_frac)) * steps)

    def sched(s: int) -> float:
        if s < warm:
            return lam_hi
        # cosine decay to lam_lo over the remaining steps
        remain = max(1, steps - warm)
        t = min(max((s - warm) / remain, 0.0), 1.0)
        return lam_lo + 0.5 * (lam_hi - lam_lo) * (1.0 + math.cos(math.pi * t))

    return sched


schedulers: dict[str, Callable] = {
    "cosine": make_lambda_cosine_decay,
    "two-phase": make_lambda_two_phase,
    "linear": make_lambda_linear_decay,
}


def get_gen_model(hint: str) -> Path | None:
    gen_paths = find_files(
        start_dir=GLOBAL_MODEL_PATH / "0_real_nvp_v2",
        prefixes=("epoch",),
        desired_ending=".ckpt",
    )
    for p in gen_paths:
        if hint in str(p):
            return p
    return None


def get_classifier(hint: str) -> Path | None:
    paths = find_files(
        start_dir=GLOBAL_MODEL_PATH,
        prefixes=("epoch",),
        desired_ending=".ckpt",
        skip_substrings=("nvp", "lpr"),
    )
    for p in paths:
        if hint in str(p):
            return p
    return None


def get_lpr(hint: str) -> Path | None:
    paths = find_files(
        start_dir=GLOBAL_MODEL_PATH / "lpr",
        prefixes=("epoch",),
        desired_ending=".ckpt",
    )
    for p in paths:
        if hint in str(p):
            return p
    return None


def eval_cond_gen(cfg: dict) -> dict[str, Any]:  # noqa: PLR0915
    assert os.getenv("GEN_MODEL") is not None
    assert os.getenv("CLASSIFIER") is not None

    gen_ckpt_path = get_gen_model(hint=os.getenv("GEN_MODEL"))
    print(f"Generator Checkpoint: {gen_ckpt_path}")
    ## Retrieve the model
    model_type_gen = get_model_type(gen_ckpt_path)
    gen_model = registery.retrieve_model(model_type_gen).load_from_checkpoint(
        gen_ckpt_path, map_location=device, strict=True
    )

    lpr_path = get_lpr(
        hint="lpr_zinc_h768-768-128_actsilu_nmbn_dp0.181982_bs288_lr0.000102926_wd0.0001_dep3_h1768_h2768_h3128_h4160"
    )
    print(f"LPR Checkpoint: {lpr_path}")
    logp_regressor = LogPRegressor.load_from_checkpoint(lpr_path, map_location=device, strict=True)

    gen_model.eval()
    logp_regressor.eval()
    for p in gen_model.parameters():
        p.requires_grad = False
    for p in logp_regressor.parameters():
        p.requires_grad = False

    gen_model.to(device)
    logp_regressor.to(device)

    # ---- Hyperparams ----
    base_dataset = cfg.get("base_dataset", "zinc")
    n_samples = cfg.get("n_samples", 100)
    latent_dim = gen_model.flat_dim
    target = logp_stats[base_dataset]["mean"] - logp_stats[base_dataset]["std"]
    lambda_hi = cfg.get("lambda_hi", 1e-2)
    lambda_lo = cfg.get("lambda_lo", 1e-4)
    steps = cfg.get("steps", 300)
    lr = cfg.get("lr", 1e-3)
    epsilon = 0.2  # success@epsilon threshold for logP

    t0_gen = time.perf_counter()
    # ---- Initialize latents ----
    base = nf.distributions.DiagGaussian(latent_dim, trainable=False).to(device)
    z = base.sample(n_samples)
    z = z.detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)

    sched = schedulers[cfg.get("scheduler")](steps=steps, lam_hi=lambda_hi, lam_lo=lambda_lo)

    min_loss = float("inf")
    # ---- Guidance loop ----
    pbar = tqdm(range(steps), desc="Steps", unit="step")
    for s in pbar:
        hdc = gen_model.decode_from_latent(z)  # needs grad
        y_hat = logp_regressor.gen_forward(hdc)

        lam = sched(s)  # <-- annealed prior weight
        target_tensor = torch.full_like(y_hat, float(target))
        loss = ((y_hat - target_tensor) ** 2).mean() + lam * z.pow(2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        min_loss = min(min_loss, loss.item())

        # update tqdm display
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "λ_prior": f"{lam:.2e}"})

    # ---- Decode once more and take the results ----
    with torch.no_grad():
        hdc = gen_model.decode_from_latent(z)
        y_pred = logp_regressor.gen_forward(hdc)

    t_gen = time.perf_counter() - t0_gen
    # ---- Success@epsilon (using model's prediction; replace with RDKit eval if you want) ----
    hits = (y_pred - target).abs() <= epsilon
    success_rate = hits.float().mean().item()

    # Only evaluate the hits
    hits = [s for s, h in zip(hdc, hits, strict=True) if h]
    n, g = gen_model.split(torch.stack(hits))
    # n, g = gen_model.split(hdc)

    decoder_settings = {
        "beam_size": 4,
        "use_pair_feasibility": True,
        "expand_on_n_anchors": 8,
        "trace_back_settings": {
            "beam_size_multiplier": 2,
            "trace_back_attempts": 3,
            "trace_back_to_last_nth_iter": 1,
            "agitated_rounds": 1,  # how many rounds after applying trace back keep the beam size larger
        },
    }
    if os.getenv("CLASSIFIER") == "SIMPLE_VOTER" and base_dataset == "qm9":  # not available for ZINC
        oracle = SimpleVoterOracle(
            model_paths=[
                GLOBAL_MODEL_PATH / "1_gin/gin-f_baseline_zinc_resume_3/models/epoch18-val0.2083.ckpt",
                GLOBAL_MODEL_PATH / "1_mlp_lightning/MLP_Lightning_qm9/models/epoch17-val0.2472.ckpt",
                GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_base_qm9_v2/models/epoch23-val0.2772.ckpt",
            ]
        )
    else:
        classifier_ckpt = get_classifier(hint=os.getenv("CLASSIFIER"))
        print(f"Classifier Checkpoint: {classifier_ckpt}")
        model_type = get_model_type(classifier_ckpt)
        classifier = (
            registery.retrieve_model(name=model_type)
            .load_from_checkpoint(classifier_ckpt, map_location="cpu", strict=True)
            .to(device)
            .eval()
        )

        # Read the metrics from training
        evals_dir = classifier_ckpt.parent.parent / "evaluations"
        epoch_metrics = pd.read_parquet(evals_dir / "epoch_metrics.parquet")

        val_loss = "val_loss" if "val_loss" in epoch_metrics.columns else "val_loss_cb"
        best = epoch_metrics.loc[epoch_metrics[val_loss].idxmin()].add_suffix("_best")
        oracle_threshold = best["val_best_thr_best"]
        print(f"Oracle Threshold: {oracle_threshold}")

        oracle = Oracle(model=classifier, model_type=model_type)
        decoder_settings["oracle_threshold"] = oracle_threshold

    generator = OracleGenerator(
        gen_model=gen_model,
        oracle=oracle,
        ds_config=QM9_SMILES_HRR_1600_CONFIG if base_dataset == "qm9" else ZINC_SMILES_HRR_7744_CONFIG,
        decoder_settings=decoder_settings,
        device=device,
    )

    nx_graphs, final_flags, sims = generator.decode(node_terms=n, graph_terms=g)
    # nx_graphs, final_flag, sims = generator.generate_most_similar(n_samples=n_samples, only_final_graphs=False)

    results = {
        "gen_model": str(gen_ckpt_path.parent.parent.stem),
        "logp_regressor": str(lpr_path.parent.parent.stem),
        "n_samples": n_samples,
        "gen_time_per_sample": t_gen / n_samples,
        "target": 0.0,
        "min_gda_loss": min_loss,
        "lambda_scheduler": cfg.get("scheduler"),
        "lambda_hi": lambda_hi,
        "lambda_low": lambda_lo,
        "steps": steps,
        "lr": lr,
        "epsilon": epsilon,
        "initial_success_rate": success_rate,
        "classifier": os.getenv("CLASSIFIER"),
        **decoder_settings,
    }

    # results.update(evaluator.evaluate(samples=nx_graphs, final_flags=final_flag, sims=sims))
    global EVALUATOR  # noqa: PLW0603
    if EVALUATOR is None:
        EVALUATOR = GenerationEvaluator(base_dataset=base_dataset, device=device)
    evals = EVALUATOR.evaluate_conditional(
        samples=nx_graphs, target=target, final_flags=final_flags, eps=epsilon, total_samples=n_samples
    )
    # Flatten the evals
    results.update(
        {f"{section}_{metric}": val for section, metrics in evals.items() for metric, val in metrics.items()}
    )
    pprint(results)

    if cfg.get("draw", False):
        base_dir = GLOBAL_ARTEFACTS_PATH / "cond_generation" / f"drawings_valid_target_{target:.1f}"
        base_dir.mkdir(parents=True, exist_ok=True)
        mols, valid_flags, sims = EVALUATOR.get_mols_and_valid_flags()
        for i, (mol, valid, sim) in enumerate(zip(mols, valid_flags, sims, strict=False)):
            if valid:
                logp = rdkit_logp(mol)
                if abs(logp - target) > epsilon:
                    out = (
                        base_dir
                        / f"{gen_ckpt_path.parent.parent.stem}__{os.getenv('CLASSIFIER')}__sim{sim:.3f}__logp{logp:.3f}{i}.png"
                    )
                    draw_mol(mol=mol, save_path=out, fmt="png")

    return results


def run_qm9_cond_gen(trial: optuna.Trial):
    cfg = {
        "lr": trial.suggest_float("lr", 5e-5, 1e-3, log=True),
        "steps": trial.suggest_int("steps", 50, 1500),
        "scheduler": trial.suggest_categorical("scheduler", ["cosine", "two-phase", "linear"]),
        "lambda_lo": trial.suggest_float("lambda_lo", 1e-5, 5e-3, log=True),
        "lambda_hi": trial.suggest_float("lambda_hi", 5e-3, 5e-2, log=True),
        "draw": False,
        "n_samples": 5,
        "base_dataset": "zinc",
    }
    pprint(cfg)

    return eval_cond_gen(cfg=cfg)


def run_zinc_cond_gen(trial: optuna.Trial):
    cfg = {
        "lr": trial.suggest_float("lr", 5e-5, 1e-3, log=True),
        "steps": trial.suggest_int("steps", 50, 1500),
        "scheduler": trial.suggest_categorical("scheduler", ["cosine", "two-phase", "linear"]),
        "lambda_lo": trial.suggest_float("lambda_lo", 1e-5, 5e-3, log=True),
        "lambda_hi": trial.suggest_float("lambda_hi", 5e-3, 5e-2, log=True),
        "draw": False,
        "n_samples": 100,
        "base_dataset": "zinc",
    }
    pprint(cfg)

    return eval_cond_gen(cfg=cfg)
