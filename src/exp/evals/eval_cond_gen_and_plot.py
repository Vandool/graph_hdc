import argparse
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from pprint import pprint
from typing import Any

import normflows as nf
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm

from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import QM9_SMILES_HRR_1600_CONFIG
from src.encoding.oracles import Oracle, SimpleVoterOracle
from src.generation.evaluator import GenerationEvaluator, rdkit_logp
from src.generation.generation import Generator
from src.generation.logp_regressor import LogPRegressor
from src.utils import registery
from src.utils.chem import draw_mol
from src.utils.utils import GLOBAL_ARTEFACTS_PATH, GLOBAL_MODEL_PATH, find_files, pick_device
from src.utils.visualisations import plot_logp_kde

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
HDC_Y_REUSE: dict[tuple[str, float], Any] = {}


def stats(arr):
    arr = np.asarray(arr)
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
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
        skip_substrings=("zinc",),
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
        skip_substrings=("zinc", "nvp"),
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
        skip_substrings=("zinc",),
    )
    for p in paths:
        if hint in str(p):
            return p
    return None


def eval_cond_gen(cfg: dict, decoder_settings: dict) -> dict[str, Any]:  # noqa: PLR0915
    global EVALUATOR  # noqa: PLW0603
    global HDC_Y_REUSE
    gen_model_hint = os.getenv("GEN_MODEL")
    assert gen_model_hint is not None
    assert os.getenv("CLASSIFIER") is not None

    gen_ckpt_path = get_gen_model(hint=gen_model_hint)
    print(f"Generator Checkpoint: {gen_ckpt_path}")
    ## Retrieve the model
    model_type_gen = get_model_type(gen_ckpt_path)
    gen_model = registery.retrieve_model(model_type_gen).load_from_checkpoint(
        gen_ckpt_path, map_location=device, strict=True
    )

    lpr_path = get_lpr(
        hint="lpr_qm9_h1280-128-448_actlrelu_nmnone_dp0.00331757_bs480_lr0.000102482_wd3e-6_dep4_h11280_h2128_h3448_h496"
    )
    print(lpr_path)
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
    base_dataset = cfg.get("base_dataset", "qm9")
    n_samples = cfg.get("n_samples", 100)
    latent_dim = gen_model.flat_dim
    target = cfg.get("target")
    lambda_hi = cfg.get("lambda_hi", 1e-2)
    lambda_lo = cfg.get("lambda_lo", 1e-4)
    steps = cfg.get("steps", 300)
    lr = cfg.get("lr", 1e-3)
    epsilon = cfg.get("epsilon", 0.02)

    if HDC_Y_REUSE.get((gen_model_hint, target)):
        hdc = HDC_Y_REUSE[(gen_model_hint, target)]["hdc"]
        y_pred = HDC_Y_REUSE[(gen_model_hint, target)]["y_pred"]
    else:
        base = nf.distributions.DiagGaussian(latent_dim, trainable=False).to(device)
        z = base.sample(n_samples)
        z = z.detach().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)

        sched, sched_name = schedulers[cfg.get("scheduler")](steps=steps, lam_hi=lambda_hi, lam_lo=lambda_lo), "cosine"

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

        HDC_Y_REUSE[(gen_model_hint, target)] = {"hdc": hdc, "y_pred": y_pred}

    # ---- Success@epsilon (using model's prediction; replace with RDKit eval if you want) ----
    hits = (y_pred - target).abs() <= epsilon
    success_rate = hits.float().mean().item()

    # Only evaluate the hits
    hits = [s for s, h in zip(hdc, hits, strict=True) if h]
    n, g = gen_model.split(torch.stack(hits))
    # n, g = gen_model.split(hdc)

    if os.getenv("CLASSIFIER") == "SIMPLE_VOTER":
        oracle = SimpleVoterOracle(
            model_paths=[
                GLOBAL_MODEL_PATH / "1_gin/gin-f_baseline_qm9_resume/models/epoch10-val0.3359.ckpt",
                GLOBAL_MODEL_PATH / "1_mlp_lightning/MLP_Lightning_qm9/models/epoch17-val0.2472.ckpt",
                GLOBAL_MODEL_PATH / "2_bah_lightning/BAH_base_qm9_v2/models/epoch23-val0.2772.ckpt",
            ],
            device=device,
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

    generator = Generator(
        gen_model=gen_model,
        oracle=oracle,
        ds_config=QM9_SMILES_HRR_1600_CONFIG,
        decoder_settings=decoder_settings,
        device=device,
    )

    nx_graphs, final_flags, sims = generator.decode(node_terms=n, graph_terms=g)
    # nx_graphs, final_flag, sims = generator.generate_most_similar(n_samples=n_samples, only_final_graphs=False)

    results = {
        "gen_model": str(gen_ckpt_path.parent.parent.stem),
        "logp_regressor": str(lpr_path.parent.parent.stem),
        "n_samples": n_samples,
        "target": cfg.get("target"),
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
    if EVALUATOR is None:
        EVALUATOR = GenerationEvaluator(base_dataset=base_dataset, device=device)

    evals = EVALUATOR.evaluate_conditional(
        samples=nx_graphs, target=target, final_flags=final_flags, eps=epsilon, total_samples=n_samples
    )
    results.update({f"eval_{k}": v for k, v in evals.items()})
    pprint(results)

    mols, valid_flags = EVALUATOR.get_mols_and_valid_flags()
    logp_gen_list = np.array(
        [rdkit_logp(mol) for mol, valid in zip(mols, valid_flags, strict=False) if valid], dtype=float
    )

    base_dir = (
        GLOBAL_ARTEFACTS_PATH
        / "cond_generation"
        / f"{base_dataset}_{os.getenv('GEN_MODEL')}_{os.getenv('CLASSIFIER')}_{n_samples}-samples"
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save for later re-plotting
    np.save(base_dir / f"{target:.3f}.npy", logp_gen_list)
    (base_dir / f"results_target_{target:.3f}.json").write_text(json.dumps(results, indent=4))

    if cfg.get("draw", False):
        base_dir.mkdir(parents=True, exist_ok=True)
        for i, (mol, valid) in enumerate(zip(mols, valid_flags, strict=False)):
            if valid:
                logp = rdkit_logp(mol)
                if abs(logp - target) > epsilon:
                    out = base_dir / f"LogP_{logp:.3f}_{i}.png"
                    draw_mol(mol=mol, save_path=out, fmt="png")

    if cfg.get("plot", False):
        ds = QM9Smiles(split="train") if base_dataset == "qm9" else ZincSmiles(split="train")
        lp = np.array(ds.logp.tolist())
        # --- dataset stats ---
        plot_logp_kde(
            dataset=base_dataset,
            lp=lp,
            lg=logp_gen_list,
            evals=evals,
            out=(base_dir / f"logp_overlay_{target:.3f}.png"),
            description=f"Classifier ({os.getenv('CLASSIFIER')}",
        )

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate conditional samples from a trained model with plots.")
    p.add_argument("--dataset", type=str, default="qm9", choices=["zinc", "qm9"])
    p.add_argument("--n_samples", type=int, default=10)
    args = p.parse_args()
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
    model_configs = {
        "nvp_qm9_h1600_f12_hid1024_s42_lr5e-4_wd0.0_an": {
            "lr": 0.0001535528683888,
            "steps": 1492,
            "scheduler": "cosine",
            "lambda_lo": 0.0001840899208055,
            "lambda_hi": 0.0052054096619994,
        }
    }
    decoder_settings = {
        "beam_size": 512,
        "use_pair_feasibility": True,
        "expand_on_n_anchors": 9,
    }
    n_samples = args.n_samples
    dataset = args.dataset
    for target_multiplier in [1, 2, 3, 4]:
        for (
            dataset,
            gen_model,
            classifier,
            samples,
        ) in [
            (
                dataset,
                "nvp_qm9_h1600_f12_hid1024_s42_lr5e-4_wd0.0_an",
                "BAH_med_hardpool_qm9",
                n_samples,
            ),
            (
                dataset,
                "nvp_qm9_h1600_f12_hid1024_s42_lr5e-4_wd0.0_an",
                "SIMPLE_VOTER",
                n_samples,
            ),
            (
                dataset,
                "nvp_qm9_h1600_f12_hid1024_s42_lr5e-4_wd0.0_an",
                "gin-f_baseline_qm9_resume",
                n_samples,
            ),
        ]:
            cfg = {
                "lr": model_configs[gen_model]["lr"],
                "steps": model_configs[gen_model]["steps"],
                "scheduler": model_configs[gen_model]["scheduler"],
                "lambda_lo": model_configs[gen_model]["lambda_lo"],
                "lambda_hi": model_configs[gen_model]["lambda_hi"],
                "draw": False,
                "n_samples": samples,
                "base_dataset": dataset,
                "plot": True,
                "target": logp_stats[dataset]["mean"] * target_multiplier,
                "epsilon": 0.25 * logp_stats[dataset]["std"],
            }
            os.environ["GEN_MODEL"] = gen_model
            os.environ["CLASSIFIER"] = classifier
            res = eval_cond_gen(cfg=cfg, decoder_settings=decoder_settings)
