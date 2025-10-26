import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from pprint import pprint
from typing import Any

import normflows as nf
import optuna
import torch
from pytorch_lightning import seed_everything
from torch_geometric.data import Batch
from torchhd import HRRTensor
from tqdm.auto import tqdm

from src.encoding.configs_and_constants import (
    SupportedDataset,
)
from src.generation.evaluator import GenerationEvaluator, rdkit_logp
from src.generation.generation import HDCGenerator
from src.generation.logp_regressor import LogPRegressor
from src.utils.chem import draw_mol
from src.utils.utils import (
    GLOBAL_ARTEFACTS_PATH,
    GLOBAL_MODEL_PATH,
    DataTransformer,
    find_files,
    pick_device,
)

# keep it modest to avoid oversubscription; tune if needed
num = max(1, min(8, os.cpu_count() or 1))
torch.set_num_threads(num)
torch.set_num_interop_threads(max(1, min(2, num)))  # coordination threads

# by defualt float64
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)


# (optional but often helps BLAS backends)
os.environ.setdefault("OMP_NUM_THREADS", str(num))
os.environ.setdefault("MKL_NUM_THREADS", str(num))

seed = 42
seed_everything(seed)
device = pick_device()
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


def make_lambda_constant(*, steps, lam_lo, lam_hi):
    assert lam_lo == lam_hi

    def sched(_: int) -> float:
        return lam_lo

    return sched


schedulers: dict[str, Callable] = {
    "cosine": make_lambda_cosine_decay,
    "two-phase": make_lambda_two_phase,
    "linear": make_lambda_linear_decay,
    "constant": make_lambda_constant,
}


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


LPR_HINT = {
    SupportedDataset.QM9_SMILES_HRR_1600_F64_G1G3: "lpr_QM9SmilesHRR1600F64G1G3_h768-896-256_actsilu_nmln_dp0.0194027_bs64_lr0.000126707_wd1e-6_dep4_h1768_h2896_h3256_h4256",
    SupportedDataset.QM9_SMILES_HRR_1600_F64_G1NG3: "lpr_QM9SmilesHRR1600F64G1NG3_h512-256-128_actgelu_nmnone_dp0.010168_bs320_lr0.000355578_wd1e-5_dep3_h1512_h2256_h3128_h4160",
}


def eval_cond_gen(cfg: dict) -> dict[str, Any]:  # noqa: PLR0915
    gen_model_hint = os.getenv("GEN_MODEL")
    assert gen_model_hint is not None
    assert os.getenv("CLASSIFIER") is not None
    print(f"Using device: {device}")
    dataset: SupportedDataset = cfg.get("dataset")
    decoder_settings = {
        "initial_limit": 2048,
        "limit": 1024,
        "beam_size": 768,
        "pruning_method": "cos_sim",
        "use_size_aware_pruning": True,
        "use_one_initial_population": True,
    }
    generator = HDCGenerator(
        gen_model_hint=gen_model_hint,
        ds_config=dataset.default_cfg,
        decoder_settings=decoder_settings,
        device=device,
        dtype=DTYPE,
    )
    gen_model = generator.gen_model

    lpr_path = get_lpr(hint=LPR_HINT.get(dataset))
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
    print(f"Gen model device: {gen_model.device}")
    print(f"LPR model device: {logp_regressor.device}")

    # ---- Hyperparams ----
    base_dataset = cfg.get("base_dataset", "zinc")
    n_samples = cfg.get("n_samples", 100)
    latent_dim = gen_model.flat_dim
    target = cfg.get("target")
    lambda_hi = cfg.get("lambda_hi", 1e-2)
    lambda_lo = cfg.get("lambda_lo", 1e-4)
    steps = cfg.get("steps", 300)
    lr = cfg.get("lr", 1e-3)
    epsilon = 0.33 * logp_stats[base_dataset]["std"]  # success@epsilon threshold for logP

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
    e, g = gen_model.split(torch.stack(hits))
    e = e.as_subclass(HRRTensor)
    g = g.as_subclass(HRRTensor)

    decoded = generator.decode(node_terms=None, edge_terms=e, graph_terms=g)
    nx_graphs = decoded["graphs"]
    final_flags = decoded["final_flags"]
    sims = decoded["similarities"]

    # Second filter nx -> hdc -> log -> is in eps?
    batch = Batch.from_data_list([DataTransformer.nx_to_pyg(g) for g in nx_graphs])
    hypernet = generator.hypernet
    res = hypernet.forward(batch, normalize=dataset.default_cfg.normalize)
    hdc_round_2 = torch.cat((res["edge_terms"], res["graph_embedding"]), dim=1)
    y_pred_round_2 = logp_regressor.gen_forward(hdc_round_2)
    hits_round_2 = (y_pred_round_2 - target).abs() <= epsilon
    success_rate_round_2 = hits_round_2.float().sum().item()

    results = {
        "gen_model": str(gen_model_hint),
        "logp_regressor": str(lpr_path.parent.parent.stem),
        "n_samples": n_samples,
        "gen_time_per_sample": t_gen / n_samples,
        "target": target,
        "min_gda_loss": min_loss,
        "lambda_scheduler": cfg.get("scheduler"),
        "lambda_hi": lambda_hi,
        "lambda_low": lambda_lo,
        "steps": steps,
        "lr": lr,
        "epsilon": epsilon,
        "initial_success_rate": success_rate,
        "final_success_rate": success_rate_round_2 / n_samples,
        **decoder_settings,
    }

    # results.update(evaluator.evaluate(samples=nx_graphs, final_flags=final_flag, sims=sims))
    global EVALUATOR  # noqa: PLW0603
    if EVALUATOR is None:
        EVALUATOR = GenerationEvaluator(base_dataset=base_dataset, device=device)
    evals = EVALUATOR.evaluate_conditional(
        samples=[g for g, hit in zip(nx_graphs, hits_round_2.tolist(), strict=True) if hit],
        target=target,
        final_flags=final_flags,
        eps=epsilon,
        total_samples=n_samples,
        sims=sims,
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
                    out = base_dir / f"{gen_model_hint}__{os.getenv('CLASSIFIER')}__sim{sim:.3f}__logp{logp:.3f}{i}.png"
                    draw_mol(mol=mol, save_path=out, fmt="png")

    return results


def run_qm9_cond_gen(trial: optuna.Trial, dataset: SupportedDataset, tgt_multiplier: int):
    cfg = {
        "lr": trial.suggest_float("lr", 5e-5, 1e-3, log=True),
        "steps": trial.suggest_int("steps", 50, 1500),
        "draw": False,
        "n_samples": 1000,
        "base_dataset": "qm9",
        "dataset": dataset,
    }
    scheduler = trial.suggest_categorical("scheduler", ["cosine", "two-phase", "linear", "constant"])
    if scheduler == "constant":
        lam = trial.suggest_float("lambda_const", 1e-5, 5e-2, log=True)
        lambda_hi = lambda_lo = lam
    else:
        lambda_lo = trial.suggest_float("lambda_lo", 1e-5, 5e-3, log=True)
        lambda_hi = trial.suggest_float("lambda_hi", 5e-3, 5e-2, log=True)

    cfg["scheduler"] = scheduler
    cfg["lambda_lo"] = lambda_lo
    cfg["lambda_hi"] = lambda_hi
    cfg["target"] = logp_stats["qm9"]["mean"] + tgt_multiplier * logp_stats["qm9"]["std"]
    pprint(cfg)

    return eval_cond_gen(cfg=cfg)


# def run_zinc_cond_gen(trial: optuna.Trial, dataset: SupportedDataset):
#     cfg = {
#         "lr": trial.suggest_float("lr", 5e-5, 1e-3, log=True),
#         "steps": trial.suggest_int("steps", 50, 1500),
#         "scheduler": trial.suggest_categorical("scheduler", ["cosine", "two-phase", "linear"]),
#         "lambda_lo": trial.suggest_float("lambda_lo", 1e-5, 5e-3, log=True),
#         "lambda_hi": trial.suggest_float("lambda_hi", 5e-3, 5e-2, log=True),
#         "draw": False,
#         "n_samples": 100,
#         "base_dataset": "zinc",
#         "dataset": dataset,
#     }
#     pprint(cfg)
#
#     return eval_cond_gen(cfg=cfg)
