import argparse
import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from pprint import pprint
from typing import Any

import normflows as nf
import numpy as np
import torch
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm

from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import (
    QM9_SMILES_HRR_1600_CONFIG_F64,
    ZINC_SMILES_HRR_7744_CONFIG_F64,
)
from src.generation.evaluator import GenerationEvaluator, rdkit_logp
from src.generation.generation import HDCGenerator
from src.generation.logp_regressor import LogPRegressor
from src.utils import registery
from src.utils.chem import draw_mol
from src.utils.registery import get_model_type
from src.utils.utils import GLOBAL_ARTEFACTS_PATH, GLOBAL_MODEL_PATH, find_files, pick_device
from src.utils.visualisations import plot_logp_kde

# keep it modest to avoid oversubscription; tune if needed
num = max(1, min(8, os.cpu_count() or 1))
torch.set_num_threads(num)
torch.set_num_interop_threads(max(1, min(2, num)))  # coordination threads
torch.set_default_dtype(torch.float64)

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


def eval_cond_gen(cfg: dict, decoder_settings: dict, gen_model_hint: str, lpr_hint: str) -> dict[str, Any]:  # noqa: PLR0915
    global EVALUATOR  # noqa: PLW0603
    global HDC_Y_REUSE

    gen_ckpt_path = get_gen_model(hint=gen_model_hint)
    print(f"Generator Checkpoint: {gen_ckpt_path}")
    ## Retrieve the model
    model_type_gen = get_model_type(gen_ckpt_path)
    gen_model = registery.retrieve_model(model_type_gen).load_from_checkpoint(
        gen_ckpt_path, map_location=device, strict=True
    )

    lpr_path = get_lpr(hint=lpr_hint)
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

    ds_config = QM9_SMILES_HRR_1600_CONFIG_F64 if base_dataset == "qm9" else ZINC_SMILES_HRR_7744_CONFIG_F64

    # Only evaluate the hits
    hits = [s for s, h in zip(hdc, hits, strict=True) if h]
    n, e, g = gen_model.split(torch.stack(hits))
    n = n.as_subclass(ds_config.vsa.tensor_class)
    e = e.as_subclass(ds_config.vsa.tensor_class)
    g = g.as_subclass(ds_config.vsa.tensor_class)

    generator = HDCGenerator(
        gen_model_hint=gen_model_hint,
        ds_config=ds_config,
        decoder_settings=decoder_settings,
        device=device,
    )

    t0_decode = time.perf_counter()
    res = generator.decode(node_terms=n, edge_terms=e, graph_terms=g)
    nx_graphs = res["graphs"]
    final_flags = res["final_flags"]
    sims = res["similarities"]
    t_decode = time.perf_counter() - t0_decode
    # nx_graphs, final_flag, sims = generator.generate_most_similar(n_samples=n_samples, only_final_graphs=False)

    results = {
        "gen_model": str(gen_ckpt_path.parent.parent.stem),
        "logp_regressor": str(lpr_path.parent.parent.stem),
        "n_samples": n_samples,
        "t_decode_per_sample": t_decode / len(hits),
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
        samples=nx_graphs, target=target, final_flags=final_flags, eps=epsilon, total_samples=n_samples, sims=sims
    )
    results.update({f"eval_{k}": v for k, v in evals.items()})
    pprint(results)

    mols, valid_flags, sims = EVALUATOR.get_mols_valid_flags_sims_and_correction_levels()

    base_dir = (
        GLOBAL_ARTEFACTS_PATH
        / "cond_generation"
        / f"{base_dataset}_{os.getenv('GEN_MODEL')}_{os.getenv('CLASSIFIER')}_{n_samples}-samples"
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save for later re-plotting
    logp_gen_list = np.array(
        [rdkit_logp(mol) for mol, valid in zip(mols, valid_flags, strict=False) if valid], dtype=float
    )
    np.save(base_dir / f"{target:.3f}.npy", logp_gen_list)
    (base_dir / f"results_target_{target:.3f}.json").write_text(json.dumps(results, indent=4))

    if cfg.get("draw", False):
        base_dir.mkdir(parents=True, exist_ok=True)
        for i, (mol, valid, sim) in enumerate(zip(mols, valid_flags, sims, strict=False)):
            if valid:
                logp = rdkit_logp(mol)
                if abs(logp - target) > epsilon:
                    out = base_dir / f"Sim_{sim:.3f}__LogP_{logp:.3f}_{i}.png"
                    draw_mol(mol=mol, save_path=out, fmt="png")

    if cfg.get("plot", False):
        ds = QM9Smiles(split="train") if base_dataset == "qm9" else ZincSmiles(split="train")
        lp = np.array(ds.logp.tolist())

        total = evals["total"]
        evals_total = {f"{k.split('_pct')[0]}": f"{v}%" for k, v in total.items() if "pct" in k}
        valids = evals["valid"]
        evals_valid = {"mae_to_target": f"{valids['mae_to_target']:.2f}"}
        evals_valid.update({f"{k.split('_pct')[0]}": f"{int(v)}%" for k, v in valids.items() if "pct" in k})

        plot_logp_kde(
            dataset=base_dataset,
            lp=lp,
            lg=logp_gen_list,
            evals_total=evals_total,
            evals_valid=evals_valid,
            epsilon=epsilon,
            target=target,
            out=(base_dir / f"logp_overlay_{target:.3f}.png"),
            description=f"Classifier ({os.getenv('CLASSIFIER')}",
        )

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate conditional samples from a trained model with plots.")
    p.add_argument("--dataset", type=str, default="qm9", choices=["zinc", "qm9"])
    p.add_argument("--n_samples", type=int, default=1000)
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
        "nvp-3d-f64_qm9_f8_hid800_lr0.000373182_wd1e-5_bs384_smf6.54123_smi2": {
            "lr": 0.00041353778859151187,
            "steps": 1234,
            "scheduler": "two-phase",
            "lambda_lo": 1.3245461546001868e-05,
            "lambda_hi": 0.019868999906189445,
        }
    }
    lpr = {
        "qm9": "lpr-3d_qm9_h896-512_actlrelu_nmln_dp0.0377275_bs448_lr0.000181621_wd1e-5_dep2_h1512_h2896_h3192_h4224"
    }
    DECODER_SETTINGS = {
        "qm9": {
            "initial_limit": 2048,
            "limit": 1024,
            "beam_size": 256,
            "pruning_method": "negative_euclidean_distance",
            "use_size_aware_pruning": True,
            "use_one_initial_population": True,
            "use_g3_instead_of_h3": True,
        }
    }
    n_samples = args.n_samples
    for target_multiplier in [0, 1, -1, 2, -2, 3, -3]:
        for (
            dataset,
            gen_model,
            classifier,
            samples,
        ) in [
            (
                args.dataset,
                "nvp-3d-f64_qm9_f8_hid800_lr0.000373182_wd1e-5_bs384_smf6.54123_smi2",
                "HDC-DECODER",
                n_samples,
            ),
        ]:
            cfg = {
                "lr": model_configs[gen_model]["lr"],
                "steps": model_configs[gen_model]["steps"],
                # "steps": 5,
                "scheduler": model_configs[gen_model]["scheduler"],
                "lambda_lo": model_configs[gen_model]["lambda_lo"],
                "lambda_hi": model_configs[gen_model]["lambda_hi"],
                "base_dataset": dataset,
                "n_samples": samples,
                "target": logp_stats[dataset]["mean"] + target_multiplier * logp_stats[dataset]["std"],
                "epsilon": 0.25 * logp_stats[dataset]["std"],
                "draw": False,
                "plot": True,
            }
            os.environ["GEN_MODEL"] = gen_model
            os.environ["CLASSIFIER"] = classifier
            os.environ["LPR"] = lpr[dataset]
            # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            res = eval_cond_gen(
                cfg=cfg,
                decoder_settings=DECODER_SETTINGS[dataset],
                gen_model_hint=gen_model,
                lpr_hint=lpr[dataset],
            )
