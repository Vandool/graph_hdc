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
import polars as pl
import torch
from pytorch_lightning import seed_everything
from torch_geometric.data import Batch
from tqdm.auto import tqdm

from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import SupportedDataset
from src.generation.evaluator import GenerationEvaluator, rdkit_logp
from src.generation.generation import HDCGenerator
from src.generation.logp_regressor import LogPRegressor
from src.utils.chem import draw_mol
from src.utils.utils import (
    GLOBAL_ARTEFACTS_PATH,
    GLOBAL_MODEL_PATH,
    ROOT,
    DataTransformer,
    find_files,
    pick_device,
    weighted_sample,
)
from src.utils.visualisations import plot_logp_kde

# keep it modest to avoid oversubscription; tune if needed
num = max(1, min(8, os.cpu_count() or 1))
torch.set_num_threads(num)
torch.set_num_interop_threads(max(1, min(2, num)))  # coordination threads

DTYPE = torch.float32
torch.set_default_dtype(DTYPE)

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
        skip_substrings=("zinc",),
    )
    for p in paths:
        if hint in str(p):
            return p
    return None


LPR_HINT = {
    SupportedDataset.QM9_SMILES_HRR_1600_F64_G1G3: "lpr_QM9SmilesHRR1600F64G1G3_h768-896-256_actsilu_nmln_dp0.0194027_bs64_lr0.000126707_wd1e-6_dep4_h1768_h2896_h3256_h4256",
    SupportedDataset.QM9_SMILES_HRR_1600_F64_G1NG3: "lpr_QM9SmilesHRR1600F64G1NG3_h512-256-128_actgelu_nmnone_dp0.010168_bs320_lr0.000355578_wd1e-5_dep3_h1512_h2256_h3128_h4160",
}


def eval_cond_gen(cfg: dict, decoder_settings: dict) -> dict[str, Any]:  # noqa: PLR0915
    global EVALUATOR  # noqa: PLW0603
    global HDC_Y_REUSE
    gen_model_hint = os.getenv("GEN_MODEL")
    assert gen_model_hint is not None

    dataset = cfg.get("dataset", SupportedDataset.ZINC_SMILES_HRR_5120_F64_G1G3)
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
    e, g = gen_model.split(torch.stack(hits))
    e = e.as_subclass(dataset.default_cfg.vsa.tensor_class)
    g = g.as_subclass(dataset.default_cfg.vsa.tensor_class)

    t0_decode = time.perf_counter()
    decoded = generator.decode(node_terms=None, edge_terms=e, graph_terms=g)
    t_decode = time.perf_counter() - t0_decode
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
        "t_decode_per_sample": t_decode / len(hits),
        "target": target,
        "lambda_scheduler": cfg.get("scheduler"),
        "lambda_hi": lambda_hi,
        "lambda_low": lambda_lo,
        "steps": steps,
        "lr": lr,
        "epsilon": epsilon,
        "initial_success_rate": success_rate,
        "round2_success_rate": success_rate_round_2 / n_samples,
        "classifier": "HDC_Decoder",
        **decoder_settings,
    }

    # results.update(evaluator.evaluate(samples=nx_graphs, final_flags=final_flag, sims=sims))
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
    results.update({f"eval_{k}": v for k, v in evals.items()})
    results.update({"hpo_metric": cfg.get("the_metric")})
    pprint(results)

    mols, valid_flags, sims = EVALUATOR.get_mols_and_valid_flags()

    dt = "f32" if torch.float32 == DTYPE else "f64"
    base_dir = (
        GLOBAL_ARTEFACTS_PATH
        / "cond_generation_v2"
        / f"{base_dataset}_{os.getenv('GEN_MODEL')}_{os.getenv('CLASSIFIER', 'HDC_Decoder')}_{dt}_{n_samples}-samples"
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

        # Weighted samples
        samples_ = n_samples // 10
        gaussian_samples, _, _ = weighted_sample(
            values=logp_gen_list, target=target, k=samples_, method="gaussian", replace=False
        )
        plot_logp_kde(
            dataset=base_dataset,
            lp=lp,
            lg=gaussian_samples,
            evals_total=evals_total,
            evals_valid=evals_valid,
            epsilon=epsilon,
            target=target,
            out=(base_dir / f"logp_overlay_{target:.3f}_gaussian_k{samples_}.png"),
            description=f"Classifier ({os.getenv('CLASSIFIER')}",
        )

        # Inverse samples
        inverse_distance_samples, _, _ = weighted_sample(
            values=logp_gen_list, target=target, k=samples_, method="inverse", replace=False
        )
        plot_logp_kde(
            dataset=base_dataset,
            lp=lp,
            lg=inverse_distance_samples,
            evals_total=evals_total,
            evals_valid=evals_valid,
            epsilon=epsilon,
            target=target,
            out=(base_dir / f"logp_overlay_{target:.3f}_invers_k{samples_}.png"),
            description=f"Classifier ({os.getenv('CLASSIFIER')}",
        )

    return results


def get_hpo_metrics(ds) -> pl.DataFrame:
    hpo_result_paths = list(
        find_files(
            start_dir=ROOT / "src/exp/cond_gen_hpo/hpo/",
            prefixes=(f"trials_R1_nvp_{ds.default_cfg.name}",),
            desired_ending=".csv",
        )
    )
    if not hpo_result_paths:
        raise FileNotFoundError(f"No HPO result CSVs found for dataset: {ds.default_cfg.name}")

    df = pl.concat([pl.read_csv(f) for f in hpo_result_paths], how="vertical")

    return df.with_columns((pl.col("valid_success_at_eps_pct") * pl.col("final_success_rate")).alias("the_metric"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate conditional samples from a trained model with plots.")
    p.add_argument(
        "--dataset",
        type=str,
        default=SupportedDataset.QM9_SMILES_HRR_1600_F64_G1NG3.value,
        choices=[ds.value for ds in SupportedDataset],
    )
    p.add_argument("--n_samples", type=int, default=1000)
    args = p.parse_args()
    dataset = SupportedDataset(args.dataset)
    base_dataset = dataset.default_cfg.base_dataset
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
    hpo_metrics = get_hpo_metrics(dataset)
    decoder_settings = {
        "initial_limit": 2048,
        "limit": 1024,
        "beam_size": 768,
        "pruning_method": "cos_sim",
        "use_size_aware_pruning": True,
        "use_one_initial_population": True,
    }
    for target_multiplier in [0, 1, -1]:
        target = logp_stats[base_dataset]["mean"] + target_multiplier * logp_stats[base_dataset]["std"]
        best = hpo_metrics.filter(abs(pl.col("meta_target") - target) < 0.1)
        if best.is_empty():
            print(f"No metrics found for target: {target}")
            continue
        best = best.sort("value", descending=True).limit(n=1)
        cfg = {
            "lr": best["lr"][0],
            "steps": int(best["steps"][0]),
            "scheduler": best["scheduler"][0],
            "lambda_lo": best["lambda_lo"][0],
            "lambda_hi": best["lambda_hi"][0],
            "the_metric": best["value"][0],
            "base_dataset": base_dataset,
            "dataset": dataset,
            "n_samples": args.n_samples,
            "target": target,
            "epsilon": 0.33 * logp_stats[base_dataset]["std"],
            "draw": False,
            "plot": True,
        }
        pprint(cfg)
        os.environ["GEN_MODEL"] = best["gen_model"][0]
        res = eval_cond_gen(cfg=cfg, decoder_settings=decoder_settings)
