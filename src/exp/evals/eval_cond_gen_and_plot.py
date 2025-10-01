import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from pprint import pprint
from typing import Any

import normflows as nf
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
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
evaluator = GenerationEvaluator(base_dataset="qm9", device=device)


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
    target = 0.0
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

    t_gen = time.perf_counter() - t0_gen
    # ---- Success@epsilon (using model's prediction; replace with RDKit eval if you want) ----
    hits = (y_pred - target).abs() <= epsilon
    success_rate = hits.float().mean().item()

    # Only evaluate the hits
    hits = [s for s, h in zip(hdc, hits, strict=True) if h]
    n, g = gen_model.split(torch.stack(hits))
    # n, g = gen_model.split(hdc)

    decoder_settings = {
        "beam_size": 64,
        "use_pair_feasibility": True,
        "expand_on_n_anchors": 9,
    }
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
        "gen_time_per_sample": t_gen / n_samples,
        "target": 0.0,
        "min_gda_loss": min_loss,
        "lambda_scheduler": sched_name,
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
    evals = evaluator.evaluate_conditional(
        samples=nx_graphs, target=target, final_flags=final_flags, eps=epsilon, total_samples=n_samples
    )
    results.update({f"eval_{k}": v for k, v in evals.items()})
    pprint(results)

    if cfg.get("draw", False):
        base_dir = GLOBAL_ARTEFACTS_PATH / "cond_generation" / f"drawings_valid_target_{target:.1f}"
        base_dir.mkdir(parents=True, exist_ok=True)
        mols, valid_flags = evaluator.get_mols_and_valid_flags()
        for i, (mol, valid) in enumerate(zip(mols, valid_flags, strict=False)):
            if valid:
                logp = rdkit_logp(mol)
                if abs(logp - target) > epsilon:
                    out = (
                        base_dir
                        / f"{gen_ckpt_path.parent.parent.stem}__{os.getenv('CLASSIFIER')}__logp{logp:.3f}{i}.png"
                    )
                    draw_mol(mol=mol, save_path=out, fmt="png")

    if cfg.get("plot", False):
        # --- dataset stats ---
        import numpy as np

        dataset = QM9Smiles(split="train") if base_dataset == "qm9" else ZincSmiles(split="train")
        logp_list = np.array([d.logp.item() for d in dataset], dtype=float)
        ds_summary = stats(logp_list)

        # --- generated stats ---
        mols, valid_flags = evaluator.get_mols_and_valid_flags()
        logp_gen_list = np.array([rdkit_logp(mol) for mol in mols], dtype=float)
        gen_summary = (
            stats(logp_gen_list) if logp_gen_list.size else {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "std": 0.0}
        )

        # --- histogram settings (aligned bins, density) ---
        # Extend range slightly so edges aren't cramped
        xmin = float(min(logp_list.min(), logp_gen_list.min() if logp_gen_list.size else logp_list.min())) - 0.25
        xmax = float(max(logp_list.max(), logp_gen_list.max() if logp_gen_list.size else logp_list.max())) + 0.25

        n_bins_ds = 100
        n_bins_gen = 50
        bins_ds = np.linspace(xmin, xmax, n_bins_ds + 1)
        bins_gen = np.linspace(xmin, xmax, n_bins_gen + 1)

        # --- simple Gaussian smoothing for prettier curves (no extra deps) ---
        def smooth_hist(y, sigma_bins=2.0):
            if len(y) < 3:
                return y
            # discrete Gaussian kernel over bin indices
            radius = int(max(1, round(3 * sigma_bins)))
            xk = np.arange(-radius, radius + 1)
            kernel = np.exp(-0.5 * (xk / float(sigma_bins)) ** 2)
            kernel /= kernel.sum()
            return np.convolve(y, kernel, mode="same")

        # compute densities
        ds_counts, ds_edges = np.histogram(logp_list, bins=bins_ds, density=True)
        ds_centers = 0.5 * (ds_edges[:-1] + ds_edges[1:])
        ds_smooth = smooth_hist(ds_counts, sigma_bins=2.0)

        if logp_gen_list.size:
            gen_counts, gen_edges = np.histogram(logp_gen_list, bins=bins_gen, density=True)
            gen_centers = 0.5 * (gen_edges[:-1] + gen_edges[1:])
            gen_smooth = smooth_hist(gen_counts, sigma_bins=1.5)
        else:
            gen_counts = np.array([])
            gen_centers = np.array([])
            gen_smooth = np.array([])

        # --- figure layout: bigger plot area, compact table below ---
        fig = plt.figure(figsize=(11, 6))
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[7, 2], hspace=0.28)
        ax = fig.add_subplot(gs[0, 0])

        # Background dataset: soft fill + smooth curve
        ax.hist(
            logp_list, bins=bins_ds, density=True, histtype="stepfilled", alpha=0.25, label="_nolegend_"
        )  # exclude from legend
        ax.plot(ds_centers, ds_smooth, linewidth=1.5, label="_nolegend_")  # dataset smooth

        # Generated overlay: step bars + smooth curve
        if logp_gen_list.size:
            ax.hist(logp_gen_list, bins=bins_gen, density=True, histtype="step", linewidth=2.0, label="_nolegend_")
            ax.plot(gen_centers, gen_smooth, linewidth=2.0, label="_nolegend_")

        # Target band (±epsilon)
        ax.axvspan(target - epsilon, target + epsilon, alpha=0.10, label="_nolegend_")
        # annotate target beneath x-axis for clarity
        ax.annotate(
            f"target={target:.2f} ± {epsilon:.2f}",
            xy=(target, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -20),
            textcoords="offset points",
            ha="center",
            va="top",
        )

        # Labels & limits
        ax.set_xlim(xmin, xmax)
        ax.set_ylabel("Density")
        ax.set_xlabel("cLogP (RDKit)")
        ax.set_title(f"cLogP Distribution — {base_dataset.upper()} vs Generated (n={len(logp_gen_list)})")

        # Legend via proxies (avoid patch legend bugs)
        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D([0], [0], linewidth=1.5, linestyle="-", label=f"{base_dataset.upper()} train (smoothed)"),
            Line2D([0], [0], linewidth=2.0, linestyle="-", label="Generated (smoothed)"),
        ]
        if logp_gen_list.size:
            ax.legend(handles=legend_handles, loc="upper right")

        # --- selective table (no clutter) ---
        # Pull key eval metrics (fall back gracefully if missing)
        getf = lambda k, dflt=np.nan: evals.get(k, dflt)
        table_rows = [
            ("validity", f"{getf('validity'):.3f}"),
            ("success@eps", f"{getf('success@eps'):.3f}"),
            ("final_success@eps", f"{getf('final_success@eps'):.3f}"),
            ("mae_to_target", f"{getf('mae_to_target'):.3f}"),
            ("uniq_overall", f"{getf('uniqueness_overall'):.3f}"),
            ("novelty_overall", f"{getf('novelty_overall'):.3f}"),
            ("diversity_hits", f"{getf('diversity_hits'):.3f}"),
            ("gen_mean±std", f"{gen_summary['mean']:.2f} ± {gen_summary['std']:.2f}"),
            ("dataset_mean±std", f"{ds_summary['mean']:.2f} ± {ds_summary['std']:.2f}"),
            ("n_samples", f"{int(getf('n_samples', len(logp_gen_list)))}"),
        ]

        ax_tbl = fig.add_subplot(gs[1, 0])
        ax_tbl.axis("off")
        table = ax_tbl.table(
            cellText=[(k, v) for k, v in table_rows],
            colLabels=["Metric", "Value"],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.1)

        # Save + show
        out_dir = GLOBAL_ARTEFACTS_PATH / "cond_generation"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"logp_overlay_{base_dataset}_{int(time.time())}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.show()

        # Combined normalized histogram: dataset vs generated
        if len(logp_list) and logp_gen_list.size:
            import numpy as np
            from matplotlib.lines import Line2D

            # common range & bins
            xmin = float(min(np.min(logp_list), np.min(logp_gen_list)))
            xmax = float(max(np.max(logp_list), np.max(logp_gen_list)))
            bins = np.linspace(xmin, xmax, 51)  # 50 bins, shared

            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(1, 1, 1)

            # plot (exclude from legend to avoid patch quirks)
            ax.hist(logp_list, bins=bins, density=True, histtype="step", linewidth=1.5, label="_nolegend_")
            ax.hist(logp_gen_list, bins=bins, density=True, histtype="step", linewidth=2.0, label="_nolegend_")

            ax.set_title("Dataset vs Generated — normalized cLogP")
            ax.set_xlabel("cLogP (RDKit)")
            ax.set_ylabel("density")

            # proxy legend (stable)
            handles = [
                Line2D([0], [0], linestyle="-", linewidth=1.5, label=f"{base_dataset.upper()}"),
                Line2D([0], [0], linestyle="-", linewidth=2.0, label="Generated"),
            ]
            ax.legend(handles=handles, loc="upper right")

            plt.tight_layout()
            plt.show()
        elif len(logp_list):
            # fallback: only dataset available
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(1, 1, 1)
            ax.hist(logp_list, bins=50, density=True, histtype="step", linewidth=1.5)
            ax.set_title(f"{base_dataset.upper()} – logP distribution (normalized)")
            ax.set_xlabel("cLogP (RDKit)")
            ax.set_ylabel("density")
            plt.tight_layout()
            plt.show()
        elif logp_gen_list.size:
            # fallback: only generated available
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(1, 1, 1)
            ax.hist(logp_gen_list, bins=50, density=True, histtype="step", linewidth=2.0)
            ax.set_title("Generated – logP distribution (normalized)")
            ax.set_xlabel("cLogP (RDKit)")
            ax.set_ylabel("density")
            plt.tight_layout()
            plt.show()

    return results


if __name__ == "__main__":
    cfg = {
        "lr": 0.0001535528683888,
        # "lr": 0.001,
        "steps": 1429,
        # "steps": 1000,
        "scheduler": "cosine",
        "lambda_lo": 0.0001840899208055,
        "lambda_hi": 0.0052054096619994,
        "draw": False,
        "n_samples": 500,
        "base_dataset": "qm9",
        "plot": True,
    }
    os.environ["GEN_MODEL"] = "nvp_qm9_h1600_f12_hid1024_s42_lr5e-4_wd0.0_an"
    os.environ["CLASSIFIER"] = "SIMPLE_VOTER"
    res = eval_cond_gen(cfg=cfg)
