#!/usr/bin/env python
"""
Spline Flow (Neural Spline Flow) Model Training Script

This script defines:
1.  The SFConfig dataclass for configuration.
2.  The SplineFlowLightning (PyTorch Lightning) model.
3.  The main `run_experiment` function, which trains and evaluates
    a model based on a given config.

This script is designed to be *imported* by an HPO script,
but can also be run directly for a single test.
"""

import datetime
import json
import os
import random
import shutil
import string
import time
from dataclasses import asdict
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.loader import DataLoader
from torchhd import HRRTensor

from src.datasets.utils import get_split
from src.encoding.correction_utilities import target_reached
from src.encoding.graph_encoders import CorrectionLevel, load_or_create_hypernet
from src.normalizing_flow.models import SFConfig
from src.utils.registery import resolve_model, retrieve_model

# --- AbstractNFModel (Base class from your NVP script) ---
# This is required for SplineFlowLightning to inherit from
from src.utils.utils import GLOBAL_MODEL_PATH, pick_device

DTYPE = torch.float32
torch.set_default_dtype(DTYPE)


# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def setup_exp(dir_name: str | None = None) -> dict:
    script_path = Path(__file__).resolve()
    experiments_path = script_path.parent
    script_stem = script_path.stem

    base_dir = experiments_path / "results" / script_stem
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Setting up experiment in {base_dir}")
    if dir_name:
        exp_dir = base_dir / dir_name
    else:
        slug = f"{datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d_%H-%M-%S')}_{''.join(random.choices(string.ascii_lowercase, k=4))}"
        exp_dir = base_dir / slug
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory created: {exp_dir}")

    dirs = {
        "exp_dir": exp_dir,
        "models_dir": exp_dir / "models",
        "evals_dir": exp_dir / "evaluations",
        "artefacts_dir": exp_dir / "artefacts",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy(script_path, exp_dir / script_path.name)
        print(f"Saved a copy of the script to {exp_dir / script_path.name}")
    except Exception as e:
        print(f"Warning: Failed to save script copy: {e}")

    return dirs


# ---------------------------------------------------------------------
# Standardization, Plotting, and Helper Functions
# (These are identical to your script)
# ---------------------------------------------------------------------


@torch.no_grad()
def fit_featurewise_standardization(
    model, loader, hv_count: int, hv_dim: int, max_batches: int | None = None, device="cpu"
):
    cnt = 0
    accum_dtype = DTYPE
    sum_vec = torch.zeros(hv_count * hv_dim, dtype=accum_dtype, device=device)
    sumsq_vec = torch.zeros(hv_count * hv_dim, dtype=accum_dtype, device=device)
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = batch.to(device)
        x = model._flat_from_batch(batch).to(accum_dtype)
        cnt += x.shape[0]
        sum_vec += x.sum(dim=0)
        sumsq_vec += (x * x).sum(dim=0)
    if cnt == 0:
        raise RuntimeError("fit_featurewise_standardization(): empty loader.")
    mu = sum_vec / cnt
    var = (sumsq_vec / cnt - mu**2).clamp_min_(0)
    sigma = var.sqrt().clamp_min_(1e-6)
    tgt_dtype = model.mu.dtype if hasattr(model, "mu") else torch.get_default_dtype()
    model.set_standardization(mu.to(tgt_dtype), sigma.to(tgt_dtype))


@torch.no_grad()
def fit_per_term_standardization(model, loader, hv_dim: int, max_batches: int | None = None, device="cpu"):
    cnt = 0
    accum_dtype = DTYPE
    sum_edge = torch.zeros(hv_dim, dtype=accum_dtype, device=device)
    sumsq_edge = torch.zeros(hv_dim, dtype=accum_dtype, device=device)
    sum_graph = torch.zeros(hv_dim, dtype=accum_dtype, device=device)
    sumsq_graph = torch.zeros(hv_dim, dtype=accum_dtype, device=device)
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = batch.to(device)
        edge = batch.edge_terms.to(accum_dtype).view(-1, hv_dim)
        graph = batch.graph_terms.to(accum_dtype).view(-1, hv_dim)
        cnt += edge.shape[0]
        sum_edge += edge.sum(dim=0)
        sumsq_edge += (edge * edge).sum(dim=0)
        sum_graph += graph.sum(dim=0)
        sumsq_graph += (graph * graph).sum(dim=0)
    if cnt == 0:
        raise RuntimeError("fit_per_term_standardization(): empty loader.")
    mu_edge = sum_edge / cnt
    var_edge = (sumsq_edge / cnt - mu_edge**2).clamp_min_(0)
    sigma_edge = var_edge.sqrt().clamp_min_(1e-6)
    mu_graph = sum_graph / cnt
    var_graph = (sumsq_graph / cnt - mu_graph**2).clamp_min_(0)
    sigma_graph = var_graph.sqrt().clamp_min_(1e-6)
    mu = torch.cat([mu_edge, mu_graph])
    sigma = torch.cat([sigma_edge, sigma_graph])
    tgt_dtype = model.mu.dtype if hasattr(model, "mu") else torch.get_default_dtype()
    model.set_standardization(mu.to(tgt_dtype), sigma.to(tgt_dtype))
    model._per_term_split = hv_dim
    log(f"Per-term standardization fitted: edge_terms [:{hv_dim}], graph_terms [{hv_dim}:]")


def plot_train_val_loss(
    df: pd.DataFrame,
    artefacts_dir: Path,
    *,
    skip_first: float = 0.1,
    min_epoch: int | None = None,
    smooth_window: int | None = None,
    clip_q: tuple[float, float] = (0.02, 0.98),
    logy: bool = False,
) -> None:
    def _first_existing(cols: list[str]) -> str | None:
        for c in cols:
            if c in df.columns:
                return c
        return None

    epoch_col = _first_existing(["epoch", "step"])
    train_col = _first_existing(["train_loss_epoch", "train_loss", "epoch_train_loss"])
    val_col = _first_existing(["val_loss", "val/loss", "validation_loss"])
    if epoch_col is None or (train_col is None and val_col is None):
        print("No epoch/metric columns; skipping plot.")
        return
    uniq = pd.unique(df[epoch_col].dropna())
    try:
        uniq = np.sort(uniq.astype(int))
    except Exception:
        uniq = np.sort(uniq)
    if min_epoch is not None:
        cutoff = min_epoch
    else:
        if isinstance(skip_first, float) and 0 < skip_first < 1:
            k = int(round(skip_first * len(uniq)))
        else:
            k = int(skip_first)
        k = max(0, min(k, max(len(uniq) - 1, 0)))
        cutoff = uniq[k] if len(uniq) else 0
    df_f = df[df[epoch_col] >= cutoff].copy()

    def _series(col: str | None):
        if col is None or col not in df_f:
            return None
        s = (
            df_f[[epoch_col, col]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .groupby(epoch_col, as_index=False)[col]
            .mean()
            .rename(columns={col: "y"})
        )
        if s.empty:
            return None
        if smooth_window and smooth_window > 1:
            s["y"] = s["y"].rolling(smooth_window, min_periods=1).mean()
        return s

    train_s, val_s = _series(train_col), _series(val_col)
    if train_s is None and val_s is None:
        print("Nothing to plot after filtering; skipping.")
        return
    plt.figure(figsize=(10, 6))
    if train_s is not None:
        plt.plot(train_s[epoch_col].to_numpy(), train_s["y"].to_numpy(), label=f"{train_col} (≥{cutoff})")
    if val_s is not None:
        plt.plot(val_s[epoch_col].to_numpy(), val_s["y"].to_numpy(), label=f"{val_col} (≥{cutoff})")
    ys = []
    if train_s is not None:
        ys.append(train_s["y"].to_numpy())
    if val_s is not None:
        ys.append(val_s["y"].to_numpy())
    if ys:
        y_all = np.concatenate(ys)
        if y_all.size:
            qlo, qhi = np.quantile(y_all, clip_q)
            if np.isfinite(qlo) and np.isfinite(qhi) and qhi > qlo:
                plt.ylim(qlo, qhi)
    if logy:
        yb, yt = plt.gca().get_ylim()
        if yb > 0 and yt > 0:
            plt.yscale("log")
        else:
            print("logy requested but data ≤ 0; using linear scale.")
    plt.xlabel(epoch_col)
    plt.ylabel("loss")
    plt.title("Train vs. Validation Loss")
    plt.legend()
    plt.tight_layout()
    artefacts_dir.mkdir(parents=True, exist_ok=True)
    out = artefacts_dir / "loss_curve.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved train/val loss plot to {out} (cutoff ≥ {cutoff})")


def on_a100() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:
        return False
    return "A100" in name


def pick_precision() -> int | str:
    p = os.getenv("PRECISION")
    if p:
        return int(p) if p.isdigit() else p
    if torch.cuda.is_available() and (on_a100() or torch.cuda.is_bf16_supported()):
        return "bf16-mixed"
    if torch.cuda.is_available():
        return "16-mixed"
    return 32


def configure_tf32(precision):
    if not torch.cuda.is_available():
        return
    name = torch.cuda.get_device_name(0)
    if "A100" in name or "H100" in name or "RTX 30" in name or "RTX 40" in name:
        if precision in ("bf16-mixed", 32, "32"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"[TF32] Enabled for {precision} on {name}")
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            print(f"[TF32] Disabled for {precision}")
    else:
        print(f"[TF32] Not supported on {name}")


class TimeLoggingCallback(Callback):
    def setup(self, trainer, pl_module, stage=None):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.start_time
        trainer.logger.log_metrics({"elapsed_time_sec": elapsed}, step=trainer.current_epoch)


def _hist(figpath: Path, data: np.ndarray, title: str, xlabel: str, bins: int = 80):
    arr = np.asarray(data).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        log(f"Skip hist {figpath.name}: empty/non-finite data")
        return
    vmin, vmax = float(arr.min()), float(arr.max())
    if np.isclose(vmin, vmax):
        span = 1.0 if vmin == 0.0 else 0.05 * abs(vmin)
        vmin, vmax, bins = vmin - span, vmax + span, 1
    plt.figure(figsize=(6, 4))
    plt.hist(arr, bins=bins, range=(vmin, vmax))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()


# ---------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------


def run_experiment(cfg: SFConfig) -> tuple[float, float]:
    """
    Runs a single training experiment for SplineFlow.
    Returns:
        (min_val_loss, incorrect_pct) for multi-objective HPO.
    """
    pprint(asdict(cfg))
    log("Starting run_experiment for SplineFlow...")
    dirs = setup_exp(cfg.exp_dir_name)
    exp_dir = Path(dirs["exp_dir"])
    models_dir = Path(dirs["models_dir"])
    evals_dir = Path(dirs["evals_dir"])
    artefacts_dir = Path(dirs["artefacts_dir"])

    (evals_dir / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
    seed_everything(cfg.seed)

    ds_cfg = cfg.dataset.default_cfg
    device = pick_device()
    log(f"Using device: {device!s}")

    log(f"Loading {ds_cfg.base_dataset} pair datasets.")
    train_dataset = get_split(split="train", ds_config=ds_cfg)
    validation_dataset = get_split(split="valid", ds_config=ds_cfg)
    train_dataset.data.edge_terms = train_dataset.data.edge_terms.to(DTYPE)
    train_dataset.data.graph_terms = train_dataset.data.graph_terms.to(DTYPE)
    validation_dataset.data.edge_terms = validation_dataset.data.edge_terms.to(DTYPE)
    validation_dataset.data.graph_terms = validation_dataset.data.graph_terms.to(DTYPE)

    log("Loading hypernet...")
    hypernet = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, cfg=ds_cfg).to(device=device, dtype=DTYPE).eval()
    hypernet.depth = ds_cfg.hypernet_depth
    log("Hypernet ready.")

    num_workers = 14 if os.getenv("CLUSTER") != "local" else 8
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
        drop_last=True,
        prefetch_factor=6,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
        drop_last=False,
        prefetch_factor=4,
    )
    log(f"Datasets ready. train={len(train_dataset)} valid={len(validation_dataset)}")

    model = resolve_model("SplineFlow", cfg=cfg).to(device=device)  # <-- This was the bug
    log(f"Model: {model!s}")

    if cfg.per_term_standardization:
        log("Using per-term standardization")
        fit_per_term_standardization(model, train_dataloader, hv_dim=cfg.hv_dim, device=device)
    else:
        log("Using global standardization")
        fit_featurewise_standardization(
            model, train_dataloader, hv_count=cfg.hv_count, hv_dim=cfg.hv_dim, device=device
        )

    csv_logger = CSVLogger(save_dir=str(evals_dir), name="logs")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath=str(models_dir),
        auto_insert_metric_name=False,
        filename="epoch{epoch:02d}-val{val_loss:.4f}",
        save_last=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    time_logger = TimeLoggingCallback()  # Make sure TimeLoggingCallback is defined
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=50,
        min_delta=0.0,
        check_finite=True,
        verbose=True,
    )
    precision = pick_precision()  # Make sure pick_precision is defined
    log(f"Using precision {precision!s}")
    configure_tf32(precision)  # Make sure configure_tf32 is defined

    trainer = Trainer(
        max_epochs=cfg.epochs,
        logger=[csv_logger],
        callbacks=[checkpoint_callback, lr_monitor, time_logger, early_stopping],
        default_root_dir=str(exp_dir),
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        log_every_n_steps=500,
        precision=precision,
        num_sanity_val_steps=0,
    )

    t_start = time.perf_counter()
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

    metrics_path = Path(csv_logger.log_dir) / "metrics.csv"
    min_val_loss = float("inf")
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        if "val_loss" in df.columns and not df["val_loss"].isnull().all():
            min_val_loss = df["val_loss"].min()
        df.to_parquet(evals_dir / "metrics.parquet", index=False)
        plot_train_val_loss(df, artefacts_dir, logy=True)  # Make sure plot_train_val_loss is defined

    best_path = checkpoint_callback.best_model_path
    if (not best_path) or ("nan" in Path(best_path).name) or (not Path(best_path).exists()):
        best_path = checkpoint_callback.last_model_path

    if not best_path or not Path(best_path).exists():
        log("No checkpoint found. Skipping post-training analysis.")
        return float("inf"), 100.0

    log(f"Loading best checkpoint: {best_path}")

    best_model = retrieve_model("SplineFlow").load_from_checkpoint(best_path)  # <-- This was the bug

    best_model.to(device).eval()
    best_model.to(dtype=DTYPE)

    with torch.no_grad():
        samples = best_model.sample_split(1000)
        edge_s = samples["edge_terms"]
        graph_s = samples["graph_terms"]

    edge_s = edge_s.as_subclass(HRRTensor)

    log("=== Starting generation quality evaluation ===")
    base_dataset = ds_cfg.base_dataset
    n_decode = 1000 if base_dataset == "qm9" else 100
    log(f"Decoding {n_decode} samples for {base_dataset} dataset...")
    edge_decode = edge_s[:n_decode]

    hypernet.base_dataset = base_dataset
    correction_levels = []
    decode_start_time = time.time()

    for i in range(n_decode):
        if (i + 1) % 100 == 0:
            log(f"Decoding sample {i + 1}/{n_decode}...")
        try:
            initial_decoded_edges = hypernet.decode_order_one_no_node_terms(edge_decode[i])
            if target_reached(initial_decoded_edges):
                correction_levels.append(CorrectionLevel.ZERO)
            else:
                correction_levels.append(CorrectionLevel.FAIL)
        except Exception:
            correction_levels.append(CorrectionLevel.FAIL)

    decode_elapsed = time.time() - decode_start_time
    log(f"Decoding completed in {decode_elapsed:.2f}s")

    n_zero_corrections = sum(1 for cl in correction_levels if cl == CorrectionLevel.ZERO)
    zero_correction_pct = 100.0 * n_zero_corrections / n_decode if n_decode else 0.0
    incorrect_pct = 100.0 - zero_correction_pct

    log(f"CorrectionLevel.ZERO percentage: {zero_correction_pct:.2f}%")
    log(f"Incorrect percentage: {incorrect_pct:.2f}%")

    metrics_dict = {
        "exp_dir_name": cfg.exp_dir_name,
        "min_val_loss": float(min_val_loss),
        "zero_correction_pct": float(zero_correction_pct),
        "incorrect_pct": float(incorrect_pct),
        # ... (rest of metrics) ...
    }
    metrics_file = evals_dir / "hpo_metrics.json"
    metrics_file.write_text(json.dumps(metrics_dict, indent=2, default=str))

    log("Experiment completed.")
    log(f"Best val NLL (loss): {min_val_loss:.4f}")
    log(f"Generation Quality (CorrectionLevel.ZERO): {zero_correction_pct:.2f}%")

    return float(min_val_loss), float(incorrect_pct)
