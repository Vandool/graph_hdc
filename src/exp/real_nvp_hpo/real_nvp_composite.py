"""
Real NVP Training Script with Composite NLL-Validity Metric

This is a modified version of real_nvp.py that incorporates molecular validity
into the optimization objective by decoding samples and computing a composite
metric that balances negative log-likelihood (NLL) and validity.

Composite Metric:
    composite = α × NLL + (1-α) × (100 - validity_pct)

Where α=0.5 gives equal weight to distribution learning and valid molecule generation.

Example Usage:
==============
# Default (BF16, compiled):
python real_nvp_composite.py

# Full precision for validation:
PRECISION=64 python real_nvp_composite.py

# Larger effective batch size:
GRAD_ACCUM=4 python real_nvp_composite.py

# Maximum optimization:
COMPILE_MODE=max-autotune GRAD_ACCUM=2 python real_nvp_composite.py

Expected Speedup: 2-4x faster training on A100/H100 GPUs
"""

import datetime
import enum
import json
import math
import os
import random
import shutil
import string
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import optuna
import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torchhd import HRRTensor

from src.datasets.utils import get_split
from src.encoding.configs_and_constants import Features, SupportedDataset
from src.encoding.graph_encoders import CorrectionLevel, load_or_create_hypernet
from src.encoding.the_types import VSAModel
from src.exp.real_nvp_hpo.hpo.folder_name import make_run_folder_name
from src.generation.analyze import analyze_terms_only
from src.utils.registery import resolve_model, retrieve_model
from src.utils.utils import GLOBAL_MODEL_PATH, pick_device

LOCAL_DEV = "LOCAL_HDC_miss"
PROJECT_NAME = "real_nvp_v2"
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)


# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
# ---------- tiny logger ----------
def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

if os.getenv("CLUSTER") == "local":
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"


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
# Model
# ---------------------------------------------------------------------


@dataclass
class FlowConfig:
    exp_dir_name: str | None = None
    seed: int = 42
    epochs: int = 800
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.0
    # is_dev: bool = os.getenv("IS_DEV", "0") == "1"
    is_dev = False
    # HDC / encoder
    hv_count: int = 3
    hv_dim: int = 40 * 40  # 1600
    vsa: VSAModel = VSAModel.HRR
    dataset: SupportedDataset = SupportedDataset.QM9_SMILES_HRR_1600

    num_flows: int = 4
    num_hidden_channels: int = 256

    smax_initial: float = 1.0
    smax_final: float = 4
    smax_warmup_epochs: float = 10

    use_act_norm: bool = True

    # Checkpointing
    continue_from: Path | None = None
    resume_retrain_last_epoch: bool = False

    # Composite metric configuration
    n_decode_override: int | None = None  # Override default decode count (1000 for QM9, 100 for ZINC)

    # Per-term standardization: separate mu/sigma for edge_terms and graph_terms
    per_term_standardization: bool = True


@torch.no_grad()
def fit_featurewise_standardization(
    model, loader, hv_count: int, hv_dim: int, max_batches: int | None = None, device="cpu"
):
    """
    Estimate per-feature mean and std (feature-wise) for the model's standardized space.
    Works safely under mixed-precision and supports [B, 3D] inputs.

    Parameters
    ----------
    model : RealNVPV2Lightning
        Model whose `_flat_from_batch` defines the feature layout.
    loader : DataLoader
        Provides graph batches.
    hv_dim : int
        Hypervector dimension D.
    max_batches : int | None
        Limit number of batches for faster estimation.
    device : str | torch.device
        Device to perform accumulation on.
    """
    cnt = 0
    accum_dtype = DTYPE
    sum_vec = torch.zeros(hv_count * hv_dim, dtype=accum_dtype, device=device)
    sumsq_vec = torch.zeros(hv_count * hv_dim, dtype=accum_dtype, device=device)

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        batch = batch.to(device)
        x = model._flat_from_batch(batch).to(accum_dtype)  # [B, 3D]
        cnt += x.shape[0]
        sum_vec += x.sum(dim=0)
        sumsq_vec += (x * x).sum(dim=0)

    if cnt == 0:
        msg = "fit_featurewise_standardization(): empty loader or no samples."
        raise RuntimeError(msg)

    mu = sum_vec / cnt
    var = (sumsq_vec / cnt - mu**2).clamp_min_(0)
    sigma = var.sqrt().clamp_min_(1e-6)

    # Match model's dtype to avoid Double↔Half/Float mismatches
    tgt_dtype = model.mu.dtype if hasattr(model, "mu") else torch.get_default_dtype()
    model.set_standardization(mu.to(tgt_dtype), sigma.to(tgt_dtype))


@torch.no_grad()
def fit_per_term_standardization(model, loader, hv_dim: int, max_batches: int | None = None, device="cpu"):
    """
    Compute separate standardization for edge_terms and graph_terms.
    Each term gets its own mu and sigma vectors, then concatenated.

    Parameters
    ----------
    model : RealNVPV2Lightning
        Model whose buffers will be updated.
    loader : DataLoader
        Provides graph batches.
    hv_dim : int
        Hypervector dimension D (NOT 2*D).
    max_batches : int | None
        Limit number of batches for faster estimation.
    device : str | torch.device
        Device to perform accumulation on.
    """
    cnt = 0
    accum_dtype = DTYPE

    # Separate accumulators for edge and graph terms
    sum_edge = torch.zeros(hv_dim, dtype=accum_dtype, device=device)
    sumsq_edge = torch.zeros(hv_dim, dtype=accum_dtype, device=device)
    sum_graph = torch.zeros(hv_dim, dtype=accum_dtype, device=device)
    sumsq_graph = torch.zeros(hv_dim, dtype=accum_dtype, device=device)

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        batch = batch.to(device)
        edge = batch.edge_terms.to(accum_dtype)  # [B, D]
        graph = batch.graph_terms.to(accum_dtype)  # [B, D]

        cnt += edge.shape[0]

        sum_edge += edge.sum(dim=0)
        sumsq_edge += (edge * edge).sum(dim=0)
        sum_graph += graph.sum(dim=0)
        sumsq_graph += (graph * graph).sum(dim=0)

    if cnt == 0:
        msg = "fit_per_term_standardization(): empty loader or no samples."
        raise RuntimeError(msg)

    # Compute statistics for each term
    mu_edge = sum_edge / cnt
    var_edge = (sumsq_edge / cnt - mu_edge**2).clamp_min_(0)
    sigma_edge = var_edge.sqrt().clamp_min_(1e-6)

    mu_graph = sum_graph / cnt
    var_graph = (sumsq_graph / cnt - mu_graph**2).clamp_min_(0)
    sigma_graph = var_graph.sqrt().clamp_min_(1e-6)

    # Concatenate to match model's expected [2*D] format
    mu = torch.cat([mu_edge, mu_graph])
    sigma = torch.cat([sigma_edge, sigma_graph])

    # Set in model (using existing method)
    tgt_dtype = model.mu.dtype if hasattr(model, "mu") else torch.get_default_dtype()
    model.set_standardization(mu.to(tgt_dtype), sigma.to(tgt_dtype))

    # Store the split point for per-term transforms
    model._per_term_split = hv_dim
    log(f"Per-term standardization fitted: edge_terms [:{hv_dim}], graph_terms [{hv_dim}:]")


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------


def plot_train_val_loss(
    df: pd.DataFrame,
    artefacts_dir: Path,
    *,
    skip_first: float = 0.1,  # int = epochs to skip; float in (0,1) = fraction to skip
    min_epoch: int | None = None,  # overrides skip_first if set
    smooth_window: int | None = None,  # rolling mean window (points)
    clip_q: tuple[float, float] = (0.02, 0.98),  # robust y-limits from quantiles
    logy: bool = False,
) -> None:
    """Plot train/val loss robustly (drops non-finite, clips outliers, averages per epoch)."""

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

    # --- define cutoff ---
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

    train_s = _series(train_col)
    val_s = _series(val_col)

    if train_s is None and val_s is None:
        print("Nothing to plot after filtering; skipping.")
        return

    plt.figure(figsize=(10, 6))
    if train_s is not None:
        plt.plot(train_s[epoch_col].to_numpy(), train_s["y"].to_numpy(), label=f"{train_col} (≥{cutoff})")
    if val_s is not None:
        plt.plot(val_s[epoch_col].to_numpy(), val_s["y"].to_numpy(), label=f"{val_col} (≥{cutoff})")

    # --- robust y-limits (two-sided quantiles on combined series) ---
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

    # --- log scale only if strictly positive ---
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
    return "A100" in name  # simple & reliable enough


def pick_precision() -> int | str:
    """
    Choose training precision. BF16-mixed provides 2-3x speedup on A100/H100 GPUs
    while maintaining stability for most deep learning workloads.

    Override via PRECISION env var: PRECISION=64 for full FP64 if needed
    """
    # explicit override via env if you want: PRECISION in {"64","32","bf16-mixed","16-mixed"}
    p = os.getenv("PRECISION")
    if p:  # trust user override
        return int(p) if p.isdigit() else p

    # Default to BF16-mixed on modern GPUs for 2-3x speedup
    # Set PRECISION=64 if you need full FP64 precision
    if torch.cuda.is_available() and on_a100():
        return "bf16-mixed"  # 2-3x faster than FP64 on A100/H100
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bf16-mixed"  # local GPUs: fast & stable
    if torch.cuda.is_available():
        return "16-mixed"  # widest fallback
    return 32


def configure_tf32(precision):
    """Enable TF32 only when using bf16-mixed or 32-bit precision on Ampere+ GPUs."""
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
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print(f"[TF32] Not supported on {name}")


class TimeLoggingCallback(Callback):
    def setup(self, trainer, pl_module, stage=None):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.start_time
        trainer.logger.log_metrics({"elapsed_time_sec": elapsed}, step=trainer.current_epoch)


class PeriodicNLLEval(Callback):
    def __init__(
        self,
        val_loader,
        artefacts_dir: Path,
        every_n_epochs: int = 100,
        max_batches: int | None = 200,
        log_hist: bool = False,
    ):
        super().__init__()
        self.val_loader = val_loader
        self.artefacts_dir = Path(artefacts_dir)
        self.every = int(every_n_epochs)
        self.max_batches = max_batches
        self.log_hist = log_hist

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = int(trainer.current_epoch) + 1
        if self.every <= 0 or (epoch % self.every) != 0:
            return

        stats = _eval_flow_metrics(
            pl_module,
            self.val_loader,
            pl_module.device,
            hv_count=pl_module.hv_count,
            hv_dim=pl_module.D,
            max_batches=self.max_batches,
        )
        if not stats:
            return

        # persist arrays (with epoch in file for easier joins)
        self.artefacts_dir.mkdir(parents=True, exist_ok=True)
        # df = pd.DataFrame(stats)
        # df["epoch"] = epoch
        # df.to_parquet(self.artefacts_dir / f"val_metrics_epoch{epoch:04d}.parquet", index=False)

        # compact scalar summary with consistent names
        summary = _summarize_arrays(stats)
        log_payload = {f"val/{k}": v for k, v in summary.items()}
        log_payload["epoch"] = epoch

        # log scalars to all attached loggers (CSV & W&B if present)
        try:
            trainer.logger.log_metrics(log_payload, step=trainer.global_step)
        except Exception:
            log(log_payload)


def _norm_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


@torch.no_grad()
def _eval_flow_metrics(model, loader, device, hv_count: int, hv_dim: int, max_batches: int | None = None):
    """
    Eval for normalizing flows trained with exact likelihood.

    We report ONLY arrays with the same length so they can be dropped into a single
    pandas DataFrame without shape errors. Per-dimension diagnostics are summarized
    to scalars and *broadcast* to per-sample length (so they're easy to join & plot).

    Why these metrics (high level):
      • nll/bpd: core objective the model optimizes; should go down if the flow
        learns a better density than a naive Gaussian after your μ/σ standardization.
      • Gaussian baseline: tells you how much "structure" the flow actually learns
        beyond mean/variance re-scaling (important because your features have wildly
        different scales; standardization already fixes some of that).
      • Δbpd (model − Gaussian): the *value add* of the flow; negative values mean
        the flow beats the Gaussian baseline in bits-per-dim (good).
      • PIT KS & 90% coverage: calibration in base space; checks whether the learned
        transform truly maps data → N(0, I). If this fails, sampling and tail behavior
        can be off even if NLL looks okay.
      • sum_log_sigma_bits / const_bpd: isolates the contribution from your fixed
        standardization step; helps detect cases where most of the "good" bpd is due
        to μ/σ rather than the flow layers (a real risk with highly heteroscedastic data).

    Returns arrays (all same length) for easy DataFrame construction:
      bpd_model, bpd_gauss, delta_bpd, bpd_stdspace, nll_model,
      sum_log_sigma_bits, const_bpd,
      pit_ks_mean/std/max, cov90_abs_err_mean/std/max
    """
    model.eval()
    ln2 = math.log(2.0)
    dim = hv_count * hv_dim

    # log p(z) of a standard normal factorizes; this is the constant term in nats
    const_term = 0.5 * dim * math.log(2 * math.pi)

    nll_model_list, bpd_std_list, bpd_gauss_list = [], [], []
    pit_ks_list, cov90_err_list = [], []

    # Helper: push from standardized z to the base N(0, I) to test calibration.
    # We try the fast path (flow.inverse on the aggregate), then safely fall back
    # to layer-by-layer inversion. This ensures reversibility is actually exercised.
    def _to_base(flow, z_std: torch.Tensor) -> torch.Tensor:
        if hasattr(flow, "inverse"):
            try:
                zb, _ = flow.inverse(z_std)
                return zb
            except Exception:
                pass
        zb = z_std
        for f in reversed(flow.flows):
            zb, _ = f.inverse(zb)
        return zb

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        batch = batch.to(device)
        mdtype = next(model.parameters()).dtype
        flat = model._flat_from_batch(batch).to(mdtype)  # [B, dim]

        # Standardize with your learned/fitted μ, σ.
        # This is a *determinantful* pre-transform; we track its log-det so that
        # likelihood is computed in the original data measure (not in z-space).
        z, log_det_corr = model._pretransform(flat)  # z ∼ data standardized

        # Likelihood under the FLOW in standardized space (the thing being learned).
        # This is the exact objective the flow trains on.
        nll_stdspace = -model.flow.log_prob(z)  # [B] nats, exact

        # Likelihood of a *plain Gaussian* in the same standardized space.
        # This is your "no flow layers" baseline: only μ/σ but no higher-order structure.
        quad = 0.5 * (z**2).sum(dim=1)  # ∑ z_i^2 / 2
        nll_gauss_stdspace = quad + const_term  # exact NLL of N(0, I)

        # Move both NLLs back to the *data* space by adding the pre-transform's
        # log-det correction. This keeps the comparison fair and truly reversible.
        nll_model = nll_stdspace + log_det_corr  # flow NLL in data space
        nll_gauss = nll_gauss_stdspace + log_det_corr  # Gaussian baseline in data space

        # Numerical hygiene: drop non-finite samples consistently.
        mask = torch.isfinite(nll_model) & torch.isfinite(nll_stdspace) & torch.isfinite(nll_gauss)
        if not mask.any():
            continue

        nll_model = nll_model[mask]
        nll_stdspace = nll_stdspace[mask]
        nll_gauss = nll_gauss[mask]
        z_masked = z[mask]

        # Bits-per-dim is the standard unit for flows (scale-invariant across dims).
        # bpd_stdspace reflects the flow's coding cost in the *standardized* space.
        bpd_stdspace = nll_stdspace / (dim * ln2)  # [B] bits/dim (model, standardized)
        bpd_gauss = nll_gauss / (dim * ln2)  # [B] bits/dim (Gaussian baseline, data space)

        nll_model_list.append(nll_model.detach().cpu())
        bpd_std_list.append(bpd_stdspace.detach().cpu())
        bpd_gauss_list.append(bpd_gauss.detach().cpu())

        # ---------------- Calibration diagnostics in base space ----------------
        # If the flow learned a correct, invertible map to N(0, I), then pushing
        # standardized data through "inverse" should yield z_base ~ N(0, I).
        z_base = _to_base(model.flow, z_masked)  # [B, dim]

        # PIT (Probability Integral Transform): Φ(z_base) should be Uniform(0,1).
        # KS statistic per-dimension quantifies mismatch; smaller is better.
        u = _norm_cdf(z_base).clamp_(0, 1)  # [B, dim] uniforms if calibrated

        # Compute KS on a random subset of dims to keep it cheap. We summarize later.
        Bm, D = u.shape
        take = min(256, D)  # Reduced from 512 for faster evaluation
        if take > 0 and Bm > 0:
            idx = torch.randperm(D, device=u.device)[:take]
            U = u[:, idx]  # [B, take]
            U_sorted, _ = torch.sort(U, dim=0)
            # Theoretical CDF grid for uniforms
            grid = (torch.arange(1, Bm + 1, device=u.device) / Bm).unsqueeze(1)  # [B, 1]
            ks = torch.max(torch.abs(U_sorted - grid), dim=0).values  # [take]
            pit_ks_list.append(ks.detach().cpu())

            # 90% central coverage per coordinate in base space should be ~0.90 for N(0,1).
            # This probes *tail calibration*. Important for your setting because
            # branch-2 had huge raw scale; after standardization + flow, bad tails would
            # show up as systematic under/over-coverage here.
            inside = (z_base.abs() <= 1.6448536269514722).float()  # [B, dim]
            cov = inside.mean(dim=0)  # per-dim coverage rate
            cov_err = (cov - 0.90).abs()
            cov90_err_list.append(cov_err.detach().cpu())

    # If nothing was collected (e.g., all masked), return empty dict.
    if not nll_model_list:
        return {}

    # Flatten per-sample metrics to numpy
    nll_model = torch.cat(nll_model_list).numpy()
    bpd_stdspace = torch.cat(bpd_std_list).numpy()
    bpd_gauss = torch.cat(bpd_gauss_list).numpy()

    # bpd_model is the final coding cost (in data space) the model assigns.
    bpd_model = nll_model / (dim * ln2)

    # Δbpd shows the *net* improvement of the learned flow over the Gaussian baseline.
    # Negative is better (fewer bits to code the data).
    delta_bpd = bpd_model - bpd_gauss

    # How much of your bpd is a constant due to standardization's log|σ|?
    # This is useful to diagnose when most "gain" comes from μ/σ rather than learned transforms.
    sum_log_sigma_bits = float(model.log_sigma.sum().item() / math.log(2.0))
    const_bpd = sum_log_sigma_bits / dim  # per-dim constant bits from the pre-transform

    # Summarize calibration diagnostics across dims (mean/std/max are enough).
    if pit_ks_list:
        pit_ks = torch.cat(pit_ks_list).numpy()
        pit_mean = float(np.mean(pit_ks))
        pit_std = float(np.std(pit_ks))
        pit_max = float(np.max(pit_ks))
    else:
        pit_mean = pit_std = pit_max = float("nan")

    if cov90_err_list:
        cov90 = torch.cat(cov90_err_list).numpy()
        cov_mean = float(np.mean(cov90))
        cov_std = float(np.std(cov90))
        cov_max = float(np.max(cov90))
    else:
        cov_mean = cov_std = cov_max = float("nan")

    # Broadcast scalar summaries so all columns align in a single DataFrame.
    def _broadcast(val: float, ref: np.ndarray) -> np.ndarray:
        return np.full_like(ref, val, dtype=np.float64)

    return {
        # Core objective (in data space): should decrease as the model learns structure.
        "bpd_model": bpd_model,  # per-sample bits-per-dim under the flow
        # Baseline with no flow layers: isolates the benefit of learning beyond μ/σ.
        "bpd_gauss": bpd_gauss,  # per-sample bits-per-dim under Gaussian baseline
        # Net benefit of the flow: negative = flow better than baseline (good).
        "delta_bpd": delta_bpd,  # per-sample improvement (model - Gaussian)
        # The same objective but in the *standardized* space (helps debug μ/σ vs flow).
        "bpd_stdspace": bpd_stdspace,  # per-sample bits-per-dim in standardized space
        # Raw NLL in data space (for completeness / alternative plotting).
        "nll_model": nll_model,  # per-sample negative log-likelihood (nats)
        # Constant contributions & calibration summaries (broadcast to match length):
        "sum_log_sigma_bits": _broadcast(sum_log_sigma_bits, bpd_model),  # total bits from log σ
        "const_bpd": _broadcast(const_bpd, bpd_model),  # per-dim constant bits
        # PIT KS (smaller is better): marginal calibration of base-space coordinates.
        "pit_ks_mean": _broadcast(pit_mean, bpd_model),
        "pit_ks_std": _broadcast(pit_std, bpd_model),
        "pit_ks_max": _broadcast(pit_max, bpd_model),
        # 90% coverage abs error (smaller is better): tail calibration in base space.
        "cov90_abs_err_mean": _broadcast(cov_mean, bpd_model),
        "cov90_abs_err_std": _broadcast(cov_std, bpd_model),
        "cov90_abs_err_max": _broadcast(cov_max, bpd_model),
    }


def _summarize_arrays(arrs: dict[str, np.ndarray]) -> dict[str, float]:
    """Mean, std, median, min, max, count for each key."""
    out = {}
    for k, v in arrs.items():
        if v.size == 0:  # skip empties
            continue
        out[f"{k}_mean"] = float(np.mean(v))
        out[f"{k}_std"] = float(np.std(v))
        out[f"{k}_median"] = float(np.median(v))
        out[f"{k}_min"] = float(np.min(v))
        out[f"{k}_max"] = float(np.max(v))
        out[f"{k}_count"] = int(v.size)
    return out


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


def _finite_clean(x: np.ndarray, *, max_abs: float | None = None) -> np.ndarray:
    a = np.asarray(x).ravel()
    m = np.isfinite(a)
    if max_abs is not None:
        m &= np.abs(a) <= max_abs
    a = a[m]
    return a


def run_experiment(cfg: FlowConfig):
    local_dev = cfg.is_dev
    local_dev = False
    pprint(cfg)
    # ----- setup dirs -----
    log("Parsing args done. Starting run_experiment …")
    dirs = setup_exp(cfg.exp_dir_name)
    exp_dir = Path(dirs["exp_dir"])
    models_dir = Path(dirs["models_dir"])
    evals_dir = Path(dirs["evals_dir"])
    artefacts_dir = Path(dirs["artefacts_dir"])

    # Save the config
    def _json_sanitize(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, enum.Enum):
            return obj.value
        return obj

    (evals_dir / "run_config.json").write_text(
        json.dumps({k: _json_sanitize(v) for k, v in asdict(cfg).items()}, indent=2)
    )

    seed_everything(cfg.seed)

    # Dataset & Encoder (HRR @ 7744)
    ds_cfg = cfg.dataset.default_cfg
    device = pick_device()
    log(f"Using device: {device!s}")

    # ----- datasets / loaders -----
    log(f"Loading {cfg.dataset.default_cfg.base_dataset} pair datasets.")
    train_dataset = get_split(split="train", ds_config=cfg.dataset.default_cfg)
    validation_dataset = get_split(split="valid", ds_config=cfg.dataset.default_cfg)
    log(
        f"Pairs loaded for {cfg.dataset.default_cfg.base_dataset}. train_pairs_full_size={len(train_dataset)} valid_pairs_full_size={len(validation_dataset)}"
    )
    # Ensure dtype compatibility
    train_dataset.data.edge_terms = train_dataset.data.edge_terms.to(DTYPE)
    train_dataset.data.graph_terms = train_dataset.data.graph_terms.to(DTYPE)
    validation_dataset.data.edge_terms = validation_dataset.data.edge_terms.to(DTYPE)
    validation_dataset.data.graph_terms = validation_dataset.data.graph_terms.to(DTYPE)

    log("Loading/creating hypernet …")
    hypernet = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, cfg=ds_cfg).to(device=device, dtype=DTYPE).eval()
    log(f"Setting hypernet depth to {ds_cfg.hypernet_depth}!")
    hypernet.depth = ds_cfg.hypernet_depth
    log("Hypernet ready.")
    rtol, atol = (1e-4, 1e-6) if torch.float32 == DTYPE else (1e-5, 1e-8)
    assert torch.allclose(
        hypernet.forward(Batch.from_data_list([train_dataset[42]]))["edge_terms"],
        train_dataset[42].edge_terms.to(device=device),
        rtol=rtol,
        atol=atol,
    ), "edge terms are not equal"
    assert torch.allclose(
        hypernet.forward(Batch.from_data_list([train_dataset[0]]))["edge_terms"],
        train_dataset[0].edge_terms.to(device=device),
        rtol=rtol,
        atol=atol,
    ), "edge terms are not equal"
    assert torch.equal(hypernet.nodes_codebook, hypernet.node_encoder_map[Features.NODE_FEATURES][0].codebook)
    assert torch.equal(hypernet.nodes_codebook, hypernet.node_encoder_map[Features.NODE_FEATURES][0].codebook)
    assert hypernet.nodes_codebook.dtype == DTYPE
    assert hypernet.node_encoder_map[Features.NODE_FEATURES][0].codebook.dtype == DTYPE
    log("Hypernet ready.")

    # pick worker counts per GPU; tune for your cluster
    num_workers = 14 if os.getenv("CLUSTER") != "local" else 8
    if local_dev:
        train_dataset = train_dataset[: cfg.batch_size]
        validation_dataset = validation_dataset[: cfg.batch_size]
        num_workers = 0

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
        drop_last=True,
        prefetch_factor=None if local_dev else 6,  # Increased from 6 for better GPU saturation
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
        drop_last=False,
        prefetch_factor=None if local_dev else 4,  # Increased from 2 for better throughput
    )
    log(f"Datasets ready. train={len(train_dataset)} valid={len(validation_dataset)}")

    # ----- model / trainer -----
    model = resolve_model("NVP", cfg=cfg).to(device=device)

    log(f"Model: {model!s}")
    log(f"Model device: {model.device}")
    log(f"Model hparams: {model.hparams}")

    # Choose standardization method
    if cfg.per_term_standardization:
        log("Using per-term standardization (separate for edge_terms and graph_terms)")
        fit_per_term_standardization(model, train_dataloader, hv_dim=cfg.hv_dim, device=device)
    else:
        log("Using global standardization (all dimensions together)")
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
    time_logger = TimeLoggingCallback()
    periodic_nll = PeriodicNLLEval(
        val_loader=validation_dataloader,
        artefacts_dir=artefacts_dir,
        every_n_epochs=100 if not local_dev else 1,
        max_batches=100 if not local_dev else 1,  # Reduced from 200 for faster training
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=50,
        min_delta=0.0,
        check_finite=True,  # stop if val becomes NaN/Inf
        verbose=True,
    )

    # ----- W&B -----
    loggers = [csv_logger]

    precision = pick_precision()
    log(f"Using precision {precision!s}")
    configure_tf32(precision)

    # Gradient accumulation for larger effective batch size (better GPU utilization)
    # Set GRAD_ACCUM=4 for 4x effective batch size with same memory usage
    grad_accum_steps = int(os.getenv("GRAD_ACCUM", "1"))
    if grad_accum_steps > 1:
        log(
            f"Using gradient accumulation: {grad_accum_steps} steps (effective batch size: {cfg.batch_size * grad_accum_steps})"
        )

    trainer = Trainer(
        max_epochs=cfg.epochs,
        logger=loggers,
        callbacks=[checkpoint_callback, lr_monitor, time_logger, periodic_nll, early_stopping],
        default_root_dir=str(exp_dir),
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        log_every_n_steps=500 if not local_dev else 1,
        enable_progress_bar=True,
        deterministic=False,
        precision=precision,
        num_sanity_val_steps=0,
        accumulate_grad_batches=grad_accum_steps,
    )

    # ----- train -----
    resume_path: Path | None = str(cfg.continue_from) if cfg.continue_from else None
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader, ckpt_path=resume_path)

    # ----- curves to parquet / png -----
    metrics_path = Path(csv_logger.log_dir) / "metrics.csv"
    min_val_loss = float("inf")
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        ## Determine best val loos
        idx = df["val_loss"].idxmin()
        min_val_loss = df.loc[idx, "val_loss"]
        df.to_parquet(evals_dir / "metrics.parquet", index=False)
        plot_train_val_loss(df, artefacts_dir)
        # Optional: print final numbers for quick scan
        train_last = df.loc[df["train_loss_epoch"].notna(), "train_loss_epoch"].tail(1)
        val_last = df.loc[df["val_loss"].notna(), "val_loss"].tail(1)
        if not train_last.empty or not val_last.empty:
            print(
                f"Final losses → train: {float(train_last.values[-1]) if not train_last.empty else 'n/a'} "
                f"| val: {float(val_last.values[-1]) if not val_last.empty else 'n/a'}"
            )

    # =================================================================
    # Post-training analysis: load best, evaluate NLL, sample & log
    # =================================================================
    best_path = checkpoint_callback.best_model_path
    if (not best_path) or ("nan" in Path(best_path).name) or (not Path(best_path).exists()):
        best_path = checkpoint_callback.last_model_path

    if not best_path or not Path(best_path).exists():
        log("No checkpoint found (best/last). Skipping post-training analysis.")
        return 0.0

    log(f"Loading best checkpoint: {best_path}")
    best_model = retrieve_model("NVP").load_from_checkpoint(best_path)
    best_model.to(device).eval()
    best_model.to(dtype=DTYPE)

    # ---- per-sample NLL (really the KL objective) on validation ----
    val_stats = _eval_flow_metrics(best_model, validation_dataloader, device, hv_count=cfg.hv_count, hv_dim=cfg.hv_dim)
    nll_arr = val_stats.get("nll_model", np.empty((0,), dtype=np.float32))
    if nll_arr.size:
        # save full arrays for later deep-dive
        pd.DataFrame(
            {
                "nll_model": val_stats["nll_model"],
                "bpd_model": val_stats["bpd_model"],
                "bpd_gauss": val_stats["bpd_gauss"],
                "bpd_stdspace": val_stats["bpd_stdspace"],
                "delta_bpd": val_stats["delta_bpd"],
            }
        ).to_parquet(evals_dir / "val_metrics_final.parquet", index=False)

        # simple hist PNGs (optional)
        _hist(artefacts_dir / "val_bpd_model_hist.png", val_stats["bpd_model"], "Validation bpd (model)", "bpd")
        _hist(artefacts_dir / "val_bpd_gauss_hist.png", val_stats["bpd_gauss"], "Validation bpd (Gaussian)", "bpd")
        _hist(
            artefacts_dir / "val_delta_bpd_hist.png",
            val_stats["delta_bpd"],
            "Validation Δbpd (model - Gaussian)",
            "Δbpd",
        )

        # scalar summary (prefixed under eval/val/*)
        summary = _summarize_arrays(val_stats)
        summary_payload = {f"eval/val/{k}": v for k, v in summary.items()}
        log(summary_payload)

    # ---- sample from the flow ----
    with torch.no_grad():
        samples = best_model.sample_split(1000)  # each [K, D]
        analyze_terms_only(samples, name=ds_cfg.name, pdf_dir=artefacts_dir)
        # node_s = samples["node_terms"]
        edge_s = samples["edge_terms"]
        graph_s = samples["graph_terms"]

    # node_s = node_s.as_subclass(HRRTensor)
    edge_s = edge_s.as_subclass(HRRTensor)
    graph_s = graph_s.as_subclass(HRRTensor)

    # log(f"node_s device: {node_s.device!s}")
    log(f"graph_s device: {graph_s.device!s}")
    log(f"Hypernet node codebook device: {hypernet.nodes_codebook.device!s}")

    # node_np = node_s.detach().cpu().numpy()
    edge_np = edge_s.detach().cpu().numpy()
    graph_np = graph_s.detach().cpu().numpy()

    # per-branch norms and pairwise cosine samples
    # node_norm = np.linalg.norm(node_np, axis=1)
    edge_norm = np.linalg.norm(edge_np, axis=1)
    graph_norm = np.linalg.norm(graph_np, axis=1)

    def _pairwise_cosine(x: np.ndarray, m: int = 2000) -> np.ndarray:
        n = x.shape[0]
        if n < 2:
            return np.array([])
        idx = np.random.choice(n, size=(min(m, n - 1), 2), replace=True)
        a = x[idx[:, 0]]
        b = x[idx[:, 1]]
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.sum(an * bn, axis=1)

    # node_cos = _pairwise_cosine(node_np, m=4000)
    edge_cos = _pairwise_cosine(edge_np, m=4000)
    graph_cos = _pairwise_cosine(graph_np, m=4000)

    # plots
    # _hist(artefacts_dir / "sample_node_norm_hist.png", node_norm, "Sample node L2 norm", "||node||")
    _hist(artefacts_dir / "sample_node_edge_hist.png", edge_norm, "Sample edge L2 norm", "||edge||")
    _hist(artefacts_dir / "sample_graph_norm_hist.png", graph_norm, "Sample graph L2 norm", "||graph||")
    # if node_cos.size:
    #     _hist(artefacts_dir / "sample_node_cos_hist.png", node_cos, "Node pairwise cosine", "cos")
    if edge_cos.size:
        _hist(artefacts_dir / "sample_edge_cos_hist.png", edge_cos, "Edge pairwise cosine", "cos")
    if graph_cos.size:
        _hist(artefacts_dir / "sample_graph_cos_hist.png", graph_cos, "Graph pairwise cosine", "cos")

    # =================================================================
    # NEW: Decode samples and compute validity for composite metric
    # =================================================================
    log("=== Starting validity evaluation for composite metric ===")

    # Determine number of samples to decode based on dataset
    base_dataset = ds_cfg.base_dataset  # "qm9" or "zinc"
    if cfg.n_decode_override is not None:
        n_decode = cfg.n_decode_override
    else:
        n_decode = 1000 if base_dataset == "qm9" else 100

    log(f"Decoding {n_decode} samples for {base_dataset} dataset...")

    # Get decoder settings from models_configs_constants
    DECODER_SETTINGS_ = {
        "qm9": {
            "iteration_budget": 1,
            "max_graphs_per_iter": 512,
            "top_k": 10,
            "sim_eps": 0.0001,
            "early_stopping": True,
            "fallback_decoder_settings": {
                "initial_limit": 2048,
                "limit": 1024,
                "beam_size": 1024,
                "pruning_method": "cos_sim",
                "use_size_aware_pruning": True,
                "use_one_initial_population": False,
                "use_g3_instead_of_h3": False,
            },
        },
        "zinc": {
            "iteration_budget": 1,
            "max_graphs_per_iter": 512,
            "top_k": 10,
            "sim_eps": 0.0001,
            "early_stopping": True,
            "fallback_decoder_settings": {
                "initial_limit": 1024,
                "limit": 64,
                "beam_size": 32,
                "pruning_method": "cos_sim",
                "use_size_aware_pruning": True,
                "use_one_initial_population": False,
                "use_g3_instead_of_h3": False,
            },
        },
    }
    decoder_settings = DECODER_SETTINGS_[base_dataset]

    log(f"Using decoder settings: {decoder_settings}")

    # Prepare samples for decoding
    edge_decode = edge_s[:n_decode]
    graph_decode = graph_s[:n_decode]

    # Decode samples
    nx_graphs = []
    final_flags = []
    sims = []
    correction_levels = []

    log("Getting Hypernet ready for decoding ...")
    nodes_set = set(map(tuple, train_dataset.x.long().tolist()))
    hypernet.limit_nodes_codebook(limit_node_set=nodes_set)
    hypernet.decoding_limit_for = base_dataset
    hypernet.base_dataset = base_dataset

    decode_start_time = time.time()
    for i in range(n_decode):
        if (i + 1) % 100 == 0 or i == 0:
            log(f"Decoding sample {i + 1}/{n_decode}...")

        try:
            decode_result = hypernet.decode_graph(
                edge_term=edge_decode[i].as_subclass(HRRTensor),
                graph_term=graph_decode[i].as_subclass(HRRTensor),
                decoder_settings=decoder_settings,
            )

            # Extract results
            nx_graphs.append(decode_result.nx_graphs[0])
            final_flags.append(decode_result.final_flags[0])
            sims.append(decode_result.similarities[0] if hasattr(decode_result, "similarities") else 0.0)
            correction_levels.append(decode_result.correction_level)

        except Exception as e:
            log(f"Warning: Failed to decode sample {i}: {e}")
            # Add placeholder for failed decoding

            nx_graphs.append(nx.Graph())
            final_flags.append(False)
            sims.append(0.0)

            correction_levels.append(CorrectionLevel.FAIL)

    decode_elapsed = time.time() - decode_start_time
    log(f"Decoding completed in {decode_elapsed:.2f}s ({decode_elapsed / n_decode:.2f}s per sample)")

    # Compute CorrectionLevel.ZERO percentage (perfect decoding without corrections)
    n_zero_corrections = sum(1 for cl in correction_levels if cl == CorrectionLevel.ZERO)
    zero_correction_pct = 100.0 * n_zero_corrections / n_decode if n_decode else 0.0

    log("Correction Level Distribution:")
    for level in CorrectionLevel:
        count = sum(1 for cl in correction_levels if cl == level)
        pct = 100.0 * count / n_decode if n_decode else 0.0
        log(f"  {level.value}: {count}/{n_decode} ({pct:.2f}%)")
    log(f"CorrectionLevel.ZERO percentage: {zero_correction_pct:.2f}%")

    log("=== Composite Metric Computation ===")
    log(f"Min validation NLL: {min_val_loss:.4f}")
    log(f"CorrectionLevel.ZERO: {zero_correction_pct:.2f}%")

    # Save comprehensive metrics to JSON
    # Store correction level distribution
    correction_level_dist = {
        level.value: sum(1 for cl in correction_levels if cl == level) for level in CorrectionLevel
    }

    metrics_dict = {
        "exp_dir_name": cfg.exp_dir_name,
        "min_val_loss": float(min_val_loss),
        "zero_correction_pct": float(zero_correction_pct),
        "incorrect_pct": 100.0 - float(zero_correction_pct),
        "n_decoded": int(n_decode),
        "base_dataset": base_dataset,
        "decode_time_sec": float(decode_elapsed),
        "decode_time_per_sample_sec": float(decode_elapsed / n_decode),
        "decoder_settings": decoder_settings,
        "correction_level_distribution": correction_level_dist,
    }

    metrics_file = evals_dir / "hpo_metrics.json"
    metrics_file.write_text(json.dumps(metrics_dict, indent=2))
    log(f"Saved comprehensive metrics to {metrics_file}")

    log("Experiment completed.")
    log(f"Best val NLL: {min_val_loss:.4f}")
    log(f"CorrectionLevel.ZERO: {zero_correction_pct:.2f}%")
    return min_val_loss, 100.0 - zero_correction_pct


def get_hidden_channel_dist(dataset: SupportedDataset):
    low, high, step = 400, 1600, 400
    # 256-dim datasets (total input: 512 = 256 edge + 256 graph)
    if dataset in [SupportedDataset.QM9_SMILES_HRR_256_F64_G1NG3, SupportedDataset.ZINC_SMILES_HRR_256_F64_5G1NG4]:
        low, high, step = 512, 2048, 512
    elif dataset == SupportedDataset.ZINC_SMILES_HRR_1024_F64_5G1NG4:
        low, high, step = 512, 1024, 256
    elif dataset == SupportedDataset.ZINC_SMILES_HRR_2048_F64_5G1NG4:
        low, high, step = 1024, 2048, 512
    return low, high, step


def get_cfg(trial: optuna.Trial, dataset: SupportedDataset):
    cfg = {
        "batch_size": trial.suggest_int("batch_size", 32, 512, step=32),
        "lr": trial.suggest_float("lr", 5e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }
    flow_cfg = FlowConfig()
    for k, v in cfg.items():
        setattr(flow_cfg, k, v)
    flow_cfg.dataset = dataset
    flow_cfg.hv_dim = dataset.default_cfg.hv_dim
    flow_cfg.hv_count = dataset.default_cfg.hv_count
    flow_cfg.vsa = dataset.default_cfg.vsa
    return flow_cfg


def run_zinc_trial(trial: optuna.Trial, dataset: SupportedDataset, norms_per: str):
    flow_cfg = get_cfg(trial, dataset=dataset)
    low, high, step = get_hidden_channel_dist(dataset)
    flow_cfg.num_hidden_channels = trial.suggest_int("num_hidden_channels", low, high, step=step)
    flow_cfg.num_flows = trial.suggest_int("num_flows", 4, 16)
    flow_cfg.smax_initial = 2.5
    flow_cfg.smax_final = 7
    flow_cfg.smax_warmup_epochs = 17
    flow_cfg.per_term_standardization = norms_per == "term"
    flow_cfg.exp_dir_name = make_run_folder_name(
        {k: getattr(flow_cfg, k) for k in keys if k in flow_cfg.__dict__}, prefix=f"nvp_comp_{dataset.default_cfg.name}"
    )
    return run_experiment(flow_cfg)


def run_qm9_trial(trial: optuna.Trial, dataset: SupportedDataset, norms_per: str):
    flow_cfg = get_cfg(trial, dataset=dataset)
    low, high, step = get_hidden_channel_dist(dataset)
    flow_cfg.num_hidden_channels = trial.suggest_int("num_hidden_channels", low, high, step=step)
    flow_cfg.num_flows = trial.suggest_int("num_flows", 4, 16)
    flow_cfg.smax_initial = 2.2
    flow_cfg.smax_final = 6.5
    flow_cfg.smax_warmup_epochs = 16
    flow_cfg.per_term_standardization = norms_per == "term"
    flow_cfg.exp_dir_name = make_run_folder_name(
        {k: getattr(flow_cfg, k) for k in keys if k in flow_cfg.__dict__}, prefix=f"nvp_comp_{dataset.default_cfg.name}"
    )
    return run_experiment(flow_cfg)


keys = {
    "batch_size",
    "lr",
    "weight_decay",
    "num_flows",
    "num_hidden_channels",
    "smax_initial",
    "smax_final",
    "smax_warmup_epochs",
    "per_term_standardization",
}
