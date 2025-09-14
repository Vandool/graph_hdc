import argparse
import contextlib
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
import normflows as nf
import numpy as np
import pandas as pd
import torch
import wandb
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch_geometric.loader import DataLoader
from torchhd import HRRTensor

from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import Features, SupportedDataset
from src.encoding.graph_encoders import load_or_create_hypernet
from src.encoding.the_types import VSAModel
from src.normalizing_flow.models import AbstractNFModel
from src.utils.utils import GLOBAL_MODEL_PATH, generated_node_edge_dist, pick_device, str2bool

LOCAL_DEV = "LOCAL_HDC_miss"

PROJECT_NAME = "real_nvp_v2"


# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
# ---------- tiny logger ----------
def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


os.environ.setdefault("PYTHONUNBUFFERED", "1")


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
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.0
    is_dev: bool = False

    # HDC / encoder
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


class BoundedMLP(torch.nn.Module):
    def __init__(self, dims, smax=6.0):
        super().__init__()
        self.net = nf.nets.MLP(dims, init_zeros=True)  # keeps identity at start
        self.smax = float(smax)
        self.last_pre = None  # for logging

    def forward(self, x):
        pre = self.net(x)
        self.last_pre = pre
        return torch.tanh(pre) * self.smax  # bound log-scale in [-smax, smax]


class RealNVPV2Lightning(AbstractNFModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        D = int(cfg.hv_dim)
        self.D = D
        self.flat_dim = 2 * D

        mask = (torch.arange(self.flat_dim) % 2).to(torch.float32)
        self.register_buffer("mask0", mask)

        # per-feature standardization params (fill later)
        self.register_buffer("mu", torch.zeros(self.flat_dim))
        self.register_buffer("log_sigma", torch.zeros(self.flat_dim))

        self.s_modules = []  # keep handles for warmup/logging
        flows = []

        # ActNorm layers
        use_act_norm = cfg.use_act_norm
        if use_act_norm and hasattr(nf.flows, "ActNorm"):
            flows.append(nf.flows.ActNorm(self.flat_dim))  # learns per-feature shift/scale

        hidden = int(cfg.num_hidden_channels)
        for _ in range(int(cfg.num_flows)):
            layers = [self.flat_dim, hidden, hidden, self.flat_dim]
            t_net = nf.nets.MLP(layers, init_zeros=True)
            s_net = BoundedMLP(layers, smax=getattr(cfg, "smax_final", 6))
            self.s_modules.append(s_net)

            flows.append(nf.flows.MaskedAffineFlow(self.mask0, t=t_net, s=s_net))
            flows.append(nf.flows.Permute(self.flat_dim, mode="shuffle"))

        base = nf.distributions.DiagGaussian(self.flat_dim, trainable=False)
        self.flow = nf.NormalizingFlow(q0=base, flows=flows)

    def set_standardization(self, mu, sigma, eps=1e-6):
        """Call once before training; mu/sigma are 1D tensors/lists length 2D."""
        mu = torch.as_tensor(mu, dtype=torch.float32, device=self.device)
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=self.device)
        self.mu.copy_(mu)
        self.log_sigma.copy_(torch.log(torch.clamp(sigma, min=eps)))

    def _pretransform(self, x):
        """z = (x - mu) / sigma ; returns (z, +sum(log sigma)) for log-det correction."""
        z = (x - self.mu) * torch.exp(-self.log_sigma)
        # log|det ∂z/∂x| = -sum(log_sigma); NLL must ADD +sum(log_sigma)
        return z, float(self.log_sigma.sum().item())

    def _posttransform(self, z):
        return self.mu + z * torch.exp(self.log_sigma)

    @torch.no_grad()
    def sample_split(self, num_samples: int):
        """
        Returns:
          node_terms:  [num_samples, D]
          graph_terms: [num_samples, D]
          logs
        """
        z, _logs = self.sample(num_samples)  # standardized space
        x = self._posttransform(z)  # back to data space
        node_terms = x[:, : self.D].contiguous()
        graph_terms = x[:, self.D :].contiguous()
        return node_terms, graph_terms, _logs

    def nf_forward_kld(self, flat):
        """Example: exact NLL with pre-transform correction."""
        z, log_det_corr = self._pretransform(flat)
        # If your nf API returns -log p(z) per-sample, add the correction:
        nll = -self.flow.log_prob(z) + log_det_corr
        return nll

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        # 5% warmup then cosine
        steps_per_epoch = max(
            1, getattr(self.trainer, "estimated_stepping_batches", 1000) // max(1, self.trainer.max_epochs)
        )
        warmup = int(0.05 * self.trainer.max_epochs) * steps_per_epoch
        total = self.trainer.max_epochs * steps_per_epoch
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda step: min(1.0, step / max(1, warmup))
            * 0.5
            * (1 + math.cos(math.pi * max(0, step - warmup) / max(1, total - warmup))),
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def on_fit_start(self):
        # H100 tip: get tensor cores w/o AMP
        with contextlib.suppress(Exception):
            torch.set_float32_matmul_precision("high")

    def on_train_epoch_start(self):
        # log-det warmup for stability: ramp smax from small → final
        warm = int(getattr(self.cfg, "smax_warmup_epochs", 15))
        s0 = float(getattr(self.cfg, "smax_initial", 1.0))
        s1 = float(getattr(self.cfg, "smax_final", 6.0))
        if warm > 0:
            t = min(1.0, self.current_epoch / max(1, warm))
            smax = (1 - t) * s0 + t * s1
            for m in self.s_modules:
                m.smax = smax

    # Lightning will move the PyG Batch; just ensure we return batch on device
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device)

    def _flat_from_batch(self, batch) -> torch.Tensor:
        D = self.D
        B = batch.num_graphs
        n, g = batch.node_terms, batch.graph_terms
        if n.dim() == 1:
            n = n.view(B, D)
        if g.dim() == 1:
            g = g.view(B, D)
        return torch.cat([n, g], dim=-1)

    def training_step(self, batch, batch_idx):
        flat = self._flat_from_batch(batch)
        obj = self.nf_forward_kld(flat)  # [B]
        # keep only finite samples (safety)
        obj = obj[torch.isfinite(obj)]
        if obj.numel() == 0:
            # log a plain float
            self.log("nan_loss_batches", 1.0, on_step=True, prog_bar=True, batch_size=flat.size(0))
            return None

        loss = obj.mean()
        # *** log pure Python float and pass batch_size ***
        self.log(
            "train_loss",
            float(loss.detach().cpu().item()),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=flat.size(0),
        )

        # optional: monitor scale pre-activation magnitude (float)
        with torch.no_grad():
            s_absmax = 0.0
            for m in getattr(self, "s_modules", []):
                if getattr(m, "last_pre", None) is not None and torch.isfinite(m.last_pre).any():
                    s_absmax = max(s_absmax, float(m.last_pre.detach().abs().max().cpu().item()))
        self.log("s_pre_absmax", s_absmax, on_step=True, prog_bar=True, batch_size=flat.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        flat = self._flat_from_batch(batch)
        obj = self.nf_forward_kld(flat)
        obj = obj[torch.isfinite(obj)]
        val = float("nan") if obj.numel() == 0 else float(obj.mean().detach().cpu().item())
        self.log("val_loss", val, on_epoch=True, prog_bar=True, batch_size=flat.size(0))
        return torch.tensor(val, device=flat.device) if val == val else None  # return NaN-safe

    # ---- helpers for sampling/inspection


@torch.no_grad()
def fit_featurewise_standardization(model, loader, hv_dim: int, max_batches: int | None = None, device="cpu"):
    # Accumulate sums per-dimension (feature-wise)
    cnt = 0
    sum_vec = torch.zeros(2 * hv_dim, dtype=torch.float64, device=device)
    sumsq_vec = torch.zeros(2 * hv_dim, dtype=torch.float64, device=device)

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = batch.to(device)
        x = model._flat_from_batch(batch).to(torch.float64)  # [B, 2D]
        cnt += x.shape[0]
        sum_vec += x.sum(dim=0)
        sumsq_vec += (x * x).sum(dim=0)

    mu = (sum_vec / max(1, cnt)).to(torch.float32)
    var = (sumsq_vec / max(1, cnt) - (sum_vec / max(1, cnt)) ** 2).clamp_min_(0).to(torch.float32)
    sigma = var.sqrt().clamp_min_(1e-6)
    model.set_standardization(mu, sigma)


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def plot_train_val_loss(
        df: pd.DataFrame,
        artefacts_dir: Path,
        *,
        skip_first: int | float = 0.1,   # int: num epochs to skip; float in (0,1): fraction of epochs to skip
        min_epoch: int | None = None,  # overrides skip_first if set
        clip_top_q: float | None = None,  # e.g. 0.98 to clip outliers on y-axis
        smooth_window: int | None = None, # rolling mean window (in points)
        logy: bool = False,
):
    # possible names across PL versions / logging configs
    train_epoch_keys = ["train_loss_epoch", "train_loss", "epoch_train_loss"]
    val_epoch_keys = ["val_loss", "val/loss", "validation_loss"]

    train_col = _first_existing(df, train_epoch_keys)
    val_col   = _first_existing(df, val_epoch_keys)
    epoch_col = _first_existing(df, ["epoch", "step"])

    if epoch_col is None or (train_col is None and val_col is None):
        log("Skip plot: no epoch/metric columns in metrics.csv for this run.")
        return

    # ---- decide the cutoff epoch ----
    uniq_epochs = pd.unique(df[epoch_col].dropna())
    try:
        uniq_epochs = np.sort(uniq_epochs.astype(int))
    except Exception:
        uniq_epochs = np.sort(uniq_epochs)

    if min_epoch is not None:
        cutoff = min_epoch
    else:
        if isinstance(skip_first, float) and 0 < skip_first < 1:
            k = int(round(skip_first * len(uniq_epochs)))
        else:
            k = int(skip_first)
        k = max(0, min(k, max(len(uniq_epochs) - 1, 0)))
        cutoff = uniq_epochs[k] if len(uniq_epochs) else 0

    # filter out burn-in
    df_f = df[df[epoch_col] >= cutoff]

    # helper: get a clean series, optionally smoothed
    def _series(col):
        s = df_f[[epoch_col, col]].dropna()
        if s.empty:
            return None, None
        if smooth_window and smooth_window > 1:
            s[col] = s[col].rolling(smooth_window, min_periods=1, center=False).mean()
        return s[epoch_col].values, s[col].values

    plt.figure(figsize=(8, 5))

    plotted = False
    if train_col is not None:
        x, y = _series(train_col)
        if x is not None:
            plt.plot(x, y, label=f"{train_col} (≥{cutoff})")
            plotted = True

    if val_col is not None:
        x, y = _series(val_col)
        if x is not None:
            plt.plot(x, y, label=f"{val_col} (≥{cutoff})")
            plotted = True

    if not plotted:
        log("Skip plot: nothing to plot after burn-in filter.")
        plt.close()
        return

    if clip_top_q is not None and 0 < clip_top_q < 1:
        # clip y-axis to reduce impact of a few spikes
        ys = []
        if train_col is not None:
            ys.append(df_f[train_col].dropna().values)
        if val_col is not None:
            ys.append(df_f[val_col].dropna().values)
        y_all = np.concatenate(ys) if ys else None
        if y_all is not None and y_all.size:
            ymax = float(np.quantile(y_all, clip_top_q))
            ymin = float(np.nanmin(y_all))
            plt.ylim(bottom=ymin, top=ymax)

    if logy:
        plt.yscale("log")

    plt.xlabel(epoch_col)
    plt.ylabel("loss")
    plt.title("Train vs. Validation Loss")
    plt.legend()
    plt.tight_layout()
    artefacts_dir.mkdir(parents=True, exist_ok=True)
    out = artefacts_dir / "train_val_loss.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved train/val loss plot to {out} (cutoff ≥ {cutoff})")



def pick_precision():
    # Works on A100/H100 if BF16 is supported by the PyTorch/CUDA build.
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "bf16-mixed"  # safest + fast on H100/A100
        return "16-mixed"  # widely supported fallback
    return 32  # CPU or MPS


def sample_and_decode_preview(
    model,
    hypernet,
    *,
    max_print: int = 16,
    logger=print,
) -> None:
    model.eval()
    with torch.no_grad():
        node_s, graph_s, _ = model.sample_split(max_print)  # [K, D] each

    # If your tensors subclass needs to be enforced (as in your code)
    node_s = node_s.as_subclass(HRRTensor)
    graph_s = graph_s.as_subclass(HRRTensor)

    logger(f"[preview] node_s device: {node_s.device}")
    logger(f"[preview] graph_s device: {graph_s.device}")
    with contextlib.suppress(Exception):
        logger(f"[preview] Hypernet node codebook device: {hypernet.nodes_codebook.device}")

    # Decode and print a few
    node_counters = hypernet.decode_order_zero_counter(node_s)
    for i, (n, ctr) in enumerate(node_counters.items()):
        if i >= max_print:
            break
        logger(f"[preview] Sample {n}: nodes={ctr.total()}, edges={sum(e + 1 for _, e, _, _ in ctr)}\n{ctr}")


class PeriodicSampleDecodeCallback(Callback):
    r"""
    Periodically samples from the model and decodes with the hypernet, printing
    a small textual preview for quick qualitative inspection.

    Triggers every ``interval_epochs`` epochs at train-epoch end.
    """

    def __init__(
        self,
        hypernet,
        *,
        interval_epochs: int = 10,
        max_print: int = 16,
        use_wandb_text: bool = True,
    ):
        super().__init__()
        self.hypernet = hypernet
        self.interval = int(interval_epochs)
        self.max_print = int(max_print)
        self.use_wandb_text = bool(use_wandb_text)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = int(getattr(trainer, "current_epoch", 0))
        if self.interval <= 0 or ((epoch + 1) % self.interval) != 0:
            return

        # Route logs to either W&B text panel (if available) or stdout
        logs_buffer = []

        def _collect(msg: str):
            logs_buffer.append(str(msg))

        sample_and_decode_preview(
            pl_module,
            self.hypernet,
            max_print=self.max_print,
            logger=_collect,
        )

        text_blob = "\n".join(logs_buffer)
        try:
            if self.use_wandb_text and wandb.run is not None:
                wandb.log(
                    {"preview/decoded_samples": wandb.Html(f"<pre>{text_blob}</pre>"), "preview/epoch": epoch + 1}
                )
            else:
                trainer.logger.log_text("preview/decoded_samples", text_blob)  # some loggers support this
                print(text_blob, flush=True)
        except Exception:
            # Fallback to stdout
            print(text_blob, flush=True)


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
        every_n_epochs: int = 50,
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
            pl_module, self.val_loader, pl_module.device, hv_dim=pl_module.D, max_batches=self.max_batches
        )
        if not stats:
            return

        # persist arrays (with epoch in file for easier joins)
        self.artefacts_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(stats)
        df["epoch"] = epoch
        df.to_parquet(self.artefacts_dir / f"val_metrics_epoch{epoch:04d}.parquet", index=False)

        # compact scalar summary with consistent names
        summary = _summarize_arrays(stats)
        log_payload = {f"val/{k}": v for k, v in summary.items()}
        log_payload["epoch"] = epoch

        # log scalars to all attached loggers (CSV & W&B if present)
        try:
            trainer.logger.log_metrics(log_payload, step=trainer.global_step)
        except Exception:
            log(log_payload)

        # histograms → only to W&B (if requested and available)
        if self.log_hist and wandb.run:
            hist_payload = {
                "val/bpd_model_hist": wandb.Histogram(stats["bpd_model"]),
                "val/bpd_gauss_hist": wandb.Histogram(stats["bpd_gauss"]),
                "val/delta_bpd_hist": wandb.Histogram(stats["delta_bpd"]),
            }
            wandb.log(hist_payload, step=trainer.global_step)


def _norm_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


@torch.no_grad()
def _eval_flow_metrics(model, loader, device, hv_dim: int, max_batches: int | None = None):
    """
    Eval for normalizing flows trained with exact likelihood.

    We report ONLY arrays with the same length so they can be dropped into a single
    pandas DataFrame without shape errors. Per-dimension diagnostics are summarized
    to scalars and *broadcast* to per-sample length (so they’re easy to join & plot).

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
        standardization step; helps detect cases where most of the “good” bpd is due
        to μ/σ rather than the flow layers (a real risk with highly heteroscedastic data).

    Returns arrays (all same length) for easy DataFrame construction:
      bpd_model, bpd_gauss, delta_bpd, bpd_stdspace, nll_model,
      sum_log_sigma_bits, const_bpd,
      pit_ks_mean/std/max, cov90_abs_err_mean/std/max
    """
    model.eval()
    ln2 = math.log(2.0)
    dim = 2 * hv_dim

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
        flat = model._flat_from_batch(batch)  # [B, dim]

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

        # Move both NLLs back to the *data* space by adding the pre-transform’s
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
        # bpd_stdspace reflects the flow’s coding cost in the *standardized* space.
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
        take = min(512, D)
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

    # How much of your bpd is a constant due to standardization’s log|σ|?
    # This is useful to diagnose when most “gain” comes from μ/σ rather than learned transforms.
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

    log("Loading/creating hypernet …")
    hypernet = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, cfg=ds_cfg).to(device=device).eval()
    log("Hypernet ready.")
    assert torch.equal(hypernet.nodes_codebook, hypernet.node_encoder_map[Features.ATOM_TYPE][0].codebook)
    encoder = hypernet.to(device).eval()
    assert torch.equal(hypernet.nodes_codebook, hypernet.node_encoder_map[Features.ATOM_TYPE][0].codebook)
    log("Hypernet ready.")

    assert torch.equal(hypernet.nodes_codebook, hypernet.node_encoder_map[Features.ATOM_TYPE][0].codebook)
    log("Hypernet ready.")

    # ----- datasets / loaders -----
    log(f"Loading {cfg.dataset.value} pair datasets.")
    if cfg.dataset == SupportedDataset.QM9_SMILES_HRR_1600:
        train_dataset = QM9Smiles(split="train", enc_suffix="HRR1600")
        validation_dataset = QM9Smiles(split="valid", enc_suffix="HRR1600")
    elif cfg.dataset == SupportedDataset.ZINC_SMILES_HRR_7744:
        train_dataset = ZincSmiles(split="train", enc_suffix="HRR7744")
        validation_dataset = ZincSmiles(split="valid", enc_suffix="HRR7744")
    log(
        f"Pairs loaded for {cfg.dataset.value}. train_pairs_full_size={len(train_dataset)} valid_pairs_full_size={len(validation_dataset)}"
    )

    # pick worker counts per GPU; tune for your cluster
    num_workers = 16 if torch.cuda.is_available() else 0
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
        # prefetch_factor=6,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
        drop_last=False,
        # prefetch_factor=6,
    )
    log(f"Datasets ready. train={len(train_dataset)} valid={len(validation_dataset)}")

    # ----- model / trainer -----
    model = RealNVPV2Lightning(cfg)
    fit_featurewise_standardization(model, train_dataloader, hv_dim=cfg.hv_dim, device=device)

    csv_logger = CSVLogger(save_dir=str(evals_dir), name="logs")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath=str(models_dir),
        auto_insert_metric_name=False,
        filename="epoch{epoch:02d}-val{val_loss:.4f}",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    time_logger = TimeLoggingCallback()
    periodic_nll = PeriodicNLLEval(
        val_loader=validation_dataloader,
        artefacts_dir=artefacts_dir,
        every_n_epochs=50 if not local_dev else 1,
        max_batches=200 if not local_dev else 1,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=20,
        min_delta=0.0,
        check_finite=True,  # stop if val becomes NaN/Inf
        verbose=True,
    )

    sample_and_decode_cb = PeriodicSampleDecodeCallback(hypernet, interval_epochs=10, max_print=16)

    # ----- W&B -----
    loggers = [csv_logger]
    if not local_dev:
        p = os.environ.get("WANDB_PROJECT", PROJECT_NAME)
        e = os.environ.get("WANDB_ENTITY")
        n = os.environ.get("WANDB_NAME", f"run_{cfg.hv_dim}_{cfg.seed}")
        log(f"W&B logging to project={p} entity={e} name={n}")
        run = wandb.run or wandb.init(
            project=p,
            entity=e,
            name=n,
            config=cfg.__dict__,
            reinit=True,
        )
        run.tags = [
            f"hv_dim={cfg.hv_dim}",
            f"vsa={cfg.vsa.value}",
            f"flows={cfg.num_flows}",
            f"hidden={cfg.num_hidden_channels}",
            f"actnorm={cfg.use_act_norm}",
            f"dataset={cfg.dataset.value}",
        ]
        loggers.append(WandbLogger(log_model=True, experiment=run))

    trainer = Trainer(
        max_epochs=cfg.epochs,
        logger=loggers,
        callbacks=[checkpoint_callback, lr_monitor, time_logger, periodic_nll, early_stopping, sample_and_decode_cb],
        default_root_dir=str(exp_dir),
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        log_every_n_steps=100 if not local_dev else 1,
        enable_progress_bar=True,
        deterministic=False,
        precision=pick_precision(),
        num_sanity_val_steps=100,
        limit_val_batches=100,
    )

    # ----- train -----
    resume_path: Path | None = str(cfg.continue_from) if cfg.continue_from else None
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader, ckpt_path=resume_path)

    # ----- curves to parquet / png -----
    metrics_path = Path(csv_logger.log_dir) / "metrics.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
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
        return

    log(f"Loading best checkpoint: {best_path}")
    best_model = RealNVPV2Lightning.load_from_checkpoint(best_path)
    best_model.to(device).eval()

    # ---- per-sample NLL (really the KL objective) on validation ----
    val_stats = _eval_flow_metrics(best_model, validation_dataloader, device, hv_dim=cfg.hv_dim)
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

        if wandb.run:
            wandb.log(summary_payload)
            # optional: histograms in W&B
            wandb.log(
                {
                    "eval/val/bpd_model_hist": wandb.Histogram(val_stats["bpd_model"]),
                    "eval/val/bpd_gauss_hist": wandb.Histogram(val_stats["bpd_gauss"]),
                    "eval/val/delta_bpd_hist": wandb.Histogram(val_stats["delta_bpd"]),
                }
            )
    else:
        log("No finite NLL values collected; skipping NLL stats.")

    # ---- sample from the flow ----
    with torch.no_grad():
        node_s, graph_s, logs = best_model.sample_split(4096)  # each [K, D]

    node_s = node_s.as_subclass(HRRTensor)
    graph_s = graph_s.as_subclass(HRRTensor)

    log(f"node_s device: {node_s.device!s}")
    log(f"graph_s device: {graph_s.device!s}")
    log(f"Hypernet node codebook device: {hypernet.nodes_codebook.device!s}")

    node_counters = hypernet.decode_order_zero_counter(node_s)
    node_counters = hypernet.decode_order_zero_counter(node_s)
    report = generated_node_edge_dist(
        node_types=node_counters, artefact_dir=artefacts_dir, wandb=wandb if wandb.run else None
    )
    pprint(report)

    node_np = node_s.detach().cpu().numpy()
    graph_np = graph_s.detach().cpu().numpy()
    logs_np = logs.detach().cpu().numpy() if torch.is_tensor(logs) else np.asarray(logs)

    # per-branch norms and pairwise cosine samples
    node_norm = np.linalg.norm(node_np, axis=1)
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

    node_cos = _pairwise_cosine(node_np, m=4000)
    graph_cos = _pairwise_cosine(graph_np, m=4000)

    # persist artefacts
    np.save(artefacts_dir / "samples_node.npy", node_np)
    np.save(artefacts_dir / "samples_graph.npy", graph_np)
    np.save(artefacts_dir / "sample_logs.npy", logs_np)
    pd.DataFrame({"node_norm": node_norm, "graph_norm": graph_norm}).to_parquet(
        evals_dir / "sample_norms.parquet", index=False
    )

    # plots
    _hist(artefacts_dir / "sample_node_norm_hist.png", node_norm, "Sample node L2 norm", "||node||")
    _hist(artefacts_dir / "sample_graph_norm_hist.png", graph_norm, "Sample graph L2 norm", "||graph||")
    if node_cos.size:
        _hist(artefacts_dir / "sample_node_cos_hist.png", node_cos, "Node pairwise cosine", "cos")
    if graph_cos.size:
        _hist(artefacts_dir / "sample_graph_cos_hist.png", graph_cos, "Graph pairwise cosine", "cos")

    # W&B logs
    if wandb.run:
        node_norm_f = _finite_clean(node_norm, max_abs=1e12)
        graph_norm_f = _finite_clean(graph_norm, max_abs=1e12)
        node_cos_f = _finite_clean(node_cos)
        graph_cos_f = _finite_clean(graph_cos)

        payload = {}

        if node_norm_f.size:
            payload.update(
                {
                    "sample_node_norm_mean": float(np.mean(node_norm_f)),
                    "sample_node_norm_std": float(np.std(node_norm_f)),
                    "sample_node_norm_hist": wandb.Histogram(node_norm_f),
                }
            )
        if graph_norm_f.size:
            payload.update(
                {
                    "sample_graph_norm_mean": float(np.mean(graph_norm_f)),
                    "sample_graph_norm_std": float(np.std(graph_norm_f)),
                    "sample_graph_norm_hist": wandb.Histogram(graph_norm_f),
                }
            )
        if node_cos_f.size:
            payload["sample_node_cos_hist"] = wandb.Histogram(node_cos_f)
        if graph_cos_f.size:
            payload["sample_graph_cos_hist"] = wandb.Histogram(graph_cos_f)

        if payload:
            wandb.log(payload)

    # quick table
    pd.DataFrame(
        {
            **{f"node_{i}": node_np[:16, i] for i in range(min(16, node_np.shape[1]))},
            **{f"graph_{i}": graph_np[:16, i] for i in range(min(16, graph_np.shape[1]))},
        }
    ).to_parquet(evals_dir / "sample_head.parquet", index=False)

    log("Experiment completed.")


if __name__ == "__main__":

    def _parse_supported_dataset(s: str) -> SupportedDataset:
        return s if isinstance(s, SupportedDataset) else SupportedDataset(s)

    def get_flow_cli_args() -> FlowConfig:
        p = argparse.ArgumentParser(description="Real NVP V2")
        p.add_argument("--exp_dir_name", type=str, default=None)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--epochs", "-e", type=int, default=50)
        p.add_argument("--batch_size", "-bs", type=int, default=64)
        p.add_argument("--vsa", "-v", type=VSAModel, default=VSAModel.HRR)
        p.add_argument("--dataset", "-ds", type=_parse_supported_dataset, default=argparse.SUPPRESS)
        p.add_argument("--hv_dim", "-hd", type=int, default=88 * 88)
        p.add_argument("--lr", type=float, default=1e-4)
        p.add_argument("--weight_decay", "-wd", type=float, default=0.0)

        p.add_argument("--is_dev", "-dev", type=str2bool, default=0.0)

        # model capacity knobs
        p.add_argument("--num_flows", type=int, default=8)
        p.add_argument("--num_hidden_channels", type=int, default=512)
        p.add_argument("--use_act_norm", action="store_true")  # safer than type=bool

        # log-scale warmup knobs (optional to expose)
        p.add_argument("--smax_initial", type=float, default=1.0)
        p.add_argument("--smax_final", type=float, default=6.0)
        p.add_argument("--smax_warmup_epochs", type=int, default=15)

        # checkpointing
        p.add_argument("--continue_from", type=Path, default=None)
        p.add_argument("--resume_retrain_last_epoch", action="store_true")

        args = p.parse_args()
        return FlowConfig(**vars(args))

    if os.environ.get(LOCAL_DEV, None):
        log("Local HDC ...")
        run_experiment(
            cfg=FlowConfig(
                exp_dir_name="DEBUG",
                seed=42,
                epochs=500,
                batch_size=128,
                lr=1e-4,
                hv_dim=88 * 88,
                vsa=VSAModel.HRR,
                num_flows=4,
                num_hidden_channels=128,
                use_act_norm=True,
                is_dev=True,
                # continue_from="/Users/arvandkaveh/Projects/kit/graph_hdc/src/exp/real_nvp_v2/results/0_real_nvp_v2/DEBUG/models/last.ckpt"
            )
        )
    else:
        log("Cluster run ...")
        cfg = get_flow_cli_args()
        run_experiment(cfg)
