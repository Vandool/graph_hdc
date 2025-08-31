import argparse
import datetime
import math
import os
import random
import shutil
import string
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import normflows as nf
import numpy as np
import pandas as pd
import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch_geometric.loader import DataLoader
from torchhd import HRRTensor

from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import DatasetConfig, FeatureConfig, Features, IndexRange
from src.encoding.feature_encoders import CombinatoricIntegerEncoder
from src.encoding.graph_encoders import HyperNet, load_or_create_hypernet
from src.encoding.the_types import VSAModel
from src.normalizing_flow.models import AbstractNFModel
from src.utils.utils import GLOBAL_MODEL_PATH

LOCAL_DEV = "LOCAL_HDC"

PROJECT_NAME = "real_nvp_v2"


# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
# ---------- tiny logger ----------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        slug = f"{datetime.now(UTC).strftime('%Y-%m-%d_%H-%M-%S')}_{''.join(random.choices(string.ascii_lowercase, k=4))}"
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
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda"

    hv_dim: int = 7744
    vsa: VSAModel = VSAModel.HRR
    num_flows: int = 8
    num_hidden_channels: int = 512

    smax_initial: float = 1.0
    smax_final: float = 12
    smax_warmup_epochs: float = 15

    use_act_norm: bool = False

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
        use_act_norm = getattr(cfg, "use_act_norm", True)
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
        graph_terms = x[:, self.D:].contiguous()
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
        steps_per_epoch = max(1, getattr(self.trainer, "estimated_stepping_batches", 1000) // max(1,
                                                                                                  self.trainer.max_epochs))
        warmup = int(0.05 * self.trainer.max_epochs) * steps_per_epoch
        total = self.trainer.max_epochs * steps_per_epoch
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda step: min(1.0, step / max(1, warmup)) * 0.5 * (
                    1 + math.cos(math.pi * max(0, step - warmup) / max(1, total - warmup)))
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def on_fit_start(self):
        # H100 tip: get tensor cores w/o AMP
        try:
            import torch

            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

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
def fit_blockwise_standardization(model, loader, hv_dim: int, max_batches: int | None = None, device="cpu"):
    cnt_node = 0
    sum_node = 0.0
    sumsq_node = 0.0
    cnt_graph = 0
    sum_graph = 0.0
    sumsq_graph = 0.0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches: break
        batch = batch.to(device)
        x = model._flat_from_batch(batch).to(torch.float32)  # [B, 2D]
        node = x[:, :hv_dim].reshape(-1)
        graph = x[:, hv_dim:].reshape(-1)
        cnt_node += node.numel()
        sum_node += float(node.sum().item())
        sumsq_node += float((node * node).sum().item())
        cnt_graph += graph.numel()
        sum_graph += float(graph.sum().item())
        sumsq_graph += float((graph * graph).sum().item())
    mu_node = sum_node / max(1, cnt_node)
    var_node = max(0.0, sumsq_node / max(1, cnt_node) - mu_node * mu_node)
    mu_graph = sum_graph / max(1, cnt_graph)
    var_graph = max(0.0, sumsq_graph / max(1, cnt_graph) - mu_graph * mu_graph)
    sigma_node = max(1e-6, var_node ** 0.5)
    sigma_graph = max(1e-6, var_graph ** 0.5)

    # broadcast to 2D vector
    mu = torch.cat([torch.full((hv_dim,), mu_node), torch.full((hv_dim,), mu_graph)])
    sigma = torch.cat([torch.full((hv_dim,), sigma_node), torch.full((hv_dim,), sigma_graph)])
    model.set_standardization(mu, sigma)


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def plot_train_val_loss(df: pd.DataFrame, artefacts_dir: Path):
    # possible names across PL versions / logging configs
    train_epoch_keys = ["train_loss_epoch", "train_loss", "epoch_train_loss"]
    val_epoch_keys   = ["val_loss", "val/loss", "validation_loss"]

    train_col = _first_existing(df, train_epoch_keys)
    val_col   = _first_existing(df, val_epoch_keys)
    epoch_col = _first_existing(df, ["epoch", "step"])

    if epoch_col is None or (train_col is None and val_col is None):
        log("Skip plot: no epoch/metric columns in metrics.csv for this run.")
        return

    plt.figure(figsize=(8, 5))
    if train_col is not None:
        train = df[df[train_col].notna()]
        if not train.empty:
            plt.plot(train[epoch_col], train[train_col], label=train_col)
    if val_col is not None:
        val = df[df[val_col].notna()]
        if not val.empty:
            plt.plot(val[epoch_col], val[val_col], label=val_col)

    plt.xlabel(epoch_col)
    plt.ylabel("loss")
    plt.title("Train vs. Validation Loss")
    plt.legend()
    plt.tight_layout()
    artefacts_dir.mkdir(exist_ok=True)
    out = artefacts_dir / "train_val_loss.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved train/val loss plot to {out}")


def pick_precision():
    # Works on A100/H100 if BF16 is supported by the PyTorch/CUDA build.
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "bf16-mixed"  # safest + fast on H100/A100
        else:
            return "16-mixed"  # widely supported fallback
    return 32  # CPU or MPS


class TimeLoggingCallback(Callback):
    def setup(self, trainer, pl_module, stage=None):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.start_time
        trainer.logger.log_metrics({"elapsed_time_sec": elapsed}, step=trainer.current_epoch)


class PeriodicNLLEval(Callback):
    def __init__(self, val_loader, artefacts_dir: Path, every_n_epochs: int = 50,
                 max_batches: int | None = 200, log_hist: bool = False):
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

        stats = _eval_flow_metrics(pl_module, self.val_loader, pl_module.device,
                                   hv_dim=pl_module.D, max_batches=self.max_batches)
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


@torch.no_grad()
def _eval_flow_metrics(model, loader, device, hv_dim: int, max_batches: int | None = None):
    """
    Returns arrays for core comparable metrics:
      - bpd_model, bpd_gauss, delta_bpd, bpd_stdspace, nll_model
    """
    model.eval()
    ln2 = math.log(2.0)
    dim = 2 * hv_dim
    const_term = 0.5 * dim * math.log(2 * math.pi)

    nll_model_list, bpd_std_list, bpd_gauss_list = [], [], []

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = batch.to(device)
        flat = model._flat_from_batch(batch)

        z, log_det_corr = model._pretransform(flat)  # z: [B, dim]; log_det_corr: float
        nll_stdspace = -model.flow.log_prob(z)  # [B] nats
        quad = 0.5 * (z ** 2).sum(dim=1)  # [B]
        nll_gauss_stdspace = quad + const_term  # [B]

        nll_model = nll_stdspace + log_det_corr  # [B]
        nll_gauss = nll_gauss_stdspace + log_det_corr  # [B]

        mask = torch.isfinite(nll_model) & torch.isfinite(nll_stdspace) & torch.isfinite(nll_gauss)
        if not mask.any():
            continue

        nll_model = nll_model[mask]
        bpd_stdspace = nll_stdspace[mask] / (dim * ln2)
        bpd_gauss = nll_gauss[mask] / (dim * ln2)

        nll_model_list.append(nll_model.detach().cpu())
        bpd_std_list.append(bpd_stdspace.detach().cpu())
        bpd_gauss_list.append(bpd_gauss.detach().cpu())

    if not nll_model_list:
        return {}

    nll_model = torch.cat(nll_model_list).numpy()
    bpd_stdspace = torch.cat(bpd_std_list).numpy()
    bpd_gauss = torch.cat(bpd_gauss_list).numpy()
    bpd_model = nll_model / (dim * ln2)
    delta_bpd = bpd_model - bpd_gauss

    return {
        "bpd_model": bpd_model,
        "bpd_gauss": bpd_gauss,
        "delta_bpd": delta_bpd,
        "bpd_stdspace": bpd_stdspace,
        "nll_model": nll_model,
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


def _evaluate_loader_nll(model, loader, device):
    stats = _eval_flow_metrics(model, loader, device, hv_dim=model.D)
    return stats.get("nll_model", np.empty((0,), dtype=np.float32))


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


def run_experiment(cfg: FlowConfig, local_dev: bool = False):
    pprint(cfg)
    # ----- setup dirs -----
    log("Parsing args done. Starting run_experiment …")
    dirs = setup_exp(cfg.exp_dir_name)
    exp_dir = Path(dirs["exp_dir"])
    models_dir = Path(dirs["models_dir"])
    evals_dir = Path(dirs["evals_dir"])
    artefacts_dir = Path(dirs["artefacts_dir"])

    seed_everything(cfg.seed)

    # ----- hypernet config (kept for provenance; not needed in this flow) -----
    ds_name = "ZincSmilesHRR7744"
    zinc_feature_bins = [9, 6, 3, 4]
    dataset_config = DatasetConfig(
        seed=cfg.seed,
        name=ds_name,
        vsa=cfg.vsa,
        hv_dim=cfg.hv_dim,
        device=cfg.device,
        node_feature_configs=OrderedDict(
            [
                (
                    Features.ATOM_TYPE,
                    FeatureConfig(
                        count=math.prod(zinc_feature_bins),
                        encoder_cls=CombinatoricIntegerEncoder,
                        index_range=IndexRange((0, 4)),
                        bins=zinc_feature_bins,
                    ),
                ),
            ]
        ),
    )

    log(f"Loading/creating hypernet … on device {cfg.device}")
    hypernet: HyperNet = (
        load_or_create_hypernet(path=GLOBAL_MODEL_PATH, cfg=dataset_config).to(cfg.device).eval()
    )
    assert torch.equal(hypernet.nodes_codebook, hypernet.node_encoder_map[Features.ATOM_TYPE][0].codebook)
    log("Hypernet ready.")

    # ----- datasets / loaders -----
    train_dataset = ZincSmiles(split="train", enc_suffix="HRR7744")
    validation_dataset = ZincSmiles(split="valid", enc_suffix="HRR7744")

    # pick worker counts per GPU; tune for your cluster
    num_workers = 8 if torch.cuda.is_available() else 0
    if local_dev:
        train_dataset = train_dataset[:cfg.batch_size]
        validation_dataset = validation_dataset[:cfg.batch_size]
        num_workers = 0

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
        drop_last=True,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
        drop_last=False,
    )
    log(f"Datasets ready. train={len(train_dataset)} valid={len(validation_dataset)}")

    # ----- model / trainer -----
    model = RealNVPV2Lightning(cfg)
    if cfg.continue_from is None and not cfg.use_act_norm:
        fit_blockwise_standardization(model, train_dataloader, hv_dim=cfg.hv_dim, device=cfg.device)

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

    # ----- W&B -----
    loggers = [csv_logger]
    if not local_dev:
        run = wandb.run or wandb.init(
            project=os.environ.get("WANDB_PROJECT", PROJECT_NAME),
            entity=os.environ.get("WANDB_ENTITY"),
            name=os.environ.get("WANDB_NAME", f"run_{cfg.hv_dim}_{cfg.seed}"),
            config=cfg.__dict__,
            reinit=True,
        )
        run.tags = [
            f"hv_dim={cfg.hv_dim}",
            f"vsa={cfg.vsa.value}",
            f"flows={cfg.num_flows}",
            f"hidden={cfg.num_hidden_channels}",
            f"actnorm={cfg.use_act_norm}",
            "dataset=ZincSmiles",
        ]
        loggers.append(WandbLogger(log_model=True, experiment=run))

    trainer = Trainer(
        max_epochs=cfg.epochs,
        logger=loggers,
        callbacks=[checkpoint_callback, lr_monitor, time_logger, periodic_nll, early_stopping],
        default_root_dir=str(exp_dir),
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        log_every_n_steps=25 if not local_dev else 1,
        enable_progress_bar=True,
        deterministic=False,
        precision=pick_precision()
    )

    # ----- train -----
    resume_path: Path | None = str(cfg.continue_from) if cfg.continue_from else None
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader, ckpt_path=resume_path)

    # ----- curves to parquet / png -----
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
            print(f"Final losses → train: {float(train_last.values[-1]) if not train_last.empty else 'n/a'} "
                  f"| val: {float(val_last.values[-1]) if not val_last.empty else 'n/a'}")

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
    best_model.to(cfg.device).eval()

    # ---- per-sample NLL (really the KL objective) on validation ----
    val_stats = _eval_flow_metrics(best_model, validation_dataloader, cfg.device, hv_dim=cfg.hv_dim)
    nll_arr = val_stats.get("nll_model", np.empty((0,), dtype=np.float32))
    if nll_arr.size:
        # save full arrays for later deep-dive
        pd.DataFrame({
            "nll_model": val_stats["nll_model"],
            "bpd_model": val_stats["bpd_model"],
            "bpd_gauss": val_stats["bpd_gauss"],
            "bpd_stdspace": val_stats["bpd_stdspace"],
            "delta_bpd": val_stats["delta_bpd"],
        }).to_parquet(evals_dir / "val_metrics_final.parquet", index=False)

        # simple hist PNGs (optional)
        _hist(artefacts_dir / "val_bpd_model_hist.png", val_stats["bpd_model"], "Validation bpd (model)", "bpd")
        _hist(artefacts_dir / "val_bpd_gauss_hist.png", val_stats["bpd_gauss"], "Validation bpd (Gaussian)", "bpd")
        _hist(artefacts_dir / "val_delta_bpd_hist.png", val_stats["delta_bpd"], "Validation Δbpd (model - Gaussian)",
              "Δbpd")

        # scalar summary (prefixed under eval/val/*)
        summary = _summarize_arrays(val_stats)
        summary_payload = {f"eval/val/{k}": v for k, v in summary.items()}
        log(summary_payload)

        if wandb.run:
            wandb.log(summary_payload)
            # optional: histograms in W&B
            wandb.log({
                "eval/val/bpd_model_hist": wandb.Histogram(val_stats["bpd_model"]),
                "eval/val/bpd_gauss_hist": wandb.Histogram(val_stats["bpd_gauss"]),
                "eval/val/delta_bpd_hist": wandb.Histogram(val_stats["delta_bpd"]),
            })
    else:
        log("No finite NLL values collected; skipping NLL stats.")

    # ---- sample from the flow ----
    with torch.no_grad():
        node_s, graph_s, logs = best_model.sample_split(1024)  # each [K, D]

    node_s = node_s.as_subclass(HRRTensor)
    graph_s = graph_s.as_subclass(HRRTensor)

    log(f"node_s device: {node_s.device!s}")
    log(f"graph_s device: {graph_s.device!s}")
    log(f"Hypernet node codebook device: {hypernet.nodes_codebook.device!s}")

    node_counters = hypernet.decode_order_zero_counter(node_s)
    for i, ctr in node_counters.items():
        if i >= 16:
            break
        log(f"Sample {i}: total nodes: {ctr.total()}, total edges: {sum(e + 1 for _, e, _, _ in ctr)} \n{ctr}")

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
        wandb.log(
            {
                "sample_node_norm_mean": float(np.mean(node_norm)),
                "sample_node_norm_std": float(np.std(node_norm)),
                "sample_graph_norm_mean": float(np.mean(graph_norm)),
                "sample_graph_norm_std": float(np.std(graph_norm)),
                "sample_node_norm_hist": wandb.Histogram(node_norm),
                "sample_graph_norm_hist": wandb.Histogram(graph_norm),
                "sample_node_cos_hist": wandb.Histogram(node_cos) if node_cos.size else None,
                "sample_graph_cos_hist": wandb.Histogram(graph_cos) if graph_cos.size else None,
            }
        )

    # quick table
    pd.DataFrame(
        {
            **{f"node_{i}": node_np[:16, i] for i in range(min(16, node_np.shape[1]))},
            **{f"graph_{i}": graph_np[:16, i] for i in range(min(16, graph_np.shape[1]))},
        }
    ).to_parquet(evals_dir / "sample_head.parquet", index=False)

    log("Experiment completed.")


def sweep_entrypoint():
    run = wandb.init()
    cfg_dict = run.config.as_dict()

    cfg = FlowConfig(
        seed=int(cfg_dict.get("seed", 42)),
        epochs=int(cfg_dict.get("epochs", 10)),
        batch_size=int(cfg_dict.get("batch_size", 64)),
        lr=float(cfg_dict.get("lr", 1e-3)),
        weight_decay=float(cfg_dict.get("weight_decay", 0.0)),
        device="cuda" if torch.cuda.is_available() else "cpu",
        hv_dim=int(cfg_dict.get("hv_dim", 7744)),
        vsa=VSAModel.HRR,
        num_flows=int(cfg_dict.get("num_flows", 8)),
        num_hidden_channels=int(cfg_dict.get("num_hidden_channels", 128)),
    )
    run_experiment(cfg)


if __name__ == "__main__":
    def get_flow_cli_args() -> FlowConfig:
        p = argparse.ArgumentParser(description="Real NVP V2")
        p.add_argument("--exp_dir_name", type=str, default=None)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--epochs", "-e", type=int, default=50)
        p.add_argument("--batch_size", "-bs", type=int, default=64)
        p.add_argument("--vsa", "-v", type=VSAModel, default=VSAModel.HRR)
        p.add_argument("--hv_dim", "-hd", type=int, default=88 * 88)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--weight_decay", "-wd", type=float, default=0.0)
        p.add_argument("--device", "-dev", choices=["cpu", "cuda", "mps"],
                       default="cuda" if torch.cuda.is_available() else "cpu")

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
        run_experiment(cfg=FlowConfig(
            exp_dir_name="DEBUG",
            seed=42,
            epochs=500,
            batch_size=128,
            lr=1e-3,
            device="mps",
            hv_dim=88 * 88,
            vsa=VSAModel.HRR,
            num_flows=4,
            num_hidden_channels=128,
            use_act_norm=True,
            continue_from="/Users/arvandkaveh/Projects/kit/graph_hdc/src/exp/real_nvp_v2/results/0_real_nvp_v2/DEBUG/models/last.ckpt"
        ), local_dev=True)
    else:
        log("Cluster run ...")
        run_experiment(get_flow_cli_args())
