import argparse
import datetime
import enum
import math
import os
import random
import shutil
import string
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import normflows as nf
import numpy as np
import pandas as pd
import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch_geometric.loader import DataLoader
from torchhd import HRRTensor

from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import Features, SupportedDataset
from src.encoding.graph_encoders import load_or_create_hypernet
from src.encoding.the_types import VSAModel
from src.normalizing_flow.models import AbstractNFModel
from src.utils.utils import GLOBAL_MODEL_PATH, pick_device, report_node_distribution


# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
# ---------- tiny logger ----------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def pick_precision():
    if not torch.cuda.is_available():
        return 32
    # A100/H100 etc → bf16
    if torch.cuda.is_bf16_supported():
        return "bf16-mixed"
    # Volta/Turing/Ampere+ (sm >= 70) → fp16 mixed
    major, minor = torch.cuda.get_device_capability()
    if major >= 7:
        return "16-mixed"
    # Maxwell/Pascal and older → plain fp32
    return 32


os.environ.setdefault("PYTHONUNBUFFERED", "1")


def setup_exp() -> dict:
    script_path = Path(__file__).resolve()
    experiments_path = script_path.parent
    script_stem = script_path.stem

    base_dir = experiments_path / "results" / script_stem
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Setting up experiment in {base_dir}")
    now = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{''.join(random.choices(string.ascii_lowercase, k=4))}"
    exp_dir = base_dir / now
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
    seed: int = 42
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0

    # HDC / encoder
    hv_dim: int = 40 * 40  # 1600
    vsa: VSAModel = VSAModel.HRR
    dataset: SupportedDataset = SupportedDataset.QM9_SMILES_HRR_1600

    # Model knobs
    num_flows: int = 8
    num_hidden_channels: int = 512

    smax_initial = 1.0
    smax_final = 12
    smax_warmup_epochs = 15


def get_flow_cli_args() -> FlowConfig:
    def _parse_supported_dataset(s: str) -> SupportedDataset:
        return s if isinstance(s, SupportedDataset) else SupportedDataset(s)

    p = argparse.ArgumentParser(description="Real NVP Flow CLI")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", "-e", type=int, default=10)
    p.add_argument("--batch_size", "-bs", type=int, default=64)
    p.add_argument("--vsa", "-v", type=VSAModel, required=True)
    p.add_argument("--hv_dim", "-hd", type=int, required=True)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", "-wd", type=float, default=0.0)
    p.add_argument("--dataset", "-ds", type=_parse_supported_dataset, default=argparse.SUPPRESS)
    return FlowConfig(**vars(p.parse_args()))


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


class RealNVPLightning(AbstractNFModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        D = int(cfg.hv_dim)
        self.D = D
        self.flat_dim = 2 * D

        mask = (torch.arange(self.flat_dim) % 2).to(torch.float32)
        self.register_buffer("mask0", mask)

        self.s_modules = []  # keep handles for warmup/logging
        flows = []
        hidden = int(cfg.num_hidden_channels)
        for _ in range(int(cfg.num_flows)):
            layers = [self.flat_dim, hidden, hidden, self.flat_dim]
            t_net = nf.nets.MLP(layers, init_zeros=True)
            s_net = BoundedMLP(layers, smax=getattr(cfg, "smax_final", 12))
            self.s_modules.append(s_net)

            flows.append(nf.flows.MaskedAffineFlow(self.mask0, t=t_net, s=s_net))
            flows.append(nf.flows.Permute(self.flat_dim, mode="shuffle"))

        base = nf.distributions.DiagGaussian(self.flat_dim, trainable=False)
        self.flow = nf.NormalizingFlow(q0=base, flows=flows)

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
    def sample_split(self, num_samples: int):
        """
        Returns:
          node_terms:  [num_samples, D]
          graph_terms: [num_samples, D]
          logs
        """
        z, _logs = self.sample(num_samples)  # [num_samples, 2D]
        node_terms = z[:, : self.D].contiguous()
        graph_terms = z[:, self.D :].contiguous()
        return node_terms, graph_terms, _logs


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------


def plot_train_val_loss(df, artefacts_dir):
    train = df[df["train_loss_epoch"].notna()]
    val = df[df["val_loss"].notna()]

    plt.figure(figsize=(8, 5))
    plt.plot(train["epoch"], train["train_loss_epoch"], label="Train Loss")
    plt.plot(val["epoch"], val["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs. Validation Loss")
    plt.legend()
    plt.tight_layout()

    artefacts_dir.mkdir(exist_ok=True)
    plt.savefig(artefacts_dir / "train_val_loss.png")
    plt.close()
    print(f"Saved train/val loss plot to {artefacts_dir / 'train_val_loss.png'}")


def get_device():
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"CUDA is available. Detected {count} GPU device{'s' if count != 1 else ''}.")
        return torch.device("cuda")
    print("CUDA is not available.")
    return torch.device("cpu")


class TimeLoggingCallback(Callback):
    def setup(self, trainer, pl_module, stage=None):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.start_time
        trainer.logger.log_metrics({"elapsed_time_sec": elapsed}, step=trainer.current_epoch)


def _evaluate_loader_nll(model, loader, device):
    model.eval()
    outs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            flat = model._flat_from_batch(batch)
            kld = model.nf_forward_kld(flat)  # [B] ideally
            if torch.is_tensor(kld):
                if kld.ndim == 0:  # library gave scalar
                    kld = kld.expand(flat.size(0))
                kld = kld[torch.isfinite(kld)]
                if kld.numel():
                    outs.append(kld.detach().cpu())
    return torch.cat(outs).numpy() if outs else np.empty((0,), dtype=np.float32)


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


def run_experiment(cfg: FlowConfig):
    # ----- setup dirs -----
    log("Parsing args done. Starting run_experiment …")
    dirs = setup_exp()
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

    # ----- W&B -----
    run = wandb.run or wandb.init(
        project="realnvp", config=cfg.__dict__, name=f"run_{cfg.hv_dim}_{cfg.seed}", reinit=True
    )
    run.tags = [f"hv_dim={cfg.hv_dim}", f"vsa={cfg.vsa.value}", "dataset=ZincSmiles"]

    wandb_logger = WandbLogger(log_model=True, experiment=run)

    # ----- datasets / loaders -----
    log(f"Loading {cfg.dataset.value} pair datasets.")
    if cfg.dataset == SupportedDataset.QM9_SMILES_HRR_1600:
        train_dataset = QM9Smiles(split="train", enc_suffix="HRR7744")
        validation_dataset = QM9Smiles(split="valid", enc_suffix="HRR7744")
    elif cfg.dataset == SupportedDataset.ZINC_SMILES_HRR_7744:
        train_dataset = ZincSmiles(split="train", enc_suffix="HRR7744")
        validation_dataset = ZincSmiles(split="valid", enc_suffix="HRR7744")
    log(
        f"Pairs loaded for {cfg.dataset.value}. train_pairs_full_size={len(train_dataset)} valid_pairs_full_size={len(validation_dataset)}"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=6,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=6,
    )
    log(f"Datasets ready. train={len(train_dataset)} valid={len(validation_dataset)}")

    # ----- model / trainer -----
    model = RealNVPLightning(cfg)

    csv_logger = CSVLogger(save_dir=str(evals_dir), name="logs")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath=str(models_dir),
        filename="epoch{epoch:02d}-val{val_loss:.4f}",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    time_logger = TimeLoggingCallback()

    trainer = Trainer(
        max_epochs=cfg.epochs,
        logger=[csv_logger, wandb_logger],
        callbacks=[checkpoint_callback, lr_monitor, time_logger],
        default_root_dir=str(exp_dir),
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        log_every_n_steps=500,
        enable_progress_bar=True,
        deterministic=False,
        precision=pick_precision(),
        num_sanity_val_steps=100,
        limit_val_batches=100,
    )

    # ----- train -----
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

    # ----- curves to parquet / png -----
    metrics_path = Path(csv_logger.log_dir) / "metrics.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        df.to_parquet(evals_dir / "metrics.parquet", index=False)
        # train/val loss plot
        train = df[df["train_loss_epoch"].notna()]
        val = df[df["val_loss"].notna()]
        plt.figure(figsize=(8, 5))
        if not train.empty:
            plt.plot(train["epoch"], train["train_loss_epoch"], label="Train")
        if not val.empty:
            plt.plot(val["epoch"], val["val_loss"], label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(artefacts_dir / "train_val_loss.png")
        plt.close()

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
    best_model = RealNVPLightning(cfg)
    state = torch.load(best_path, map_location="cpu", weights_only=False)
    best_model.load_state_dict(state["state_dict"])
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    best_model.to(device).eval()

    # ---- per-sample NLL (really the KL objective) on validation ----
    val_nll = _evaluate_loader_nll(best_model, validation_dataloader, device)
    if val_nll.size:
        # report bits per dimension
        bpd = val_nll / ((2 * cfg.hv_dim) * np.log(2))  # [node|graph]
        df = pd.DataFrame({"nll": val_nll, "bpd": bpd})
        df.to_parquet(evals_dir / "val_nll.parquet", index=False)

        _hist(artefacts_dir / "val_bpd_hist.png", bpd, "Validation bpd", "bpd")
        _hist(artefacts_dir / "val_nll_hist.png", val_nll, "Validation NLL", "NLL")
        wandb.log(
            {
                "val_nll_mean": float(np.mean(val_nll)),
                "val_nll_std": float(np.std(val_nll)),
                "val_nll_min": float(np.min(val_nll)),
                "val_nll_max": float(np.max(val_nll)),
                "val_nll_hist": wandb.Histogram(val_nll),
                "val_bpd_mean": float(np.mean(bpd)),
                "val_bpd_std": float(np.std(bpd)),
                "val_bpd_min": float(np.min(bpd)),
                "val_bpd_max": float(np.max(bpd)),
                "val_bpd_hist": wandb.Histogram(bpd),
            }
        )
    else:
        log("No finite NLL values collected; skipping NLL stats.")

    # ---- sample from the flow ----
    with torch.no_grad():
        node_s, graph_s, logs = best_model.sample_split(4096)  # each [K, D]


    node_s = node_s.as_subclass(HRRTensor)
    graph_s = graph_s.as_subclass(HRRTensor)

    node_counters = hypernet.decode_order_zero_counter(node_s)
    report = report_node_distribution(node_types=node_counters, artefact_dir=artefacts_dir)
    log(report["summary"])
    log(report["paths"])


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

    seed_everything(cfg.seed)
    run_experiment(cfg)


if __name__ == "__main__":
    if os.environ.get("WANDB_SWEEP", "0") == "1":
        sweep_entrypoint()
    else:
        run_experiment(get_flow_cli_args())
