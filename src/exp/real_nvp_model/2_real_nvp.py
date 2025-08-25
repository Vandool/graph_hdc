import argparse
import datetime
import os
import random
import shutil
import string
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from math import prod
from pathlib import Path

import matplotlib.pyplot as plt
import normflows as nf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch_geometric.loader import DataLoader

from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import DatasetConfig, Features, FeatureConfig, IndexRange
from src.encoding.feature_encoders import CombinatoricIntegerEncoder
from src.encoding.graph_encoders import load_or_create_hypernet
from src.encoding.the_types import VSAModel
from src.normalizing_flow.models import AbstractNFModel
from src.utils.utils import GLOBAL_MODEL_PATH


# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
# ---------- tiny logger ----------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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
    device: str = "cuda"

    hv_dim: int = 7744
    vsa: VSAModel = VSAModel.HRR

    proj_graph_dim: int = 512
    proj_node_dim: int = 512
    use_layernorm: bool = True

    num_flows: int = 8
    num_hidden_channels: int = 512


def get_flow_cli_args() -> FlowConfig:
    parser = argparse.ArgumentParser(description="Real NVP Flow CLI")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--batch_size", "-bs", type=int, default=64)
    parser.add_argument("--vsa", "-v", type=VSAModel, required=True)
    parser.add_argument("--hv_dim", "-hd", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", "-wd", type=float, default=0.0)
    parser.add_argument("--device", "-dev", type=str, choices=["cpu","cuda"],
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return FlowConfig(**vars(parser.parse_args()))


class RealNVPLightning(AbstractNFModel):
    """
    Featurization:
      flat = [ LN(g) → Linear(D→Pg),  LN(mean(node_terms)) → Linear(D→Pn) ]
      latent_dim = Pg + Pn
    Flow:
      [MaskedAffineFlow ∘ Permute] × num_flows, base = DiagGaussian
    """
    def __init__(self, cfg: FlowConfig):
        super().__init__(cfg)
        D = cfg.hv_dim
        Pg, Pn = cfg.proj_graph_dim, cfg.proj_node_dim

        self.g_ln = nn.LayerNorm(D) if cfg.use_layernorm else nn.Identity()
        self.n_ln = nn.LayerNorm(D) if cfg.use_layernorm else nn.Identity()
        self.g_proj = nn.Linear(D, Pg, bias=False)
        self.n_proj = nn.Linear(D, Pn, bias=False)
        self.latent_dim = Pg + Pn

        mask = torch.tensor([(i % 2) for i in range(self.latent_dim)], dtype=torch.float32)
        self.register_buffer("mask", mask)

        flows = []
        for _ in range(cfg.num_flows):
            t_net = nf.nets.MLP([self.latent_dim, cfg.num_hidden_channels, cfg.num_hidden_channels, self.latent_dim],
                                init_zeros=True)
            s_net = nf.nets.MLP([self.latent_dim, cfg.num_hidden_channels, cfg.num_hidden_channels, self.latent_dim],
                                init_zeros=True)
            flows.append(nf.flows.MaskedAffineFlow(self.mask, t=t_net, s=s_net))
            flows.append(nf.flows.Permute(self.latent_dim, mode="swap"))
        base = nf.distributions.DiagGaussian(self.latent_dim, trainable=False)
        self.flow = nf.NormalizingFlow(q0=base, flows=flows)

    # ensure Lightning moves PyG Batch properly
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device)

    # ---- featurizer on a PyG Batch
    def _flat_from_batch(self, batch) -> torch.Tensor:
        """
        batch.graph_terms: [B, D]
        batch.node_terms : [sum_N, D]
        batch.batch      : [sum_N] graph ids
        returns flat     : [B, Pg+Pn]
        """
        assert hasattr(batch, "graph_terms") and hasattr(batch, "node_terms") and hasattr(batch, "batch")
        g = batch.graph_terms         # [B, D]
        n = batch.node_terms          # [sum_N, D]
        B, D = g.size(0), g.size(1)

        # per-graph mean over node_terms with pure PyTorch (no torch_scatter):
        # sums per graph id
        sums = torch.zeros(B, D, device=n.device, dtype=n.dtype)
        sums.index_add_(0, batch.batch, n)
        # counts per graph id
        counts = torch.bincount(batch.batch, minlength=B).clamp_min(1).unsqueeze(1).to(n.dtype)
        n_mean = sums / counts        # [B, D]

        g_feat = self.g_proj(self.g_ln(g))
        n_feat = self.n_proj(self.n_ln(n_mean))
        return torch.cat([g_feat, n_feat], dim=-1)  # [B, latent_dim]

    def training_step(self, batch, batch_idx):
        flat = self._flat_from_batch(batch)
        loss = self.nf_forward_kld(flat).mean()
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        flat = self._flat_from_batch(batch)
        loss = self.nf_forward_kld(flat).mean()
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

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


def _evaluate_loader_nll(model: RealNVPLightning, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            flat = model._flat_from_batch(batch)
            kld = model.nf_forward_kld(flat)  # [B]
            outs.append(kld.detach().cpu())
    return torch.cat(outs).numpy()


def _hist(figpath: Path, data: np.ndarray, title: str, xlabel: str, bins: int = 80):
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel);
    plt.ylabel("count");
    plt.tight_layout()
    plt.savefig(figpath);
    plt.close()


def _pairwise_cosine(x: np.ndarray, m: int = 2000) -> np.ndarray:
    # sample up to m pairs
    n = x.shape[0]
    if n < 2: return np.array([])
    idx = np.random.choice(n, size=(min(m, n - 1), 2), replace=True)
    a = x[idx[:, 0]];
    b = x[idx[:, 1]]
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.sum(an * bn, axis=1)


def run_experiment(cfg: FlowConfig):
    # ----- setup dirs -----
    log("Parsing args done. Starting run_experiment …")
    dirs = setup_exp()
    exp_dir = Path(dirs["exp_dir"])
    models_dir = Path(dirs["models_dir"])
    evals_dir = Path(dirs["evals_dir"])
    artefacts_dir = Path(dirs["artefacts_dir"])

    # ----- hypernet config (kept for provenance; not needed in this flow) -----
    ds_name = "ZincSmilesHRR7744"
    zinc_feature_bins = [9, 6, 3, 4]
    dataset_config = DatasetConfig(
        seed=cfg.seed,
        name=ds_name,
        vsa=cfg.vsa,
        hv_dim=cfg.hv_dim,
        device=cfg.device,
        node_feature_configs=OrderedDict([
            (Features.ATOM_TYPE, FeatureConfig(
                count=prod(zinc_feature_bins),
                encoder_cls=CombinatoricIntegerEncoder,
                index_range=IndexRange((0, 4)),
                bins=zinc_feature_bins,
            )),
        ]),
    )

    log("Loading/creating hypernet …")
    _ = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, ds_name=ds_name, cfg=dataset_config).to(cfg.device).eval()
    log("Hypernet ready.")

    # ----- W&B -----
    run = wandb.run or wandb.init(
        project="realnvp-hdc-overfit",
        config=cfg.__dict__,
        name=f"run_{cfg.hv_dim}_{cfg.seed}",
        reinit=True
    )
    run.tags = [f"hv_dim={cfg.hv_dim}", f"vsa={cfg.vsa.value}", "dataset=ZincSmiles"]

    wandb_logger = WandbLogger(log_model=True, experiment=run)

    # ----- datasets / loaders -----
    train_dataset = ZincSmiles(split="train", enc_suffix="HRR7744")[:64]
    validation_dataset = ZincSmiles(split="valid", enc_suffix="HRR7744")[:64]

    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=2, pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=2, pin_memory=torch.cuda.is_available(), drop_last=False,
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
        log_every_n_steps=10,
        enable_progress_bar=True,
        detect_anomaly=False,
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
        plt.xlabel("Epoch");
        plt.ylabel("Loss");
        plt.title("Loss")
        plt.legend();
        plt.tight_layout()
        plt.savefig(artefacts_dir / "train_val_loss.png");
        plt.close()

    # =================================================================
    # Post-training analysis: load best, evaluate NLL, sample & log
    # =================================================================
    best_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    assert best_path and Path(best_path).exists(), "No checkpoint saved."
    log(f"Loading best checkpoint: {best_path}")

    # Lightning checkpoints include state_dict; instantiate with cfg and load
    best_model = RealNVPLightning(cfg)
    state = torch.load(best_path, map_location="cpu")
    best_model.load_state_dict(state["state_dict"])
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    best_model.to(device).eval()

    # ---- per-sample NLL (really the KL objective) on validation ----
    val_nll = _evaluate_loader_nll(best_model, validation_dataloader, device)
    pd.DataFrame({"nll": val_nll}).to_parquet(evals_dir / "val_nll.parquet", index=False)
    _hist(artefacts_dir / "val_nll_hist.png", val_nll, "Validation NLL", "NLL")
    wandb.log({
        "val_nll_mean": float(np.mean(val_nll)),
        "val_nll_std": float(np.std(val_nll)),
        "val_nll_min": float(np.min(val_nll)),
        "val_nll_max": float(np.max(val_nll)),
        "val_nll_hist": wandb.Histogram(val_nll),
    })

    # ---- sample from the flow ----
    with torch.no_grad():
        samples, logs = best_model.sample(2048)  # samples ~ [K, latent_dim]
    samples_np = samples.detach().cpu().numpy()
    logs_np = logs.detach().cpu().numpy() if torch.is_tensor(logs) else np.asarray(logs)

    # sample stats
    s_norm = np.linalg.norm(samples_np, axis=1)
    s_cos = _pairwise_cosine(samples_np, m=4000)

    pd.DataFrame({"sample_norm": s_norm}).to_parquet(evals_dir / "sample_norms.parquet", index=False)
    np.save(artefacts_dir / "samples.npy", samples_np)
    np.save(artefacts_dir / "sample_logs.npy", logs_np)

    _hist(artefacts_dir / "sample_norm_hist.png", s_norm, "Sample L2 norm", "||z||")
    if s_cos.size:
        _hist(artefacts_dir / "sample_cos_hist.png", s_cos, "Sample pairwise cosine", "cos")

    wandb.log({
        "sample_norm_mean": float(np.mean(s_norm)),
        "sample_norm_std": float(np.std(s_norm)),
        "sample_norm_hist": wandb.Histogram(s_norm),
        "sample_cos_hist": wandb.Histogram(s_cos) if s_cos.size else None,
    })

    # quick table of a few samples
    tbl = pd.DataFrame(samples_np[:32])
    tbl_path = evals_dir / "sample_head.parquet"
    tbl.to_parquet(tbl_path, index=False)
    wandb.save(str(tbl_path))

    log("==== The Experiment is done! ====")


def sweep_entrypoint():
    run = wandb.init()
    cfg_dict = run.config.as_dict()

    # coerce types
    cfg = FlowConfig(
        seed=int(cfg_dict.get("seed", 42)),
        epochs=int(cfg_dict.get("epochs", 10)),
        batch_size=int(cfg_dict.get("batch_size", 64)),
        lr=float(cfg_dict.get("lr", 1e-3)),
        weight_decay=float(cfg_dict.get("weight_decay", 0.0)),
        device="cuda" if torch.cuda.is_available() else "cpu",
        hv_dim=int(cfg_dict.get("hv_dim", 7744)),
        vsa=VSAModel(cfg_dict.get("vsa", "HRR")),
        proj_graph_dim=512,
        proj_node_dim=512,
        use_layernorm=True,
        num_flows=int(cfg_dict.get("num_flows", 8)),
        num_hidden_channels=int(cfg_dict.get("num_hidden_channels", 512)),
    )

    run_experiment(cfg)


if __name__ == "__main__":
    if os.environ.get("WANDB_SWEEP", "0") == "1":
        sweep_entrypoint()
    else:
        run_experiment(get_flow_cli_args())
