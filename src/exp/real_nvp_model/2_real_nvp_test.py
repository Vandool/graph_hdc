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
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch_geometric.loader import DataLoader

from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import DatasetConfig, Features, FeatureConfig, IndexRange
from src.encoding.feature_encoders import CombinatoricIntegerEncoder
from src.encoding.graph_encoders import load_or_create_hypernet, HyperNet
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
    parser.add_argument("--device", "-dev", type=str, choices=["cpu", "cuda"],
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return FlowConfig(**vars(parser.parse_args()))


class RealNVPLightning(AbstractNFModel):
    """
    Exact RealNVP on flat = concat([node_terms, graph_terms]) ∈ R^(2*D).
    No projections, no pooling. Fully invertible in the original space.

    Masks: alternating 0101... with random permutations inserted between blocks.
    Base: standard diagonal Gaussian.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        D = int(cfg.hv_dim)
        self.D = D
        self.flat_dim = 2 * D  # node_terms (D) + graph_terms (D)

        # --- Build RealNVP stack ---
        # Start with a simple alternating mask
        mask = torch.arange(self.flat_dim) % 2  # 0,1,0,1,...
        mask = mask.to(torch.float32)
        self.register_buffer("mask0", mask)

        flows = []
        for k in range(int(cfg.num_flows)):
            # Subnets for shift (t) and log-scale (s)
            # Keep them small; params blow up fast with 15,488 dims.
            hidden = int(cfg.num_hidden_channels)
            layers = [self.flat_dim, hidden, hidden, self.flat_dim]

            t_net = nf.nets.MLP(layers, init_zeros=True)  # identity at start
            s_net = nf.nets.MLP(layers, init_zeros=True)

            flows.append(nf.flows.MaskedAffineFlow(self.mask0, t=t_net, s=s_net))

            # Permute to change which dims are updated next block
            flows.append(nf.flows.Permute(self.flat_dim, mode="shuffle"))

        base = nf.distributions.DiagGaussian(self.flat_dim, trainable=False)
        self.flow = nf.NormalizingFlow(q0=base, flows=flows)

    # Lightning will move the PyG Batch; just ensure we return batch on device
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device)

    # ---- flattener: exact concatenation, no projections/pooling
    def _flat_from_batch(self, batch) -> torch.Tensor:
        """
        Expects:
          batch.node_terms: [B, D]
          batch.graph_terms: [B, D]
        Returns:
          flat: [B, 2D]
        """
        assert hasattr(batch, "node_terms") and hasattr(batch, "graph_terms"), \
            "Exact RealNVP needs node_terms and graph_terms per-graph, each shape [B, D]."
        n = batch.node_terms
        g = batch.graph_terms
        assert n.dim() == 2 and g.dim() == 2, f"Expected 2D tensors; got {n.shape=} and {g.shape=}."
        assert n.shape == g.shape and n.shape[1] == self.D, \
            f"Expected both [B,{self.D}]; got {n.shape} and {g.shape}."
        return torch.cat([n, g], dim=-1)  # [B, 2D]

    # ---- standard Lightning steps
    def training_step(self, batch, batch_idx):
        flat = self._flat_from_batch(batch)
        # normflows.forward_kld returns per-sample objective
        loss = self.nf_forward_kld(flat).mean()
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        flat = self._flat_from_batch(batch)
        loss = self.nf_forward_kld(flat).mean()
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

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
        node_terms = z[:, :self.D].contiguous()
        graph_terms = z[:, self.D:].contiguous()
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
    hypernet: HyperNet = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, ds_name=ds_name, cfg=dataset_config).to(cfg.device).eval()
    log("Hypernet ready.")

    # ----- W&B -----
    run = wandb.run or wandb.init(
        project="realnvp-test",
        config=cfg.__dict__,
        name=f"run_{cfg.hv_dim}_{cfg.seed}",
        reinit=True
    )
    run.tags = [f"hv_dim={cfg.hv_dim}", f"vsa={cfg.vsa.value}", "dataset=ZincSmiles"]

    wandb_logger = WandbLogger(log_model=True, experiment=run)

    # ----- datasets / loaders -----
    train_dataset = ZincSmiles(split="train", enc_suffix="HRR7744")[:128]
    # validation_dataset = ZincSmiles(split="valid", enc_suffix="HRR7744")[:]

    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=2, pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    # validation_dataloader = DataLoader(
    #     validation_dataset, batch_size=cfg.batch_size, shuffle=False,
    #     num_workers=2, pin_memory=torch.cuda.is_available(), drop_last=False,
    # )
    # log(f"Datasets ready. train={len(train_dataset)} valid={len(validation_dataset)}")

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
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=train_dataloader)

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
    val_nll = _evaluate_loader_nll(best_model, train_dataloader, device)
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
        node_s, graph_s, logs = best_model.sample_split(2048)  # each [K, D]

    node_counters = hypernet.decode_order_zero_counter(node_s)
    for i, ctr in node_counters.items():
        log(f"Sample {i}: total nodes: {ctr.total()}, \n{ctr}")

    node_np = node_s.detach().cpu().numpy()
    graph_np = graph_s.detach().cpu().numpy()
    logs_np = logs.detach().cpu().numpy() if torch.is_tensor(logs) else np.asarray(logs)

    # per-branch norms and pairwise cosine samples
    node_norm = np.linalg.norm(node_np, axis=1)
    graph_norm = np.linalg.norm(graph_np, axis=1)

    def _pairwise_cosine(x: np.ndarray, m: int = 2000) -> np.ndarray:
        n = x.shape[0]
        if n < 2: return np.array([])
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
    if node_cos.size:  _hist(artefacts_dir / "sample_node_cos_hist.png", node_cos, "Node pairwise cosine", "cos")
    if graph_cos.size: _hist(artefacts_dir / "sample_graph_cos_hist.png", graph_cos, "Graph pairwise cosine", "cos")

    # W&B logs
    wandb.log({
        "sample_node_norm_mean": float(np.mean(node_norm)),
        "sample_node_norm_std": float(np.std(node_norm)),
        "sample_graph_norm_mean": float(np.mean(graph_norm)),
        "sample_graph_norm_std": float(np.std(graph_norm)),
        "sample_node_norm_hist": wandb.Histogram(node_norm),
        "sample_graph_norm_hist": wandb.Histogram(graph_norm),
        "sample_node_cos_hist": wandb.Histogram(node_cos) if node_cos.size else None,
        "sample_graph_cos_hist": wandb.Histogram(graph_cos) if graph_cos.size else None,
    })

    # quick table
    pd.DataFrame({
        **{f"node_{i}": node_np[:16, i] for i in range(min(16, node_np.shape[1]))},
        **{f"graph_{i}": graph_np[:16, i] for i in range(min(16, graph_np.shape[1]))},
    }).to_parquet(evals_dir / "sample_head.parquet", index=False)


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
