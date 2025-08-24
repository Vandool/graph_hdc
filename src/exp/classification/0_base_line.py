import argparse
import random
import shutil
import string
from collections import OrderedDict
from dataclasses import dataclass, asdict
from datetime import datetime
from math import prod
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
from tqdm.auto import tqdm

from src.datasets.zinc_pairs import ZincPairs
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import DatasetConfig, Features, FeatureConfig, IndexRange
from src.encoding.feature_encoders import CombinatoricIntegerEncoder
from src.encoding.graph_encoders import AbstractGraphEncoder, load_or_create_hypernet
from src.encoding.the_types import VSAModel
from src.utils.utils import GLOBAL_MODEL_PATH


def setup_exp() -> dict:
    """
    Sets up experiment directories based on the current script location.

    Returns:
        dict: Dictionary containing paths to various directories.
    """
    # Resolve script location
    script_path = Path(__file__).resolve()
    experiments_path = script_path.parent
    script_stem = script_path.stem  # without .py

    # Resolve base and project directories
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

    # Save a copy of the script
    try:
        shutil.copy(script_path, exp_dir / script_path.name)
        print(f"Saved a copy of the script to {exp_dir / script_path.name}")
    except Exception as e:
        print(f"Warning: Failed to save script copy: {e}")

    return dirs


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class LogisticPairBaseline(nn.Module):
    def __init__(self, dim: int = 7744, use_cosine: bool = True):
        super().__init__()
        self.dim = dim
        self.use_cosine = use_cosine
        extra = 6 if use_cosine else 5  # [n1, n2, dot, (cos), k1, k2]
        self.linear = nn.Linear(2 * dim + extra, 1)

    def forward(self, h1, h2, k1k2):  # h1,h2: [B,D], k1k2: [B,2]
        d = torch.abs(h1 - h2)
        m = h1 * h2
        n1 = torch.linalg.vector_norm(h1, dim=-1, keepdim=True)
        n2 = torch.linalg.vector_norm(h2, dim=-1, keepdim=True)
        dot = (h1 * h2).sum(dim=-1, keepdim=True)
        if self.use_cosine:
            cos = dot / (n1 * n2 + 1e-8)
            scalars = torch.cat([n1, n2, dot, cos, k1k2.float()], dim=-1)
        else:
            scalars = torch.cat([n1, n2, dot, k1k2.float()], dim=-1)
        z = torch.cat([d, m, scalars], dim=-1)
        return self.linear(z).squeeze(-1)  # logits


# ---------------------------------------------------------------------
# Dataset wrapper with per-worker LRU for G2 encodings
# ---------------------------------------------------------------------
class LRU:
    def __init__(self, max_items=256):
        self.max_items = max_items
        self._d = OrderedDict()

    def get(self, key, factory):
        if key in self._d:
            val = self._d.pop(key)
            self._d[key] = val
            return val
        val = factory()
        self._d[key] = val
        if len(self._d) > self.max_items:
            self._d.popitem(last=False)
        return val


class EncodedPairsDataset(Dataset):
    """
    Wraps ZincPairs and returns (h1, h2, k1k2, y).
    - h1: enc(G1) computed on the fly (no cache)
    - h2: enc(G2) cached per worker by parent_idx
    NOTE: encoder.forward expects a BATCH; we wrap each Data in a Batch of size 1.
    """

    def __init__(self, pairs_ds: Dataset, encoder: AbstractGraphEncoder, g2_cache_size: int = 512, device='cpu'):
        self.ds = pairs_ds
        self.encoder = encoder.to(device=device)
        self.device = device
        self.g2_cache = LRU(g2_cache_size)

    def __len__(self):
        return len(self.ds)

    def _encode_one(self, g: Data) -> torch.Tensor:
        batch = Batch.from_data_list([g]).to(self.device)
        with torch.no_grad():
            out = self.encoder.forward(batch)  # expects a Batch
            hv = out["graph_embedding"]  # [B, D]
            if hv.dim() == 1:
                hv = hv.unsqueeze(0)
            return hv[0].detach().cpu().to(torch.float32)

    def __getitem__(self, idx):
        item = self.ds[idx]
        # Rebuild G1 and G2 as PyG Data
        g1 = Data(x=item.x1, edge_index=item.edge_index1)
        g2 = Data(x=item.x2, edge_index=item.edge_index2)

        # Encode on the fly
        h1 = self._encode_one(g1)
        parent_idx = int(item.parent_idx.view(-1)[0].item()) if hasattr(item, 'parent_idx') else -1
        h2 = self.g2_cache.get(parent_idx, lambda: self._encode_one(g2))

        k1 = g1.x.size(0)
        k2 = g2.x.size(0)
        y = int(item.y.view(-1)[0].item())

        return {
            "h1": h1,
            "h2": h2,
            "k1k2": torch.tensor([k1, k2], dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
        }


def default_collate(batch):
    h1 = torch.stack([b["h1"] for b in batch], dim=0)
    h2 = torch.stack([b["h2"] for b in batch], dim=0)
    k1k2 = torch.stack([b["k1k2"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    return h1, h2, k1k2, y


# ---------------------------------------------------------------------
# Config (aligned with baseline classifier)
# ---------------------------------------------------------------------
@dataclass
class OracleConfig:
    # General
    project_dir: Path
    seed: int = 42
    epochs: int = 5
    batch_size: int = 1024

    # HDC / encoder
    hv_dim: int = 88 * 88  # 7744
    vsa: VSAModel = VSAModel.HRR
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Optim
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Loader / caching
    num_workers: int = 8
    g2_cache_size: int = 512


def get_args() -> OracleConfig:
    parser = argparse.ArgumentParser(description="Logistic baseline classifier args")
    parser.add_argument("--project_dir", "-pdir", type=Path,
                        default=Path("/home/ka/ka_iti/ka_zi9629/projects/graph_hdc"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", "-e", type=int, default=5)
    parser.add_argument("--batch_size", "-bs", type=int, default=1024)
    parser.add_argument("--hv_dim", "-hd", type=int, default=7744)
    parser.add_argument("--vsa", "-v", type=VSAModel, default=VSAModel.HRR)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--device", "-dev", type=str, choices=["cpu", "cuda"],
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--g2_cache_size", type=int, default=512)
    cfg = OracleConfig(**vars(parser.parse_args()))
    return cfg


# ---------------------------------------------------------------------
# Eval + Train
# ---------------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for h1, h2, k1k2, y in loader:
            h1, h2, k1k2 = h1.to(device), h2.to(device), k1k2.to(device)
            logit = model(h1, h2, k1k2)
            prob = torch.sigmoid(logit).cpu()
            ys.append(y)
            ps.append(prob)
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    return {"auc": roc_auc_score(y, p), "ap": average_precision_score(y, p)}


def _count_labels_fast(pairs_ds: ZincPairs) -> tuple[int, int]:
    pos = neg = 0
    for i in range(len(pairs_ds)):
        y = int(pairs_ds[i].y.view(-1)[0].item())
        if y == 1:
            pos += 1
        else:
            neg += 1
    return pos, neg


def train(
        train_pairs, valid_pairs,
        encoder: AbstractGraphEncoder,
        models_dir: Path,
        evals_dir: Path,
        artefacts_dir: Path,
        cfg: OracleConfig,
):

    # Dataset wrappers
    # IMPORTANT: if you want GPU encoding, set num_workers=0 to avoid CUDA in dataloader workers.
    enc_device = "cpu"  # keep encoding in CPU workers; train model on cfg.device
    train_ds = EncodedPairsDataset(train_pairs, encoder, g2_cache_size=cfg.g2_cache_size, device=enc_device)
    valid_ds = EncodedPairsDataset(valid_pairs, encoder, g2_cache_size=cfg.g2_cache_size, device=enc_device)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=default_collate, persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=default_collate, persistent_workers=(cfg.num_workers > 0),
    )

    # Model
    model = LogisticPairBaseline(dim=cfg.hv_dim).to(cfg.device)

    # Class imbalance
    tr_pos, tr_neg = _count_labels_fast(train_pairs)
    pos_weight = torch.tensor([max(1.0, tr_neg / max(1, tr_pos))], device=cfg.device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Traceability payload (constant fields)
    base_row = {
        **asdict(cfg),
        "train_size": len(train_pairs),
        "valid_size": len(valid_pairs),
        "train_pos": tr_pos,
        "train_neg": tr_neg,
        "model_params": sum(p.numel() for p in model.parameters()),
    }

    rows = []
    train_losses, valid_aucs = [], []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for h1, h2, k1k2, y in pbar:
            h1, h2, k1k2, y = h1.to(cfg.device), h2.to(cfg.device), k1k2.to(cfg.device), y.to(cfg.device)
            logit = model(h1, h2, k1k2)
            loss = criterion(logit, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            epoch_loss += loss.item() * y.size(0)
            pbar.set_postfix(loss=loss.item())
        epoch_loss /= len(train_ds)
        train_losses.append(epoch_loss)

        metrics = evaluate(model, valid_loader, cfg.device)
        valid_aucs.append(metrics["auc"])
        print(
            f"[epoch {epoch}] train_loss={epoch_loss:.4f}  valid_auc={metrics['auc']:.4f}  valid_ap={metrics['ap']:.4f}")

        # Row for parquet
        rows.append({
            **base_row,
            "epoch": epoch,
            "train_loss": epoch_loss,
            "valid_auc": float(metrics["auc"]),
            "valid_ap": float(metrics["ap"]),
            "pos_weight": float(pos_weight.item()),
        })

        # Save best
        if metrics["auc"] >= max(valid_aucs):
            torch.save(model.state_dict(), models_dir / "best.pt")

    # Save metrics parquet (append if exists)
    df = pd.DataFrame(rows)
    parquet_path = evals_dir / "metrics.parquet"
    if parquet_path.exists():
        old = pd.read_parquet(parquet_path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_parquet(parquet_path, index=False)

    # Plots
    plt.figure()
    plt.plot(train_losses)
    plt.title("Train loss")
    plt.xlabel("epoch")
    plt.ylabel("BCE")
    plt.savefig(artefacts_dir / "loss.png")
    plt.close()

    plt.figure()
    plt.plot(valid_aucs)
    plt.title("Valid ROC-AUC")
    plt.xlabel("epoch")
    plt.ylabel("AUC")
    plt.savefig(artefacts_dir / "auc.png")
    plt.close()


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------
def run_experiment(cfg: OracleConfig):
    print("Running experiment")
    pprint(asdict(cfg), indent=2)

    # Setup experiment directories (using your project setup)
    dirs = setup_exp()
    exp_dir = Path(dirs["exp_dir"])
    models_dir = Path(dirs["models_dir"])
    evals_dir = Path(dirs["evals_dir"])
    artefacts_dir = Path(dirs["artefacts_dir"])

    # Dataset & Encoder (HRR @ 7744)
    ds_name = "ZincPairsEncodings"
    zinc_feature_bins = [9, 6, 3, 4]

    dataset_config = DatasetConfig(
        seed=cfg.seed,
        name=ds_name,
        vsa=cfg.vsa,
        hv_dim=cfg.hv_dim,
        device="cpu",  # encoder runs in dataloader workers (CPU)
        node_feature_configs=OrderedDict([
            (
                Features.ATOM_TYPE,
                FeatureConfig(
                    count=prod(zinc_feature_bins),  # 9 * 6 * 3 * 4
                    encoder_cls=CombinatoricIntegerEncoder,
                    index_range=IndexRange((0, 4)),
                    bins=zinc_feature_bins,
                ),
            ),
        ]),
    )

    # Build/Load encoder
    hypernet = load_or_create_hypernet(
        path=GLOBAL_MODEL_PATH,
        ds_name=ds_name,
        cfg=dataset_config
    )

    # Base pair datasets
    train_pairs = ZincPairs(split="train", base_dataset=ZincSmiles("train"))
    valid_pairs = ZincPairs(split="valid", base_dataset=ZincSmiles("valid"))

    # Train
    torch.manual_seed(cfg.seed)
    train(
        train_pairs=train_pairs,
        valid_pairs=valid_pairs,
        encoder=hypernet,
        models_dir=models_dir,
        evals_dir=evals_dir,
        artefacts_dir=artefacts_dir,
        cfg=cfg,
    )


if __name__ == "__main__":
    run_experiment(get_args())
