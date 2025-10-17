import contextlib
import math

import torch.multiprocessing as mp

with contextlib.suppress(RuntimeError):
    mp.set_sharing_strategy("file_system")

import argparse
import os
import random
import shutil
import string
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime
from math import prod
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
import torch
from pytorch_lightning import seed_everything
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Batch, Data
from tqdm.auto import tqdm

from src.datasets.zinc_pairs import ZincPairs
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import DSHDCConfig, FeatureConfig, Features, IndexRange
from src.encoding.feature_encoders import CombinatoricIntegerEncoder
from src.encoding.graph_encoders import AbstractGraphEncoder, load_or_create_hypernet
from src.encoding.the_types import VSAModel
from src.utils.utils import GLOBAL_MODEL_PATH


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
# MLP classifier on concatenated (h1, h2) – no normalization, GELU, no dropout
class MLPClassifier(nn.Module):
    def __init__(self, hv_dim: int, hidden_dims: list[int]):
        """
        hv_dim: dimension of each HRR vector (e.g., 7744)
        hidden_dims: e.g., [4096, 2048, 512, 128]
        """
        super().__init__()
        d_in = hv_dim * 2
        layers: list[nn.Module] = []
        last = d_in
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.GELU()]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        # h1,h2: [B, hv_dim]
        x = torch.cat([h1, h2], dim=-1)  # [B, 2*D]
        return self.net(x).squeeze(-1)   # [B]

# ---------------------------------------------------------------------
# Dataset wrapper that returns graphs (we encode in the training loop)
# ---------------------------------------------------------------------
class PairsGraphsDataset(Dataset):
    """
    Returns raw graphs (no encodings): (g1, g2, k1k2, y, parent_idx)
    """

    def __init__(self, pairs_ds: ZincPairs):
        self.ds = pairs_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        g1 = Data(x=item.x1, edge_index=item.edge_index1)
        g2 = Data(x=item.x2, edge_index=item.edge_index2)
        y = float(item.y.view(-1)[0].item())
        parent_idx = int(item.parent_idx.view(-1)[0].item()) if hasattr(item, "parent_idx") else -1
        return g1, g2, torch.tensor(y, dtype=torch.float32), parent_idx


def collate_pairs(batch):
    g1_list, g2_list, y, parent_idx = zip(*batch, strict=False)
    g1_b = Batch.from_data_list(list(g1_list))
    g2_b = Batch.from_data_list(list(g2_list))
    return g1_b, g2_b, torch.stack(y, 0), torch.tensor(parent_idx, dtype=torch.long)




class ParentH2Cache:
    """CPU LRU cache of G2 embeddings, keyed by parent_idx."""

    def __init__(self, max_items: int = 20000):
        self.max_items = max_items
        self._d: OrderedDict[int, torch.Tensor] = OrderedDict()

    def get(self, k: int):
        if k in self._d:
            v = self._d.pop(k)
            self._d[k] = v
            return v
        return None

    def set(self, k: int, v: torch.Tensor):
        # store on CPU to keep GPU RAM small
        v = v.detach().to("cpu")
        if k in self._d:
            self._d.pop(k)
        self._d[k] = v
        if len(self._d) > self.max_items:
            self._d.popitem(last=False)


@torch.no_grad()
def encode_g2_with_cache(
    encoder: AbstractGraphEncoder,
    g2_b: Batch,
    parent_ids: torch.Tensor,  # shape [B], Long
    device: torch.device,
    cache: ParentH2Cache,
    micro_bs: int,
) -> torch.Tensor:
    """
    Returns h2 for the whole batch [B, D], encoding each parent only once.
    Cache lives on CPU; we copy gathered rows to GPU for the step.
    """
    parent_ids_cpu = parent_ids.detach().cpu()
    data_list = g2_b.to_data_list()  # per-graph list aligned with parent_ids

    # 1) figure out which parents are missing
    missing = []
    for i, pid in enumerate(parent_ids_cpu.tolist()):
        if cache.get(pid) is None:
            missing.append((pid, i))

    # 2) encode missing parents in micro-batches
    if missing:
        uniq_graphs = []
        uniq_pids = []
        seen = set()
        for pid, i in missing:
            if pid in seen:
                continue
            seen.add(pid)
            uniq_graphs.append(data_list[i])
            uniq_pids.append(pid)

        # encode the unique parents
        # (use the existing micro-batch encoder)
        encoded = []
        for j in range(0, len(uniq_graphs), micro_bs):
            chunk = Batch.from_data_list(uniq_graphs[j : j + micro_bs]).to(device)
            out = encoder.forward(chunk)["graph_embedding"]  # [b, D] on device
            encoded.append(out.to("cpu"))
            del chunk
            if device.type == "cuda":
                torch.cuda.empty_cache()
        encoded = torch.cat(encoded, dim=0)  # [U, D]

        # fill cache
        for pid, vec in zip(uniq_pids, encoded, strict=False):
            cache.set(pid, vec)

    # 3) assemble batch h2 by gathering from cache (then move to device)
    h2_cpu = torch.stack([cache.get(int(pid)) for pid in parent_ids_cpu], dim=0)  # [B, D] cpu
    return h2_cpu.to(device)


@torch.no_grad()
def _encode_batch(
    encoder: AbstractGraphEncoder,
    g_batch: Batch,
    *,
    device: torch.device,
    micro_bs: int,
) -> torch.Tensor:
    """
    Encode a big PyG Batch in micro-chunks to avoid OOM.
    Returns: [B, hv_dim] on the same device as encoder.
    """
    # Split into micro batches by graphs
    data_list = g_batch.to_data_list()  # (keeps tensors; we’ll move to device below)
    outs = []
    for i in range(0, len(data_list), micro_bs):
        chunk_list = data_list[i : i + micro_bs]
        chunk = Batch.from_data_list(chunk_list).to(device)
        with torch.no_grad():
            out = encoder.forward(chunk)["graph_embedding"]  # [b, D]
        outs.append(out)
        # free ASAP
        del chunk
        torch.cuda.empty_cache() if device.type == "cuda" else None
    H = torch.cat(outs, dim=0)  # [B, D]
    return H


# ---------------------------------------------------------------------
# Config (aligned with baseline classifier)
# ---------------------------------------------------------------------
@dataclass
class Config:
    # General
    project_dir: Path
    seed: int = 42
    epochs: int = 5
    batch_size: int = 256
    train_parents: int = 2000
    valid_parents: int = 500

    hidden_dims: list[int] = (4096, 2048, 512, 128)

    # HDC / encoder
    hv_dim: int = 88 * 88  # 7744
    vsa: VSAModel = VSAModel.HRR
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Optim
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Loader
    num_workers: int = 0
    prefetch_factor: int = 1
    pin_memory: bool = False
    micro_bs: int = 64
    hv_scale: float | None = None


def get_args() -> Config:
    parser = argparse.ArgumentParser(description="Logistic baseline classifier args")
    parser.add_argument(
        "--project_dir", "-pdir", type=Path, default=Path("/home/ka/ka_iti/ka_zi9629/projects/graph_hdc")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", "-e", type=int, default=5)
    parser.add_argument("--batch_size", "-bs", type=int, default=1024)
    parser.add_argument("--hv_dim", "-hd", type=int, default=7744)
    parser.add_argument("--vsa", "-v", type=VSAModel, default=VSAModel.HRR)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument(
        "--device", "-dev", type=str, choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--train_parents", type=int, default=2000)
    parser.add_argument("--valid_parents", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--micro_bs", type=int, default=64)
    parser.add_argument("--hv_scale", type=float, default=None)
    # Add hidden_dims as a comma-separated list
    parser.add_argument(
        "--hidden_dims", type=lambda s: [int(item) for item in s.split(",")],
        default="4096,2048,512,128",
        help="Comma-separated list of hidden layer sizes, e.g., '4096,2048,512,128'"
    )
    return Config(**vars(parser.parse_args()))


def _gpu_mem(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    return f"gpu_mem: alloc={a:.2f}G reserved={r:.2f}G"


# ---------------------------------------------------------------------
# Eval + Train
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, encoder, loader, device, criterion, cfg, h2_cache, *, return_details: bool=False):
    model.eval()
    encoder.eval()
    ys, ps, ls = [], [], []  # labels, probs, logits
    total_loss, total_n = 0.0, 0

    for g1_b, g2_b, y, parent_ids in loader:
        y = y.to(device)

        h1 = _encode_batch(encoder, g1_b, device=device, micro_bs=cfg.micro_bs)
        h2 = encode_g2_with_cache(encoder, g2_b, parent_ids, device, h2_cache, cfg.micro_bs)

        logits = model(h1, h2)
        loss = criterion(logits, y)

        prob = torch.sigmoid(logits).detach().cpu()
        ys.append(y.detach().cpu())
        ps.append(prob)
        if return_details:
            ls.append(logits.detach().cpu())

        total_loss += float(loss.item()) * y.size(0)
        total_n += y.size(0)

    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    out = {
        "auc": roc_auc_score(y, p),
        "ap": average_precision_score(y, p),
        "loss": total_loss / max(1, total_n),
    }
    if return_details:
        out["y"] = y
        out["p"] = p
        out["logits"] = torch.cat(ls).numpy()
    return out


def make_loader(ds, batch_size, shuffle, cfg, collate_fn):
    kwargs = {
        "dataset": ds,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": cfg.num_workers,
        "pin_memory": (cfg.device == "cuda" and cfg.pin_memory),
        "collate_fn": collate_fn,
        "persistent_workers": False,  # important for memory
        "worker_init_fn": lambda _: torch.set_num_threads(1),  # keep per-worker light
    }
    if cfg.num_workers > 0:  # only valid when workers > 0
        kwargs["prefetch_factor"] = max(1, cfg.prefetch_factor)
    return DataLoader(**kwargs)


def _sanitize_for_parquet(d: dict) -> dict:
    """Make dict Arrow-friendly (Path/Enum/etc → str, tensors → int/float)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, VSAModel):
            out[k] = v.value
        elif torch.is_tensor(v):
            out[k] = v.item() if v.numel() == 1 else v.detach().cpu().tolist()
        else:
            out[k] = v
    return out

@torch.no_grad()
def probe_encoder_edge_sensitivity(encoder, pairs_ds, device, max_pairs=1000):
    """
    For matched (pos,neg) with same parent + k + node multiset, compare enc(G1).
    If cosine~1.0 and L2~0 → the graph_embedding is effectively edge-insensitive *for these negatives*.
    """
    from collections import defaultdict
    from torch_geometric.data import Data, Batch

    def label_multiset_hash(x_tensor):
        # order-invariant hash of node feature rows
        rows = [tuple(int(v) for v in row.tolist()) for row in x_tensor]
        return hash(tuple(sorted(rows)))

    groups = defaultdict(lambda: {"pos": [], "neg": []})
    for i in range(len(pairs_ds)):
        item = pairs_ds[i]
        pid = int(item.parent_idx) if hasattr(item, "parent_idx") else -1
        kk  = int(item.k) if hasattr(item, "k") else item.x1.size(0)
        mhash = label_multiset_hash(item.x1)
        y = int(item.y)
        key = (pid, kk, mhash)
        (groups[key]["pos"] if y==1 else groups[key]["neg"]).append(i)

    pairs = []
    for key, bucket in groups.items():
        if bucket["pos"] and bucket["neg"]:
            pairs.append((bucket["pos"][0], bucket["neg"][0]))
            if len(pairs) >= max_pairs:
                break
    if not pairs:
        print("[probe] No matched (pos,neg) pairs found.")
        return

    cos_list, l2_list = [], []
    for pi, ni in pairs:
        ip = pairs_ds[pi]; ineg = pairs_ds[ni]
        gp = Data(x=ip.x1, edge_index=ip.edge_index1)
        gn = Data(x=ineg.x1, edge_index=ineg.edge_index1)
        bp = Batch.from_data_list([gp]).to(device)
        bn = Batch.from_data_list([gn]).to(device)
        hp = encoder.forward(bp)["graph_embedding"]
        hn = encoder.forward(bn)["graph_embedding"]
        cos = torch.nn.functional.cosine_similarity(hp, hn).item()
        l2  = torch.norm(hp - hn).item()
        cos_list.append(cos); l2_list.append(l2)

    print(f"[probe] matched={len(pairs)}  cos μ={sum(cos_list)/len(cos_list):.6f} "
          f"l2 μ={sum(l2_list)/len(l2_list):.6f} | "
          f"cos min/max=({min(cos_list):.6f},{max(cos_list):.6f}) "
          f"l2 min/max=({min(l2_list):.6f},{max(l2_list):.6f})")

def train(
    train_pairs,
    valid_pairs,
    encoder: AbstractGraphEncoder,
    models_dir: Path,
    evals_dir: Path,
    artefacts_dir: Path,
    cfg: Config,
):
    import json

    from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

    log("In Training ... ")
    log("Setting up datasets …")
    train_ds = PairsGraphsDataset(train_pairs)
    valid_ds = PairsGraphsDataset(valid_pairs)
    log(f"Datasets ready. train_size={len(train_ds):,} valid_size={len(valid_ds):,}")

    log("Setting up dataloaders …")
    train_loader = make_loader(train_ds, cfg.batch_size, True, cfg, collate_pairs)
    valid_loader = make_loader(valid_ds, cfg.batch_size, False, cfg, collate_pairs)
    log("In Training ... Data loaders ready.")

    device = torch.device(cfg.device)
    encoder = encoder.to(device).eval()

    probe_encoder_edge_sensitivity(encoder, train_pairs, device, max_pairs=1000)

    log("Training finished.")

# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------
def run_experiment(cfg: Config):
    log("Parsing args done. Starting run_experiment …")
    dirs = setup_exp()
    exp_dir = Path(dirs["exp_dir"])
    models_dir = Path(dirs["models_dir"])
    evals_dir = Path(dirs["evals_dir"])
    artefacts_dir = Path(dirs["artefacts_dir"])

    # Dataset & Encoder (HRR @ 7744)
    ds_name = "ZincPairsEncodings"
    zinc_feature_bins = [9, 6, 3, 4]

    dataset_config = DSHDCConfig(
        seed=cfg.seed,
        name=ds_name,
        vsa=cfg.vsa,
        hv_dim=cfg.hv_dim,
        device=cfg.device,  # encoder lives on main process device now
        node_feature_configs=OrderedDict(
            [
                (
                    Features.ATOM_TYPE,
                    FeatureConfig(
                        count=prod(zinc_feature_bins),  # 9 * 6 * 3 * 4
                        encoder_cls=CombinatoricIntegerEncoder,
                        index_range=IndexRange((0, 4)),
                        bins=zinc_feature_bins,
                    ),
                ),
            ]
        ),
    )

    log("Loading/creating hypernet …")
    hypernet = (
        load_or_create_hypernet(path=GLOBAL_MODEL_PATH, ds_name=ds_name, cfg=dataset_config).to(cfg.device).eval()
    )
    log("Hypernet ready.")

    # Datasets
    log("Loading pair datasets …")
    train_full = ZincPairs(split="train", base_dataset=ZincSmiles(split="train"))
    valid_full = ZincPairs(split="valid", base_dataset=ZincSmiles(split="valid"))
    log(f"Pairs loaded. train_pairs={len(train_full)} valid_pairs={len(valid_full)}")

    # --- pick N parents & wrap as Subset ---
    # estimated at 500 pairs per parent
    train_small = torch.utils.data.Subset(train_full, range(cfg.train_parents * 500))
    valid_small = torch.utils.data.Subset(valid_full, range(cfg.valid_parents * 500))
    log(f"[subset] train_indices={len(train_small):,}  valid_indices={len(valid_small):,}")

    torch.manual_seed(cfg.seed)
    train(
        train_pairs=train_small,
        valid_pairs=train_small,
        encoder=hypernet,
        models_dir=models_dir,
        evals_dir=evals_dir,
        artefacts_dir=artefacts_dir,
        cfg=cfg,
    )



if __name__ == "__main__":
    log(f"Running {Path(__file__).resolve()}")
    cfg = get_args()
    pprint(asdict(cfg), indent=2)
    seed_everything(cfg.seed)
    run_experiment(cfg)
