import argparse
import shutil
import string
from dataclasses import dataclass, field
import random
from typing import Optional

import torch.multiprocessing as mp

from src.datasets.zinc_pairs_v2 import ZincPairsV2
from src.encoding.the_types import VSAModel
import contextlib
with contextlib.suppress(RuntimeError):
    mp.set_sharing_strategy("file_system")

import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

import torch

from src.encoding.graph_encoders import AbstractGraphEncoder


# ----------------- Single unified experiment config -----------------
@dataclass
class Config:
    # General
    project_dir: Path | None = None
    exp_dir_name: str | None = None
    seed: int = 42
    epochs: int = 5
    batch_size: int = 256

    # (optional) parent-range slicing
    train_parents_start: int | None = None
    train_parents_end: int | None = None
    valid_parents_start: int | None = None
    valid_parents_end: int | None = None

    # Model (shared knobs)
    hidden_dims: list[int] = field(default_factory=lambda: [4096, 2048, 512, 128])

    # HDC / encoder
    hv_dim: int = 88 * 88  # 7744
    vsa: VSAModel = VSAModel.HRR

    # Optim
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Loader
    num_workers: int = 0
    prefetch_factor: int = 1
    pin_memory: bool = False
    micro_bs: int = 64
    hv_scale: float | None = None

    # Checkpointing
    save_every_seconds: int = 3600  # every 60 minutes
    keep_last_k: int = 2            # rolling snapshots to keep
    continue_from: Path | None = None
    resume_retrain_last_epoch: bool = False

# ----------------- CLI parsing that never clobbers defaults -----------------
def _parse_hidden_dims(s: str) -> list[int]:
    # accept "4096,2048,512,128" or with spaces
    return [int(tok) for tok in s.replace(" ", "").split(",") if tok]

def _parse_vsa(s: str) -> VSAModel:
    # Accepts e.g. "HRR", not VSAModel.HRR
    if isinstance(s, VSAModel):
        return s
    return VSAModel(s)

def get_args(argv: Optional[list[str]] = None) -> Config:
    """
    Build a Config by starting from dataclass defaults and then
    applying ONLY the CLI options the user actually provided.
    NOTE: For --vsa, pass a string like "HRR", not VSAModel.HRR.
    """
    cfg = Config()  # start with your defaults

    p = argparse.ArgumentParser(description="Experiment Config (unified)")

    # IMPORTANT: default=SUPPRESS so unspecified flags don't overwrite dataclass defaults
    p.add_argument("--project_dir", "-pdir", type=Path, default=argparse.SUPPRESS,
                   help="Project root (will be created if missing)")
    p.add_argument("--exp_dir_name", type=str, default=argparse.SUPPRESS,
                   help="Optional experiment subfolder name")

    p.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    p.add_argument("--epochs", "-e", type=int, default=argparse.SUPPRESS)
    p.add_argument("--batch_size", "-bs", type=int, default=argparse.SUPPRESS)

    # Ranges for selecting parents
    p.add_argument("--train_parents_start", type=int, default=argparse.SUPPRESS)
    p.add_argument("--train_parents_end", type=int, default=argparse.SUPPRESS)
    p.add_argument("--valid_parents_start", type=int, default=argparse.SUPPRESS)
    p.add_argument("--valid_parents_end", type=int, default=argparse.SUPPRESS)

    # Model knobs
    p.add_argument("--hidden_dims", type=_parse_hidden_dims, default=argparse.SUPPRESS,
                   help="Comma-separated: e.g. '4096,2048,512,128'")

    # HDC
    p.add_argument("--hv_dim", "-hd", type=int, default=argparse.SUPPRESS)
    p.add_argument("--vsa", "-v", type=_parse_vsa, default=argparse.SUPPRESS,
                   choices=[m.value for m in VSAModel])  # accepts strings like "HRR"
    p.add_argument("--device", "-dev", type=str, choices=["cpu", "cuda"], default=argparse.SUPPRESS)

    # Optim
    p.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    p.add_argument("--weight_decay", "-wd", type=float, default=argparse.SUPPRESS)

    # Loader
    p.add_argument("--num_workers", type=int, default=argparse.SUPPRESS)
    p.add_argument("--prefetch_factor", type=int, default=argparse.SUPPRESS)
    p.add_argument("--pin_memory", action="store_true", default=argparse.SUPPRESS)
    p.add_argument("--micro_bs", type=int, default=argparse.SUPPRESS)
    p.add_argument("--hv_scale", type=float, default=argparse.SUPPRESS)

    # Checkpointing
    p.add_argument("--save_every_seconds", type=int, default=argparse.SUPPRESS)
    p.add_argument("--keep_last_k", type=int, default=argparse.SUPPRESS)
    p.add_argument("--continue_from", type=Path, default=argparse.SUPPRESS)
    p.add_argument("--resume_retrain_last_epoch", type=bool, default=argparse.SUPPRESS)

    ns = p.parse_args(argv)
    provided = vars(ns)  # only the keys the user actually passed

    # Apply only provided keys onto cfg
    for k, v in provided.items():
        # Make sure VSAModel parsed if user typed the enum value directly
        if k == "vsa" and isinstance(v, str):
            v = VSAModel(v)
        setattr(cfg, k, v)

    return cfg


def setup_exp(dir_name: str | None = None) -> dict:
    script_path = Path(__file__).resolve()
    experiments_path = script_path.parent
    script_stem = script_path.stem

    base_dir = experiments_path / "results" / script_stem
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Setting up experiment in {base_dir}")
    now = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{''.join(random.choices(string.ascii_lowercase, k=4))}"
    exp_dir = base_dir / now if not dir_name else base_dir / dir_name
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


def cleanup_old_snapshots(models_dir: Path, keep_last_k: int):
    snaps = sorted(models_dir.glob("autosnap_*.pt"))
    for p in snaps[:-keep_last_k]:
        p.unlink(missing_ok=True)


def atomic_save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def gpu_mem(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    return f"gpu_mem: alloc={a:.2f}G reserved={r:.2f}G"


# ---------------------------------------------------------------------
# Dataset wrapper that returns graphs (we encode in the training loop)
# ---------------------------------------------------------------------
class PairsGraphsDataset(Dataset):
    """
    Returns raw graphs (no encodings): (g1, g2, k1k2, y, parent_idx)
    """

    def __init__(self, pairs_ds: ZincPairsV2):
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
            chunk = Batch.from_data_list(uniq_graphs[j: j + micro_bs]).to(device)
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
def encode_batch(
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
        chunk_list = data_list[i: i + micro_bs]
        chunk = Batch.from_data_list(chunk_list).to(device)
        with torch.no_grad():
            out = encoder.forward(chunk)["graph_embedding"]  # [b, D]
        outs.append(out)
        # free ASAP
        del chunk
        torch.cuda.empty_cache() if device.type == "cuda" else None
    H = torch.cat(outs, dim=0)  # [B, D]
    return H
