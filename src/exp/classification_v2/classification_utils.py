import argparse
import contextlib
import math
import os
import random
from bisect import bisect_left
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from src.datasets.zinc_pairs_v2 import ZincPairsV2, PairType
from src.encoding.graph_encoders import AbstractGraphEncoder
from src.encoding.the_types import VSAModel
from src.utils.utils import str2bool

with contextlib.suppress(RuntimeError):
    mp.set_sharing_strategy("file_system")


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
    use_layer_norm: bool = False
    use_batch_norm: bool = False

    # Evals
    oracle_num_evals: int = 1
    oracle_beam_size: int = 8

    # HDC / encoder
    hv_dim: int = 88 * 88  # 7744
    vsa: VSAModel = VSAModel.HRR

    # Optim
    lr: float = 1e-3
    weight_decay: float = 0.0

    # Loader
    num_workers: int = 0
    prefetch_factor: int = 1
    pin_memory: bool = False
    micro_bs: int = 64
    hv_scale: float | None = None

    # Checkpointing
    save_every_seconds: int = 3600  # every 60 minutes
    keep_last_k: int = 2  # rolling snapshots to keep
    continue_from: Path | None = None
    resume_retrain_last_epoch: bool = False

    # Stratification
    stratify: bool = True
    p_per_parent: int = 20
    n_per_parent: int = 20
    exclude_negs: set[int] = field(default_factory=list)
    resample_training_data_on_batch: bool = False


# ----------------- CLI parsing that never clobbers defaults -----------------
def _parse_hidden_dims(s: str) -> list[int]:
    # accept "4096,2048,512,128" or with spaces
    return [int(tok) for tok in s.replace(" ", "").split(",") if tok]


def _parse_vsa(s: str) -> VSAModel:
    # Accepts e.g. "HRR", not VSAModel.HRR
    if isinstance(s, VSAModel):
        return s
    return VSAModel(s)


def get_args(argv: list[str] | None = None) -> Config:
    """
    Build a Config by starting from dataclass defaults and then
    applying ONLY the CLI options the user actually provided.
    NOTE: For --vsa, pass a string like "HRR", not VSAModel.HRR.
    """
    cfg = Config()  # start with your defaults

    p = argparse.ArgumentParser(description="Experiment Config (unified)")

    # IMPORTANT: default=SUPPRESS so unspecified flags don't overwrite dataclass defaults
    p.add_argument(
        "--project_dir", "-pdir", type=Path, default=argparse.SUPPRESS, help="Project root (will be created if missing)"
    )
    p.add_argument("--exp_dir_name", type=str, default=argparse.SUPPRESS, help="Optional experiment subfolder name")

    p.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    p.add_argument("--epochs", "-e", type=int, default=argparse.SUPPRESS)
    p.add_argument("--batch_size", "-bs", type=int, default=argparse.SUPPRESS)

    # Ranges for selecting parents
    p.add_argument("--train_parents_start", type=int, default=argparse.SUPPRESS)
    p.add_argument("--train_parents_end", type=int, default=argparse.SUPPRESS)
    p.add_argument("--valid_parents_start", type=int, default=argparse.SUPPRESS)
    p.add_argument("--valid_parents_end", type=int, default=argparse.SUPPRESS)

    # Evals
    p.add_argument("--oracle_num_evals", type=int, default=argparse.SUPPRESS)
    p.add_argument("--oracle_beam_size", type=int, default=argparse.SUPPRESS)

    # Model knobs
    p.add_argument(
        "--hidden_dims",
        type=_parse_hidden_dims,
        default=argparse.SUPPRESS,
        help="Comma-separated: e.g. '4096,2048,512,128'",
    )
    p.add_argument("--use_batch_norm", type=str2bool, nargs="?", const=True, default=argparse.SUPPRESS)
    p.add_argument("--use_layer_norm", type=str2bool, nargs="?", const=True, default=argparse.SUPPRESS)

    # HDC
    p.add_argument("--hv_dim", "-hd", type=int, default=argparse.SUPPRESS)

    p.add_argument(
        "--vsa", "-v", type=_parse_vsa, default=argparse.SUPPRESS, choices=[m.value for m in VSAModel]
    )  # accepts strings like "HRR"
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
    p.add_argument("--resume_retrain_last_epoch", type=str2bool, default=argparse.SUPPRESS)

    # Stratification
    p.add_argument("--stratify", type=str2bool, default=argparse.SUPPRESS)
    p.add_argument("--p_per_parent", type=int, default=argparse.SUPPRESS)
    p.add_argument("--n_per_parent", type=int, default=argparse.SUPPRESS)
    p.add_argument("--exclude_negs", type=set[int], default=argparse.SUPPRESS)
    p.add_argument("--resample_training_data_on_batch", type=str2bool, default=argparse.SUPPRESS)

    ns = p.parse_args(argv)
    provided = vars(ns)  # only the keys the user actually passed

    # Apply only provided keys onto cfg
    for k, v in provided.items():
        # Make sure VSAModel parsed if user typed the enum value directly
        if k == "vsa" and isinstance(v, str):
            v = VSAModel(v)
        setattr(cfg, k, v)

    return cfg


def cleanup_old_snapshots(models_dir: Path, keep_last_k: int):
    snaps = sorted(models_dir.glob("autosnap_*.pt"))
    for p in snaps[:-keep_last_k]:
        p.unlink(missing_ok=True)


def atomic_save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


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


def stratified_per_parent_indices(
    ds,
    *,
    pos_per_parent: int,
    neg_per_parent: int,
    exclude_neg_types: set[int] = frozenset(),
    seed: int = 42,
    log_every_shards: int = 50,
) -> list[int]:
    """
    Uniform random sampling per parent using per-parent reservoirs.

    Returns global dataset indices suitable for torch.utils.data.Subset(ds, indices).

    Notes
    -----
    - Works with ZincPairsV2 storage: iterates shards, reads only
      y / neg_type / parent_idx columns from collated tensors.
    - Deterministic w.r.t. `seed`, shard order, and ds content.
    - If a parent has fewer than the requested quota, it returns whatever exists.
    """
    rng = random.Random(seed)

    # per-parent reservoirs and seen counters
    pos_seen = defaultdict(int)
    neg_seen = defaultdict(int)
    pos_res = defaultdict(list)  # parent_idx -> list[global_idx]
    neg_res = defaultdict(list)

    def _reservoir_push(lst, seen_count, k, idx) -> None:
        if len(lst) < k:
            lst.append(idx)
        else:
            j = rng.randrange(seen_count)  # 0..seen_count-1
            if j < k:
                lst[j] = idx

    global_offset = 0
    num_shards = len(ds._shards)

    for shard_id in range(num_shards):
        data, slices = ds._get_shard(shard_id)

        # Number of samples in this shard = len(slices['y']) - 1
        m = int(slices["y"].shape[0] - 1)

        y_all = data.y
        pid_all = data.parent_idx
        nt_all = data.neg_type

        sy = slices["y"]
        spid = slices["parent_idx"]
        snt = slices["neg_type"]

        for li in range(m):
            gidx = global_offset + li

            # scalar reads (each is a 1-length slice)
            y = int(y_all[sy[li] : sy[li + 1]].item())
            pid = int(pid_all[spid[li] : spid[li + 1]].item())

            if y == 1:
                pos_seen[pid] += 1
                _reservoir_push(pos_res[pid], pos_seen[pid], pos_per_parent, gidx)
            else:
                nt = int(nt_all[snt[li] : snt[li + 1]].item())
                if nt in exclude_neg_types:
                    continue
                neg_seen[pid] += 1
                _reservoir_push(neg_res[pid], neg_seen[pid], neg_per_parent, gidx)

        global_offset += m
        if log_every_shards and (shard_id + 1) % log_every_shards == 0:
            print(f"[sample] scanned shard {shard_id + 1}/{num_shards}", flush=True)

    # flatten and make order deterministic (no DataLoader shuffle)
    selected = []
    for lst in pos_res.values():
        selected.extend(lst)
    for lst in neg_res.values():
        selected.extend(lst)

    # You can sort to keep a monotonic traversal over the underlying dataset (resume-friendly).
    selected.sort()
    return selected


def stratified_per_parent_indices_with_caps(
    ds,
    *,
    pos_per_parent: int,
    neg_per_parent: int,
    exclude_neg_types: set[int] = frozenset(),
    seed: int = 42,
    log_every_shards: int = 50,
) -> list[int]:
    """
    Uniform random sampling per parent using per-parent reservoirs.

    Returns global dataset indices suitable for torch.utils.data.Subset(ds, indices).

    Notes
    -----
    - Works with ZincPairsV2 storage: iterates shards, reads only
      y / neg_type / parent_idx columns from collated tensors.
    - Deterministic w.r.t. `seed`, shard order, and ds content.
    - If a parent has fewer than the requested quota, it returns whatever exists.
    """
    rng = random.Random(seed)

    # --- per-type caps (5% each for type 4 and type 5) ---
    cap_cross_parent_at = 0.05
    cap_rewire_at = 0.05
    cap_cross_parent = int(neg_per_parent * cap_cross_parent_at)
    cap_rewire = int(neg_per_parent * cap_rewire_at)

    print(f"Capping cross parent at {cap_cross_parent_at} and rewire at {cap_rewire_at}", flush=True)

    # per-parent reservoirs and seen counters
    pos_seen = defaultdict(int)
    neg_seen = defaultdict(int)  # used for "other" negative types (not 4/5)
    pos_res = defaultdict(list)  # parent_idx -> list[global_idx]
    neg_res = defaultdict(list)  # "other" negatives (not 4/5)

    # --- dedicated reservoirs & counters for Type4 and Type5 ---
    neg4_seen = defaultdict(int)  # Type4 (rewire)
    neg5_seen = defaultdict(int)  # Type5 (cross-parent)
    neg4_res = defaultdict(list)
    neg5_res = defaultdict(list)

    def _reservoir_push(lst, seen_count, k, idx):
        if k <= 0:
            return  # respect zero-cap
        if len(lst) < k:
            lst.append(idx)
        else:
            j = rng.randrange(seen_count)  # 0..seen_count-1
            if j < k:
                lst[j] = idx

    global_offset = 0
    num_shards = len(ds._shards)

    for shard_id in range(num_shards):
        data, slices = ds._get_shard(shard_id)

        # Number of samples in this shard = len(slices['y']) - 1
        m = int(slices["y"].shape[0] - 1)

        y_all = data.y
        pid_all = data.parent_idx
        nt_all = data.neg_type

        sy = slices["y"]
        spid = slices["parent_idx"]
        snt = slices["neg_type"]

        for li in range(m):
            gidx = global_offset + li

            # scalar reads (each is a 1-length slice)
            y = int(y_all[sy[li] : sy[li + 1]].item())
            pid = int(pid_all[spid[li] : spid[li + 1]].item())

            if y == 1:
                pos_seen[pid] += 1
                _reservoir_push(pos_res[pid], pos_seen[pid], pos_per_parent, gidx)
            else:
                nt = int(nt_all[snt[li] : snt[li + 1]].item())
                if nt in exclude_neg_types:
                    continue

                # --- route negatives by type; cap type 4/5 at 5% each ---
                if nt == PairType.CROSS_PARENT:
                    neg4_seen[pid] += 1
                    _reservoir_push(neg4_res[pid], neg4_seen[pid], cap_cross_parent, gidx)
                elif nt == PairType.REWIRE:
                    neg5_seen[pid] += 1
                    _reservoir_push(neg5_res[pid], neg5_seen[pid], cap_rewire, gidx)
                else:
                    # Keep a larger reservoir for "other" types; final trimming happens per-parent below.
                    neg_seen[pid] += 1
                    _reservoir_push(neg_res[pid], neg_seen[pid], neg_per_parent, gidx)

        global_offset += m
        if log_every_shards and (shard_id + 1) % log_every_shards == 0:
            print(f"[sample] scanned shard {shard_id + 1}/{num_shards}", flush=True)

    # --- build negatives per parent respecting caps first, then fill remainder from others ---
    selected = []

    # Positives
    for lst in pos_res.values():
        selected.extend(lst)

    # Negatives: combine per parent
    parent_ids = set(pos_res.keys()) | set(neg_res.keys()) | set(neg4_res.keys()) | set(neg5_res.keys())
    for pid in parent_ids:
        take = []

        # Guaranteed: capped via their reservoirs
        take.extend(neg4_res.get(pid, []))
        take.extend(neg5_res.get(pid, []))

        # Fill the remainder from "other" negatives up to neg_per_parent
        rem = neg_per_parent - len(take)
        if rem > 0:
            others = neg_res.get(pid, [])
            if others:
                if len(others) > rem:
                    # random but deterministic subset from the reservoir
                    others = rng.sample(others, rem)
                take.extend(others)

        selected.extend(take)

    # Keep deterministic order (no DataLoader shuffle)
    selected.sort()

    # --- Sanity check — print final distribution over the SELECTED indices ---
    total = len(selected)
    pos_cnt = 0
    neg_cnt = 0
    neg_hist = defaultdict(int)

    global_offset = 0  # re-walk shards and only touch rows we actually selected
    sel = selected  # alias
    si = 0          # moving pointer into `sel` (since both are sorted)

    for shard_id in range(num_shards):
        data, slices = ds._get_shard(shard_id)
        m = int(slices["y"].shape[0] - 1)

        y_all = data.y
        nt_all = data.neg_type
        sy = slices["y"]
        snt = slices["neg_type"]

        # Range of global indices covered by this shard
        lo = global_offset
        hi = global_offset + m

        # Slice [start:end) of `sel` that lies in this shard
        start = bisect_left(sel, lo, lo=si)  # we can start from last si to be linear
        end = bisect_left(sel, hi, lo=start)
        for gidx in sel[start:end]:
            li = gidx - global_offset
            y = int(y_all[sy[li] : sy[li + 1]].item())
            if y == 1:
                pos_cnt += 1
            else:
                neg_cnt += 1
                nt = int(nt_all[snt[li] : snt[li + 1]].item())
                neg_hist[nt] += 1
        si = end
        global_offset = hi

    if total > 0:
        pos_pct = 100.0 * pos_cnt / total
        neg_pct = 100.0 * neg_cnt / total
        # stable order by neg_type
        hist_items = sorted(neg_hist.items())
        hist_str = ", ".join(f"{t}:{c} ({(0 if neg_cnt==0 else 100.0*c/neg_cnt):.1f}%)" for t, c in hist_items)
        print("=== Sanity summary (selected subset) ===", flush=True)
        print(f"[SEL] size={total} | positives={pos_cnt} ({pos_pct:.1f}%) | "
              f"negatives={neg_cnt} ({neg_pct:.1f}%)", flush=True)
        print(f"[SEL] neg_type histogram: {hist_str}", flush=True)
    else:
        print("=== Sanity summary: no samples selected ===", flush=True)

    return selected


def exact_representative_validation_indices(
    ds,
    *,
    target_total: int,
    exclude_neg_types=frozenset(),
    by_neg_type: bool = True,
    seed: int = 42,
    log_fn=print,
):
    """
    Select exactly `min(target_total, available_after_exclusion)` validation indices.

    :param ds: ZincPairsV2
    :param target_total: desired number of validation samples
    :param exclude_neg_types: set of neg_type codes to drop entirely
    :param by_neg_type: if True, preserve negatives' per-type proportions exactly
    :param seed: RNG seed for deterministic reservoirs
    :param log_fn: callable(str) used for logging

    Behavior
    --------
    • Computes class totals (pos/neg) and, if `by_neg_type`, per-`neg_type` totals
      after applying `exclude_neg_types`.
    • Allocates integer targets by proportional apportionment (largest remainders),
      capped by availability.
    • Runs independent reservoirs to meet those exact integer targets.
    """
    rng = random.Random(seed)

    # ---------- 1) Availability ----------
    total_pos = 0
    neg_counts_by_type = defaultdict(int)

    for shard_id in range(len(ds._shards)):
        data, slices = ds._get_shard(shard_id)
        m = int(slices["y"].shape[0] - 1)
        y, nt = data.y, data.neg_type
        sy, snt = slices["y"], slices["neg_type"]
        for li in range(m):
            yy = int(y[sy[li] : sy[li + 1]].item())
            if yy == 1:
                total_pos += 1
            else:
                t = int(nt[snt[li] : snt[li + 1]].item())
                if t not in exclude_neg_types:
                    neg_counts_by_type[t] += 1

    total_neg = sum(neg_counts_by_type.values())
    total_avail = total_pos + total_neg

    log_fn(f"[val-sample] avail(after exclude): pos={total_pos:,} neg={total_neg:,} total={total_avail:,}")
    if by_neg_type and total_neg > 0:
        for t in sorted(neg_counts_by_type):
            log_fn(f"[val-sample]   neg_type={t}: avail={neg_counts_by_type[t]:,}")

    if total_avail == 0:
        log_fn("[val-sample] nothing available after exclusions → returning [].")
        return []

    # ---------- 2) Targets (integer, exact sum) ----------
    # class split
    pos_share = total_pos / total_avail
    pos_target = round(target_total * pos_share)
    neg_target = target_total - pos_target

    capped = False
    if pos_target > total_pos:
        pos_target = total_pos
        neg_target = min(total_neg, target_total - pos_target)
        capped = True
    if neg_target > total_neg:
        neg_target = total_neg
        pos_target = min(total_pos, target_total - neg_target)
        capped = True

    log_fn(
        f"[val-sample] target_total={target_total:,} → pos_target={pos_target:,} neg_target={neg_target:,}"
        + (" (capped by availability)" if capped else "")
    )

    # per-neg_type targets
    if by_neg_type and neg_target > 0 and total_neg > 0:
        nt_targets = dict.fromkeys(neg_counts_by_type, 0)
        shares = {t: (neg_counts_by_type[t] / total_neg) * neg_target for t in neg_counts_by_type}
        # floors
        for t, s in shares.items():
            nt_targets[t] = min(neg_counts_by_type[t], math.floor(s))
        assigned = sum(nt_targets.values())
        remaining = neg_target - assigned
        # largest remainders
        rema = sorted(((t, shares[t] - math.floor(shares[t])) for t in shares), key=lambda x: x[1], reverse=True)
        i = 0
        while remaining > 0 and i < len(rema):
            t = rema[i][0]
            if nt_targets[t] < neg_counts_by_type[t]:
                nt_targets[t] += 1
                remaining -= 1
            i += 1
        # round-robin fill if still remaining (rare)
        if remaining > 0:
            for t in sorted(nt_targets):
                while remaining > 0 and nt_targets[t] < neg_counts_by_type[t]:
                    nt_targets[t] += 1
                    remaining -= 1
                if remaining == 0:
                    break
        # log per-type targets
        for t in sorted(nt_targets):
            log_fn(f"[val-sample]   neg_type={t}: target={nt_targets[t]:,} (avail {neg_counts_by_type[t]:,})")
    else:
        nt_targets = None
        if neg_target > 0:
            log_fn("[val-sample] negatives pooled (by_neg_type=False).")

    # ---------- 3) Reservoirs ----------
    def reservoir_push(res_list, seen_count, k, idx):
        if k <= 0:
            return
        if len(res_list) < k:
            res_list.append(idx)
        else:
            j = rng.randrange(seen_count)  # 0..seen_count-1
            if j < k:
                res_list[j] = idx

    pos_res = []
    pos_seen = 0
    neg_res_by_type = {t: [] for t in nt_targets} if nt_targets is not None else {-1: []}
    neg_seen_by_type = defaultdict(int)

    global_offset = 0
    for shard_id in range(len(ds._shards)):
        data, slices = ds._get_shard(shard_id)
        m = int(slices["y"].shape[0] - 1)
        y, nt = data.y, data.neg_type
        sy, snt = slices["y"], slices["neg_type"]
        for li in range(m):
            gidx = global_offset + li
            yy = int(y[sy[li] : sy[li + 1]].item())
            if yy == 1:
                pos_seen += 1
                reservoir_push(pos_res, pos_seen, pos_target, gidx)
            else:
                t = int(nt[snt[li] : snt[li + 1]].item())
                if t in exclude_neg_types:
                    continue
                key = t if nt_targets is not None else -1
                neg_seen_by_type[key] += 1
                k = nt_targets[key] if nt_targets is not None else neg_target
                reservoir_push(neg_res_by_type[key], neg_seen_by_type[key], k, gidx)
        global_offset += m

    # ---------- 4) Finalize ----------
    selected = list(pos_res)
    for _, lst in neg_res_by_type.items():
        selected.extend(lst)
    selected.sort()

    log_fn(f"[val-sample] selected={len(selected):,} (pos={len(pos_res):,}, neg={len(selected) - len(pos_res):,})")
    return selected
