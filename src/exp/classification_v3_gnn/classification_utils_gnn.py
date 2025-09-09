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

from src.datasets.zinc_pairs_v2 import PairType, ZincPairsV2
from src.encoding.graph_encoders import AbstractGraphEncoder
from src.encoding.the_types import VSAModel
from src.utils.utils import str2bool

with contextlib.suppress(RuntimeError):
    mp.set_sharing_strategy("file_system")


# ----------------- Single unified experiment config -----------------



# def stratified_per_parent_indices_with_caps(
#     ds,
#     *,
#     pos_per_parent: int,
#     neg_per_parent: int,
#     exclude_neg_types: set[int] = frozenset(),
#     seed: int = 42,
#     log_every_shards: int = 50,
# ) -> list[int]:
#     """
#     Uniform random sampling per parent using per-parent reservoirs.
#
#     Returns global dataset indices suitable for torch.utils.data.Subset(ds, indices).
#
#     Notes
#     -----
#     - Works with ZincPairsV2 storage: iterates shards, reads only
#       y / neg_type / parent_idx columns from collated tensors.
#     - Deterministic w.r.t. `seed`, shard order, and ds content.
#     - If a parent has fewer than the requested quota, it returns whatever exists.
#     """
#     rng = random.Random(seed)
#
#     # --- per-type caps (5% each for type 4 and type 5) ---
#     cap_cross_parent_at = 0.05
#     cap_rewire_at = 0.05
#     cap_cross_parent = int(neg_per_parent * cap_cross_parent_at)
#     cap_rewire = int(neg_per_parent * cap_rewire_at)
#
#     print(f"Capping cross parent at {cap_cross_parent_at} and rewire at {cap_rewire_at}", flush=True)
#
#     # per-parent reservoirs and seen counters
#     pos_seen = defaultdict(int)
#     neg_seen = defaultdict(int)  # used for "other" negative types (not 4/5)
#     pos_res = defaultdict(list)  # parent_idx -> list[global_idx]
#     neg_res = defaultdict(list)  # "other" negatives (not 4/5)
#
#     # --- dedicated reservoirs & counters for Type4 and Type5 ---
#     neg4_seen = defaultdict(int)  # Type4 (rewire)
#     neg5_seen = defaultdict(int)  # Type5 (cross-parent)
#     neg4_res = defaultdict(list)
#     neg5_res = defaultdict(list)
#
#     def _reservoir_push(lst, seen_count, k, idx):
#         if k <= 0:
#             return  # respect zero-cap
#         if len(lst) < k:
#             lst.append(idx)
#         else:
#             j = rng.randrange(seen_count)  # 0..seen_count-1
#             if j < k:
#                 lst[j] = idx
#
#     global_offset = 0
#     num_shards = len(ds._shards)
#
#     for shard_id in range(num_shards):
#         data, slices = ds._get_shard(shard_id)
#
#         # Number of samples in this shard = len(slices['y']) - 1
#         m = int(slices["y"].shape[0] - 1)
#
#         y_all = data.y
#         pid_all = data.parent_idx
#         nt_all = data.neg_type
#
#         sy = slices["y"]
#         spid = slices["parent_idx"]
#         snt = slices["neg_type"]
#
#         for li in range(m):
#             gidx = global_offset + li
#
#             # scalar reads (each is a 1-length slice)
#             y = int(y_all[sy[li] : sy[li + 1]].item())
#             pid = int(pid_all[spid[li] : spid[li + 1]].item())
#
#             if y == 1:
#                 pos_seen[pid] += 1
#                 _reservoir_push(pos_res[pid], pos_seen[pid], pos_per_parent, gidx)
#             else:
#                 nt = int(nt_all[snt[li] : snt[li + 1]].item())
#                 if nt in exclude_neg_types:
#                     continue
#
#                 # --- route negatives by type; cap type 4/5 at 5% each ---
#                 if nt == PairType.CROSS_PARENT:
#                     neg4_seen[pid] += 1
#                     _reservoir_push(neg4_res[pid], neg4_seen[pid], cap_cross_parent, gidx)
#                 elif nt == PairType.REWIRE:
#                     neg5_seen[pid] += 1
#                     _reservoir_push(neg5_res[pid], neg5_seen[pid], cap_rewire, gidx)
#                 else:
#                     # Keep a larger reservoir for "other" types; final trimming happens per-parent below.
#                     neg_seen[pid] += 1
#                     _reservoir_push(neg_res[pid], neg_seen[pid], neg_per_parent, gidx)
#
#         global_offset += m
#         if log_every_shards and (shard_id + 1) % log_every_shards == 0:
#             print(f"[sample] scanned shard {shard_id + 1}/{num_shards}", flush=True)
#
#     # --- build negatives per parent respecting caps first, then fill remainder from others ---
#     selected = []
#
#     # Positives
#     for lst in pos_res.values():
#         selected.extend(lst)
#
#     # Negatives: combine per parent
#     parent_ids = set(pos_res.keys()) | set(neg_res.keys()) | set(neg4_res.keys()) | set(neg5_res.keys())
#     for pid in parent_ids:
#         take = []
#
#         # Guaranteed: capped via their reservoirs
#         take.extend(neg4_res.get(pid, []))
#         take.extend(neg5_res.get(pid, []))
#
#         # Fill the remainder from "other" negatives up to neg_per_parent
#         rem = neg_per_parent - len(take)
#         if rem > 0:
#             others = neg_res.get(pid, [])
#             if others:
#                 if len(others) > rem:
#                     # random but deterministic subset from the reservoir
#                     others = rng.sample(others, rem)
#                 take.extend(others)
#
#         selected.extend(take)
#
#     # Keep deterministic order (no DataLoader shuffle)
#     selected.sort()
#
#     # --- Sanity check — print final distribution over the SELECTED indices ---
#     total = len(selected)
#     pos_cnt = 0
#     neg_cnt = 0
#     neg_hist = defaultdict(int)
#
#     global_offset = 0  # re-walk shards and only touch rows we actually selected
#     sel = selected  # alias
#     si = 0  # moving pointer into `sel` (since both are sorted)
#
#     for shard_id in range(num_shards):
#         data, slices = ds._get_shard(shard_id)
#         m = int(slices["y"].shape[0] - 1)
#
#         y_all = data.y
#         nt_all = data.neg_type
#         sy = slices["y"]
#         snt = slices["neg_type"]
#
#         # Range of global indices covered by this shard
#         lo = global_offset
#         hi = global_offset + m
#
#         # Slice [start:end) of `sel` that lies in this shard
#         start = bisect_left(sel, lo, lo=si)  # we can start from last si to be linear
#         end = bisect_left(sel, hi, lo=start)
#         for gidx in sel[start:end]:
#             li = gidx - global_offset
#             y = int(y_all[sy[li] : sy[li + 1]].item())
#             if y == 1:
#                 pos_cnt += 1
#             else:
#                 neg_cnt += 1
#                 nt = int(nt_all[snt[li] : snt[li + 1]].item())
#                 neg_hist[nt] += 1
#         si = end
#         global_offset = hi
#
#     if total > 0:
#         pos_pct = 100.0 * pos_cnt / total
#         neg_pct = 100.0 * neg_cnt / total
#         # stable order by neg_type
#         hist_items = sorted(neg_hist.items())
#         hist_str = ", ".join(f"{t}:{c} ({(0 if neg_cnt == 0 else 100.0 * c / neg_cnt):.1f}%)" for t, c in hist_items)
#         print("=== Sanity summary (selected subset) ===", flush=True)
#         print(
#             f"[SEL] size={total} | positives={pos_cnt} ({pos_pct:.1f}%) | negatives={neg_cnt} ({neg_pct:.1f}%)",
#             flush=True,
#         )
#         print(f"[SEL] neg_type histogram: {hist_str}", flush=True)
#     else:
#         print("=== Sanity summary: no samples selected ===", flush=True)
#
#     return selected


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
