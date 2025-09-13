import contextlib
import math
import random
from collections import Counter, defaultdict

import torch.multiprocessing as mp

from src.datasets.qm9_pairs import PairType

with contextlib.suppress(RuntimeError):
    mp.set_sharing_strategy("file_system")


# ----------------- Single unified experiment config -----------------


def stratified_per_parent_indices_with_type_mix(
    ds,
    *,
    pos_per_parent: int,
    neg_per_parent: int,
    type_mix: dict[int | PairType, float] | None = None,
    balance_k2: bool = True,
    exclude_neg_types: set[int] = frozenset(),
    seed: int = 42,
    log_every_shards: int = 50,
) -> list[int]:
    """
    Uniform random sampling *per parent*, with fine-grained negative type control and k==2 balancing.

    Parameters
    ----------
    pos_per_parent : int
        Target positives per parent (sum of POSITIVE and POSITIVE_EDGE).
    neg_per_parent : int
        Target negatives per parent.
    type_mix : dict[int|PairType, float], optional
        Desired **fractions of negatives per parent** by negative type.
        Fractions are absolute caps (per parent): q_t <= floor(neg_per_parent * frac).
        Any remainder is filled from available other types.
        By default:
            CROSS_PARENT -> 0.05
            REWIRE       -> 0.05
            MISSING_EDGE -> 0.25
            ANCHOR_AWARE -> 0.35
            WRONG_EDGE_ALLOWED -> 0.15
            FORBIDDEN_ADD -> 0.10
        NEGATIVE_EDGE is *not* in the mix and is handled by k==2 balancing.
    balance_k2 : bool
        If True, enforce approx balance per parent: #NEGATIVE_EDGE ~= #POSITIVE_EDGE.
        (Subject to availability and neg_per_parent budget.)
    exclude_neg_types : set[int]
        Negative types to exclude entirely from sampling.
    seed : int
        RNG seed.
    log_every_shards : int
        Progress log cadence over shards.

    Returns
    -------
    list[int]
        Sorted global indices to use with `Subset(ds, indices)`.
    """
    rng = random.Random(seed)

    # ------------ defaults for the new dataset (your stated preferences) ------------
    if type_mix is None:
        type_mix = {
            int(PairType.CROSS_PARENT): 0.05,  # stricter cap
            int(PairType.REWIRE): 0.05,  # stricter cap
            int(PairType.MISSING_EDGE): 0.25,  # high priority
            int(PairType.ANCHOR_AWARE): 0.35,  # high priority
            int(PairType.WRONG_EDGE_ALLOWED): 0.15,
            int(PairType.FORBIDDEN_ADD): 0.10,
            # NEGATIVE_EDGE intentionally omitted (balanced via k==2)
        }

    # Normalize keys to ints
    type_mix = {int(k): float(v) for k, v in type_mix.items() if k not in exclude_neg_types and v > 0.0}
    # Sanity: fractions shouldn't exceed 1 (NEGATIVE_EDGE handled separately).
    if sum(type_mix.values()) > 1.0:
        # Clamp by proportional scaling (keep relative importance)
        s = sum(type_mix.values())
        type_mix = {k: v / s for k, v in type_mix.items()}

    # Per-type *reservoir* capacities: a bit of slack to reduce variance
    def _cap_for_type(t: int) -> int:
        return max(1, int(math.floor(neg_per_parent * type_mix.get(t, 0.0))))

    # --------- per-parent reservoirs ----------
    # Positives split so we can count POSITIVE_EDGE for k==2 balancing
    pos_edge_res = defaultdict(list)  # POSITIVE_EDGE only
    pos_edge_seen = defaultdict(int)
    pos_other_res = defaultdict(list)  # POSITIVE (k>2)
    pos_other_seen = defaultdict(int)

    # Negatives per type
    neg_by_type_res = defaultdict(lambda: defaultdict(list))  # pid -> {type -> [idx]}
    neg_by_type_seen = defaultdict(lambda: defaultdict(int))

    # Special reservoirs
    neg_edge_res = defaultdict(list)  # NEGATIVE_EDGE only (balanced vs POSITIVE_EDGE)
    neg_edge_seen = defaultdict(int)

    # Generic helper
    def _reservoir_push(lst, seen_count, k, idx):
        if k <= 0:
            return
        if len(lst) < k:
            lst.append(idx)
        else:
            j = rng.randrange(seen_count)
            if j < k:
                lst[j] = idx

    # ---------- scan shards, build reservoirs ----------
    global_offset = 0
    num_shards = len(ds._shards)

    for shard_id in range(num_shards):
        data, slices = ds._get_shard(shard_id)
        m = int(slices["y"].shape[0] - 1)

        y_all = data.y
        pid_all = data.parent_idx
        nt_all = data.neg_type

        sy = slices["y"]
        spid = slices["parent_idx"]
        snt = slices["neg_type"]

        for li in range(m):
            gidx = global_offset + li
            y = int(y_all[sy[li] : sy[li + 1]].item())
            pid = int(pid_all[spid[li] : spid[li + 1]].item())
            nt = int(nt_all[snt[li] : snt[li + 1]].item())

            if y == 1:
                # Positive types: POSITIVE_EDGE (-1) vs POSITIVE (0)
                if nt == int(PairType.POSITIVE_EDGE):
                    pos_edge_seen[pid] += 1
                    _reservoir_push(pos_edge_res[pid], pos_edge_seen[pid], pos_per_parent, gidx)
                else:
                    pos_other_seen[pid] += 1
                    _reservoir_push(pos_other_res[pid], pos_other_seen[pid], pos_per_parent, gidx)
            else:
                if nt in exclude_neg_types:
                    continue
                if nt == int(PairType.NEGATIVE_EDGE):
                    neg_edge_seen[pid] += 1
                    _reservoir_push(neg_edge_res[pid], neg_edge_seen[pid], neg_per_parent, gidx)
                else:
                    # Type-specific reservoir with slack (2x cap) to give selection room
                    cap_t = max(1, 2 * _cap_for_type(nt))
                    neg_by_type_seen[pid][nt] += 1
                    _reservoir_push(neg_by_type_res[pid][nt], neg_by_type_seen[pid][nt], cap_t, gidx)

        global_offset += m
        if log_every_shards and (shard_id + 1) % log_every_shards == 0:
            print(f"[sample/train] scanned shard {shard_id + 1}/{num_shards}", flush=True)

    # ---------- build final selection per parent ----------
    selected = []
    parents = (
        set(pos_edge_res.keys()) | set(pos_other_res.keys()) | set(neg_edge_res.keys()) | set(neg_by_type_res.keys())
    )

    # For diagnostics
    diag = {
        "parents": len(parents),
        "pos": 0,
        "neg": 0,
        "k2_pos_edge": 0,
        "k2_neg_edge": 0,
        "neg_by_type": Counter(),
    }

    for pid in sorted(parents):
        # ---- positives: sample up to pos_per_parent from the union ----
        pos_pool = list(pos_edge_res.get(pid, ())) + list(pos_other_res.get(pid, ()))
        pos_take = set(rng.sample(pos_pool, pos_per_parent)) if len(pos_pool) > pos_per_parent else set(pos_pool)

        # Count POSITIVE_EDGE selected (k==2)
        pos_edge_selected = len(pos_take & set(pos_edge_res.get(pid, ())))
        diag["k2_pos_edge"] += pos_edge_selected

        # Emit positives
        selected.extend(pos_take)
        diag["pos"] += len(pos_take)

        # ---- negatives: enforce k==2 balance first (NEGATIVE_EDGE ~= POSITIVE_EDGE) ----
        neg_limit = neg_per_parent
        neg_take = []

        k2_needed = pos_edge_selected if balance_k2 else 0
        k2_avail = len(neg_edge_res.get(pid, ()))
        k2_use = min(k2_needed, k2_avail, neg_limit)
        if k2_use > 0:
            neg_take.extend(
                rng.sample(neg_edge_res[pid], k2_use) if k2_use < k2_avail else list(neg_edge_res[pid])[:k2_use]
            )
            neg_limit -= k2_use
            diag["k2_neg_edge"] += k2_use
            diag["neg_by_type"][int(PairType.NEGATIVE_EDGE)] += k2_use

        if neg_limit <= 0:
            selected.extend(neg_take)
            diag["neg"] += len(neg_take)
            continue

        # ---- per-type quotas (absolute caps) for the remaining negatives (excluding NEGATIVE_EDGE) ----
        # Apply caps as floor(neg_per_parent * frac)
        caps_abs = {t: _cap_for_type(t) for t in type_mix}
        # If some budget consumed by k2, scale the remaining proportions to remaining budget
        remaining_budget = neg_limit

        # First pass: take up to cap from each type, subject to availability
        staged = []
        leftovers_pool = []  # if some caps are underfilled, we’ll fill from whatever is left later

        for t, cap_t in caps_abs.items():
            if remaining_budget <= 0:
                break
            avail = len(neg_by_type_res[pid].get(t, ()))
            if avail <= 0 or cap_t <= 0:
                continue
            want = min(cap_t, remaining_budget, avail)
            pick = rng.sample(neg_by_type_res[pid][t], want) if want < avail else list(neg_by_type_res[pid][t])[:want]
            staged.extend((t, idx) for idx in pick)
            remaining_budget -= want

            # Leftover pool candidates (what remains after cap)
            rest = avail - want
            if rest > 0:
                # add the "spare" ones to a pool
                pool_idxs = list(set(neg_by_type_res[pid][t]) - set(pick))
                leftovers_pool.extend((t, z) for z in pool_idxs)

        # Second pass: if still budget, fill from any remaining types (excluding NEGATIVE_EDGE)
        if remaining_budget > 0 and leftovers_pool:
            rng.shuffle(leftovers_pool)
            take_more = leftovers_pool[:remaining_budget]
            staged.extend(take_more)
            remaining_budget -= len(take_more)

        # Emit negatives
        if staged:
            neg_take.extend([idx for (_t, idx) in staged])
            for _t, _ in staged:
                diag["neg_by_type"][_t] += 1

        # If we still haven’t met neg_per_parent (rare, not enough supply), try to backoff into any negatives available
        if remaining_budget > 0:
            # try any other negatives from any type we haven't already taken (still excluding NEGATIVE_EDGE beyond balance)
            spill = []
            for t, lst in neg_by_type_res.get(pid, {}).items():
                rest = list(set(lst) - set(neg_take))
                spill.extend((t, z) for z in rest)
            if spill:
                rng.shuffle(spill)
                take_more = spill[:remaining_budget]
                neg_take.extend([idx for (_t, idx) in take_more])
                for _t, _ in take_more:
                    diag["neg_by_type"][_t] += 1

        selected.extend(neg_take)
        diag["neg"] += len(neg_take)

    selected.sort()

    # ------------- logging -------------
    total = len(selected)
    pos_cnt = diag["pos"]
    neg_cnt = diag["neg"]
    pos_pct = (100.0 * pos_cnt / total) if total else 0.0
    neg_pct = (100.0 * neg_cnt / total) if total else 0.0

    # Render neg type histogram
    # Include NEGATIVE_EDGE explicitly
    neg_hist = {int(PairType.NEGATIVE_EDGE): diag["k2_neg_edge"], **diag["neg_by_type"]}
    # sort by type id
    hist_items = sorted(neg_hist.items())
    hist_str = ", ".join(f"{PairType(t).name}:{c}" for t, c in hist_items if c > 0)

    print("=== Train sampling summary ===", flush=True)
    print(f"[parents] {diag['parents']} | pos/parent={pos_per_parent} | neg/parent={neg_per_parent}", flush=True)
    print(f"[selected] total={total}  pos={pos_cnt} ({pos_pct:.1f}%)  neg={neg_cnt} ({neg_pct:.1f}%)", flush=True)
    if balance_k2:
        print(f"[k==2 balance] POSITIVE_EDGE={diag['k2_pos_edge']}  NEGATIVE_EDGE={diag['k2_neg_edge']}", flush=True)
    print(f"[neg mix] {hist_str}", flush=True)

    return selected


def balanced_indices_for_validation(
    ds,
    *,
    seed: int = 0,
    log_every_shards: int = 50,
) -> list[int]:
    """
    Build a balanced validation subset: select min(#pos, #neg) of each.
    Types are *ignored*; only y is used.

    Returns sorted global indices suitable for Subset(ds, indices).
    """
    rng = random.Random(seed)

    pos_all, neg_all = [], []

    global_offset = 0
    num_shards = len(ds._shards)
    for shard_id in range(num_shards):
        data, slices = ds._get_shard(shard_id)
        m = int(slices["y"].shape[0] - 1)

        y_all = data.y
        sy = slices["y"]

        for li in range(m):
            gidx = global_offset + li
            y = int(y_all[sy[li] : sy[li + 1]].item())
            if y == 1:
                pos_all.append(gidx)
            else:
                neg_all.append(gidx)

        global_offset += m
        if log_every_shards and (shard_id + 1) % log_every_shards == 0:
            print(f"[val] scanned shard {shard_id + 1}/{num_shards}", flush=True)

    n = min(len(pos_all), len(neg_all))
    if n == 0:
        print("[val] WARNING: cannot build balanced set (one class empty). Returning all.", flush=True)
        out = sorted(pos_all + neg_all)
        print(f"[val] size={len(out)} pos={len(pos_all)} neg={len(neg_all)}", flush=True)
        return out

    pos_sel = rng.sample(pos_all, n) if len(pos_all) > n else pos_all
    neg_sel = rng.sample(neg_all, n) if len(neg_all) > n else neg_all

    out = sorted(pos_sel + neg_sel)
    print("=== Validation sampling summary ===", flush=True)
    print(f"[selected] total={len(out)}  pos={n} (50.0%)  neg={n} (50.0%)", flush=True)
    return out
