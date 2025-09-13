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
        Desired **fractions of negatives per parent** by negative type (excluding NEGATIVE_EDGE).
        Fractions are absolute caps (per parent): q_t <= floor(neg_budget * frac).
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
        If True, enforce per-parent NEGATIVE_EDGE == POSITIVE_EDGE (subject to availability and budget).
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
    import math
    import random
    from collections import Counter, defaultdict

    rng = random.Random(seed)

    # ---- defaults for negative type caps (excluding NEGATIVE_EDGE) ----
    if type_mix is None:
        type_mix = {
            int(PairType.CROSS_PARENT): 0.05,
            int(PairType.REWIRE): 0.05,
            int(PairType.MISSING_EDGE): 0.25,
            int(PairType.ANCHOR_AWARE): 0.35,
            int(PairType.WRONG_EDGE_ALLOWED): 0.15,
            int(PairType.FORBIDDEN_ADD): 0.10,
        }

    # normalize keys, drop excluded/nonpositive fracs, and renormalize if >1
    type_mix = {int(k): float(v) for k, v in type_mix.items() if k not in exclude_neg_types and v > 0.0}
    s = sum(type_mix.values())
    if s > 1.0:
        type_mix = {k: v / s for k, v in type_mix.items()}

    def _cap_for_type_with_budget(t: int, budget: int) -> int:
        # Per-type absolute cap for THIS parent given remaining negative budget.
        # 0 is allowed (we'll fill with leftovers later).
        return int(math.floor(budget * type_mix.get(t, 0.0)))

    # --------- reservoirs (per parent) ----------
    pos_edge_res = defaultdict(list)       # POSITIVE_EDGE only
    pos_edge_seen = defaultdict(int)
    pos_other_res = defaultdict(list)      # POSITIVE (k>2)
    pos_other_seen = defaultdict(int)

    neg_edge_res = defaultdict(list)       # NEGATIVE_EDGE only
    neg_edge_seen = defaultdict(int)
    neg_by_type_res = defaultdict(lambda: defaultdict(list))  # pid -> {neg_type -> [idx]}
    neg_by_type_seen = defaultdict(lambda: defaultdict(int))

    def _reservoir_push(lst, seen_count, k, idx):
        if k <= 0:
            return
        if len(lst) < k:
            lst.append(idx)
        else:
            j = rng.randrange(seen_count)
            if j < k:
                lst[j] = idx

    # ---------- scan shards ----------
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
                    # give slack in reservoirs to reduce variance (2x cap against the *max* budget)
                    slack_cap = max(1, 2 * _cap_for_type_with_budget(nt, neg_per_parent))
                    neg_by_type_seen[pid][nt] += 1
                    _reservoir_push(neg_by_type_res[pid][nt], neg_by_type_seen[pid][nt], slack_cap, gidx)

        global_offset += m
        if log_every_shards and (shard_id + 1) % log_every_shards == 0:
            print(f"[sample/train] scanned shard {shard_id + 1}/{num_shards}", flush=True)

    # ---------- assemble per parent ----------
    selected = []
    parents = (
            set(pos_edge_res.keys())
            | set(pos_other_res.keys())
            | set(neg_edge_res.keys())
            | set(neg_by_type_res.keys())
    )

    diag = {
        "parents": len(parents),
        "pos": 0,
        "neg": 0,
        "k2_pos_edge": 0,
        "k2_neg_edge": 0,
        "neg_by_type": Counter(),
    }

    for pid in sorted(parents):
        pos_edge_pool  = list(pos_edge_res.get(pid, ()))
        pos_other_pool = list(pos_other_res.get(pid, ()))
        neg_edge_pool  = list(neg_edge_res.get(pid, ()))

        pe_avail = len(pos_edge_pool)
        po_avail = len(pos_other_pool)
        ne_avail = len(neg_edge_pool)

        # ---- decide positive targets ----
        if balance_k2:
            # POSITIVE_EDGE we choose must be matchable by NEGATIVE_EDGE and fit budgets
            k2_cap_by_neg = min(ne_avail, neg_per_parent)
            pe_target = min(pe_avail, k2_cap_by_neg, pos_per_parent)
        else:
            pe_target = min(pe_avail, pos_per_parent)

        po_target = min(po_avail, max(0, pos_per_parent - pe_target))

        # If POSITIVE supply (po) is short, try to backfill with extra POSITIVE_EDGE (still matchable if k2)
        if pe_target + po_target < pos_per_parent:
            deficit = pos_per_parent - (pe_target + po_target)
            extra_pe_cap = pe_avail - pe_target
            if balance_k2:
                extra_pe_cap = min(extra_pe_cap, max(0, min(ne_avail, neg_per_parent) - pe_target))
            pe_target += min(deficit, extra_pe_cap)

        rng.shuffle(pos_edge_pool)
        rng.shuffle(pos_other_pool)
        pos_take = pos_edge_pool[:pe_target] + pos_other_pool[:po_target]
        selected.extend(pos_take)

        pos_edge_selected = pe_target  # by construction
        pos_final = len(pos_take)

        diag["pos"] += pos_final
        diag["k2_pos_edge"] += pos_edge_selected

        # ---- negatives: total negatives per parent cannot exceed actual positives ----
        neg_limit = min(neg_per_parent, pos_final)

        neg_take = []

        # First, k==2: take exactly as many NEGATIVE_EDGE as POSITIVE_EDGE (subject to availability & budget)
        k2_use = min(pos_edge_selected, ne_avail, neg_limit) if balance_k2 else 0
        if k2_use > 0:
            rng.shuffle(neg_edge_pool)
            neg_take.extend(neg_edge_pool[:k2_use])
            neg_limit -= k2_use
            diag["k2_neg_edge"] += k2_use
            diag["neg_by_type"][int(PairType.NEGATIVE_EDGE)] += k2_use

        if neg_limit > 0:
            # Per-type caps for the remaining negatives (excluding NEGATIVE_EDGE), using *remaining* budget
            staged = []
            leftovers_pool = []

            # 1st pass: up to cap for each type
            for t, frac in type_mix.items():
                if neg_limit <= 0:
                    break
                avail_list = list(neg_by_type_res[pid].get(t, ()))
                if not avail_list:
                    continue
                rng.shuffle(avail_list)
                cap_t = _cap_for_type_with_budget(t, neg_limit)
                want = min(cap_t, neg_limit, len(avail_list))
                if want > 0:
                    pick = avail_list[:want]
                    staged.extend((t, idx) for idx in pick)
                    neg_limit -= want
                    rest = avail_list[want:]
                    if rest:
                        leftovers_pool.extend((t, z) for z in rest)

            # 2nd pass: fill remaining budget from any leftover types
            if neg_limit > 0 and leftovers_pool:
                rng.shuffle(leftovers_pool)
                take_more = leftovers_pool[:neg_limit]
                staged.extend(take_more)
                neg_limit -= len(take_more)

            if staged:
                neg_take.extend([idx for (_t, idx) in staged])
                for _t, _ in staged:
                    diag["neg_by_type"][_t] += 1

            # 3rd pass (optional): if STILL short, try any other non-NEGATIVE_EDGE remaining
            if neg_limit > 0:
                spill = []
                for t, lst in neg_by_type_res.get(pid, {}).items():
                    rest = list(set(lst) - set(neg_take))
                    if rest:
                        spill.extend((t, z) for z in rest)
                if spill:
                    rng.shuffle(spill)
                    take_more = spill[:neg_limit]
                    neg_take.extend([idx for (_t, idx) in take_more])
                    for _t, _ in take_more:
                        diag["neg_by_type"][_t] += 1
                    neg_limit -= len(take_more)

        selected.extend(neg_take)
        diag["neg"] += len(neg_take)

    selected.sort()

    # ------------- logging -------------
    total = len(selected)
    pos_cnt = diag["pos"]
    neg_cnt = diag["neg"]
    pos_pct = (100.0 * pos_cnt / total) if total else 0.0
    neg_pct = (100.0 * neg_cnt / total) if total else 0.0

    # NEG type histogram (include NEGATIVE_EDGE explicitly)
    neg_hist = {int(PairType.NEGATIVE_EDGE): diag["k2_neg_edge"], **diag["neg_by_type"]}
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
