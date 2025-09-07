# scripts/build_and_sanity_all_splits.py
import collections
import random

import torch
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from src.datasets.zinc_pairs import (
    PairConfig,
    ZincPairs,
    is_induced_subgraph_feature_aware,
    pyg_to_nx,
)
from src.datasets.zinc_smiles_generation import ZincSmiles


def compute_split_stats(ds: ZincPairs):
    """Return (n, pos, neg, neg_hist Counter)."""
    n = len(ds)
    pos = 0
    neg = 0
    neg_hist = collections.Counter()

    for i in range(n):
        item = ds[i]
        y = int(item.y.view(-1)[0].item())
        if y == 1:
            pos += 1
        else:
            neg += 1
            neg_type = int(item.neg_type.view(-1)[0].item())
            neg_hist[neg_type] += 1
    return n, pos, neg, neg_hist


def print_split_summary(name: str, n: int, pos: int, neg: int, neg_hist: collections.Counter):
    if n == 0:
        print(f"[{name}] size=0")
        return
    pos_pct = f"{(pos / n):.1%}"
    neg_pct = f"{(neg / n):.1%}"
    print(f"[{name}] size={n} | positives={pos} ({pos_pct}) | negatives={neg} ({neg_pct})")
    if neg:
        hist_str = ", ".join(f"{t}:{c} ({c/neg:.1%})" for t, c in sorted(neg_hist.items()))
        print(f"[{name}] neg_type histogram: {hist_str}")


def summarize_split(ds: ZincPairs, name: str):
    """Compute + print summary; also return stats for aggregation."""
    n, pos, neg, neg_hist = compute_split_stats(ds)
    print_split_summary(name, n, pos, neg, neg_hist)
    return n, pos, neg, neg_hist


def sanity_check_split(pairs_ds: ZincPairs, split_name: str, n_samples: int = 500, seed: int = 0) -> None:
    """Sample up to n_samples from pairs_ds and verify labels via VF2 (feature-aware induced-subgraph)."""
    N = len(pairs_ds)
    if N == 0:
        print(f"[{split_name}] WARNING: no pair samples to test.")
        return

    k = min(n_samples, N)
    rng = random.Random(seed)
    idxs = rng.sample(range(N), k)

    subset = Subset(pairs_ds, idxs)
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    n_pos = 0
    n_neg = 0
    for i, batch in enumerate(tqdm(loader, total=len(subset), desc=f"VF2 sanity [{split_name}]")):
        y = int(batch.y.item())
        neg_code = int(batch.neg_type.item())

        g1 = Data(x=batch.x1.cpu(), edge_index=batch.edge_index1.cpu())
        g2 = Data(x=batch.x2.cpu(), edge_index=batch.edge_index2.cpu())

        is_sub = is_induced_subgraph_feature_aware(
            pyg_to_nx(g1),
            pyg_to_nx(g2),
        )

        if y == 1:
            assert is_sub, f"[{split_name} idx={idxs[i]}] Positive mislabeled: not found as induced subgraph."
            n_pos += 1
        elif y == 0:
            assert not is_sub, f"[{split_name} idx={idxs[i]}] Negative mislabeled (neg_type={neg_code})."
            n_neg += 1
        else:
            raise AssertionError(f"[{split_name} idx={idxs[i]}] Unexpected label y={y} (expected 0/1).")

    print(f"[{split_name}] Sanity OK ✓  positives: {n_pos}, negatives: {n_neg}")


def run_pipeline_for_split(split_name: str, cfg: PairConfig, sanity_n: int, seed: int, k):
    print(f"\n=== Split: {split_name} ===")
    base = ZincSmiles(split=split_name)[:k]
    print(f"[{split_name}] base graphs: {len(base)}")

    pairs = ZincPairs(base_dataset=base, split=split_name, cfg=cfg, dev=True)
    print(f"[{split_name}] pair samples: {len(pairs)}")

    stats = summarize_split(pairs, split_name)
    sanity_check_split(pairs, split_name, n_samples=sanity_n, seed=seed)
    return stats  # (n, pos, neg, neg_hist)


def main():
    cfg = PairConfig()
    sanity_check_samples = 10_000  # capped by dataset size

    # Run all splits
    t_n, t_pos, t_neg, t_hist = run_pipeline_for_split("test",  cfg, sanity_check_samples, seed=0, k=200)
    v_n, v_pos, v_neg, v_hist = run_pipeline_for_split("valid", cfg, sanity_check_samples, seed=1, k=25)
    tr_n, tr_pos, tr_neg, tr_hist = run_pipeline_for_split("train", cfg, sanity_check_samples, seed=2,k=25)

    # Final aggregated summary
    print("\n=== Aggregated summary (train+valid+test) ===")
    total_n = tr_n + v_n + t_n
    total_pos = tr_pos + v_pos + t_pos
    total_neg = tr_neg + v_neg + t_neg
    total_hist = tr_hist + v_hist + t_hist

    print_split_summary("ALL", total_n, total_pos, total_neg, total_hist)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()