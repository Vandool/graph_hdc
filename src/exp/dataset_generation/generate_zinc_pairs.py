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


def summarize_split(ds: ZincPairs, name: str) -> None:
    n = len(ds)
    if n == 0:
        print(f"[{name}] size=0")
        return

    pos = 0
    neg = 0
    neg_hist = collections.Counter()

    for i in range(n):
        item = ds[i]
        # robust scalar extraction
        y = int(item.y.view(-1)[0].item())
        if y == 1:
            pos += 1
        else:
            neg += 1
            neg_type = int(item.neg_type.view(-1)[0].item())
            neg_hist[neg_type] += 1

    print(f"[{name}] size={n} | positives={pos} ({pos/n:.1%}) | negatives={neg} ({neg/n:.1%})")
    if neg:
        hist_str = ", ".join(
            f"{t}:{c} ({c/neg:.1%})" for t, c in sorted(neg_hist.items())
        )
        print(f"[{name}] neg_type histogram: {hist_str}")


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
        # batch_size=1 => scalars below are 0-d tensors
        y = int(batch.y.item())
        neg_code = int(batch.neg_type.item())

        # Rebuild candidate (G1) and parent (G2)
        g1 = Data(x=batch.x1.cpu(), edge_index=batch.edge_index1.cpu())
        g2 = Data(x=batch.x2.cpu(), edge_index=batch.edge_index2.cpu())

        # Feature-aware induced-subgraph test (label-preserving)
        is_sub = is_induced_subgraph_feature_aware(
            pyg_to_nx(g1),  # small graph
            pyg_to_nx(g2),  # big/parent graph
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


def main():
    # Build base splits
    train_ds = ZincSmiles(split="train")
    valid_ds = ZincSmiles(split="valid")
    test_ds = ZincSmiles(split="test")

    # Build pair datasets (processed files will be cached under root/processed)
    cfg = PairConfig()
    test_pairs = ZincPairs(base_dataset=test_ds, split="test", cfg=cfg)
    # valid_pairs = ZincPairs(base_dataset=valid_ds, split="valid", cfg=cfg)

    # Global label mix + neg_type mix
    # summarize_split(valid_pairs, "valid")
    summarize_split(test_pairs, "test")

    # Run sanity checks (random samples each)
    sanity_check_samples = 10_000
    # sanity_check_split(valid_pairs, "valid", n_samples=sanity_check_samples, seed=1)
    sanity_check_split(test_pairs, "test", n_samples=sanity_check_samples, seed=2)

    # Build train pairs last (takes longest)
    train_pairs = ZincPairs(base_dataset=train_ds, split="train", cfg=cfg)

    # Global label mix + neg_type mix
    summarize_split(train_pairs, "train")

    # Run sanity checks (random samples each)
    sanity_check_split(train_pairs, "train", n_samples=sanity_check_samples, seed=0)



if __name__ == "__main__":
    # Optional: make torch deterministic-ish for reproducibility of any transforms
    torch.manual_seed(0)
    main()
