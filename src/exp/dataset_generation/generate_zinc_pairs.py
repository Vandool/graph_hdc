# scripts/build_and_sanity_all_splits.py
import sys
import random
from pathlib import Path

import torch
from tqdm.auto import tqdm
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.datasets.zinc_smiles_generation import ZincSmiles
from src.datasets.zinc_pairs import (
    ZincPairs,
    PairConfig,
    pyg_to_nx,
    is_induced_subgraph_feature_aware,
)

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
            pyg_to_nx(g1),    # small graph
            pyg_to_nx(g2),    # big/parent graph
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
    train_ds = ZincSmiles(split="train")[:10]
    valid_ds = ZincSmiles(split="valid")[:10]
    test_ds  = ZincSmiles(split="test")[:10]

    # Build pair datasets (processed files will be cached under root/processed)
    cfg = PairConfig()
    test_pairs  = ZincPairs(base_dataset=test_ds,  split="test",  cfg=cfg)
    valid_pairs = ZincPairs(base_dataset=valid_ds, split="valid", cfg=cfg)
    train_pairs = ZincPairs(base_dataset=train_ds, split="train", cfg=cfg)

    print(f"base graphs: train={len(train_ds)}, valid={len(valid_ds)}, test={len(test_ds)}")
    print(f"pair samples: train={len(train_pairs)}, valid={len(valid_pairs)}, test={len(test_pairs)}")

    # Run sanity checks (up to 500 random samples each)
    sanity_check_samples = 10
    sanity_check_split(train_pairs, "train", n_samples=sanity_check_samples, seed=0)
    sanity_check_split(valid_pairs, "valid", n_samples=sanity_check_samples, seed=1)
    sanity_check_split(test_pairs,  "test",  n_samples=sanity_check_samples, seed=2)

if __name__ == "__main__":
    # Optional: make torch deterministic-ish for reproducibility of any transforms
    torch.manual_seed(0)
    main()