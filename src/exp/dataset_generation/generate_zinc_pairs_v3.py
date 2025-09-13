import argparse
from collections import Counter

import torch
from torch_geometric import seed_everything
from torch_geometric.data import Data
from tqdm.auto import tqdm

from src.datasets.zinc_pairs_v3 import PairData, PairType, ZincPairV3Config, ZincPairsV3
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.decoder import is_induced_subgraph_by_features
from src.utils.utils import DataTransformer


def sanity_check_pairs(pairs: ZincPairsV3, base_ds, *, limit: int | None = None) -> dict:
    r"""
    Validate a :class:`ZincPairsV3` dataset and RETURN stats (no printing).

    Returns
    -------
    dict
        {
          "total": int,
          "n_pos": int,
          "n_neg": int,
          "per_k": {k: count, ...},
          "per_type_total": {name: count, ...},
          "per_type_pos":   {name: count, ...},
          "per_type_neg":   {name: count, ...},
          "breakdown": {
             "positives": [{"type": name, "count": c, "pct_pos": p1, "pct_total": p2}, ...],
             "negatives": [{"type": name, "count": c, "pct_neg": p1, "pct_total": p2}, ...],
          }
        }
    """
    # Precompute parent NX once
    parent_nx = [DataTransformer.pyg_to_nx(base_ds[i], strict_undirected=True) for i in range(len(base_ds))]

    n_pos = n_neg = 0
    per_type = Counter()
    per_type_pos = Counter()
    per_type_neg = Counter()
    per_k = Counter()

    it = range(len(pairs)) if limit is None else range(min(limit, len(pairs)))

    with torch.inference_mode():
        for i in tqdm(it, total=(len(it) if limit is not None else len(pairs)), desc="VF2 sanity"):
            p: PairData = pairs.get(i)

            y = int(p.y.item())
            k = int(p.k.item())
            tcode = int(p.neg_type.item())
            pid = int(p.parent_idx.item())

            per_type[tcode] += 1
            per_k[k] += 1

            # candidate (g1) and parent (G2)
            g1 = Data(x=p.x1.cpu(), edge_index=p.edge_index1.cpu())
            G1 = DataTransformer.pyg_to_nx(g1, strict_undirected=True)
            G2 = parent_nx[pid]

            is_sub = is_induced_subgraph_by_features(G1, G2, require_connected=True)

            # k==2 shape invariants
            if k == 2:
                if tcode == PairType.POSITIVE_EDGE:
                    assert G1.number_of_nodes() == 2 and G1.number_of_edges() == 1, (
                        f"[{i}] POSITIVE_EDGE must be 2 nodes / 1 edge."
                    )
                elif tcode == PairType.NEGATIVE_EDGE:
                    assert G1.number_of_nodes() == 2 and G1.number_of_edges() == 1, (
                        f"[{i}] NEGATIVE_EDGE must be 2 nodes / 1 edges."
                    )

            if y == 1:
                assert is_sub, f"[{i}] Positive mislabeled (not induced). type={PairType(tcode).name}, k={k}, pid={pid}"
                n_pos += 1
                per_type_pos[tcode] += 1
            elif y == 0:
                assert not is_sub, f"[{i}] Negative mislabeled (embeds). type={PairType(tcode).name}, k={k}, pid={pid}"
                n_neg += 1
                per_type_neg[tcode] += 1
            else:
                raise AssertionError(f"[{i}] Unexpected y={y} (expected 0/1).")

    total = n_pos + n_neg

    def pct(num: int, den: int) -> float:
        return (100.0 * num / den) if den > 0 else 0.0

    # Named dicts
    per_type_total_named = {PairType(t).name: c for t, c in sorted(per_type.items(), key=lambda kv: kv[0])}
    per_type_pos_named = {PairType(t).name: c for t, c in sorted(per_type_pos.items(), key=lambda kv: kv[0])}
    per_type_neg_named = {PairType(t).name: c for t, c in sorted(per_type_neg.items(), key=lambda kv: kv[0])}

    # Detailed breakdown lists
    pos_types = [PairType.POSITIVE_EDGE, PairType.POSITIVE]
    neg_types = [
        PairType.FORBIDDEN_ADD,
        PairType.WRONG_EDGE_ALLOWED,
        PairType.CROSS_PARENT,
        PairType.MISSING_EDGE,
        PairType.REWIRE,
        PairType.ANCHOR_AWARE,
        PairType.NEGATIVE_EDGE,
    ]

    pos_breakdown = []
    for t in pos_types:
        c = per_type_pos.get(t.value, 0)
        pos_breakdown.append(
            {
                "type": t.name,
                "count": c,
                "pct_pos": round(pct(c, n_pos), 1),
                "pct_total": round(pct(c, total), 1),
            }
        )

    neg_breakdown = []
    for t in neg_types:
        c = per_type_neg.get(t.value, 0)
        neg_breakdown.append(
            {
                "type": t.name,
                "count": c,
                "pct_neg": round(pct(c, n_neg), 1),
                "pct_total": round(pct(c, total), 1),
            }
        )

    return {
        "total": total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "per_k": dict(per_k),
        "per_type_total": per_type_total_named,
        "per_type_pos": per_type_pos_named,
        "per_type_neg": per_type_neg_named,
        "breakdown": {
            "positives": pos_breakdown,
            "negatives": neg_breakdown,
        },
    }


def print_sanity_summary(stats_by_split: dict[str, dict]) -> None:
    """Pretty-print the sanity stats for multiple splits in one go."""
    for split, S in stats_by_split.items():
        print(f"\n=== {split.upper()} ===")
        print(f"positives: {S['n_pos']}, negatives: {S['n_neg']}, total: {S['total']}")
        print("By k (counts):", S["per_k"])

        print("\nPositives breakdown (within positives / of total):")
        for row in S["breakdown"]["positives"]:
            print(
                f"  {row['type']:<16} count={row['count']:>8}   %pos={row['pct_pos']:5.1f}%   %total={row['pct_total']:5.1f}%"
            )

        print("\nNegatives breakdown (within negatives / of total):")
        for row in S["breakdown"]["negatives"]:
            print(
                f"  {row['type']:<16} count={row['count']:>8}   %neg={row['pct_neg']:5.1f}%   %total={row['pct_total']:5.1f}%"
            )

        print("\nBy type (counts over all samples):", S["per_type_total"])


def build_zinc_pairs_for_split(
    split: str,
    *,
    cfg: ZincPairV3Config | None = None,
    force_reprocess: bool = False,
    debug: bool = False,
    is_dev: bool = False,
    n: int | None = None,
    sanity_limit: int | None = None,
) -> tuple[ZincPairsV3, dict]:
    """
    Build ONE split and sanity-check it. Returns (dataset, stats).
    If `n`/`sanity_limit` are None, sensible defaults are chosen based on `is_dev`.
    """
    assert split in {"train", "valid", "test"}

    # Defaults: small dev slice; full otherwise
    if n is None:
        n = {"train": 200, "valid": 20, "test": 20}[split] if is_dev else None
    if sanity_limit is None:
        sanity_limit = {"train": 1_000_000, "valid": 100_000, "test": 100_000}[split] if is_dev else None

    print(f"\n=== Building pairs for split='{split}' n={n} dev={is_dev} ===")
    base_full = ZincSmiles(split=split)
    base = base_full[:n] if n is not None else base_full

    pairs = ZincPairsV3(
        base_dataset=base,
        split=split,
        cfg=cfg or ZincPairV3Config(),
        dev=is_dev,
        debug=debug,
        force_reprocess=force_reprocess,
    )

    print(f"  base graphs: {len(base)}")
    print(f"  pair samples: {len(pairs)}")
    print(f"  processed_dir: {pairs.processed_dir}")

    stats = sanity_check_pairs(pairs, base, limit=sanity_limit)
    print_sanity_summary({split: stats})
    return pairs, stats


def main():
    p = argparse.ArgumentParser(description="Generate ZincPairsV3 for a single split.")
    p.add_argument("--split", required=True, choices=["train", "valid", "test"])
    p.add_argument("--dev", action="store_true", help="Use small dev sizes (train=200, valid=20, test=20).")
    p.add_argument("--n", type=int, default=None, help="Number of base molecules (overrides --dev default).")
    p.add_argument("--sanity-limit", type=int, default=None, help="VF2 check cap (overrides --dev default).")
    args = p.parse_args()

    cfg = ZincPairV3Config()
    seed_everything(cfg.seed)

    _, stats = build_zinc_pairs_for_split(
        split=args.split,
        cfg=cfg,
        force_reprocess=False,
        debug=False,
        is_dev=True,
        n=args.n,
        sanity_limit=args.sanity_limit,
    )

    _, stats = build_zinc_pairs_for_split(
        split=args.split,
        cfg=cfg,
        force_reprocess=False,
        debug=False,
        is_dev=False,
        n=args.n,
        sanity_limit=args.sanity_limit,
    )


if __name__ == "__main__":
    main()
