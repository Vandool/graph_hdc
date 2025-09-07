#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from src.datasets.zinc_pairs_v2 import ZincPairsV2

# Project imports
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import ZINC_SMILES_HRR_7744_CONFIG
from src.encoding.graph_encoders import load_or_create_hypernet
from src.utils.utils import GLOBAL_MODEL_PATH


@dataclass
class GraphPairs:
    parent: Data
    pos: list[Data]
    negs: dict[int, list[Data]]  # key = neg_type (0 if not provided)


def _parent_key(d: Data) -> Hashable:
    # Prefer a true id if your dataset exposes it
    if hasattr(d, "parent_id"):
        t = d.parent_id
        return int(t.item()) if isinstance(t, torch.Tensor) else int(t)
    # Fallback: deterministic shape signature (OK within one run)
    return "shape_sig", d.x2.size(), d.edge_index2.size()


def _to_int(x, default: int = 0) -> int:
    if x is None:
        return default
    if isinstance(x, torch.Tensor):
        return int(x.item())
    return int(x)


def collect_pairs(dataset: Iterable[Data], *, batch_size: int = 1, shuffle: bool = False) -> list[GraphPairs]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    if batch_size != 1:
        raise ValueError("Use batch_size=1 to avoid PyG Batch handling in this collector.")

    groups: dict[Hashable, GraphPairs] = {}

    for data in loader:
        key = _parent_key(data)

        if key not in groups:
            parent = Data(x=data.x2, edge_index=data.edge_index2)
            groups[key] = GraphPairs(parent=parent, pos=[], negs=defaultdict(list))

        label = _to_int(getattr(data, "y", None), default=-1)

        # Candidate is always (x1, edge_index1)
        cand = Data(x=data.x1, edge_index=data.edge_index1)

        if label == 1:
            groups[key].pos.append(cand)
        else:
            # treat anything != 1 as negative
            neg_type = _to_int(getattr(data, "neg_type", None), default=0)
            groups[key].negs[neg_type].append(cand)

    # Normalize defaultdict -> dict
    out: list[GraphPairs] = []
    for gp in groups.values():
        gp.negs = dict(gp.negs)
        out.append(gp)
    return out


if __name__ == "__main__":
    # Build datasets (use the split you want)
    base = ZincSmiles(split="test")
    pairs = ZincPairsV2(split="test", base_dataset=base, dev=True)

    # Quick probe: count labels to verify assumptions
    probe_loader = DataLoader(pairs, batch_size=1, shuffle=False)
    lbl_counts = defaultdict(int)
    for d in probe_loader:
        y = int(d.y.item()) if hasattr(d, "y") else -1
        lbl_counts[y] += 1
    print(f"Label counts: {dict(lbl_counts)}")

    collection = collect_pairs(pairs, batch_size=1, shuffle=False)

    print(f"Collected {len(collection)} parent groups.")
    total_pos = sum(len(gp.pos) for gp in collection)
    total_neg = sum(sum(len(v) for v in gp.negs.values()) for gp in collection)
    print(f"Total positives: {total_pos}")
    print(f"Total negatives: {total_neg}")

    show = min(5, len(collection))
    for i in range(show):
        gp = collection[i]
        neg_breakdown = {k: len(v) for k, v in gp.negs.items()}
        print(f"[{i}] pos={len(gp.pos)} negs={neg_breakdown}")

    ## Encode in hypervector
    hypernet = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, cfg=ZINC_SMILES_HRR_7744_CONFIG, depth=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hypernet.to(device)
    print("Device:", device)

    # ----- Encode on the correct device -----
    gp = collection[2]  # pick any group with both pops and negs

    parent_batch = Batch.from_data_list([gp.parent]).to(device)
    pos_batch = Batch.from_data_list(gp.pos).to(device) if gp.pos else None
    neg_batches = {k: Batch.from_data_list(v).to(device) for k, v in gp.negs.items() if v}

    with torch.no_grad():
        parent_hv = hypernet(parent_batch)["graph_embedding"]  # [1, D]
        pos_hv = hypernet(pos_batch)["graph_embedding"] if pos_batch else None
        neg_hvs = {k: hypernet(b)["graph_embedding"] for k, b in neg_batches.items()}  # k -> [B, D]

    # ----- PCA + Plot -----
    import math

    def _to_np(t: torch.Tensor | None) -> np.ndarray | None:
        return None if t is None else t.detach().cpu().numpy()

    P = _to_np(parent_hv)  # (1, D)
    POS = _to_np(pos_hv)  # (Np, D) or None
    NEGS = {k: _to_np(v) for k, v in neg_hvs.items()}  # k -> (N_k, D)

    # Build a single matrix to fit PCA on all points for a consistent projection:
    blocks = [P]
    labels = [("parent", 1)]
    if POS is not None and POS.shape[0] > 0:
        blocks.append(POS)
        labels.append(("pos", POS.shape[0]))
    for k, arr in NEGS.items():
        if arr is not None and arr.shape[0] > 0:
            blocks.append(arr)
            labels.append((f"neg[{k}]", arr.shape[0]))

    X = np.concatenate(blocks, axis=0)  # (N_total, D)

    # PCA to 2D (pure NumPy/Torch fallback to avoid extra deps)
    # Center:
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    # SVD on covariance surrogate (Xc = U S V^T). Top-2 PCs are first two rows of V^T.
    # For numerical stability on tall matrices, use economy SVD:
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    PCs = Vt[:2, :]  # (2, D)
    Z = Xc @ PCs.T  # (N_total, 2)

    # Split back into groups
    idx = 0
    parent_2d = Z[idx : idx + labels[0][1]]
    idx += labels[0][1]
    pos_2d = None
    if len(labels) > 1 and labels[1][0] == "pos":
        n = labels[1][1]
        pos_2d = Z[idx : idx + n]
        idx += n

    neg2d = {}
    for name, n in labels[2 if pos_2d is not None else 1 :]:
        if name.startswith("neg["):
            k = int(name[4:-1])  # extract inside brackets
            neg2d[k] = Z[idx : idx + n]
        idx += n

    # ----- Plot -----
    fig, ax = plt.subplots(figsize=(16, 14))

    # Parent: big filled blue circle
    ax.scatter(
        parent_2d[:, 0], parent_2d[:, 1], s=220, marker="o", c="blue", alpha=0.3, edgecolor="none", label="parent"
    )

    # Positives: empty circle + green plus sign
    if pos_2d is not None and len(pos_2d) > 0:
        # empty circle
        ax.scatter(
            pos_2d[:, 0],
            pos_2d[:, 1],
            s=90,
            facecolors="none",
            edgecolors="green",
            linewidths=1.5,
            marker="o",
            label="pos (circle)",
        )
        # plus sign in the middle
        ax.scatter(pos_2d[:, 0], pos_2d[:, 1], s=90, c="green", marker="+", linewidths=1.5, label="pos (+)")

    # Negatives: distinct red tones
    neg_keys = sorted(neg2d.keys())
    if neg_keys:
        reds = plt.cm.Reds
        shades = np.linspace(0.55, 0.95, num=len(neg_keys))
        for shade, k in zip(shades, neg_keys, strict=False):
            pts = neg2d[k]
            ax.scatter(pts[:, 0], pts[:, 1], s=50, marker="o", edgecolor="none", label=f"neg[{k}]", c=[reds(shade)])

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("PCA (2D) of Graph Embeddings")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(loc="best", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.show()

    # ----- UMAP Projection -----
    import numpy as np
    import umap
    import matplotlib.pyplot as plt

    def _to_np(t):
        return None if t is None else t.detach().cpu().numpy()

    P = _to_np(parent_hv)                         # (1, D)
    POS = _to_np(pos_hv)                          # (Np, D)
    NEGS = {k: _to_np(v) for k, v in neg_hvs.items()}  # dict of (N_k, D)

    blocks = [P]
    labels = [("parent", 1)]
    if POS is not None and len(POS) > 0:
        blocks.append(POS)
        labels.append(("pos", POS.shape[0]))
    for k, arr in NEGS.items():
        if arr is not None and len(arr) > 0:
            blocks.append(arr)
            labels.append((f"neg[{k}]", arr.shape[0]))

    X = np.concatenate(blocks, axis=0)

    reducer = umap.UMAP(n_components=2, random_state=42)
    Z = reducer.fit_transform(X)   # (N_total, 2)

    # Split back into groups
    idx = 0
    parent_2d = Z[idx:idx + labels[0][1]]; idx += labels[0][1]
    pos_2d = None
    if len(labels) > 1 and labels[1][0] == "pos":
        n = labels[1][1]
        pos_2d = Z[idx:idx + n]
        idx += n
    neg2d = {}
    for name, n in labels[2 if pos_2d is not None else 1:]:
        if name.startswith("neg["):
            k = int(name[4:-1])
            neg2d[k] = Z[idx:idx + n]
        idx += n

    # ----- Plot UMAP -----
    fig, ax = plt.subplots(figsize=(7, 6))

    # Parent: big filled blue circle
    ax.scatter(parent_2d[:, 0], parent_2d[:, 1], s=220, marker="o",
               c="blue", alpha=0.3, edgecolor="none", label="parent")

    # Positives: empty circle + plus
    if pos_2d is not None and len(pos_2d) > 0:
        ax.scatter(pos_2d[:, 0], pos_2d[:, 1],
                   s=90, facecolors="none", edgecolors="green",
                   linewidths=1.5, marker="o", label="pos (circle)")
        ax.scatter(pos_2d[:, 0], pos_2d[:, 1],
                   s=90, c="green", marker="+", linewidths=1.5, label="pos (+)")

    # Negatives: different red tones
    reds = plt.cm.Reds
    neg_keys = sorted(neg2d.keys())
    if neg_keys:
        shades = np.linspace(0.55, 0.95, num=len(neg_keys))
        for shade, k in zip(shades, neg_keys):
            pts = neg2d[k]
            ax.scatter(pts[:, 0], pts[:, 1], s=50, marker="o", edgecolor="none",
                       label=f"neg[{k}]", c=[reds(shade)])

    ax.set_title("UMAP (2D) of Graph Embeddings")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(loc="best", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.show()
