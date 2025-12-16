"""
Dataset presentation plots for QM9 and ZINC molecular datasets.

Run: python dataset_presentation_plots.py
Outputs saved to: src/datasets/plots/plots/
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem

from src.datasets.utils import get_dataset_props, get_split

# ===== KIT Color Palette =====
KIT_COLORS = {
    "teal": (0 / 255, 155 / 255, 127 / 255),  # kitteal - dataset
    "orange": (217 / 255, 102 / 255, 31 / 255),  # kitorange - generated 1
    "blue": (41 / 255, 94 / 255, 138 / 255),  # kitblue - generated 2
    "red": (180 / 255, 40 / 255, 40 / 255),  # kitred - target lines
}

TARGET_COLORS = [
    KIT_COLORS["orange"],
    KIT_COLORS["blue"],
    (0.6, 0.4, 0.8),
    (0.4, 0.7, 0.4),
]

DATASET_COLORS = {"qm9": KIT_COLORS["teal"], "zinc": KIT_COLORS["orange"]}

# ===== Matplotlib Configuration =====
plt.rcParams.update(
    {
        "font.family": "Source Sans Pro",
        "mathtext.fontset": "dejavusans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

OUTPUT_DIR = Path(__file__).parent / "plots"


def _stats(arr):
    arr = np.asarray(arr)
    return {
        "mean": float(np.mean(arr)) if arr.size else 0,
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0,
    }


def plot_dataset_summary(dataset: str):
    """Create comprehensive summary figure for a dataset."""
    props = get_dataset_props(dataset, splits=["train"])
    ds = get_split("train", base_dataset=dataset)
    color = DATASET_COLORS[dataset]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Node count
    ax1 = fig.add_subplot(gs[0, 0])
    num_nodes = [int(data.num_nodes) for data in ds]
    ax1.hist(num_nodes, bins=30, color=color, alpha=0.8, edgecolor="white")
    ax1.axvline(np.mean(num_nodes), color=KIT_COLORS["red"], linestyle="--", linewidth=1.5)
    ax1.set_xlabel("Number of Atoms")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Atom Count")

    # 2. LogP
    ax2 = fig.add_subplot(gs[0, 1])
    logp = [v for v in props.logp if not np.isnan(v)]
    ax2.hist(logp, bins=40, color=color, alpha=0.8, edgecolor="white")
    ax2.axvline(np.mean(logp), color=KIT_COLORS["red"], linestyle="--", linewidth=1.5)
    ax2.set_xlabel("LogP")
    ax2.set_ylabel("Frequency")
    ax2.set_title("LogP Distribution")

    # 3. QED
    ax3 = fig.add_subplot(gs[0, 2])
    qed = [v for v in props.qed if not np.isnan(v)]
    ax3.hist(qed, bins=40, color=color, alpha=0.8, edgecolor="white")
    ax3.axvline(np.mean(qed), color=KIT_COLORS["red"], linestyle="--", linewidth=1.5)
    ax3.set_xlabel("QED")
    ax3.set_ylabel("Frequency")
    ax3.set_title("QED Distribution")

    # 4. SA Score
    ax4 = fig.add_subplot(gs[1, 0])
    sa = [v for v in props.sa_score if not np.isnan(v)]
    ax4.hist(sa, bins=40, color=color, alpha=0.8, edgecolor="white")
    ax4.axvline(np.mean(sa), color=KIT_COLORS["red"], linestyle="--", linewidth=1.5)
    ax4.set_xlabel("SA Score")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Synthetic Accessibility")

    # 5. Molecular Weight
    ax5 = fig.add_subplot(gs[1, 1])
    mw = [v for v in props.mw if not np.isnan(v)]
    ax5.hist(mw, bins=40, color=color, alpha=0.8, edgecolor="white")
    ax5.axvline(np.mean(mw), color=KIT_COLORS["red"], linestyle="--", linewidth=1.5)
    ax5.set_xlabel("Molecular Weight (Da)")
    ax5.set_ylabel("Frequency")
    ax5.set_title("Molecular Weight")

    # 6. Number of Rings
    ax6 = fig.add_subplot(gs[1, 2])
    ring_counter = Counter(props.num_rings)
    sorted_rings = sorted(ring_counter.items())
    if sorted_rings:
        labels, counts = zip(*sorted_rings)
        ax6.bar(labels, counts, color=color, alpha=0.8, edgecolor="white")
        ax6.set_xticks(labels)
    ax6.set_xlabel("Number of Rings")
    ax6.set_ylabel("Frequency")
    ax6.set_title("Rings per Molecule")

    # 7. Atom types
    ax7 = fig.add_subplot(gs[2, 0])
    atom_counter = Counter()
    for data in ds:
        mol = Chem.MolFromSmiles(data.smiles)
        if mol:
            for atom in mol.GetAtoms():
                atom_counter[atom.GetSymbol()] += 1
    sorted_atoms = atom_counter.most_common(10)
    if sorted_atoms:
        labels, counts = zip(*sorted_atoms)
        ax7.bar(range(len(labels)), counts, color=color, alpha=0.8, edgecolor="white")
        ax7.set_xticks(range(len(labels)))
        ax7.set_xticklabels(labels)
    ax7.set_xlabel("Atom Type")
    ax7.set_ylabel("Count")
    ax7.set_title("Atom Types (Top 10)")

    # 8. Ring sizes
    ax8 = fig.add_subplot(gs[2, 1])
    ring_size_counter = Counter()
    for data in ds:
        mol = Chem.MolFromSmiles(data.smiles)
        if mol:
            for ring in mol.GetRingInfo().AtomRings():
                ring_size_counter[len(ring)] += 1
    sorted_sizes = sorted(ring_size_counter.items())
    if sorted_sizes:
        labels, counts = zip(*sorted_sizes)
        ax8.bar(labels, counts, color=color, alpha=0.8, edgecolor="white")
        ax8.set_xticks(labels)
    ax8.set_xlabel("Ring Size")
    ax8.set_ylabel("Count")
    ax8.set_title("Ring Size Distribution")

    # 9. TPSA
    ax9 = fig.add_subplot(gs[2, 2])
    tpsa = [v for v in props.tpsa if not np.isnan(v)]
    ax9.hist(tpsa, bins=40, color=color, alpha=0.8, edgecolor="white")
    ax9.axvline(np.mean(tpsa), color=KIT_COLORS["red"], linestyle="--", linewidth=1.5)
    ax9.set_xlabel("TPSA (A^2)")
    ax9.set_ylabel("Frequency")
    ax9.set_title("Polar Surface Area")

    fig.suptitle(f"{dataset.upper()} Dataset Summary (Train Split, n={len(ds):,})", fontsize=14, y=1.01)

    fig.savefig(OUTPUT_DIR / f"{dataset}_summary.pdf")
    fig.savefig(OUTPUT_DIR / f"{dataset}_summary.png")
    plt.close(fig)


def plot_node_edge_distributions(dataset: str):
    """Plot node and edge count distributions."""
    ds = get_split("train", base_dataset=dataset)
    color = DATASET_COLORS[dataset]

    num_nodes = [int(data.num_nodes) for data in ds]
    num_edges = [int(data.num_edges) // 2 for data in ds]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(num_nodes, bins=30, color=color, alpha=0.8, edgecolor="white")
    axes[0].axvline(np.mean(num_nodes), color=KIT_COLORS["red"], linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Number of Nodes")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"{dataset.upper()} - Node Count")

    axes[1].hist(num_edges, bins=30, color=color, alpha=0.8, edgecolor="white")
    axes[1].axvline(np.mean(num_edges), color=KIT_COLORS["red"], linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Number of Edges")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"{dataset.upper()} - Edge Count")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{dataset}_node_edge.pdf")
    fig.savefig(OUTPUT_DIR / f"{dataset}_node_edge.png")
    plt.close(fig)


def plot_molecular_properties(dataset: str):
    """Plot all molecular property distributions in a grid."""
    props = get_dataset_props(dataset, splits=["train"])
    color = DATASET_COLORS[dataset]

    property_configs = [
        ("logp", "LogP"),
        ("qed", "QED"),
        ("sa_score", "SA Score"),
        ("mw", "Molecular Weight"),
        ("tpsa", "TPSA"),
        ("num_rings", "Number of Rings"),
        ("num_aromatic_rings", "Aromatic Rings"),
        ("num_rotatable_bonds", "Rotatable Bonds"),
        ("bertz_ct", "Bertz Complexity"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (prop_name, label) in enumerate(property_configs):
        ax = axes[idx]
        values = getattr(props, prop_name, None)
        if values is None:
            ax.set_visible(False)
            continue

        values = [v for v in values if v is not None and not np.isnan(v)]
        if not values:
            ax.set_visible(False)
            continue

        ax.hist(values, bins=40, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(np.mean(values), color=KIT_COLORS["red"], linestyle="--", linewidth=1.5)
        ax.set_xlabel(label)
        ax.set_ylabel("Frequency")

    fig.suptitle(f"{dataset.upper()} - Molecular Properties", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{dataset}_properties.pdf")
    fig.savefig(OUTPUT_DIR / f"{dataset}_properties.png")
    plt.close(fig)


def plot_atom_types(dataset: str):
    """Plot atom type distribution."""
    ds = get_split("train", base_dataset=dataset)
    color = DATASET_COLORS[dataset]

    atom_counter = Counter()
    for data in ds:
        mol = Chem.MolFromSmiles(data.smiles)
        if mol:
            for atom in mol.GetAtoms():
                atom_counter[atom.GetSymbol()] += 1

    sorted_atoms = atom_counter.most_common()
    labels, counts = zip(*sorted_atoms) if sorted_atoms else ([], [])

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), counts, color=color, alpha=0.8, edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Atom Type")
    ax.set_ylabel("Count")
    ax.set_title(f"{dataset.upper()} - Atom Type Distribution")

    for bar, count in zip(bars, counts):
        ax.annotate(f"{count:,}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{dataset}_atom_types.pdf")
    fig.savefig(OUTPUT_DIR / f"{dataset}_atom_types.png")
    plt.close(fig)


def plot_degree_distribution(dataset: str):
    """Plot degree distribution."""
    ds = get_split("train", base_dataset=dataset)
    color = DATASET_COLORS[dataset]

    degree_counter = Counter()
    for data in ds:
        mol = Chem.MolFromSmiles(data.smiles)
        if mol:
            for atom in mol.GetAtoms():
                degree_counter[atom.GetDegree()] += 1

    sorted_degrees = sorted(degree_counter.items())
    labels, counts = zip(*sorted_degrees) if sorted_degrees else ([], [])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, counts, color=color, alpha=0.8, edgecolor="white")
    ax.set_xticks(labels)
    ax.set_xlabel("Node Degree")
    ax.set_ylabel("Count")
    ax.set_title(f"{dataset.upper()} - Degree Distribution")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{dataset}_degree.pdf")
    fig.savefig(OUTPUT_DIR / f"{dataset}_degree.png")
    plt.close(fig)


def plot_ring_distributions(dataset: str):
    """Plot ring size and count distributions."""
    ds = get_split("train", base_dataset=dataset)
    props = get_dataset_props(dataset, splits=["train"])
    color = DATASET_COLORS[dataset]

    # Ring size distribution
    ring_size_counter = Counter()
    for data in ds:
        mol = Chem.MolFromSmiles(data.smiles)
        if mol:
            for ring in mol.GetRingInfo().AtomRings():
                ring_size_counter[len(ring)] += 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Ring sizes
    sorted_sizes = sorted(ring_size_counter.items())
    if sorted_sizes:
        labels, counts = zip(*sorted_sizes)
        axes[0].bar(labels, counts, color=color, alpha=0.8, edgecolor="white")
        axes[0].set_xticks(labels)
    axes[0].set_xlabel("Ring Size")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Ring Size Distribution")

    # Rings per molecule
    ring_counter = Counter(props.num_rings)
    sorted_rings = sorted(ring_counter.items())
    if sorted_rings:
        labels, counts = zip(*sorted_rings)
        axes[1].bar(labels, counts, color=color, alpha=0.8, edgecolor="white")
        axes[1].set_xticks(labels)
    axes[1].set_xlabel("Number of Rings")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Rings per Molecule")

    # Max ring size
    max_ring_counter = Counter(props.max_ring_size_calc)
    sorted_max = sorted(max_ring_counter.items())
    if sorted_max:
        labels, counts = zip(*sorted_max)
        axes[2].bar(labels, counts, color=color, alpha=0.8, edgecolor="white")
        axes[2].set_xticks(labels)
    axes[2].set_xlabel("Max Ring Size")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Maximum Ring Size")

    fig.suptitle(f"{dataset.upper()} - Ring Analysis", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{dataset}_rings.pdf")
    fig.savefig(OUTPUT_DIR / f"{dataset}_rings.png")
    plt.close(fig)


def plot_comparison():
    """Plot property comparison between QM9 and ZINC."""
    qm9_props = get_dataset_props("qm9", splits=["train"])
    zinc_props = get_dataset_props("zinc", splits=["train"])

    property_configs = [
        ("logp", "LogP"),
        ("qed", "QED"),
        ("sa_score", "SA Score"),
        ("mw", "Molecular Weight"),
        ("tpsa", "TPSA"),
        ("num_rings", "Number of Rings"),
        ("num_aromatic_rings", "Aromatic Rings"),
        ("num_atoms", "Number of Atoms"),
        ("num_bonds", "Number of Bonds"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (prop_name, label) in enumerate(property_configs):
        ax = axes[idx]

        qm9_values = getattr(qm9_props, prop_name, None)
        zinc_values = getattr(zinc_props, prop_name, None)

        if qm9_values is None or zinc_values is None:
            ax.set_visible(False)
            continue

        qm9_values = [v for v in qm9_values if v is not None and not np.isnan(v)]
        zinc_values = [v for v in zinc_values if v is not None and not np.isnan(v)]

        if not qm9_values or not zinc_values:
            ax.set_visible(False)
            continue

        all_values = qm9_values + zinc_values
        bin_edges = np.histogram_bin_edges(all_values, bins=40)

        ax.hist(qm9_values, bins=bin_edges, color=DATASET_COLORS["qm9"], alpha=0.6,
                label=f'QM9 (mean={np.mean(qm9_values):.2f})', density=True, edgecolor="white")
        ax.hist(zinc_values, bins=bin_edges, color=DATASET_COLORS["zinc"], alpha=0.6,
                label=f'ZINC (mean={np.mean(zinc_values):.2f})', density=True, edgecolor="white")

        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle("QM9 vs ZINC - Property Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison_all.pdf")
    fig.savefig(OUTPUT_DIR / "comparison_all.png")
    plt.close(fig)


def main():
    """Generate all plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in ["qm9", "zinc"]:
        print(f"Generating plots for {dataset.upper()}...")
        plot_dataset_summary(dataset)
        plot_node_edge_distributions(dataset)
        plot_molecular_properties(dataset)
        plot_atom_types(dataset)
        plot_degree_distribution(dataset)
        plot_ring_distributions(dataset)

    print("Generating comparison plots...")
    plot_comparison()

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
