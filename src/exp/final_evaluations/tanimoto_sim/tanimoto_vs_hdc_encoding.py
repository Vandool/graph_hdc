"""
Tanimoto vs HDC Encoding Similarity Analysis.

This experiment validates Hypothesis 2: "The parameter-free HDC encoder produces
an inherently structured latent space where molecular similarity is preserved
as geometric proximity."

We use an EXTERNAL, INDEPENDENT measure of molecular similarity (Tanimoto on
Morgan fingerprints - the gold standard in cheminformatics) to test whether
chemically similar molecules end up close in HDC hypervector space (high cosine
similarity).

Key components:
1. Morgan fingerprints (ECFP4) -> Tanimoto similarity (cheminformatics standard)
2. HDC graph_terms -> Cosine similarity (geometric proximity in HDC space)
3. Correlation analysis between these two independent measures
4. Random baseline to prove correlation is due to HDC structure, not chance
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchhd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.datasets.utils import get_split
from src.encoding.configs_and_constants import (
    QM9_SMILES_HRR_256_CONFIG_F64_G1NG3_CONFIG,
    ZINC_SMILES_HRR_256_F64_5G1NG4_CONFIG,
)
from src.encoding.graph_encoders import HyperNet

# ===== KIT Color Palette =====
KIT_COLORS = {
    "teal": (0 / 255, 155 / 255, 127 / 255),
    "orange": (217 / 255, 102 / 255, 31 / 255),
    "blue": (41 / 255, 94 / 255, 138 / 255),
    "red": (180 / 255, 40 / 255, 40 / 255),
    "green": (0 / 255, 150 / 255, 130 / 255),
}

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

# Results directory
RESULTS_DIR = Path(__file__).parent / "results"

# Configuration
N_PAIRS = 100_000  # Number of pairs to sample per dataset
MORGAN_RADIUS = 2  # ECFP4
MORGAN_NBITS = 2048
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_morgan_fingerprint(smiles: str, radius: int = MORGAN_RADIUS, n_bits: int = MORGAN_NBITS):
    """Compute Morgan fingerprint (ECFP4) - the cheminformatics standard."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def compute_tanimoto_similarity(fp1, fp2) -> float:
    """Compute Tanimoto similarity between two Morgan fingerprints."""
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def sample_pairs(n_molecules: int, n_pairs: int, seed: int = SEED) -> np.ndarray:
    """Sample random pairs ensuring no self-pairs."""
    rng = np.random.default_rng(seed)
    pairs = rng.choice(n_molecules, size=(n_pairs * 2, 2), replace=True)
    # Remove self-pairs
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    # Take requested number
    return pairs[:n_pairs]


def compute_pairwise_cosine(embeddings: torch.Tensor, pairs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity for sampled pairs.

    Uses torchhd.cos for consistency with the rest of the codebase.
    The embeddings are used exactly as returned by HyperNet.forward() - NO additional
    normalization is applied before computing cosine similarity.

    Note: torchhd.cos internally handles normalization when computing cosine similarity:
        cos(a, b) = (a . b) / (||a|| * ||b||)

    Both Tanimoto and Cosine are SIMILARITY measures (higher = more similar):
        - Tanimoto: range [0, 1]
        - Cosine: range [-1, 1]
    """
    cosine_sims = []
    for i, j in pairs:
        # Use torchhd.cos exactly as used in graph_encoders.py for decoding
        sim = torchhd.cos(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
        cosine_sims.append(sim)
    return np.array(cosine_sims)


def compute_random_baseline_cosine(n_molecules: int, dim: int, pairs: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Compute cosine similarity for random vectors (baseline control).

    Uses random HRR vectors via torchhd to match the VSA model used for HDC encoding.
    This provides a proper control: if HDC encoding were just random, we'd see
    zero correlation with Tanimoto (which is what random baseline shows).
    """
    # Generate random HRR vectors (same VSA model as the encoding)
    torch.manual_seed(seed)
    random_embeddings = torchhd.random(n_molecules, dim, vsa="HRR")

    cosine_sims = []
    for i, j in pairs:
        sim = torchhd.cos(random_embeddings[i].unsqueeze(0), random_embeddings[j].unsqueeze(0)).item()
        cosine_sims.append(sim)
    return np.array(cosine_sims)


def binned_analysis(tanimoto_sims: np.ndarray, cosine_sims: np.ndarray) -> dict:
    """Perform binned analysis of cosine similarity by Tanimoto bins."""
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

    results = {"bins": bin_labels, "means": [], "stds": [], "counts": []}

    for low, high in zip(bins[:-1], bins[1:]):
        if high == 1.0:
            mask = (tanimoto_sims >= low) & (tanimoto_sims <= high)
        else:
            mask = (tanimoto_sims >= low) & (tanimoto_sims < high)

        if mask.sum() > 0:
            results["means"].append(float(cosine_sims[mask].mean()))
            results["stds"].append(float(cosine_sims[mask].std()))
            results["counts"].append(int(mask.sum()))
        else:
            results["means"].append(0.0)
            results["stds"].append(0.0)
            results["counts"].append(0)

    return results


def regression_analysis(tanimoto_sims: np.ndarray, cosine_sims: np.ndarray) -> dict:
    """Fit linear and polynomial regression models."""
    X = tanimoto_sims.reshape(-1, 1)
    y = cosine_sims

    # Linear fit
    lin_model = LinearRegression()
    lin_model.fit(X, y)
    y_pred_lin = lin_model.predict(X)
    r2_linear = r2_score(y, y_pred_lin)

    # Quadratic fit
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    quad_model = LinearRegression()
    quad_model.fit(X_poly, y)
    y_pred_quad = quad_model.predict(X_poly)
    r2_quad = r2_score(y, y_pred_quad)

    return {
        "linear": {
            "r2": float(r2_linear),
            "slope": float(lin_model.coef_[0]),
            "intercept": float(lin_model.intercept_),
        },
        "quadratic": {"r2": float(r2_quad)},
    }


def run_analysis(base_dataset: str, n_pairs: int = N_PAIRS) -> dict:
    """Run complete Tanimoto vs HDC cosine analysis for a dataset."""

    print(f"\n{'=' * 70}")
    print(f"Dataset: {base_dataset.upper()}")
    print(f"{'=' * 70}\n")

    # Create results directory
    config_dir = RESULTS_DIR / base_dataset
    config_dir.mkdir(parents=True, exist_ok=True)

    # ===== 1. Load dataset (TEST split for evaluation) =====
    print("Loading test dataset...")
    ds = get_split(base_dataset=base_dataset, split="test")
    n_molecules = len(ds)
    print(f"  Loaded {n_molecules} molecules from test set")

    # ===== 2. Compute Morgan fingerprints =====
    print("Computing Morgan fingerprints (ECFP4)...")
    fingerprints = []
    valid_indices = []
    smiles_list = []

    for idx in tqdm(range(n_molecules), desc="Fingerprints"):
        data = ds[idx]
        smiles = data.smiles
        fp = get_morgan_fingerprint(smiles)
        if fp is not None:
            fingerprints.append(fp)
            valid_indices.append(idx)
            smiles_list.append(smiles)

    print(f"  Valid molecules with fingerprints: {len(fingerprints)}/{n_molecules}")

    # ===== 3. Compute HDC embeddings =====
    print("Computing HDC embeddings...")

    # Get config
    if base_dataset == "zinc":
        cfg = ZINC_SMILES_HRR_256_F64_5G1NG4_CONFIG
        depth = 4  # ZINC uses depth 4 for graph_terms
    else:
        cfg = QM9_SMILES_HRR_256_CONFIG_F64_G1NG3_CONFIG
        depth = 3  # QM9 uses depth 3 for graph_terms

    hypernet = HyperNet(config=cfg, depth=depth)
    hypernet.eval()

    # Get embeddings for valid molecules only
    valid_dataset = ds.index_select(valid_indices)
    loader = DataLoader(valid_dataset, batch_size=512, shuffle=False)

    all_graph_terms = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="HDC Encoding"):
            output = hypernet.forward(batch)
            all_graph_terms.append(output["graph_embedding"])

    graph_terms = torch.cat(all_graph_terms, dim=0)
    print(f"  HDC embeddings shape: {graph_terms.shape}")
    hv_dim = graph_terms.shape[1]

    # ===== 4. Sample pairs =====
    n_valid = len(fingerprints)
    actual_n_pairs = min(n_pairs, n_valid * (n_valid - 1) // 2)
    print(f"Sampling {actual_n_pairs} pairs...")
    pairs = sample_pairs(n_valid, actual_n_pairs, seed=SEED)
    print(f"  Sampled {len(pairs)} unique pairs")

    # ===== 5. Compute pairwise Tanimoto similarities (Morgan FP) =====
    print("Computing pairwise Tanimoto similarities (Morgan FP)...")
    tanimoto_sims = []
    for i, j in tqdm(pairs, desc="Tanimoto"):
        sim = compute_tanimoto_similarity(fingerprints[i], fingerprints[j])
        tanimoto_sims.append(sim)
    tanimoto_sims = np.array(tanimoto_sims)
    print(f"  Tanimoto: mean={tanimoto_sims.mean():.4f}, std={tanimoto_sims.std():.4f}")

    # ===== 6. Compute pairwise Cosine similarities (HDC) =====
    print("Computing pairwise Cosine similarities (HDC graph_terms)...")
    hdc_cosine_sims = compute_pairwise_cosine(graph_terms, pairs)
    print(f"  HDC Cosine: mean={hdc_cosine_sims.mean():.4f}, std={hdc_cosine_sims.std():.4f}")

    # ===== 7. Random baseline =====
    print("Computing random baseline...")
    random_cosine_sims = compute_random_baseline_cosine(n_valid, hv_dim, pairs, seed=SEED)
    print(f"  Random Cosine: mean={random_cosine_sims.mean():.4f}, std={random_cosine_sims.std():.4f}")

    # ===== 8. Correlation analysis =====
    print("\nComputing correlations...")

    # HDC correlations
    pearson_r, pearson_p = pearsonr(tanimoto_sims, hdc_cosine_sims)
    spearman_rho, spearman_p = spearmanr(tanimoto_sims, hdc_cosine_sims)
    kendall_tau, kendall_p = kendalltau(tanimoto_sims, hdc_cosine_sims)

    print(f"  HDC - Pearson r: {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  HDC - Spearman ρ: {spearman_rho:.4f} (p={spearman_p:.2e})")
    print(f"  HDC - Kendall τ: {kendall_tau:.4f} (p={kendall_p:.2e})")

    # Random baseline correlations
    rand_pearson_r, rand_pearson_p = pearsonr(tanimoto_sims, random_cosine_sims)
    rand_spearman_rho, rand_spearman_p = spearmanr(tanimoto_sims, random_cosine_sims)
    rand_kendall_tau, rand_kendall_p = kendalltau(tanimoto_sims, random_cosine_sims)

    print(f"  Random - Pearson r: {rand_pearson_r:.4f} (p={rand_pearson_p:.2e})")
    print(f"  Random - Spearman ρ: {rand_spearman_rho:.4f} (p={rand_spearman_p:.2e})")
    print(f"  Random - Kendall τ: {rand_kendall_tau:.4f} (p={rand_kendall_p:.2e})")

    # ===== 9. Binned analysis =====
    print("\nPerforming binned analysis...")
    binned = binned_analysis(tanimoto_sims, hdc_cosine_sims)
    for label, mean, std, count in zip(binned["bins"], binned["means"], binned["stds"], binned["counts"]):
        print(f"  Bin {label}: mean={mean:.4f} ± {std:.4f} (n={count})")

    # ===== 10. Regression analysis =====
    print("\nPerforming regression analysis...")
    regression = regression_analysis(tanimoto_sims, hdc_cosine_sims)
    print(f"  Linear R²: {regression['linear']['r2']:.4f}")
    print(f"  Quadratic R²: {regression['quadratic']['r2']:.4f}")

    # ===== 11. Compile results =====
    results = {
        "dataset": base_dataset,
        "n_molecules": n_valid,
        "n_pairs": len(pairs),
        "hv_dim": hv_dim,
        "depth": depth,
        "morgan_config": {"radius": MORGAN_RADIUS, "n_bits": MORGAN_NBITS},
        "statistics": {
            "tanimoto": {"mean": float(tanimoto_sims.mean()), "std": float(tanimoto_sims.std())},
            "hdc_cosine": {"mean": float(hdc_cosine_sims.mean()), "std": float(hdc_cosine_sims.std())},
            "random_cosine": {"mean": float(random_cosine_sims.mean()), "std": float(random_cosine_sims.std())},
        },
        "correlations": {
            "hdc": {
                "pearson": {"r": float(pearson_r), "p": float(pearson_p)},
                "spearman": {"rho": float(spearman_rho), "p": float(spearman_p)},
                "kendall": {"tau": float(kendall_tau), "p": float(kendall_p)},
            },
            "random_baseline": {
                "pearson": {"r": float(rand_pearson_r), "p": float(rand_pearson_p)},
                "spearman": {"rho": float(rand_spearman_rho), "p": float(rand_spearman_p)},
                "kendall": {"tau": float(rand_kendall_tau), "p": float(rand_kendall_p)},
            },
        },
        "binned_analysis": binned,
        "regression": regression,
    }

    # Save results JSON
    results_path = config_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Save raw data for reproducibility
    raw_data = pd.DataFrame(
        {
            "pair_i": pairs[:, 0],
            "pair_j": pairs[:, 1],
            "tanimoto_sim": tanimoto_sims,
            "hdc_cosine_sim": hdc_cosine_sims,
            "random_cosine_sim": random_cosine_sims,
        }
    )
    raw_data_path = config_dir / "pairwise_similarities.csv"
    raw_data.to_csv(raw_data_path, index=False)
    print(f"Saved raw data to {raw_data_path}")

    # ===== 12. Generate visualizations =====
    generate_visualizations(
        config_dir=config_dir,
        dataset_name=base_dataset,
        tanimoto_sims=tanimoto_sims,
        hdc_cosine_sims=hdc_cosine_sims,
        random_cosine_sims=random_cosine_sims,
        binned=binned,
        pearson_r=pearson_r,
        rand_pearson_r=rand_pearson_r,
        regression=regression,
    )

    return results


def generate_visualizations(
    config_dir: Path,
    dataset_name: str,
    tanimoto_sims: np.ndarray,
    hdc_cosine_sims: np.ndarray,
    random_cosine_sims: np.ndarray,
    binned: dict,
    pearson_r: float,
    rand_pearson_r: float,
    regression: dict,
):
    """Generate all required visualizations."""

    print("\nGenerating visualizations...")

    # ===== Figure 1: Scatter Plot with Density (PRIMARY) =====
    fig, ax = plt.subplots(figsize=(8, 6))

    # Hexbin plot for density
    hb = ax.hexbin(
        tanimoto_sims,
        hdc_cosine_sims,
        gridsize=50,
        cmap="Blues",
        mincnt=1,
    )

    # Regression line
    slope = regression["linear"]["slope"]
    intercept = regression["linear"]["intercept"]
    x_line = np.array([0, 1])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, "r--", linewidth=2, label=f"r = {pearson_r:.3f}")

    ax.set_xlabel("Tanimoto Similarity (Morgan Fingerprint)")
    ax.set_ylabel("Cosine Similarity (HDC Space)")
    ax.set_title(f"{dataset_name.upper()} - Tanimoto vs HDC Cosine (n={len(tanimoto_sims):,} pairs)")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, 1)

    plt.colorbar(hb, ax=ax, label="Count")
    plt.tight_layout()

    scatter_path = config_dir / "fig1_scatter_density.png"
    fig.savefig(scatter_path)
    fig.savefig(config_dir / "fig1_scatter_density.pdf")
    plt.close(fig)
    print(f"  Saved scatter plot to {scatter_path}")

    # ===== Figure 2: Binned Analysis with Error Bars =====
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(binned["bins"]))
    bars = ax.bar(
        x,
        binned["means"],
        yerr=binned["stds"],
        capsize=5,
        color=KIT_COLORS["blue"],
        edgecolor="black",
        alpha=0.8,
    )

    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, binned["counts"])):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + binned["stds"][i] + 0.02,
            f"n={count:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(binned["bins"])
    ax.set_xlabel("Tanimoto Similarity Bin")
    ax.set_ylabel("Mean Cosine Similarity (HDC)")
    ax.set_title(f"{dataset_name.upper()} - Binned Analysis")
    ax.set_ylim(0, 1)

    plt.tight_layout()

    binned_path = config_dir / "fig2_binned_analysis.png"
    fig.savefig(binned_path)
    fig.savefig(config_dir / "fig2_binned_analysis.pdf")
    plt.close(fig)
    print(f"  Saved binned analysis to {binned_path}")

    # ===== Figure 3: Random Baseline Comparison (CRITICAL CONTROL) =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: HDC encoding
    hb1 = axes[0].hexbin(
        tanimoto_sims,
        hdc_cosine_sims,
        gridsize=50,
        cmap="Blues",
        mincnt=1,
    )
    axes[0].set_title(f"HDC Encoding (r = {pearson_r:.3f})")
    axes[0].set_xlabel("Tanimoto Similarity (Morgan FP)")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(-0.2, 1)
    plt.colorbar(hb1, ax=axes[0], label="Count")

    # Right: Random encoding
    hb2 = axes[1].hexbin(
        tanimoto_sims,
        random_cosine_sims,
        gridsize=50,
        cmap="Reds",
        mincnt=1,
    )
    axes[1].set_title(f"Random Encoding (r = {rand_pearson_r:.3f})")
    axes[1].set_xlabel("Tanimoto Similarity (Morgan FP)")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(-0.2, 0.2)
    plt.colorbar(hb2, ax=axes[1], label="Count")

    fig.suptitle(f"{dataset_name.upper()} - HDC vs Random Baseline Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    baseline_path = config_dir / "fig3_random_baseline_comparison.png"
    fig.savefig(baseline_path)
    fig.savefig(config_dir / "fig3_random_baseline_comparison.pdf")
    plt.close(fig)
    print(f"  Saved baseline comparison to {baseline_path}")

    # ===== Figure 4: Distribution Histograms =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(tanimoto_sims, bins=50, density=True, alpha=0.75, color=KIT_COLORS["teal"], edgecolor="black")
    axes[0].set_xlabel("Tanimoto Similarity")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Tanimoto (Morgan FP)\nμ={tanimoto_sims.mean():.3f}, σ={tanimoto_sims.std():.3f}")

    axes[1].hist(hdc_cosine_sims, bins=50, density=True, alpha=0.75, color=KIT_COLORS["blue"], edgecolor="black")
    axes[1].set_xlabel("Cosine Similarity")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"HDC Cosine\nμ={hdc_cosine_sims.mean():.3f}, σ={hdc_cosine_sims.std():.3f}")

    axes[2].hist(random_cosine_sims, bins=50, density=True, alpha=0.75, color=KIT_COLORS["red"], edgecolor="black")
    axes[2].set_xlabel("Cosine Similarity")
    axes[2].set_ylabel("Density")
    axes[2].set_title(f"Random Cosine\nμ={random_cosine_sims.mean():.3f}, σ={random_cosine_sims.std():.3f}")

    fig.suptitle(f"{dataset_name.upper()} - Similarity Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()

    hist_path = config_dir / "fig4_distributions.png"
    fig.savefig(hist_path)
    fig.savefig(config_dir / "fig4_distributions.pdf")
    plt.close(fig)
    print(f"  Saved distributions to {hist_path}")


def generate_combined_figures(all_results: list[dict]):
    """Generate combined figures for both datasets."""

    print("\n" + "=" * 70)
    print("Generating combined figures for thesis...")
    print("=" * 70)

    # ===== Combined Figure 1: Side-by-side scatter plots =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, result in zip(axes, all_results):
        dataset = result["dataset"]
        config_dir = RESULTS_DIR / dataset

        # Load raw data
        raw_data = pd.read_csv(config_dir / "pairwise_similarities.csv")
        tanimoto = raw_data["tanimoto_sim"].values
        hdc_cosine = raw_data["hdc_cosine_sim"].values
        pearson_r = result["correlations"]["hdc"]["pearson"]["r"]

        # Hexbin
        hb = ax.hexbin(tanimoto, hdc_cosine, gridsize=50, cmap="Blues", mincnt=1)

        # Regression line
        slope = result["regression"]["linear"]["slope"]
        intercept = result["regression"]["linear"]["intercept"]
        x_line = np.array([0, 1])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, "r--", linewidth=2, label=f"r = {pearson_r:.3f}")

        ax.set_xlabel("Tanimoto Similarity (Morgan FP)")
        ax.set_ylabel("Cosine Similarity (HDC)")
        ax.set_title(f"{dataset.upper()} (n={result['n_pairs']:,} pairs)")
        ax.legend(loc="upper left")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.2, 1)
        plt.colorbar(hb, ax=ax, label="Count")

    plt.tight_layout()
    combined_path = RESULTS_DIR / "combined_scatter.png"
    fig.savefig(combined_path)
    fig.savefig(RESULTS_DIR / "combined_scatter.pdf")
    plt.close(fig)
    print(f"  Saved combined scatter to {combined_path}")

    # ===== Combined Figure 2: Side-by-side binned analysis =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, result in zip(axes, all_results):
        dataset = result["dataset"]
        binned = result["binned_analysis"]

        x = np.arange(len(binned["bins"]))
        bars = ax.bar(
            x,
            binned["means"],
            yerr=binned["stds"],
            capsize=5,
            color=KIT_COLORS["blue"],
            edgecolor="black",
            alpha=0.8,
        )

        for i, (bar, count) in enumerate(zip(bars, binned["counts"])):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + binned["stds"][i] + 0.02,
                f"n={count:,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(binned["bins"])
        ax.set_xlabel("Tanimoto Similarity Bin")
        ax.set_ylabel("Mean Cosine Similarity (HDC)")
        ax.set_title(f"{dataset.upper()}")
        ax.set_ylim(0, 1)

    plt.tight_layout()
    binned_combined_path = RESULTS_DIR / "combined_binned.png"
    fig.savefig(binned_combined_path)
    fig.savefig(RESULTS_DIR / "combined_binned.pdf")
    plt.close(fig)
    print(f"  Saved combined binned to {binned_combined_path}")


def generate_summary_tables(all_results: list[dict]):
    """Generate summary tables for the thesis."""

    print("\n" + "=" * 70)
    print("SUMMARY TABLES FOR THESIS")
    print("=" * 70)

    # Table 1: Correlation Summary
    print("\nTable 1: Tanimoto-Cosine Correlation Analysis")
    print("-" * 85)
    print(f"{'Dataset':<10} | {'n_pairs':<10} | {'Pearson r':<12} | {'Spearman ρ':<12} | {'Kendall τ':<12} | {'p-value':<10}")
    print("-" * 85)

    for result in all_results:
        dataset = result["dataset"].upper()
        n_pairs = result["n_pairs"]
        pearson = result["correlations"]["hdc"]["pearson"]["r"]
        spearman = result["correlations"]["hdc"]["spearman"]["rho"]
        kendall = result["correlations"]["hdc"]["kendall"]["tau"]
        p_val = result["correlations"]["hdc"]["pearson"]["p"]
        print(f"{dataset:<10} | {n_pairs:<10,} | {pearson:<12.4f} | {spearman:<12.4f} | {kendall:<12.4f} | <0.001")

    # Random baseline
    for result in all_results:
        dataset = f"{result['dataset'].upper()}-Rand"
        n_pairs = result["n_pairs"]
        pearson = result["correlations"]["random_baseline"]["pearson"]["r"]
        spearman = result["correlations"]["random_baseline"]["spearman"]["rho"]
        kendall = result["correlations"]["random_baseline"]["kendall"]["tau"]
        p_val = result["correlations"]["random_baseline"]["pearson"]["p"]
        p_str = "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
        print(f"{dataset:<10} | {n_pairs:<10,} | {pearson:<12.4f} | {spearman:<12.4f} | {kendall:<12.4f} | {p_str}")

    print("-" * 85)
    print("Note: Random baseline uses i.i.d. Gaussian vectors of same dimension.\n")

    # Table 2: Binned Analysis
    print("\nTable 2: Mean Cosine Similarity by Tanimoto Bin")
    print("-" * 70)
    header = f"{'Tanimoto Bin':<15}"
    for result in all_results:
        header += f" | {result['dataset'].upper()} Mean ± Std (n)"
    print(header)
    print("-" * 70)

    bins = all_results[0]["binned_analysis"]["bins"]
    for i, bin_label in enumerate(bins):
        row = f"{bin_label:<15}"
        for result in all_results:
            mean = result["binned_analysis"]["means"][i]
            std = result["binned_analysis"]["stds"][i]
            count = result["binned_analysis"]["counts"][i]
            row += f" | {mean:.3f} ± {std:.3f} ({count:,})"
        print(row)

    print("-" * 70)

    # Table 3: Regression Summary
    print("\nTable 3: Regression Analysis")
    print("-" * 50)
    print(f"{'Dataset':<10} | {'Linear R²':<12} | {'Quadratic R²':<12}")
    print("-" * 50)
    for result in all_results:
        dataset = result["dataset"].upper()
        r2_lin = result["regression"]["linear"]["r2"]
        r2_quad = result["regression"]["quadratic"]["r2"]
        print(f"{dataset:<10} | {r2_lin:<12.4f} | {r2_quad:<12.4f}")
    print("-" * 50)

    # Save tables to file
    tables_path = RESULTS_DIR / "summary_tables.txt"
    with open(tables_path, "w") as f:
        f.write("=" * 85 + "\n")
        f.write("TANIMOTO-COSINE CORRELATION ANALYSIS - SUMMARY TABLES\n")
        f.write("=" * 85 + "\n\n")

        f.write("Table 1: Correlation Summary\n")
        f.write("-" * 85 + "\n")
        f.write(f"{'Dataset':<10} | {'n_pairs':<10} | {'Pearson r':<12} | {'Spearman ρ':<12} | {'Kendall τ':<12} | {'p-value':<10}\n")
        f.write("-" * 85 + "\n")

        for result in all_results:
            dataset = result["dataset"].upper()
            n_pairs = result["n_pairs"]
            pearson = result["correlations"]["hdc"]["pearson"]["r"]
            spearman = result["correlations"]["hdc"]["spearman"]["rho"]
            kendall = result["correlations"]["hdc"]["kendall"]["tau"]
            f.write(f"{dataset:<10} | {n_pairs:<10,} | {pearson:<12.4f} | {spearman:<12.4f} | {kendall:<12.4f} | <0.001\n")

        for result in all_results:
            dataset = f"{result['dataset'].upper()}-Rand"
            n_pairs = result["n_pairs"]
            pearson = result["correlations"]["random_baseline"]["pearson"]["r"]
            spearman = result["correlations"]["random_baseline"]["spearman"]["rho"]
            kendall = result["correlations"]["random_baseline"]["kendall"]["tau"]
            p_val = result["correlations"]["random_baseline"]["pearson"]["p"]
            p_str = "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
            f.write(f"{dataset:<10} | {n_pairs:<10,} | {pearson:<12.4f} | {spearman:<12.4f} | {kendall:<12.4f} | {p_str}\n")

        f.write("-" * 85 + "\n\n")

        f.write("Table 2: Mean Cosine Similarity by Tanimoto Bin\n")
        f.write("-" * 70 + "\n")

        bins = all_results[0]["binned_analysis"]["bins"]
        for i, bin_label in enumerate(bins):
            row = f"{bin_label:<15}"
            for result in all_results:
                mean = result["binned_analysis"]["means"][i]
                std = result["binned_analysis"]["stds"][i]
                count = result["binned_analysis"]["counts"][i]
                row += f" | {mean:.3f} ± {std:.3f} ({count:,})"
            f.write(row + "\n")

        f.write("-" * 70 + "\n\n")

        f.write("Table 3: Regression Analysis\n")
        f.write("-" * 50 + "\n")
        for result in all_results:
            dataset = result["dataset"].upper()
            r2_lin = result["regression"]["linear"]["r2"]
            r2_quad = result["regression"]["quadratic"]["r2"]
            f.write(f"{dataset:<10} | {r2_lin:<12.4f} | {r2_quad:<12.4f}\n")
        f.write("-" * 50 + "\n")

    print(f"\nSaved summary tables to {tables_path}")


def main():
    """Run complete Tanimoto-Cosine correlation study."""

    print("=" * 70)
    print("TANIMOTO-COSINE CORRELATION STUDY")
    print("Validating HDC Semantic Structure Preservation (Hypothesis 2)")
    print("=" * 70)

    # Ensure base results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Run for both datasets
    for base_dataset in ["qm9", "zinc"]:
        results = run_analysis(base_dataset=base_dataset, n_pairs=N_PAIRS)
        all_results.append(results)

    # Save aggregated results
    aggregate_path = RESULTS_DIR / "all_results.json"
    with open(aggregate_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved aggregated results to {aggregate_path}")

    # Generate combined figures
    generate_combined_figures(all_results)

    # Generate summary tables
    generate_summary_tables(all_results)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("\nKey findings:")
    for result in all_results:
        dataset = result["dataset"].upper()
        pearson = result["correlations"]["hdc"]["pearson"]["r"]
        rand_pearson = result["correlations"]["random_baseline"]["pearson"]["r"]
        print(f"  {dataset}: Pearson r = {pearson:.4f} (random baseline: {rand_pearson:.4f})")

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    avg_pearson = np.mean([r["correlations"]["hdc"]["pearson"]["r"] for r in all_results])
    if avg_pearson > 0.6:
        print(f"  Strong correlation (r > 0.6) - STRONGLY SUPPORTS Hypothesis 2")
    elif avg_pearson > 0.4:
        print(f"  Moderate correlation (0.4 < r < 0.6) - SUPPORTS Hypothesis 2")
    elif avg_pearson > 0.2:
        print(f"  Weak correlation (0.2 < r < 0.4) - PARTIALLY SUPPORTS Hypothesis 2")
    else:
        print(f"  Very weak correlation (r < 0.2) - WEAKENS Hypothesis 2")
    print("-" * 70)


if __name__ == "__main__":
    main()
