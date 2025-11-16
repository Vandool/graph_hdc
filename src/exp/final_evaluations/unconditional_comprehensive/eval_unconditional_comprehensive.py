"""
Comprehensive Evaluation: Unconditional Molecular Generation
=============================================================

This script implements the complete benchmarking protocol for unconditional generation
as specified in the Molecular Graph Generation Benchmarking Protocols document.

Default: Quick evaluation with 10 samples and 1 seed for fast testing
MOSES benchmark: Use --n_samples 30000 --n_seeds 3 for full evaluation

Comprehensive metrics:
- Core Distributional Metrics: Validity, Uniqueness, Novelty, NUV
- Diversity Metrics: IntDiv1 (p=1), IntDiv2 (p=2)
- Correction Level Distribution: Analysis of HDC decoding correction levels
- Distribution Similarity: KL divergence, Wasserstein distance
- Property Distribution Analysis: LogP, QED, SA Score, Max Ring Size, MW, TPSA
- Synthesizability Assessment: SAScore, Ring Penalties
- Comprehensive statistical reporting with plots

Usage:
    # Quick test (default: 10 samples, 1 seed)
    python eval_unconditional_comprehensive.py \
        --dataset QM9_SMILES_HRR_1600_F64_G1NG3

    # Comparisons
    pixi run -e local python eval_unconditional_comprehensive.py \
        --dataset ZINC_SMILES_HRR_256_F64_5G1NG4 --n_samples 100 --no-corrections --moses
    pixi run -e local python eval_unconditional_comprehensive.py \
        --dataset ZINC_SMILES_HRR_256_F64_5G1NG4 --n_samples 100 --moses
    pixi run -e local python eval_unconditional_comprehensive.py \
        --dataset ZINC_SMILES_HRR_256_F64_5G1NG4 --n_samples 100 --no-corrections --moses --use-ring-structure
    pixi run -e local python eval_unconditional_comprehensive.py \
        --dataset ZINC_SMILES_HRR_256_F64_5G1NG4 --n_samples 100 --moses --use-ring-structure


    # Full MOSES benchmark
    python eval_unconditional_comprehensive.py \
        --dataset QM9_SMILES_HRR_1600_F64_G1NG3 \
        --n_samples 30000 \
        --n_seeds 3

Author: Based on benchmarking protocols document
"""

import argparse
import json
import os
import random
import time
import traceback
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from fcd import get_fcd
from pytorch_lightning import seed_everything
from rdkit import Chem
from rdkit.Chem import BRICS, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from scipy.stats import gaussian_kde
from tqdm.auto import tqdm

from src.datasets.utils import get_dataset_props
from src.encoding.configs_and_constants import BaseDataset, DecoderSettings, SupportedDataset
from src.encoding.graph_encoders import CorrectionLevel
from src.exp.final_evaluations.models_configs_constants import GENERATOR_REGISTRY
from src.generation.evaluator import (
    GenerationEvaluator,
    rdkit_logp,
    rdkit_max_ring_size,
    rdkit_qed,
    rdkit_sa_score,
)
from src.generation.generation import HDCGenerator
from src.utils.chem import draw_mol
from src.utils.utils import pick_device

# --- Matplotlib/Seaborn styling ---
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "font.size": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

# --- Thread configuration ---
num = max(1, min(8, os.cpu_count() or 1))
torch.set_num_threads(num)
torch.set_num_interop_threads(max(1, min(2, num)))
os.environ.setdefault("OMP_NUM_THREADS", str(num))
os.environ.setdefault("MKL_NUM_THREADS", str(num))

DTYPE = torch.float32
torch.set_default_dtype(DTYPE)
device = pick_device()

# Default sample size (set to 10 for quick testing, use 30000 for MOSES benchmark)
DEFAULT_SAMPLE_SIZE = 10

EVALUATOR: dict[BaseDataset, GenerationEvaluator | None] = {"qm9": None, "zinc": None}

# MOSES 9 properties for KL divergence calculation
MOSES_PROPERTIES = [
    "BertzCT",
    "MolLogP",
    "MolWt",
    "TPSA",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "NumAliphaticRings",
    "NumAromaticRings",
]


def calculate_molecular_properties(mol, data=None) -> dict[str, float]:
    """
    Calculate comprehensive molecular properties for evaluation.

    Args:
        mol: RDKit molecule object

    Returns:
        Dictionary of property name -> value
    """
    if mol is None:
        return None

    try:
        props = {
            "mw": Descriptors.MolWt(mol),
            "tpsa": Descriptors.TPSA(mol),
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
            "num_rings": Chem.Descriptors.RingCount(mol),
            "num_aromatic_rings": Chem.Descriptors.NumAromaticRings(mol),
            "num_hba": Chem.Descriptors.NumHAcceptors(mol),
            "num_hbd": Chem.Descriptors.NumHDonors(mol),
        }
        if data is None:
            props.update(
                {
                    "logp": rdkit_logp(mol),
                    "qed": rdkit_qed(mol),
                    "sa_score": rdkit_sa_score(mol),
                    "max_ring_size": rdkit_max_ring_size(mol),
                }
            )
        return props
    except Exception as e:
        print(f"Error calculating properties: {e}")
        return None


def calculate_moses_properties(mol) -> dict[str, float]:
    """
    Calculate the 9 MOSES properties for a molecule.

    Args:
        mol: RDKit molecule object

    Returns:
        Dictionary of MOSES property name -> value
    """
    if mol is None:
        return None

    try:
        from rdkit.Chem import Descriptors

        props = {
            "BertzCT": Descriptors.BertzCT(mol),
            "MolLogP": Descriptors.MolLogP(mol),
            "MolWt": Descriptors.MolWt(mol),
            "TPSA": Descriptors.TPSA(mol),
            "NumHAcceptors": Descriptors.NumHAcceptors(mol),
            "NumHDonors": Descriptors.NumHDonors(mol),
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            "NumAliphaticRings": Descriptors.NumAliphaticRings(mol),
            "NumAromaticRings": Descriptors.NumAromaticRings(mol),
        }
        return props
    except Exception as e:
        print(f"Error calculating MOSES properties: {e}")
        return None


def calculate_moses_kl_divergence(
    dataset_props: dict[str, list[float]],
    generated_props: dict[str, list[float]],
) -> dict[str, float]:
    """
    Calculate KL divergence for the 9 MOSES properties.

    Uses the MOSES paper formula:
    - Create histograms with 50 bins
    - Compute KL(generated || dataset)
    - Average across 9 properties

    Args:
        dataset_props: MOSES properties from training dataset
        generated_props: MOSES properties from generated molecules

    Returns:
        Dictionary with individual KL divergences and average
    """
    kl_divergences = {}
    kl_values = []

    for prop in MOSES_PROPERTIES:
        if prop not in dataset_props or prop not in generated_props:
            continue

        ds_vals = np.array(dataset_props[prop])
        gen_vals = np.array(generated_props[prop])

        # Create common bins
        min_val = min(ds_vals.min(), gen_vals.min())
        max_val = max(ds_vals.max(), gen_vals.max())
        bins = np.linspace(min_val, max_val, 50)

        hist_ds, _ = np.histogram(ds_vals, bins=bins, density=True)
        hist_gen, _ = np.histogram(gen_vals, bins=bins, density=True)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        hist_ds = hist_ds + eps
        hist_gen = hist_gen + eps

        # Normalize
        hist_ds = hist_ds / hist_ds.sum()
        hist_gen = hist_gen / hist_gen.sum()

        # KL divergence
        kl_div = stats.entropy(hist_gen, hist_ds)
        kl_divergences[f"moses_kl_{prop}"] = float(kl_div)
        kl_values.append(kl_div)

    # Average KL divergence (MOSES metric)
    if kl_values:
        kl_divergences["moses_kl_avg"] = float(np.mean(kl_values))

        # Final MOSES score: average of exp(-D_KL) for each descriptor
        # Per UNCONDITIONAL.md: S = avg(exp(-D_KL)) across 9 properties
        exp_neg_kl_values = [np.exp(-kl) for kl in kl_values]
        kl_divergences["moses_kl_final_score"] = float(np.mean(exp_neg_kl_values))

    return kl_divergences


def calculate_fcd_metric(
    generated_smiles: list[str],
    dataset_smiles: list[str],
) -> dict[str, float]:
    """
    Calculate Fréchet ChemNet Distance (FCD) metric.

    Args:
        generated_smiles: List of generated SMILES
        dataset_smiles: List of dataset SMILES (sample 10k)

    Returns:
        Dictionary with FCD score and metric
    """

    try:
        # Sample 10k from dataset for consistency
        dataset_smiles_sample = random.sample(dataset_smiles, 10000) if len(dataset_smiles) > 10000 else dataset_smiles

        # Calculate FCD
        fcd_score = get_fcd(generated_smiles, dataset_smiles_sample)

        # MOSES metric: S = exp(-0.2 * FCD)
        fcd_metric = np.exp(-0.2 * fcd_score)

        return {
            "fcd_score": float(fcd_score),
            "fcd_metric": float(fcd_metric),
        }
    except Exception as e:
        print(f"Error calculating FCD: {e}")
        return {"fcd_score": None, "fcd_metric": None}


def calculate_fragment_metrics(generated_mols: list, dataset_mols: list) -> dict[str, float]:
    """
    Calculate BRICS fragment novelty and uniqueness metrics.

    These metrics address mode collapse by measuring diversity at the fragment level.

    Args:
        generated_mols: List of RDKit molecule objects from generated set
        dataset_mols: List of RDKit molecule objects from training dataset

    Returns:
        Dictionary with fragment metrics:
        - fragment_uniqueness: Average number of unique fragments per molecule
        - fragment_novelty: Percentage of fragments not in training set
        - num_unique_fragments: Total number of unique fragments in generated set
    """
    try:
        # Extract fragments from generated molecules
        gen_fragments = set()
        gen_mol_count = 0
        for mol in generated_mols:
            if mol is not None:
                try:
                    fragments = BRICS.BRICSDecompose(mol)
                    gen_fragments.update(fragments)
                    gen_mol_count += 1
                except:
                    pass

        # Extract fragments from dataset
        ds_fragments = set()
        for mol in dataset_mols:
            if mol is not None:
                try:
                    fragments = BRICS.BRICSDecompose(mol)
                    ds_fragments.update(fragments)
                except:
                    pass

        # Calculate metrics
        fragment_uniqueness = len(gen_fragments) / gen_mol_count if gen_mol_count > 0 else 0.0
        fragment_novelty = len(gen_fragments - ds_fragments) / len(gen_fragments) if gen_fragments else 0.0

        return {
            "fragment_uniqueness": float(fragment_uniqueness),
            "fragment_novelty": float(fragment_novelty),
            "num_unique_fragments": len(gen_fragments),
        }
    except Exception as e:
        print(f"Error calculating fragment metrics: {e}")
        return {
            "fragment_uniqueness": 0.0,
            "fragment_novelty": 0.0,
            "num_unique_fragments": 0,
        }


def calculate_scaffold_metrics(generated_mols: list, dataset_mols: list) -> dict[str, float]:
    """
    Calculate Murcko scaffold diversity, novelty, and uniqueness metrics.

    These metrics address mode collapse by measuring diversity at the scaffold level.

    Args:
        generated_mols: List of RDKit molecule objects from generated set
        dataset_mols: List of RDKit molecule objects from training dataset

    Returns:
        Dictionary with scaffold metrics:
        - scaffold_diversity: Total number of unique scaffolds in generated set
        - scaffold_uniqueness: Average number of unique scaffolds per molecule
        - scaffold_novelty: Percentage of scaffolds not in training set
    """
    try:
        # Extract scaffolds from generated molecules
        gen_scaffolds = set()
        gen_mol_count = 0
        for mol in generated_mols:
            if mol is not None:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                if scaffold is not None:
                    scaffold_smiles = Chem.MolToSmiles(scaffold)
                    gen_scaffolds.add(scaffold_smiles)
                    gen_mol_count += 1

        # Extract scaffolds from dataset
        ds_scaffolds = set()
        for mol in dataset_mols:
            if mol is not None:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                if scaffold is not None:
                    scaffold_smiles = Chem.MolToSmiles(scaffold)
                    ds_scaffolds.add(scaffold_smiles)

        # Calculate metrics
        scaffold_diversity = len(gen_scaffolds)
        scaffold_uniqueness = len(gen_scaffolds) / gen_mol_count if gen_mol_count > 0 else 0.0
        scaffold_novelty = len(gen_scaffolds - ds_scaffolds) / len(gen_scaffolds) if gen_scaffolds else 0.0

        return {
            "scaffold_diversity": scaffold_diversity,
            "scaffold_uniqueness": float(scaffold_uniqueness),
            "scaffold_novelty": float(scaffold_novelty),
        }
    except Exception as e:
        print(f"Error calculating scaffold metrics: {e}")
        return {
            "scaffold_diversity": 0,
            "scaffold_uniqueness": 0.0,
            "scaffold_novelty": 0.0,
        }


def calculate_distribution_metrics(
    dataset_props: dict[str, list[float]],
    generated_props: dict[str, list[float]],
) -> dict[str, Any]:
    """
    Calculate distribution similarity metrics between dataset and generated molecules.

    Args:
        dataset_props: Properties from training dataset
        generated_props: Properties from generated molecules

    Returns:
        Dictionary of distribution metrics
    """
    metrics = {}

    # Calculate KL divergence and Wasserstein distance for continuous properties
    continuous_props = ["logp", "qed", "sa_score", "mw", "tpsa"]

    for prop in continuous_props:
        if prop not in dataset_props or prop not in generated_props:
            continue

        ds_vals = np.array(dataset_props[prop])
        gen_vals = np.array(generated_props[prop])

        # KL divergence (using histograms)
        try:
            # Create common bins
            min_val = min(ds_vals.min(), gen_vals.min())
            max_val = max(ds_vals.max(), gen_vals.max())
            bins = np.linspace(min_val, max_val, 50)

            hist_ds, _ = np.histogram(ds_vals, bins=bins, density=True)
            hist_gen, _ = np.histogram(gen_vals, bins=bins, density=True)

            # Add small epsilon to avoid log(0)
            eps = 1e-10
            hist_ds = hist_ds + eps
            hist_gen = hist_gen + eps

            # Normalize
            hist_ds = hist_ds / hist_ds.sum()
            hist_gen = hist_gen / hist_gen.sum()

            kl_div = stats.entropy(hist_gen, hist_ds)
            metrics[f"{prop}_kl_divergence"] = float(kl_div)
        except Exception as e:
            print(f"Error computing KL divergence for {prop}: {e}")

        # Wasserstein distance
        try:
            wasserstein = stats.wasserstein_distance(ds_vals, gen_vals)
            metrics[f"{prop}_wasserstein"] = float(wasserstein)
        except Exception as e:
            print(f"Error computing Wasserstein distance for {prop}: {e}")

        # Basic statistics comparison
        metrics[f"{prop}_mean_diff"] = float(gen_vals.mean() - ds_vals.mean())
        metrics[f"{prop}_std_diff"] = float(gen_vals.std() - ds_vals.std())

    # Discrete property comparison (ring sizes, counts)
    discrete_props = ["max_ring_size", "num_rings", "num_aromatic_rings"]

    for prop in discrete_props:
        if prop not in dataset_props or prop not in generated_props:
            continue

        ds_vals = np.array(dataset_props[prop])
        gen_vals = np.array(generated_props[prop])

        # Chi-square test for discrete distributions
        try:
            unique_vals = np.unique(np.concatenate([ds_vals, gen_vals]))
            ds_counts = np.array([np.sum(ds_vals == v) for v in unique_vals])
            gen_counts = np.array([np.sum(gen_vals == v) for v in unique_vals])

            # Normalize to same total count
            gen_counts_norm = gen_counts * (ds_counts.sum() / gen_counts.sum())

            chi2, p_value = stats.chisquare(gen_counts_norm, ds_counts)
            metrics[f"{prop}_chi2"] = float(chi2)
            metrics[f"{prop}_chi2_pvalue"] = float(p_value)
        except Exception as e:
            print(f"Error computing chi-square for {prop}: {e}")

    return metrics


def create_comprehensive_plots(
    dataset_name: str,
    dataset_props: dict[str, list[float]],
    generated_props: dict[str, list[float]],
    save_dir: Path,
):
    """
    Create comprehensive visualization of property distributions.

    Args:
        dataset_name: Name of the dataset
        dataset_props: Properties from training dataset
        generated_props: Properties from generated molecules
        save_dir: Directory to save plots
    """
    # Create subplots for all properties
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    axes = axes.flatten()

    properties = [
        ("logp", "LogP", "continuous"),
        ("qed", "QED", "continuous"),
        ("sa_score", "SA Score", "continuous"),
        ("max_ring_size", "Max Ring Size", "discrete"),
        ("mw", "Molecular Weight", "continuous"),
        ("tpsa", "TPSA", "continuous"),
        ("num_atoms", "Number of Atoms", "discrete"),
        ("num_bonds", "Number of Bonds", "discrete"),
        ("num_rings", "Number of Rings", "discrete"),
        ("num_aromatic_rings", "Aromatic Rings", "discrete"),
        ("num_hba", "H-Bond Acceptors", "discrete"),
        ("num_hbd", "H-Bond Donors", "discrete"),
    ]

    for idx, (prop_key, prop_label, prop_type) in enumerate(properties):
        if idx >= len(axes):
            break

        ax = axes[idx]

        if prop_key not in dataset_props or prop_key not in generated_props:
            ax.set_visible(False)
            continue

        ds_vals = np.array(dataset_props[prop_key])
        gen_vals = np.array(generated_props[prop_key])

        if prop_type == "continuous":
            # KDE plots for continuous variables
            lo = min(ds_vals.min(), gen_vals.min())
            hi = max(ds_vals.max(), gen_vals.max())
            x = np.linspace(lo, hi, 200)

            kde_ds = gaussian_kde(ds_vals)
            kde_gen = gaussian_kde(gen_vals)

            ax.plot(x, kde_ds(x), label="Dataset", linewidth=2, color="blue")
            ax.plot(x, kde_gen(x), label="Generated", linewidth=2, linestyle="--", color="orange")
        else:
            # Bar plots for discrete variables
            unique_vals = np.unique(np.concatenate([ds_vals, gen_vals]))
            ds_counts = np.array([np.sum(ds_vals == v) for v in unique_vals]) / len(ds_vals)
            gen_counts = np.array([np.sum(gen_vals == v) for v in unique_vals]) / len(gen_vals)

            x = np.arange(len(unique_vals))
            width = 0.35

            ax.bar(x - width / 2, ds_counts, width, label="Dataset", color="blue", alpha=0.7)
            ax.bar(x + width / 2, gen_counts, width, label="Generated", color="orange", alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(unique_vals)

        ax.set_xlabel(prop_label)
        ax.set_ylabel("Density" if prop_type == "continuous" else "Frequency")
        ax.set_title(f"{prop_label}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(properties), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"{dataset_name} - Property Distribution Comparison", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / "comprehensive_property_distributions.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    # Create individual high-quality plots for key properties
    key_properties = ["logp", "qed", "sa_score", "max_ring_size"]

    for prop_key in key_properties:
        if prop_key not in dataset_props or prop_key not in generated_props:
            continue

        plt.figure(figsize=(8, 6))

        ds_vals = np.array(dataset_props[prop_key])
        gen_vals = np.array(generated_props[prop_key])

        # Create violin plot
        data_df = pd.DataFrame(
            {
                "Value": np.concatenate([ds_vals, gen_vals]),
                "Source": ["Dataset"] * len(ds_vals) + ["Generated"] * len(gen_vals),
            }
        )

        sns.violinplot(data=data_df, x="Source", y="Value", inner="box")
        plt.title(f"{dataset_name} - {prop_key.upper()} Distribution")
        plt.ylabel(prop_key.upper())
        plt.tight_layout()
        plt.savefig(save_dir / f"violin_{prop_key}.pdf", bbox_inches="tight", dpi=300)
        plt.close()


def draw_molecules_with_metadata(
    mols: list,
    valid_flags: list[bool],
    correction_levels: list,
    similarities: list[float],
    properties_list: list[dict],
    save_dir: Path,
    max_molecules: int = 100,
):
    """
    Draw molecules with filenames encoding metadata.

    Args:
        mols: List of RDKit molecules
        valid_flags: Validity flags for each molecule
        correction_levels: Correction level for each molecule
        similarities: Cosine similarities for each molecule
        properties_list: List of property dictionaries for each molecule
        save_dir: Directory to save molecule images
        max_molecules: Maximum number of molecules to draw (default: 100)
    """
    molecules_dir = save_dir / "molecules"
    molecules_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDrawing up to {max_molecules} molecules to {molecules_dir}...")

    n_drawn = 0
    for idx, (mol, valid, corr_level, sim, props) in enumerate(
        zip(mols, valid_flags, correction_levels, similarities, properties_list, strict=False)
    ):
        if n_drawn >= max_molecules:
            break

        if not valid or mol is None:
            continue

        # Extract key properties (use 0 if missing)
        logp = props.get("logp", 0.0)
        qed = props.get("qed", 0.0)
        sa = props.get("sa_score", 0.0)
        ring = props.get("max_ring_size", 0)

        # Map correction level to short code
        corr_code_map = {
            CorrectionLevel.ZERO: "L0",
            CorrectionLevel.ONE: "L1",
            CorrectionLevel.TWO: "L2",
            CorrectionLevel.THREE: "L3",
            CorrectionLevel.FAIL: "FAIL",
        }
        corr_code = corr_code_map.get(corr_level, "UNK")

        # Create informative filename
        # Format: mol_{idx:04d}_sim{sim:.3f}_logp{logp:.2f}_qed{qed:.2f}_sa{sa:.2f}_ring{ring}_{corr_code}.png
        filename = f"mol_{idx:04d}_sim{sim:.3f}_logp{logp:+.2f}_qed{qed:.2f}_sa{sa:.2f}_ring{int(ring)}_{corr_code}.png"

        save_path = molecules_dir / filename

        try:
            draw_mol(mol, save_path=str(save_path), size=(400, 400), fmt="png")
            n_drawn += 1
        except Exception as e:
            print(f"Warning: Failed to draw molecule {idx}: {e}")
            continue

    print(f"Drew {n_drawn} valid molecules")

    # Create a summary CSV with all metadata
    summary_data = []
    for idx, (mol, valid, corr_level, sim, props) in enumerate(
        zip(mols, valid_flags, correction_levels, similarities, properties_list, strict=False)
    ):
        if not valid or mol is None:
            continue

        smiles = Chem.MolToSmiles(mol) if mol else None

        summary_data.append(
            {
                "index": idx,
                "smiles": smiles,
                "similarity": sim,
                "correction_level": corr_level.value if hasattr(corr_level, "value") else str(corr_level),
                **props,
            }
        )

    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv(save_dir / "molecules_metadata.csv", index=False)
        print(f"Saved metadata for {len(summary_data)} valid molecules")


def plot_correction_level_distribution(
    dataset_name: str,
    correction_levels: list,
    save_dir: Path,
):
    """
    Create bar plot showing distribution of correction levels.

    Args:
        dataset_name: Name of the dataset
        correction_levels: List of CorrectionLevel enum values
        save_dir: Directory to save plot
    """
    if not correction_levels:
        print("No correction levels to plot")
        return

    # Count correction levels
    counts = {
        "Level 0\n(No correction)": sum(1 for cl in correction_levels if cl == CorrectionLevel.ZERO),
        "Level 1\n(Edge add/remove)": sum(1 for cl in correction_levels if cl == CorrectionLevel.ONE),
        "Level 2\n(Re-decode)": sum(1 for cl in correction_levels if cl == CorrectionLevel.TWO),
        "Level 3\n(L2 + corrections)": sum(1 for cl in correction_levels if cl == CorrectionLevel.THREE),
        "FAIL\n(Greedy fallback)": sum(1 for cl in correction_levels if cl == CorrectionLevel.FAIL),
    }

    # Convert to percentages
    total = len(correction_levels)
    percentages = {k: (v / total * 100) for k, v in counts.items()}

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = list(percentages.keys())
    values = list(percentages.values())
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#95a5a6"]  # Green, blue, orange, red, gray

    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

    # Add percentage labels on bars
    for bar, pct in zip(bars, values, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Percentage of Samples (%)", fontsize=12)
    ax.set_title(f"{dataset_name} - Correction Level Distribution\n(n={total} samples)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "correction_level_distribution.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    # Also create a stacked version with counts
    fig, ax = plt.subplots(figsize=(8, 6))

    count_values = list(counts.values())
    count_labels = [f"{label.split(chr(10))[0]}\n({count})" for label, count in zip(labels, count_values, strict=False)]

    bars = ax.bar(count_labels, count_values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

    # Add count labels on bars
    for bar, cnt in zip(bars, count_values, strict=False):
        height = bar.get_height()
        if cnt > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height / 2,
                f"{cnt}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white",
            )

    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title(f"{dataset_name} - Correction Level Counts", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(count_values) * 1.1 if max(count_values) > 0 else 1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "correction_level_counts.pdf", bbox_inches="tight", dpi=300)
    plt.close()


def generate_moses_compliant(
    generator: HDCGenerator,
    evaluator: GenerationEvaluator,
    n_samples: int,
    max_attempts_multiplier: int = 5,
) -> dict[str, Any]:
    """
    Generate molecules following MOSES methodology:
    - Sample until we have N valid unique molecules
    - Track total_samples_drawn for true validity metric

    Args:
        generator: HDC generator instance
        evaluator: Generation evaluator instance
        n_samples: Number of valid unique molecules to collect
        max_attempts_multiplier: Maximum attempts = n_samples * this value

    Returns:
        Dictionary containing:
        - generated_smiles_set: Set of valid unique SMILES
        - generated_mol_list: List of Chem.Mol objects
        - nx_graphs: List of NetworkX graphs
        - correction_levels: List of correction levels
        - similarities: List of cosine similarities
        - total_samples_drawn: Total number of sampling attempts
        - validity_moses: True validity = n_samples / total_samples_drawn
    """
    generated_smiles_set = set()
    generated_mol_list = []
    nx_graphs_list = []
    correction_levels_list = []
    similarities_list = []

    total_samples_drawn = 0
    max_attempts = n_samples * max_attempts_multiplier

    print(f"\n{'=' * 80}")
    print("MOSES-Compliant Generation")
    print(f"Target valid unique molecules: {n_samples}")
    print(f"Maximum attempts: {max_attempts} (multiplier={max_attempts_multiplier})")
    print(f"{'=' * 80}\n")

    pbar = tqdm(total=n_samples, desc="Collecting valid unique molecules")

    while len(generated_smiles_set) < n_samples and total_samples_drawn < max_attempts:
        # Generate a single sample
        samples = generator.generate_most_similar(n_samples=1)
        total_samples_drawn += 1

        # Skip if no results
        if len(samples["graphs"]) == 0:
            continue

        # Extract results
        nx_graph = samples["graphs"][0]
        corr_level = samples["correction_levels"][0]
        sim = samples["similarities"][0]

        # Use evaluator's to_mols_and_validate method to convert and validate
        mols, valid_flags = evaluator.to_mols_and_validate([nx_graph])
        mol = mols[0]
        is_valid = valid_flags[0]

        if is_valid and mol is not None:
            # Get canonical SMILES
            smiles = Chem.MolToSmiles(mol)

            # Check uniqueness
            if smiles not in generated_smiles_set:
                generated_smiles_set.add(smiles)
                generated_mol_list.append(mol)
                nx_graphs_list.append(nx_graph)
                correction_levels_list.append(corr_level)
                similarities_list.append(sim)
                pbar.update(1)

    pbar.close()

    n_collected = len(generated_smiles_set)
    validity_moses = n_collected / total_samples_drawn if total_samples_drawn > 0 else 0.0

    print("\nMOSES Generation Summary:")
    print(f"  Total samples drawn: {total_samples_drawn}")
    print(f"  Valid unique molecules collected: {n_collected}")
    print(f"  Validity (MOSES): {validity_moses:.4f} ({n_collected}/{total_samples_drawn})")

    if n_collected < n_samples:
        print(f"  WARNING: Only collected {n_collected}/{n_samples} molecules after {total_samples_drawn} attempts")

    return {
        "generated_smiles_set": generated_smiles_set,
        "generated_mol_list": generated_mol_list,
        "nx_graphs": nx_graphs_list,
        "correction_levels": correction_levels_list,
        "similarities": similarities_list,
        "total_samples_drawn": total_samples_drawn,
        "validity_moses": validity_moses,
        "n_collected": n_collected,
    }


def evaluate_single_model(
    dataset: SupportedDataset,
    gen_model_hint: str,
    n_samples: int,
    decoder_settings: DecoderSettings,
    seed: int = 42,
    dataset_props=None,
    draw_molecules: bool = False,
    moses_mode: bool = False,
) -> dict[str, Any]:
    """
    Evaluate a single generative model following MOSES protocol.

    Args:
        dataset: Dataset configuration
        gen_model_hint: Model identifier/path hint
        n_samples: Number of molecules to generate (default: 30,000 for MOSES)
        seed: Random seed
        draw_molecules: Whether to draw individual molecules

    Returns:
        Tuple containing:
        - results: Comprehensive evaluation metrics dictionary
        - dataset_props: Dataset property distributions
        - generated_props: Generated molecule property distributions
        - correction_levels: Correction level for each molecule
        - mols: List of RDKit molecules
        - valid_flags: Validity flag for each molecule
        - sims: Cosine similarities for each molecule
        - all_properties: Property dictionary for each molecule
    """
    seed_everything(seed)

    # Initialize generator
    generator = HDCGenerator(
        gen_model_hint=gen_model_hint,
        ds_config=dataset.default_cfg,
        device=device,
        dtype=DTYPE,
    )
    generator.decoder_settings = decoder_settings

    # Generate samples
    print(f"\n{'=' * 80}")
    print(f"Mode: {'MOSES-Compliant' if moses_mode else 'Comprehensive'}")
    print(f"Generating {n_samples} samples for {dataset.value}")
    print(f"Model: {gen_model_hint}")
    print(f"Seed: {seed}")
    print("Decoder Settings:")
    pprint(decoder_settings)
    print(f"{'=' * 80}\n")

    base_dataset = dataset.default_cfg.base_dataset
    # Initialize evaluator before generation
    global EVALUATOR  # noqa: PLW0602
    if EVALUATOR[base_dataset] is None:
        EVALUATOR[base_dataset] = GenerationEvaluator(base_dataset=base_dataset, device=device)
    evaluator = EVALUATOR[base_dataset]

    t0_gen = time.perf_counter()

    if moses_mode:
        # MOSES-compliant generation: sample until N valid unique molecules
        moses_results = generate_moses_compliant(
            generator=generator,
            evaluator=evaluator,
            n_samples=n_samples,
            max_attempts_multiplier=30,
        )
        nx_graphs = moses_results["nx_graphs"]
        correction_levels = moses_results["correction_levels"]
        sims = moses_results["similarities"]
        total_samples_drawn = moses_results["total_samples_drawn"]
        validity_moses = moses_results["validity_moses"]

        # Create dummy values for compatibility
        final_flags = [True] * len(nx_graphs)  # All are valid by construction
        tgt_reached = [True] * len(nx_graphs)  # Not applicable in MOSES mode
    else:
        # Comprehensive mode: generate N samples and evaluate
        samples = generator.generate_most_similar(n_samples=n_samples)
        nx_graphs = samples["graphs"]
        final_flags = samples["final_flags"]
        sims = samples["similarities"]
        tgt_reached = samples["intermediate_target_reached"]
        correction_levels = samples["correction_levels"]
        total_samples_drawn = n_samples  # In comprehensive mode, this equals n_samples
        validity_moses = None  # Not applicable in comprehensive mode

    t_gen = time.perf_counter() - t0_gen

    # Evaluate with standard metrics (evaluator already initialized above)
    evals = evaluator.evaluate(
        n_samples=n_samples, samples=nx_graphs, final_flags=final_flags, sims=sims, correction_levels=correction_levels
    )

    # Get molecules and validity flags
    mols, valid_flags, sims, correction_levels = evaluator.get_mols_valid_flags_sims_and_correction_levels()

    # Calculate comprehensive molecular properties for valid molecules
    print("Calculating comprehensive molecular properties...")
    generated_props = defaultdict(list)
    valid_smiles = []
    all_properties = []  # Store properties for all molecules (for drawing)

    for mol, valid in tqdm(zip(mols, valid_flags, strict=False), total=len(mols), desc="Properties"):
        if valid and mol is not None:
            props = calculate_molecular_properties(mol)
            if props:
                for k, v in props.items():
                    generated_props[k].append(v)
                valid_smiles.append(Chem.MolToSmiles(mol))
                all_properties.append(props)
            else:
                all_properties.append({})
        else:
            all_properties.append({})

    if dataset_props == None:
        # Load dataset properties using cached get_dataset_props
        print("Loading dataset properties from cache...")

        base_dataset = "qm9" if dataset.default_cfg.base_dataset == "qm9" else "zinc"
        dataset_props_obj = get_dataset_props(base_dataset=base_dataset, splits=["train"])

        # Convert DatasetProps object to dict format expected by plotting functions
        dataset_props = {
            "smiles": dataset_props_obj.smiles,
            "logp": dataset_props_obj.logp,
            "qed": dataset_props_obj.qed,
            "sa_score": dataset_props_obj.sa_score,
            "max_ring_size": dataset_props_obj.max_ring_size_data,
            "mw": dataset_props_obj.mw,
            "tpsa": dataset_props_obj.tpsa,
            "num_atoms": dataset_props_obj.num_atoms,
            "num_bonds": dataset_props_obj.num_bonds,
            "num_rings": dataset_props_obj.num_rings,
            "num_rotatable_bonds": dataset_props_obj.num_rotatable_bonds,
            "num_hba": dataset_props_obj.num_hba,
            "num_hbd": dataset_props_obj.num_hbd,
            "num_aliphatic_rings": dataset_props_obj.num_aliphatic_rings,
            "num_aromatic_rings": dataset_props_obj.num_aromatic_rings,
            "max_ring_size_calc": dataset_props_obj.max_ring_size_calc,
            "bertz_ct": dataset_props_obj.bertz_ct,
            "pen_logp": dataset_props_obj.pen_logp,
        }

    # Calculate distribution metrics
    distribution_metrics = calculate_distribution_metrics(dataset_props, generated_props)

    # MOSES-specific metrics
    moses_metrics = {}
    if moses_mode:
        print("Calculating MOSES-specific metrics...")

        # Calculate MOSES properties for generated molecules
        moses_generated_props = defaultdict(list)
        for mol, valid in tqdm(zip(mols, valid_flags, strict=False), total=len(mols), desc="MOSES properties"):
            if valid and mol is not None:
                props = calculate_moses_properties(mol)
                if props:
                    for k, v in props.items():
                        moses_generated_props[k].append(v)

        # Map cached dataset properties to MOSES property names
        if "BertzCT" not in dataset_props:
            print("Mapping cached dataset properties to MOSES format...")
            dataset_props["BertzCT"] = dataset_props["bertz_ct"]
            dataset_props["MolLogP"] = dataset_props["logp"]
            dataset_props["MolWt"] = dataset_props["mw"]
            dataset_props["TPSA"] = dataset_props["tpsa"]
            dataset_props["NumHAcceptors"] = dataset_props["num_hba"]
            dataset_props["NumHDonors"] = dataset_props["num_hbd"]
            dataset_props["NumRotatableBonds"] = dataset_props["num_rotatable_bonds"]
            dataset_props["NumAliphaticRings"] = dataset_props["num_aliphatic_rings"]
            dataset_props["NumAromaticRings"] = dataset_props["num_aromatic_rings"]

        # Calculate KL divergence for MOSES properties
        moses_kl_metrics = calculate_moses_kl_divergence(dataset_props, moses_generated_props)
        moses_metrics.update(moses_kl_metrics)

        # Calculate FCD using cached dataset SMILES
        dataset_smiles = dataset_props["smiles"]
        fcd_metrics = calculate_fcd_metric(list(valid_smiles), dataset_smiles)
        moses_metrics.update(fcd_metrics)

        # Add MOSES validity (different from comprehensive validity)
        moses_metrics["validity_moses"] = validity_moses
        moses_metrics["total_samples_drawn"] = total_samples_drawn
        moses_metrics["uniqueness_moses"] = 1.0  # By construction in MOSES mode

        # Calculate fragment and scaffold metrics (mode collapse pre-emption)
        print("Computing fragment and scaffold metrics...")

        # Convert cached dataset SMILES to molecules
        print("Converting cached dataset SMILES to molecules...")
        dataset_mols = []
        for smiles in tqdm(dataset_props["smiles"], desc="Loading dataset molecules"):
            m = Chem.MolFromSmiles(smiles)
            if m is not None:
                dataset_mols.append(m)

        # Calculate fragment metrics (BRICS)
        fragment_metrics = calculate_fragment_metrics(mols, dataset_mols)
        moses_metrics.update(fragment_metrics)

        # Calculate scaffold metrics (Murcko)
        scaffold_metrics = calculate_scaffold_metrics(mols, dataset_mols)
        moses_metrics.update(scaffold_metrics)

    correction_level_counts = {
        "correction_level_0_pct": 0.0,
        "correction_level_1_pct": 0.0,
        "correction_level_2_pct": 0.0,
        "correction_level_3_pct": 0.0,
        "correction_level_fail_pct": 0.0,
    }
    if correction_levels:
        total = len(correction_levels)
        correction_level_counts["correction_level_0_pct"] = (
            sum(1 for cl in correction_levels if cl == CorrectionLevel.ZERO) / total * 100
        )
        correction_level_counts["correction_level_1_pct"] = (
            sum(1 for cl in correction_levels if cl == CorrectionLevel.ONE) / total * 100
        )
        correction_level_counts["correction_level_2_pct"] = (
            sum(1 for cl in correction_levels if cl == CorrectionLevel.TWO) / total * 100
        )
        correction_level_counts["correction_level_3_pct"] = (
            sum(1 for cl in correction_levels if cl == CorrectionLevel.THREE) / total * 100
        )
        correction_level_counts["correction_level_fail_pct"] = (
            sum(1 for cl in correction_levels if cl == CorrectionLevel.FAIL) / total * 100
        )

    # Compile results
    results = {
        "model": gen_model_hint,
        "dataset": dataset.value,
        "seed": seed,
        "moses_mode": moses_mode,
        "n_samples_requested": n_samples,
        "n_samples_generated": len(nx_graphs),
        "generation_time_total": t_gen,
        "generation_time_per_sample": t_gen / n_samples,
        "final_flags": evals.get("final_flags"),
        **evals.get("cos_sim"),
        # Core MOSES metrics
        "validity": evals.get("validity", 0.0),
        "uniqueness": evals.get("uniqueness", 0.0),
        "novelty": evals.get("novelty", 0.0),
        "nuv": evals.get("nuv", 0.0),
        # Diversity metrics
        "internal_diversity_p1": evals.get("internal_diversity_p1", 0.0),
        "internal_diversity_p2": evals.get("internal_diversity_p2", 0.0),
        # Property statistics (mean ± std)
        "logp_mean": float(np.mean(generated_props.get("logp", [0]))),
        "logp_std": float(np.std(generated_props.get("logp", [0]))),
        "qed_mean": float(np.mean(generated_props.get("qed", [0]))),
        "qed_std": float(np.std(generated_props.get("qed", [0]))),
        "sa_score_mean": float(np.mean(generated_props.get("sa_score", [0]))),
        "sa_score_std": float(np.std(generated_props.get("sa_score", [0]))),
        "max_ring_size_mean": float(np.mean(generated_props.get("max_ring_size", [0]))),
        "max_ring_size_std": float(np.std(generated_props.get("max_ring_size", [0]))),
        "mw_mean": float(np.mean(generated_props.get("mw", [0]))),
        "mw_std": float(np.std(generated_props.get("mw", [0]))),
        # Correction level distribution
        **correction_level_counts,
        # Distribution similarity metrics
        **distribution_metrics,
        # MOSES-specific metrics (if in MOSES mode)
        **moses_metrics,
        # Additional statistics
        "intermediate_target_reached": sum(tgt_reached) / len(tgt_reached) if tgt_reached else 0.0,
        "n_valid_unique": len(set(valid_smiles)),
        "decoder_settings": decoder_settings.to_dict(),
    }

    return results, dataset_props, generated_props, correction_levels, mols, valid_flags, sims, all_properties


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of unconditional molecular generation (MOSES protocol)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=SupportedDataset.ZINC_SMILES_HRR_256_F64_5G1NG4.value,
        choices=[ds.value for ds in SupportedDataset],
        help="Dataset to use for evaluation",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of molecules to generate (default: {DEFAULT_SAMPLE_SIZE}, use 30000 for MOSES benchmark)",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=1,
        help="Number of random seeds for evaluation (default: 1, use 3 for robust statistics)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Specific seeds to use (default: 42, 43, 44)",
    )
    parser.add_argument(
        "--draw",
        action="store_true",
        help="Draw individual molecules",
    )
    parser.add_argument(
        "--no-corrections",
        action="store_true",
        help="Draw individual molecules",
    )
    parser.add_argument(
        "--use-ring-structure",
        action="store_true",
        help="Draw individual molecules",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip creating plots",
    )
    parser.add_argument(
        "--moses",
        action="store_true",
        help="Use MOSES-compliant evaluation (sample until N valid unique molecules)",
    )

    args = parser.parse_args()
    dataset = SupportedDataset(args.dataset)

    # Determine seeds
    seeds = args.seeds or [42 + i for i in range(args.n_seeds)]

    # Get list of generators for this dataset
    generators = GENERATOR_REGISTRY.get(dataset, [])

    if not generators:
        print(f"\n{'=' * 80}")
        print(f"WARNING: No generators configured for {dataset.value}")
        print("Please update GENERATOR_REGISTRY in this script with trained model hints.")
        print(f"{'=' * 80}\n")
        return

    # Setup output directory
    dt = "f32" if torch.float32 == DTYPE else "f64"
    decoder_settings = DecoderSettings.get_default_for(base_dataset=dataset.default_cfg.base_dataset)
    if args.no_corrections:
        decoder_settings.use_correction = False
    if args.use_ring_structure:
        decoder_settings.validate_ring_structure = True

    base_dir = (
        Path(__file__).resolve().parent
        / f"{dataset.value}_{dt}_{args.n_samples}_rs{int(decoder_settings.validate_ring_structure)}_correction{int(decoder_settings.use_correction)}"
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each generator with multiple seeds
    all_results = []

    dataset_props = None
    for gen_idx, gen_hint in enumerate(generators):
        model_results = []

        for seed in seeds:
            try:
                print(f"\nEvaluating {gen_hint} with seed {seed}...")

                results, dataset_props, generated_props, correction_levels, mols, valid_flags, sims, all_properties = (
                    evaluate_single_model(
                        dataset=dataset,
                        gen_model_hint=gen_hint,
                        n_samples=args.n_samples,
                        decoder_settings=decoder_settings,
                        seed=seed,
                        dataset_props=dataset_props,
                        draw_molecules=args.draw and seed == seeds[0],  # Only draw for first seed
                        moses_mode=args.moses,
                    )
                )

                model_results.append(results)

                # Create plots for first seed only
                if not args.no_plot and seed == seeds[0]:
                    model_dir = base_dir / f"{gen_hint}_seed{seed}_idx{gen_idx}_s{seeds}"
                    model_dir.mkdir(parents=True, exist_ok=True)

                    create_comprehensive_plots(dataset.value, dataset_props, generated_props, model_dir)
                    plot_correction_level_distribution(dataset.value, correction_levels, model_dir)

                    # Draw molecules if requested
                    if args.draw:
                        draw_molecules_with_metadata(
                            mols=mols,
                            valid_flags=valid_flags,
                            correction_levels=correction_levels,
                            similarities=sims,
                            properties_list=all_properties,
                            save_dir=model_dir,
                            max_molecules=100,  # Draw up to 100 molecules
                        )

                    # Save raw data
                    np.save(model_dir / "dataset_properties.npy", dict(dataset_props))
                    np.save(model_dir / "generated_properties.npy", dict(generated_props))

            except Exception as e:
                print(f"\nERROR evaluating {gen_hint} with seed {seed}: {e}")
                traceback.print_exc()
                continue

        # Aggregate results across seeds
        if model_results:
            aggregated = {
                "model": gen_hint,
                "dataset": dataset.value,
                "n_seeds": len(model_results),
            }

            # Calculate mean and std for each metric
            metric_keys = [
                k
                for k in model_results[0]
                if k
                not in ["model", "dataset", "seed", "decoder_settings", "n_samples_generated", "n_samples_requested"]
            ]

            for key in metric_keys:
                values = [r[key] for r in model_results if key in r]
                if values:
                    aggregated[f"{key}_mean"] = float(np.mean(values))
                    aggregated[f"{key}_std"] = float(np.std(values))

            all_results.append(aggregated)

            # Save individual model results
            model_results_path = base_dir / f"{gen_hint}_all_seeds.json"
            with open(model_results_path, "w") as f:
                json.dump(model_results, f, indent=2)

    # Save aggregated results
    if all_results:
        # JSON format
        summary_path = base_dir / f"summary_{dataset.value}.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)

        # CSV format for easy analysis
        df = pd.DataFrame(all_results)
        csv_path = base_dir / f"summary_{dataset.value}.csv"
        df.to_csv(csv_path, index=False)

        print(f"\n{'=' * 80}")
        print("Evaluation complete!")
        print(f"Results saved to: {base_dir}")
        print(f"Summary JSON: {summary_path}")
        print(f"Summary CSV: {csv_path}")
        print(f"{'=' * 80}\n")

        # Print summary table
        print("\nSummary Results (mean ± std across seeds):")
        print("=" * 120)

        key_metrics = [
            "validity_mean",
            "uniqueness_mean",
            "novelty_mean",
            "nuv_mean",
            "internal_diversity_p1_mean",
            "internal_diversity_p2_mean",
        ]

        # MOSES-specific metrics (if available)
        moses_metrics_list = [
            "moses_kl_avg_mean",
            "moses_kl_final_score_mean",
            "fcd_score_mean",
            "fcd_metric_mean",
        ]

        for result in all_results:
            print(f"\nModel: {result['model']}")
            for metric in key_metrics:
                if metric in result:
                    metric_base = metric.replace("_mean", "")
                    mean_val = result[metric]
                    std_val = result.get(f"{metric_base}_std", 0.0)
                    print(f"  {metric_base}: {mean_val:.4f} ± {std_val:.4f}")

            # Print MOSES metrics if available
            print("\n  MOSES Metrics (if in MOSES mode):")
            for metric in moses_metrics_list:
                if metric in result:
                    metric_base = metric.replace("_mean", "")
                    mean_val = result[metric]
                    std_val = result.get(f"{metric_base}_std", 0.0)
                    print(f"    {metric_base}: {mean_val:.4f} ± {std_val:.4f}")

            # Highlight the final combined KLD score
            if "moses_kl_final_score_mean" in result:
                print(f"\n  >>> Final MOSES KLD Score: {result['moses_kl_final_score_mean']:.4f} (higher is better)")


if __name__ == "__main__":
    main()
