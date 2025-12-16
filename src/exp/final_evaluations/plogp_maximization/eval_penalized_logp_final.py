#!/usr/bin/env python
"""
Penalized LogP Maximization Final Evaluation with Dual Reporting.
"""

import argparse
import json
import math
import pathlib
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import normflows as nf
import numpy as np
import pandas as pd
import torch
from lightning_fabric import seed_everything
from rdkit import Chem
from torch import nn
from tqdm.auto import tqdm

from src.datasets.utils import get_split
from src.encoding.configs_and_constants import DecoderSettings, SupportedDataset
from src.encoding.graph_encoders import CorrectionLevel
from src.exp.final_evaluations.models_configs_constants import (
    DATASET_STATS,
    GENERATOR_REGISTRY,
    REGRESSOR_REGISTRY,
    get_pr_path,
)
from src.generation.evaluator import (
    GenerationEvaluator,
    rdkit_logp,
    rdkit_max_ring_size,
    rdkit_sa_score,
)
from src.generation.generation import HDCGenerator
from src.utils.chem import draw_mol
from src.utils.registery import retrieve_model
from src.utils.utils import pick_device

# Default float32
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)
DEVICE = pick_device()

seed = 42
seed_everything(seed)

# ===== Literature Baselines =====
LITERATURE_BASELINES = {
    "constrained": {
        "GCPN": {"top1": 7.98, "type": "Legacy (RL-based)"},
        "JT-VAE": {"top1": 5.30, "type": "Standard VAE"},
        "GP-MoLFormer": {"top1": 9.35, "type": "Modern Autoregressive"},
    },
    "unconstrained": {
        "FRATTVAE+MSO": {"top1": 16.84, "type": "Modern VAE"},
        "GP-MoLFormer": {"top1": 19.59, "type": "Modern Autoregressive"},
        "β-CVAE": {
            "top1": 104.29,
            "type": "Outlier",
            "note": "Likely exploits pLogP metric via extreme structures",
        },
    },
}


# ===== Property Calculation =====
def calculate_penalized_logp_rdkit(mol) -> float:
    """Calculate penalized logP using RDKit."""
    if mol is None:
        return -float("inf")
    try:
        logp = rdkit_logp(mol)
        sa = rdkit_sa_score(mol)
        ring_size = rdkit_max_ring_size(mol)
        return float(logp) - float(sa) - max(ring_size - 6, 0)
    except Exception:
        return -float("inf")


def count_heavy_atoms(mol) -> int:
    """Count non-hydrogen atoms in molecule."""
    if mol is None:
        return 0
    try:
        return mol.GetNumHeavyAtoms()
    except Exception:
        return 0


# ===== Scheduler Classes =====
class CosineScheduler:
    def __init__(self, steps: int, lam_hi: float, lam_lo: float):
        self.steps = steps
        self.lam_hi = lam_hi
        self.lam_lo = lam_lo

    def __call__(self, step: int) -> float:
        if step >= self.steps:
            return self.lam_lo
        alpha = 0.5 * (1 + math.cos(math.pi * step / self.steps))
        return self.lam_lo + (self.lam_hi - self.lam_lo) * alpha


class TwoPhaseScheduler:
    def __init__(self, steps: int, lam_hi: float, lam_lo: float, phase1_ratio: float = 0.3):
        self.phase1_steps = int(steps * phase1_ratio)
        self.phase2_steps = steps - self.phase1_steps
        self.lam_hi = lam_hi
        self.lam_lo = lam_lo

    def __call__(self, step: int) -> float:
        if step < self.phase1_steps:
            return self.lam_hi
        phase2_step = step - self.phase1_steps
        if phase2_step >= self.phase2_steps:
            return self.lam_lo
        alpha = 0.5 * (1 + math.cos(math.pi * phase2_step / self.phase2_steps))
        return self.lam_lo + (self.lam_hi - self.lam_lo) * alpha


SCHEDULER_REGISTRY = {"cosine": CosineScheduler, "two-phase": TwoPhaseScheduler}


# ===== Data Classes =====
@dataclass
class PenalizedLogPConfig:
    """Configuration for pLogP maximization experiment."""

    dataset: SupportedDataset
    n_samples: int
    gen_model_idx: int
    max_heavy_atoms: int = 38
    device: str = "cuda"


@dataclass
class PenalizedLogPFinalResults:
    """Results from pLogP maximization with dual reporting."""

    # Standard metrics
    validity: float
    uniqueness: float
    novelty: float
    n_valid: int

    # Unconstrained results (no size limit)
    unconstrained_top1: float
    unconstrained_top10_mean: float
    unconstrained_top100_mean: float
    unconstrained_n_molecules: int
    unconstrained_heavy_atom_mean: float
    unconstrained_heavy_atom_std: float
    unconstrained_heavy_atom_max: int

    # Constrained results (≤38 heavy atoms)
    constrained_top1: float
    constrained_top10_mean: float
    constrained_top100_mean: float
    constrained_n_molecules: int
    constrained_pass_rate: float

    # Property distributions
    plogp_mean: float
    plogp_std: float
    plogp_min: float
    plogp_max: float
    logp_mean: float
    logp_std: float
    sa_score_mean: float
    sa_score_std: float
    ring_size_mean: float
    ring_size_std: float

    # Metadata
    correction_levels: dict[str, float]
    cos_sim_mean: float
    cos_sim_std: float
    optimization_time: float
    decoding_time: float
    total_time: float


# ===== Utility Functions =====
def filter_by_heavy_atom_count(rdkit_mols: list, plogp_list: list[float], max_atoms: int = 38) -> tuple:
    """
    Filter molecules by heavy atom count using existing RDKit objects.

    Returns:
        (filtered_molecules, filtered_plogp, heavy_atom_counts, pass_indices)
    """
    filtered_molecules = []
    filtered_plogp = []
    heavy_atom_counts = []
    pass_indices = []

    for i, (mol, plogp) in enumerate(zip(rdkit_mols, plogp_list, strict=False)):
        if mol is None:
            continue
        try:
            n_heavy = count_heavy_atoms(mol)
            heavy_atom_counts.append(n_heavy)
            if n_heavy <= max_atoms:
                filtered_molecules.append(mol)
                filtered_plogp.append(plogp)
                pass_indices.append(i)
        except Exception:
            continue

    return filtered_molecules, filtered_plogp, heavy_atom_counts, pass_indices


def calculate_top_k_metrics(plogp_values: list[float], k_values: list[int] = None) -> dict:
    """Calculate top-k statistics."""
    if k_values is None:
        k_values = [1, 10, 100]

    if not plogp_values:
        return {f"top{k}": 0.0 for k in k_values}

    sorted_plogp = sorted(plogp_values, reverse=True)
    metrics = {}

    for k in k_values:
        if k == 1:
            metrics["top1"] = sorted_plogp[0] if sorted_plogp else 0.0
        else:
            n_actual = min(k, len(sorted_plogp))
            metrics[f"top{k}_mean"] = np.mean(sorted_plogp[:n_actual]) if n_actual > 0 else 0.0

    return metrics


def load_best_trial_from_csv(csv_path: pathlib.Path) -> dict:
    """Load best trial configuration from HPO CSV."""
    if not csv_path.exists():
        raise ValueError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["state"] == "COMPLETE"]
    if df.empty:
        raise ValueError(f"No completed trials found in {csv_path}")

    best_idx = df["value"].idxmax()
    best_trial = df.loc[best_idx]

    return {
        "lr": best_trial["params_lr"],
        "steps": int(best_trial["params_steps"]),
        "scheduler": best_trial["params_scheduler"],
        "lambda_lo": best_trial["params_lambda_lo"],
        "lambda_hi": best_trial["params_lambda_hi"],
        "lambda_diversity": best_trial.get("params_lambda_diversity", 0.1),
        "optimizer": best_trial.get("params_optimizer", "adam"),
        "grad_clip": best_trial["params_grad_clip"],
        "trial_number": int(best_idx),
        "objective_value": best_trial["value"],
        "plogp_mean": best_trial.get("plogp_mean", 0),
        "plogp_max": best_trial.get("plogp_max", 0),
    }


# ===== Penalized LogP Maximization Implementation =====
class PenalizedLogPOptimizer:
    """Handles gradient-based pLogP optimization in latent space."""

    def __init__(
        self,
        generator: HDCGenerator,
        regressors: dict[str, nn.Module],
        config: PenalizedLogPConfig,
    ):
        self.generator = generator
        self.gen_model = generator.gen_model
        self.logp_regressor = regressors["logp"]
        self.sa_regressor = regressors["sa_score"]
        self.ring_regressor = regressors["max_ring_size"]
        self.hypernet = generator.hypernet
        self.config = config

        self.gen_model.to(DEVICE).eval()
        self.logp_regressor.to(DEVICE).eval()
        self.sa_regressor.to(DEVICE).eval()
        self.ring_regressor.to(DEVICE).eval()
        self.hypernet.to(DEVICE).eval()

        base_dataset = config.dataset.default_cfg.base_dataset
        self.dataset_stats = DATASET_STATS[base_dataset]
        self.decoder_settings = DecoderSettings.get_default_for(base_dataset)
        self.evaluator = GenerationEvaluator(base_dataset=base_dataset, device=DEVICE)

    def optimize_latent(
        self,
        lr: float,
        steps: int,
        scheduler_name: str,
        lambda_lo: float,
        lambda_hi: float,
        lambda_diversity: float,
        optimizer_name: str,
        grad_clip: float,
    ) -> dict[str, Any]:
        """Optimize latent codes to maximize penalized LogP."""
        n_samples = self.config.n_samples

        base = nf.distributions.DiagGaussian(self.gen_model.flat_dim, trainable=False).to(DEVICE)
        z = base.sample(n_samples).detach().requires_grad_(True)

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam([z], lr=lr)
        else:
            optimizer = torch.optim.SGD([z], lr=lr, momentum=0.9)

        scheduler = SCHEDULER_REGISTRY[scheduler_name](steps=steps, lam_hi=lambda_hi, lam_lo=lambda_lo)

        best_loss = float("inf")
        best_z = z.clone()
        losses = []

        pbar = tqdm(range(steps), desc="pLogP Maximization", unit="step")
        for s in pbar:
            hdc = self.gen_model.decode_from_latent(z)
            e, g = self.gen_model.split(hdc)

            # Predict properties
            logp_pred = self.logp_regressor.gen_forward(hdc)
            sa_pred = self.sa_regressor.gen_forward(hdc)
            ring_pred = self.ring_regressor.gen_forward(hdc)

            # Calculate penalized logP
            ring_penalty = torch.relu(ring_pred - 6.0)
            plogp_pred = logp_pred - sa_pred - ring_penalty

            # Maximize pLogP (minimize negative pLogP)
            lam = scheduler(s)
            plogp_loss = -plogp_pred.mean()
            prior_loss = z.pow(2).mean()

            # Diversity constraint (prevent mode collapse)
            if lambda_diversity > 0:
                g_norm = torch.nn.functional.normalize(g, dim=1)
                sim_matrix = torch.mm(g_norm, g_norm.t())
                diversity_loss = sim_matrix.triu(diagonal=1).mean()
            else:
                diversity_loss = 0.0

            loss = plogp_loss + lam * prior_loss + lambda_diversity * diversity_loss

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_z = z.clone()

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([z], grad_clip)
            optimizer.step()

            mean_plogp = plogp_pred.mean().item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "pLogP": f"{mean_plogp:.4f}", "λ": f"{lam:.2e}"})

        # Use best z for final generation
        z = best_z

        with torch.no_grad():
            x = self.gen_model.decode_from_latent(z)
            edge_terms, graph_terms = self.gen_model.split(x)

        # Decode ALL molecules
        print(f"\nOptimization complete. Decoding {len(edge_terms)} samples...")
        decoded = self.generator.decode(edge_terms=edge_terms, graph_terms=graph_terms)

        return {
            "molecules": decoded["graphs"],
            "similarities": decoded["similarities"],
            "correction_levels": decoded["correction_levels"],
            "final_flags": decoded["final_flags"],
            "optimization_losses": losses,
        }

    def evaluate_and_get_results(
        self,
        molecules: list,
        similarities: list,
        correction_levels: list,
        final_flags: list,
    ) -> tuple[PenalizedLogPFinalResults, list[float], list, list[float], list[float], list[float]]:
        """
        Perform final evaluation with dual reporting.

        Returns:
            (PenalizedLogPFinalResults, plogp_list, valid_rdkit_mols, logp_list, sa_list, ring_list)
        """
        if not molecules:
            return self._empty_results(), [], [], [], [], []

        # Evaluate - evaluator handles reconstruction and validation internally
        eval_results = self.evaluator.evaluate(
            n_samples=len(molecules),
            samples=molecules,
            final_flags=final_flags,
            sims=similarities,
            correction_levels=correction_levels,
        )

        # Get reconstructed molecules directly from evaluator to avoid re-conversion
        mols, valid_flags, _, valid_correction_levels = self.evaluator.get_mols_valid_flags_sims_and_correction_levels()

        # Filter for valid RDKit molecules only
        valid_rdkit_mols = [m for m, f in zip(mols, valid_flags, strict=False) if f]

        # Calculate properties using existing RDKit objects
        plogp_list = [calculate_penalized_logp_rdkit(m) for m in valid_rdkit_mols]
        logp_list = [rdkit_logp(m) for m in valid_rdkit_mols]
        sa_list = [rdkit_sa_score(m) for m in valid_rdkit_mols]
        ring_list = [rdkit_max_ring_size(m) for m in valid_rdkit_mols]

        # Unconstrained metrics (all molecules)
        unconstrained_metrics = calculate_top_k_metrics(plogp_list)

        # Get heavy atom counts for all molecules
        heavy_atom_counts = [count_heavy_atoms(m) for m in valid_rdkit_mols]

        # Constrained metrics (≤38 heavy atoms) using RDKit objects directly
        _, constrained_plogp, all_heavy_atoms, _ = filter_by_heavy_atom_count(
            valid_rdkit_mols, plogp_list, self.config.max_heavy_atoms
        )
        constrained_metrics = calculate_top_k_metrics(constrained_plogp)

        correction_stats = self._analyze_correction_levels(valid_correction_levels)

        # Build results
        results = PenalizedLogPFinalResults(
            validity=eval_results["validity"],
            uniqueness=eval_results["uniqueness"],
            novelty=eval_results["novelty"],
            n_valid=len(valid_rdkit_mols),
            # Unconstrained
            unconstrained_top1=unconstrained_metrics["top1"],
            unconstrained_top10_mean=unconstrained_metrics["top10_mean"],
            unconstrained_top100_mean=unconstrained_metrics["top100_mean"],
            unconstrained_n_molecules=len(plogp_list),
            unconstrained_heavy_atom_mean=np.mean(all_heavy_atoms) if all_heavy_atoms else 0.0,
            unconstrained_heavy_atom_std=np.std(all_heavy_atoms) if all_heavy_atoms else 0.0,
            unconstrained_heavy_atom_max=int(np.max(all_heavy_atoms)) if all_heavy_atoms else 0,
            # Constrained
            constrained_top1=constrained_metrics["top1"],
            constrained_top10_mean=constrained_metrics["top10_mean"],
            constrained_top100_mean=constrained_metrics["top100_mean"],
            constrained_n_molecules=len(constrained_plogp),
            constrained_pass_rate=(100.0 * len(constrained_plogp) / len(plogp_list) if plogp_list else 0.0),
            # Property distributions
            plogp_mean=np.mean(plogp_list) if plogp_list else 0.0,
            plogp_std=np.std(plogp_list) if plogp_list else 0.0,
            plogp_min=np.min(plogp_list) if plogp_list else 0.0,
            plogp_max=np.max(plogp_list) if plogp_list else 0.0,
            logp_mean=np.mean(logp_list) if logp_list else 0.0,
            logp_std=np.std(logp_list) if logp_list else 0.0,
            sa_score_mean=np.mean(sa_list) if sa_list else 0.0,
            sa_score_std=np.std(sa_list) if sa_list else 0.0,
            ring_size_mean=np.mean(ring_list) if ring_list else 0.0,
            ring_size_std=np.std(ring_list) if ring_list else 0.0,
            # Metadata
            correction_levels=correction_stats,
            cos_sim_mean=eval_results["cos_sim"].get("final_sim_mean", 0),
            cos_sim_std=eval_results["cos_sim"].get("final_sim_std", 0),
            optimization_time=0,
            decoding_time=0,
            total_time=0,
        )

        return results, plogp_list, valid_rdkit_mols, logp_list, sa_list, ring_list

    def _analyze_correction_levels(self, correction_levels: list[CorrectionLevel]) -> dict[str, float]:
        if not correction_levels:
            return {"level_0_pct": 0.0, "level_1_pct": 0.0, "level_2_pct": 0.0, "level_3_pct": 0.0, "fail_pct": 0.0}
        counts = defaultdict(int)
        for level in correction_levels:
            counts[level] += 1
        total = len(correction_levels)
        return {
            "level_0_pct": 100.0 * counts.get(CorrectionLevel.ZERO, 0) / total,
            "level_1_pct": 100.0 * counts.get(CorrectionLevel.ONE, 0) / total,
            "level_2_pct": 100.0 * counts.get(CorrectionLevel.TWO, 0) / total,
            "level_3_pct": 100.0 * counts.get(CorrectionLevel.THREE, 0) / total,
            "fail_pct": 100.0 * counts.get(CorrectionLevel.FAIL, 0) / total,
        }

    def _empty_results(self) -> PenalizedLogPFinalResults:
        """Return empty results structure."""
        return PenalizedLogPFinalResults(
            validity=0,
            uniqueness=0,
            novelty=0,
            n_valid=0,
            unconstrained_top1=0,
            unconstrained_top10_mean=0,
            unconstrained_top100_mean=0,
            unconstrained_n_molecules=0,
            unconstrained_heavy_atom_mean=0,
            unconstrained_heavy_atom_std=0,
            unconstrained_heavy_atom_max=0,
            constrained_top1=0,
            constrained_top10_mean=0,
            constrained_top100_mean=0,
            constrained_n_molecules=0,
            constrained_pass_rate=0,
            plogp_mean=0,
            plogp_std=0,
            plogp_min=0,
            plogp_max=0,
            logp_mean=0,
            logp_std=0,
            sa_score_mean=0,
            sa_score_std=0,
            ring_size_mean=0,
            ring_size_std=0,
            correction_levels={},
            cos_sim_mean=0,
            cos_sim_std=0,
            optimization_time=0,
            decoding_time=0,
            total_time=0,
        )


# ===== Plotting Functions =====
def plot_dual_plogp_distribution(
    plogp_list: list[float],
    constrained_plogp: list[float],
    save_dir: pathlib.Path,
):
    """Plot pLogP distributions for both unconstrained and constrained."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Unconstrained
    if plogp_list:
        ax1.hist(plogp_list, bins=50, alpha=0.7, color="blue", edgecolor="black")
        ax1.axvline(np.mean(plogp_list), color="red", linestyle="--", label=f"Mean: {np.mean(plogp_list):.2f}")
        ax1.axvline(np.max(plogp_list), color="green", linestyle="--", label=f"Max: {np.max(plogp_list):.2f}")
    ax1.set_xlabel("Penalized LogP", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("UNCONSTRAINED (All molecules)", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Constrained
    if constrained_plogp:
        ax2.hist(constrained_plogp, bins=50, alpha=0.7, color="orange", edgecolor="black")
        ax2.axvline(
            np.mean(constrained_plogp), color="red", linestyle="--", label=f"Mean: {np.mean(constrained_plogp):.2f}"
        )
        ax2.axvline(
            np.max(constrained_plogp), color="green", linestyle="--", label=f"Max: {np.max(constrained_plogp):.2f}"
        )
    ax2.set_xlabel("Penalized LogP", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("CONSTRAINED (≤38 heavy atoms)", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "plogp_distribution_dual.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_heavy_atom_distribution(heavy_atoms: list[int], max_atoms: int, save_dir: pathlib.Path):
    """Plot distribution of heavy atom counts."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(heavy_atoms, bins=50, alpha=0.7, color="purple", edgecolor="black")
    ax.axvline(max_atoms, color="red", linestyle="--", linewidth=2, label=f"ZINC250k Max ({max_atoms} atoms)")
    ax.axvline(np.mean(heavy_atoms), color="green", linestyle="--", label=f"Mean: {np.mean(heavy_atoms):.1f}")

    ax.set_xlabel("Number of Heavy Atoms", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Heavy Atom Count Distribution", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "heavy_atom_distribution.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_baseline_comparison(results: PenalizedLogPFinalResults, save_dir: pathlib.Path):
    """Plot comparison with literature baselines."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Constrained comparison
    constrained_methods = ["GCPN", "JT-VAE", "GP-MoLFormer", "Ours"]
    constrained_scores = [7.98, 5.30, 9.35, results.constrained_top1]
    colors_c = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    bars1 = ax1.bar(constrained_methods, constrained_scores, color=colors_c)
    ax1.set_ylabel("Top-1 Penalized LogP", fontsize=12)
    ax1.set_title("CONSTRAINED (≤38 heavy atoms)", fontsize=14)
    ax1.grid(True, alpha=0.3, axis="y")

    for bar, score in zip(bars1, constrained_scores, strict=False):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height, f"{score:.2f}", ha="center", va="bottom", fontsize=10)

    # Unconstrained comparison
    unconstrained_methods = ["FRATTVAE+MSO", "GP-MoLFormer", "Ours"]
    unconstrained_scores = [16.84, 19.59, results.unconstrained_top1]
    colors_u = ["#ff7f0e", "#2ca02c", "#d62728"]

    bars2 = ax2.bar(unconstrained_methods, unconstrained_scores, color=colors_u)
    ax2.set_ylabel("Top-1 Penalized LogP", fontsize=12)
    ax2.set_title("UNCONSTRAINED (No size limit)", fontsize=14)
    ax2.grid(True, alpha=0.3, axis="y")

    for bar, score in zip(bars2, unconstrained_scores, strict=False):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height, f"{score:.2f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_dir / "baseline_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_optimization_history(losses: list[float], save_dir: pathlib.Path):
    """Plot optimization loss history."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses, linewidth=2)
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Loss")
    ax.set_title("pLogP Maximization Optimization History")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(save_dir / "optimization_history.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def save_top_molecules(
    valid_rdkit_mols: list,
    plogp_list: list[float],
    save_path: pathlib.Path,
    n_top: int = 100,
    max_heavy_atoms: int = None,
):
    """Save top N molecules to CSV with properties."""
    if not plogp_list:
        return

    # Sort by pLogP descending
    sorted_indices = np.argsort(plogp_list)[::-1]
    n_actual = min(n_top, len(sorted_indices))
    top_indices = sorted_indices[:n_actual]

    rows = []
    for rank, idx in enumerate(top_indices, 1):
        mol = valid_rdkit_mols[idx]
        plogp = plogp_list[idx]

        try:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                logp = rdkit_logp(mol)
                sa = rdkit_sa_score(mol)
                ring = rdkit_max_ring_size(mol)
                heavy_atoms = count_heavy_atoms(mol)

                # Skip if constrained and doesn't meet criteria
                if max_heavy_atoms is not None and heavy_atoms > max_heavy_atoms:
                    continue

                rows.append(
                    {
                        "rank": rank,
                        "penalized_logp": plogp,
                        "logp": logp,
                        "sa_score": sa,
                        "ring_size": ring,
                        "heavy_atoms": heavy_atoms,
                        "smiles": smiles,
                    }
                )
        except Exception as e:
            print(f"Error processing molecule {idx}: {e}")
            continue

    df = pd.DataFrame(rows)
    # Re-rank after filtering
    df["rank"] = range(1, len(df) + 1)
    df.to_csv(save_path, index=False)
    print(f"Saved top {len(df)} molecules to {save_path}")


def draw_molecules_with_metadata(
    valid_rdkit_mols: list,
    plogp_list: list[float],
    logp_list: list[float],
    sa_list: list[float],
    ring_list: list[float],
    training_smiles: set[str],
    save_dir: pathlib.Path,
    max_draw: int = 100,
    fmt: str = "svg",
):
    """
    Draw molecules with pLogP values and metadata in filenames.
    """
    if not valid_rdkit_mols:
        print("No valid molecules to draw")
        return

    # Create molecules subdirectory
    mol_dir = save_dir / "molecules"
    mol_dir.mkdir(exist_ok=True)

    # Sort by pLogP descending to get top molecules
    sorted_indices = np.argsort(plogp_list)[::-1]
    n_draw = min(max_draw, len(sorted_indices))
    top_indices = sorted_indices[:n_draw]

    print(f"\nDrawing top {n_draw} molecules to {mol_dir}/")

    drawn_count = 0
    for rank, idx in enumerate(tqdm(top_indices, desc="Drawing molecules"), start=1):
        mol = valid_rdkit_mols[idx]
        if mol is None:
            continue

        try:
            plogp = plogp_list[idx]
            logp = logp_list[idx]
            sa = sa_list[idx]
            ring = int(ring_list[idx])

            # Get SMILES for novelty check
            smiles = Chem.MolToSmiles(mol)
            is_novel = smiles not in training_smiles
            novelty = "novel" if is_novel else "known"

            # Get heavy atom count
            heavy_atoms = count_heavy_atoms(mol)

            # Create filename with 4 decimal places for pLogP
            filename = (
                f"mol_{rank:03d}_plogp{plogp:.4f}_logp{logp:.2f}_sa{sa:.2f}_ring{ring}_ha{heavy_atoms}_{novelty}.{fmt}"
            )
            save_path = mol_dir / filename

            # Draw molecule using src.utils.chem.draw_mol
            draw_mol(mol, save_path=str(save_path), fmt=fmt)
            drawn_count += 1

        except Exception as e:
            print(f"\nWarning: Failed to draw molecule {rank} (idx {idx}): {e}")
            continue

    print(f"Successfully drew {drawn_count}/{n_draw} molecules")


def save_all_valid_molecules(
    valid_rdkit_mols: list,
    plogp_list: list[float],
    logp_list: list[float],
    sa_list: list[float],
    ring_list: list[float],
    save_path: pathlib.Path,
):
    """Save ALL valid molecules with properties to CSV."""
    if not valid_rdkit_mols:
        print("No valid molecules to save")
        return

    rows = []
    for idx, mol in enumerate(valid_rdkit_mols):
        if mol is None:
            continue

        try:
            smiles = Chem.MolToSmiles(mol)
            heavy_atoms = count_heavy_atoms(mol)

            rows.append(
                {
                    "idx": idx + 1,
                    "penalized_logp": plogp_list[idx],
                    "logp": logp_list[idx],
                    "sa_score": sa_list[idx],
                    "ring_size": int(ring_list[idx]),
                    "heavy_atoms": heavy_atoms,
                    "smiles": smiles,
                }
            )
        except Exception as e:
            print(f"Error processing molecule {idx}: {e}")
            continue

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} molecules to {save_path}")


# ===== Main Execution =====
def run_final_evaluation(
    hpo_dir: pathlib.Path,
    dataset: SupportedDataset,
    gen_model_idx: int,
    n_samples: int,
    max_heavy_atoms: int,
    output_dir: pathlib.Path,
    draw: bool = True,
    max_draw: int = 100,
    draw_format: str = "svg",
):
    """Run final evaluation with best HPO configuration and dual reporting."""
    device = pick_device()

    # Load best trial from CSV
    csv_files = list(hpo_dir.glob("trials_*.csv"))
    if not csv_files:
        raise ValueError(f"No trials CSV found in {hpo_dir}")

    csv_path = csv_files[0]
    print(f"\nLoading best trial from: {csv_path}")
    best_trial = load_best_trial_from_csv(csv_path)

    print("\n" + "=" * 80)
    print("Best Trial Configuration:")
    print(f"  Trial Number: {best_trial['trial_number']}")
    print(f"  Objective Value: {best_trial['objective_value']:.6f}")
    print(f"  pLogP Mean: {best_trial['plogp_mean']:.4f}")
    print(f"  pLogP Max: {best_trial['plogp_max']:.6f}")
    print(f"  Learning Rate: {best_trial['lr']:.6f}")
    print(f"  Steps: {best_trial['steps']}")
    print(f"  Scheduler: {best_trial['scheduler']}")
    print(f"  Lambda Diversity: {best_trial['lambda_diversity']:.6f}")
    print(f"  Optimizer: {best_trial['optimizer']}")
    print("=" * 80 + "\n")

    # Load models
    gen_model_hint = GENERATOR_REGISTRY[dataset][gen_model_idx]
    generator = HDCGenerator(
        gen_model_hint=gen_model_hint,
        ds_config=dataset.default_cfg,
        device=device,
        decoder_settings=DecoderSettings.get_default_for(dataset.default_cfg.base_dataset),
    )

    # Load property regressors
    regressors = {}
    for prop in ["logp", "sa_score", "max_ring_size"]:
        reg_hints = REGRESSOR_REGISTRY[dataset].get(prop, [])
        if not reg_hints:
            raise ValueError(f"No {prop} regressor available for {dataset.value}")
        pr_path = get_pr_path(hint=reg_hints[0])
        regressors[prop] = retrieve_model(name="PR").load_from_checkpoint(pr_path).to(device).eval()

    base_dataset = dataset.default_cfg.base_dataset
    ds = get_split(split="train", base_dataset=base_dataset)
    # ds_props = get_dataset_props(base_dataset=base_dataset)

    print(f"Dataset pLogP stats: mean={ds.pen_logp.mean():.4f}, std={ds.pen_logp.std():.4f}\n")

    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_dir / f"{dataset.value}_plogp_final_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Save configuration
    config_dict = {
        "dataset": dataset.value,
        "property": "penalized_logp_maximization",
        "gen_model_hint": gen_model_hint,
        "gen_model_idx": gen_model_idx,
        "regressor_hints": {
            prop: REGRESSOR_REGISTRY[dataset][prop][0] for prop in ["logp", "sa_score", "max_ring_size"]
        },
        "n_samples": n_samples,
        "max_heavy_atoms": max_heavy_atoms,
        "timestamp": timestamp,
        "device": str(device),
        "hpo_dir": str(hpo_dir),
        "best_trial": best_trial,
    }
    (experiment_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

    # Run optimization
    config = PenalizedLogPConfig(
        dataset=dataset,
        n_samples=n_samples,
        gen_model_idx=gen_model_idx,
        max_heavy_atoms=max_heavy_atoms,
        device=str(device),
    )

    optimizer_obj = PenalizedLogPOptimizer(generator=generator, regressors=regressors, config=config)

    print(f"Starting optimization with {n_samples} samples...")
    start_time = time.time()
    opt_results = optimizer_obj.optimize_latent(
        lr=best_trial["lr"],
        steps=best_trial["steps"],
        scheduler_name=best_trial["scheduler"],
        lambda_lo=best_trial["lambda_lo"],
        lambda_hi=best_trial["lambda_hi"],
        lambda_diversity=best_trial["lambda_diversity"],
        optimizer_name=best_trial["optimizer"],
        grad_clip=best_trial["grad_clip"],
    )
    optimization_time = time.time() - start_time

    decode_start = time.time()
    results, plogp_list, valid_rdkit_mols, logp_list, sa_list, ring_list = optimizer_obj.evaluate_and_get_results(
        molecules=opt_results["molecules"],
        similarities=opt_results["similarities"],
        correction_levels=opt_results["correction_levels"],
        final_flags=opt_results["final_flags"],
    )
    decoding_time = time.time() - decode_start

    results.optimization_time = optimization_time
    results.decoding_time = decoding_time
    results.total_time = optimization_time + decoding_time

    # Get constrained pLogP values for plotting (using RDKit molecules)
    _, constrained_plogp, heavy_atoms, _ = filter_by_heavy_atom_count(valid_rdkit_mols, plogp_list, max_heavy_atoms)

    # Save metrics
    metrics_dict = {
        "validity": results.validity,
        "uniqueness": results.uniqueness,
        "novelty": results.novelty,
        "n_valid": results.n_valid,
        "unconstrained": {
            "top1": results.unconstrained_top1,
            "top10_mean": results.unconstrained_top10_mean,
            "top100_mean": results.unconstrained_top100_mean,
            "n_molecules": results.unconstrained_n_molecules,
            "heavy_atom_mean": results.unconstrained_heavy_atom_mean,
            "heavy_atom_std": results.unconstrained_heavy_atom_std,
            "heavy_atom_max": results.unconstrained_heavy_atom_max,
        },
        "constrained": {
            "top1": results.constrained_top1,
            "top10_mean": results.constrained_top10_mean,
            "top100_mean": results.constrained_top100_mean,
            "n_molecules": results.constrained_n_molecules,
            "pass_rate": results.constrained_pass_rate,
        },
        "property_distributions": {
            "plogp": {
                "mean": results.plogp_mean,
                "std": results.plogp_std,
                "min": results.plogp_min,
                "max": results.plogp_max,
            },
            "logp": {"mean": results.logp_mean, "std": results.logp_std},
            "sa_score": {"mean": results.sa_score_mean, "std": results.sa_score_std},
            "ring_size": {"mean": results.ring_size_mean, "std": results.ring_size_std},
        },
        "timing": {
            "optimization_time": results.optimization_time,
            "decoding_time": results.decoding_time,
            "total_time": results.total_time,
        },
    }
    (experiment_dir / "metrics.json").write_text(json.dumps(metrics_dict, indent=2, default=float))

    # Save unconstrained and constrained results separately
    (experiment_dir / "unconstrained_results.json").write_text(
        json.dumps(metrics_dict["unconstrained"], indent=2, default=float)
    )
    (experiment_dir / "constrained_results.json").write_text(
        json.dumps(metrics_dict["constrained"], indent=2, default=float)
    )

    # Save baseline comparison
    baseline_comparison = {
        "our_results": {
            "unconstrained_top1": results.unconstrained_top1,
            "constrained_top1": results.constrained_top1,
        },
        "literature_baselines": LITERATURE_BASELINES,
    }
    (experiment_dir / "baseline_comparison.json").write_text(json.dumps(baseline_comparison, indent=2))

    # Save pLogP values
    np.save(experiment_dir / "plogp_values.npy", np.array(plogp_list))

    # Save top molecules (both unconstrained and constrained)
    save_top_molecules(valid_rdkit_mols, plogp_list, experiment_dir / "top100_unconstrained_molecules.csv", n_top=100)
    save_top_molecules(
        valid_rdkit_mols,
        plogp_list,
        experiment_dir / "top100_constrained_molecules.csv",
        n_top=100,
        max_heavy_atoms=max_heavy_atoms,
    )

    # Save ALL valid molecules with properties
    print("\nSaving all valid molecules...")
    save_all_valid_molecules(
        valid_rdkit_mols=valid_rdkit_mols,
        plogp_list=plogp_list,
        logp_list=logp_list,
        sa_list=sa_list,
        ring_list=ring_list,
        save_path=experiment_dir / "all_valid_molecules.csv",
    )

    # Draw molecules with metadata
    if draw:
        print(f"\n{'=' * 80}")
        print("Drawing molecules with metadata...")
        print(f"{'=' * 80}")
        draw_molecules_with_metadata(
            valid_rdkit_mols=valid_rdkit_mols,
            plogp_list=plogp_list,
            logp_list=logp_list,
            sa_list=sa_list,
            ring_list=ring_list,
            training_smiles=optimizer_obj.evaluator.T,
            save_dir=experiment_dir,
            max_draw=max_draw,
            fmt=draw_format,
        )

    # Generate plots
    print("\nGenerating plots...")
    plot_optimization_history(opt_results["optimization_losses"], plots_dir)
    plot_dual_plogp_distribution(plogp_list, constrained_plogp, plots_dir)
    plot_heavy_atom_distribution(heavy_atoms, max_heavy_atoms, plots_dir)
    plot_baseline_comparison(results, plots_dir)

    # Print summary with dual reporting
    print("\n" + "=" * 80)
    print("FINAL EVALUATION RESULTS - DUAL REPORTING")
    print("=" * 80)
    print("\nGeneral Metrics:")
    print(f"  Validity: {results.validity:.2f}%")
    print(f"  Uniqueness: {results.uniqueness:.2f}%")
    print(f"  Novelty: {results.novelty:.2f}%")
    print(f"  Total valid molecules: {results.n_valid}")
    print("\n" + "=" * 80)
    print("UNCONSTRAINED RESULTS (No size limit)")
    print("=" * 80)
    print(f"  Top-1 pLogP:      {results.unconstrained_top1:.4f} (cf. GP-MoLFormer: 19.59)")
    print(f"  Top-10 Mean:      {results.unconstrained_top10_mean:.4f}")
    print(f"  Top-100 Mean:     {results.unconstrained_top100_mean:.4f}")
    print(f"  N molecules:      {results.unconstrained_n_molecules}")
    print(
        f"  Heavy atoms:      {results.unconstrained_heavy_atom_mean:.1f} ± {results.unconstrained_heavy_atom_std:.1f}"
    )
    print(f"  Max heavy atoms:  {results.unconstrained_heavy_atom_max}")
    print("\n" + "=" * 80)
    print(f"CONSTRAINED RESULTS (≤{max_heavy_atoms} heavy atoms)")
    print("=" * 80)
    print(f"  Top-1 pLogP:      {results.constrained_top1:.4f} (cf. GCPN: 7.98, GP-MoLFormer: 9.35)")
    print(f"  Top-10 Mean:      {results.constrained_top10_mean:.4f}")
    print(f"  Top-100 Mean:     {results.constrained_top100_mean:.4f}")
    print(f"  N molecules:      {results.constrained_n_molecules}")
    print(f"  Pass rate:        {results.constrained_pass_rate:.2f}%")
    print("\n" + "=" * 80)
    print("Property Distributions (All valid molecules):")
    print(f"  pLogP:     {results.plogp_mean:.4f} ± {results.plogp_std:.4f}")
    print(f"  LogP:      {results.logp_mean:.4f} ± {results.logp_std:.4f}")
    print(f"  SA Score:  {results.sa_score_mean:.4f} ± {results.sa_score_std:.4f}")
    print(f"  Ring Size: {results.ring_size_mean:.2f} ± {results.ring_size_std:.2f}")
    print("\nTiming:")
    print(f"  Optimization: {results.optimization_time:.2f}s")
    print(f"  Decoding: {results.decoding_time:.2f}s")
    print(f"  Total: {results.total_time:.2f}s")
    print("=" * 80)
    print(f"\nResults saved to: {experiment_dir}")


def main():
    parser = argparse.ArgumentParser(description="Final evaluation for penalized logP maximization with dual reporting")
    parser.add_argument(
        "--hpo_dir",
        type=pathlib.Path,
        default="/home/akaveh/Projects/kit/graph_hdc/src/exp/final_evaluations/plogp_maximization/hpo_results/ZINC_SMILES_HRR_256_F64_5G1NG4_plogp_maximization_20251117_160645",
        help="HPO results directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=SupportedDataset.ZINC_SMILES_HRR_256_F64_5G1NG4.value,
        choices=[d.value for d in SupportedDataset],
        help="Dataset to use",
    )
    parser.add_argument("--model_idx", type=int, default=0, help="Index of generator model in registry")
    parser.add_argument("--n_samples", type=int, default=2, help="Number of samples to generate")
    parser.add_argument(
        "--max_heavy_atoms",
        type=int,
        default=38,
        help="Maximum heavy atom count for constrained evaluation (ZINC250k max = 38)",
    )
    parser.add_argument("--output_dir", type=str, default="final_results", help="Output directory")
    parser.add_argument("--draw", action="store_true", default=True, help="Draw molecules (enabled by default)")
    parser.add_argument("--no_draw", action="store_false", dest="draw", help="Disable molecule drawing")
    parser.add_argument("--max_draw", type=int, default=100, help="Maximum number of molecules to draw")
    parser.add_argument(
        "--draw_format", type=str, default="svg", choices=["svg", "png"], help="Format for molecule drawings"
    )

    args = parser.parse_args()
    dataset = SupportedDataset(args.dataset)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_final_evaluation(
        hpo_dir=args.hpo_dir,
        dataset=dataset,
        gen_model_idx=args.model_idx,
        n_samples=args.n_samples,
        max_heavy_atoms=args.max_heavy_atoms,
        output_dir=output_dir,
        draw=args.draw,
        max_draw=args.max_draw,
        draw_format=args.draw_format,
    )


if __name__ == "__main__":
    main()
