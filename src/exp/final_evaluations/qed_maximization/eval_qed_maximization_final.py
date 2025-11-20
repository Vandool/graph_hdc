#!/usr/bin/env python
"""
QED Maximization Final Evaluation.

This script takes the HPO directory and re-runs the best configuration to get final metrics.
Following the exact protocol:
1. Load best model and hyperparameters from HPO
2. Generate 10,000 candidates via gradient-based optimization
3. Filter invalid molecules and remove duplicates
4. Select top 100 by QED score
5. Report GuacaMol score, novelty, diversity, and compound quality metrics
6. Draw top 100 molecules with comprehensive metadata (enabled by default)

Usage Examples:
--------------

1. Standard evaluation with mean_qed criterion (default):
   python eval_qed_maximization_final.py \\
       --hpo_dir hpo_results/ZINC_SMILES_HRR_256_F64_5G1NG4_qed_maximization_20251117_144852 \\
       --dataset ZINC_SMILES_HRR_256_F64_5G1NG4 \\
       --n_samples 10000 \\
       --selection_criterion mean_qed

2. Evaluation with max_qed criterion:
   python eval_qed_maximization_final.py \\
       --hpo_dir hpo_results/ZINC_SMILES_HRR_256_F64_5G1NG4_qed_maximization_20251117_144852 \\
       --dataset ZINC_SMILES_HRR_256_F64_5G1NG4 \\
       --n_samples 10000 \\
       --selection_criterion max_qed

3. With custom drawing options (top 50 molecules in PNG format):
   python eval_qed_maximization_final.py \\
       --hpo_dir hpo_results/ZINC_SMILES_HRR_256_F64_5G1NG4_qed_maximization_20251117_144852 \\
       --dataset ZINC_SMILES_HRR_256_F64_5G1NG4 \\
       --n_samples 10000 \\
       --selection_criterion mean_qed \\
       --max_draw 50 \\
       --draw_format png

4. Disable molecule drawing:
   python eval_qed_maximization_final.py \\
       --hpo_dir hpo_results/ZINC_SMILES_HRR_256_F64_5G1NG4_qed_maximization_20251117_144852 \\
       --dataset ZINC_SMILES_HRR_256_F64_5G1NG4 \\
       --n_samples 10000 \\
       --selection_criterion mean_qed \\
       --no_draw

5. QM9 dataset with mean_qed criterion:
   python eval_qed_maximization_final.py \\
       --hpo_dir hpo_results/QM9_SMILES_HRR_1600_F64_G1NG3_qed_maximization_20251115_120000 \\
       --dataset QM9_SMILES_HRR_1600_F64_G1NG3 \\
       --n_samples 10000 \\
       --selection_criterion mean_qed

6. QM9 dataset with max_qed criterion:
   python eval_qed_maximization_final.py \\
       --hpo_dir hpo_results/QM9_SMILES_HRR_1600_F64_G1NG3_qed_maximization_20251115_120000 \\
       --dataset QM9_SMILES_HRR_1600_F64_G1NG3 \\
       --n_samples 10000 \\
       --selection_criterion max_qed

Selection Criteria:
------------------
- mean_qed: Selects HPO trial with highest average QED across generated molecules
            (more stable, recommended for production)
- max_qed:  Selects HPO trial with highest peak QED value
            (more aggressive, may sacrifice average quality for top performers)

Output Structure:
----------------
experiment_dir/
├── config.json                        # Experiment configuration
├── metrics.json                       # Comprehensive metrics (validity, novelty, diversity, etc.)
├── guacamol_scores.json              # GuacaMol benchmark scores
├── top100_molecules.csv              # Top 100 molecules with properties
├── qed_values.npy                    # All QED values
├── plots/
│   ├── optimization_history.pdf      # Loss curves during optimization
│   ├── qed_distribution.pdf          # QED distribution vs dataset
│   └── guacamol_components.pdf       # GuacaMol component breakdown
└── molecules/                        # Molecule drawings (if --draw enabled)
    ├── mol_001_qed0.9482_novel_ro5pass_L0.svg
    ├── mol_002_qed0.9475_known_ro5pass_L1.svg
    └── ... (up to 100 molecules with rank, QED, novelty, Ro5, correction level)
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
from rdkit.Chem import Crippen, Descriptors
from scipy import stats
from torch import nn
from tqdm.auto import tqdm

from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import DecoderSettings, SupportedDataset
from src.encoding.graph_encoders import CorrectionLevel
from src.exp.final_evaluations.models_configs_constants import (
    DATASET_STATS,
    GENERATOR_REGISTRY,
    REGRESSOR_REGISTRY,
    get_pr_path,
)
from src.generation.evaluator import GenerationEvaluator, rdkit_qed
from src.generation.generation import HDCGenerator
from src.utils.chem import draw_mol, is_valid_molecule, reconstruct_for_eval_v2
from src.utils.registery import retrieve_model
from src.utils.utils import DataTransformer, pick_device

# Default float32
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)
DEVICE = pick_device()

seed = 42
seed_everything(seed)

# ===== Property Functions =====
PROPERTY_FUNCTIONS = {
    "qed": rdkit_qed,
}


# ===== Scheduler Classes (Same as HPO) =====
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


class LinearScheduler:
    def __init__(self, steps: int, lam_hi: float, lam_lo: float):
        self.steps = steps
        self.lam_hi = lam_hi
        self.lam_lo = lam_lo

    def __call__(self, step: int) -> float:
        if step >= self.steps:
            return self.lam_lo
        alpha = 1.0 - step / self.steps
        return self.lam_lo + (self.lam_hi - self.lam_lo) * alpha


class ConstantScheduler:
    def __init__(self, steps: int, lam_hi: float, lam_lo: float):
        self.lam_const = (lam_hi + lam_lo) / 2

    def __call__(self, step: int) -> float:
        return self.lam_const


SCHEDULER_REGISTRY = {
    "cosine": CosineScheduler,
    "two-phase": TwoPhaseScheduler,
    "linear": LinearScheduler,
    "constant": ConstantScheduler,
}


# ===== Data Classes =====
@dataclass
class QEDMaximizationConfig:
    """Configuration for QED maximization experiment."""

    dataset: SupportedDataset
    n_samples: int
    gen_model_idx: int
    device: str = "cuda"


@dataclass
class QEDMaximizationResults:
    """Results from QED maximization evaluation."""

    # Final evaluation results
    validity: float
    uniqueness: float
    novelty: float
    diversity_p1: float
    diversity_p2: float
    qed_mean: float
    qed_std: float
    qed_min: float
    qed_max: float
    n_samples: int  # This is n_valid
    property_stats: dict[str, dict[str, float]]
    correction_levels: dict[str, float]
    cos_sim_mean: float
    cos_sim_std: float

    # GuacaMol scores
    guacamol_score: float
    top1_qed: float
    top10_mean_qed: float
    top100_mean_qed: float

    # Top 100 analysis
    top100_novelty: float
    top100_diversity_p1: float
    top100_diversity_p2: float
    top100_compound_quality_pass_rate: float

    # Optimization metrics
    optimization_time: float
    decoding_time: float
    total_time: float


# ===== Utility Functions =====
def calculate_guacamol_score(qed_values: list[float]) -> dict:
    """
    Calculate GuacaMol QED optimization score.

    Formula: S = 1/3 * (best + mean_top10 + mean_top100)
    """
    if not qed_values:
        return {
            "guacamol_score": 0.0,
            "top1_qed": 0.0,
            "top10_mean_qed": 0.0,
            "top100_mean_qed": 0.0,
        }

    sorted_qed = sorted(qed_values, reverse=True)
    best = sorted_qed[0] if sorted_qed else 0.0
    mean_top10 = np.mean(sorted_qed[:10]) if len(sorted_qed) >= 10 else np.mean(sorted_qed)
    mean_top100 = np.mean(sorted_qed[:100]) if len(sorted_qed) >= 100 else np.mean(sorted_qed)
    guacamol_score = (best + mean_top10 + mean_top100) / 3.0

    return {
        "guacamol_score": guacamol_score,
        "top1_qed": best,
        "top10_mean_qed": mean_top10,
        "top100_mean_qed": mean_top100,
    }


def apply_medicinal_chemistry_filters(mol: Chem.Mol) -> dict:
    """
    Apply medicinal chemistry filters (Lipinski's Rule of Five, etc.).

    Returns dictionary with filter results and molecular properties.
    """
    # Lipinski's Rule of Five
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    ro5_pass = mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10

    # Additional quality checks
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)

    return {
        "ro5_pass": ro5_pass,
        "mw": mw,
        "logp": logp,
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": rotatable_bonds,
        "tpsa": tpsa,
    }


def load_best_trial_from_csv(csv_path: pathlib.Path, selection_criterion: str = "mean_qed") -> dict:
    """
    Load best trial configuration from HPO CSV.

    Args:
        csv_path: Path to trials CSV file
        selection_criterion: Criterion for selecting best trial
            - "mean_qed": Select trial with highest mean QED (default)
            - "max_qed": Select trial with highest max QED

    Returns:
        Dictionary with best trial configuration and metadata
    """
    if not csv_path.exists():
        raise ValueError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Filter to successful trials
    df = df[df["state"] == "COMPLETE"]
    if df.empty:
        raise ValueError(f"No completed trials found in {csv_path}")

    # Find best trial based on selection criterion
    if selection_criterion == "mean_qed":
        # Use 'value' column (which is qed_mean) or fall back to qed_mean
        if "value" in df.columns:
            best_idx = df["value"].idxmax()
        else:
            best_idx = df["qed_mean"].idxmax()
        criterion_col = "qed_mean"
    elif selection_criterion == "max_qed":
        best_idx = df["qed_max"].idxmax()
        criterion_col = "qed_max"
    else:
        raise ValueError(f"Invalid selection_criterion: {selection_criterion}. Must be 'mean_qed' or 'max_qed'")

    best_trial = df.loc[best_idx]

    return {
        "lr": best_trial["lr"],
        "steps": int(best_trial["steps"]),
        "scheduler": best_trial["scheduler"],
        "lambda_lo": best_trial["lambda_lo"],
        "lambda_hi": best_trial["lambda_hi"],
        "grad_clip": best_trial["grad_clip"],
        "trial_number": int(best_trial["number"]),
        "objective_value": best_trial["value"],
        "qed_mean": best_trial.get("qed_mean", 0),
        "qed_max": best_trial.get("qed_max", 0),
        "selection_criterion": selection_criterion,
        "selection_value": best_trial[criterion_col],
    }


# ===== QED Maximization Implementation =====
class QEDMaximizationOptimizer:
    """Handles gradient-based QED optimization in latent space."""

    def __init__(
        self,
        generator: HDCGenerator,
        qed_regressor: nn.Module,
        config: QEDMaximizationConfig,
    ):
        self.generator = generator
        self.gen_model = generator.gen_model
        self.qed_regressor = qed_regressor
        self.hypernet = generator.hypernet
        self.config = config

        self.gen_model.to(DEVICE).eval()
        self.qed_regressor.to(DEVICE).eval()
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
        grad_clip: float,
    ) -> dict[str, Any]:
        """Optimize latent codes to maximize QED."""
        n_samples = self.config.n_samples

        base = nf.distributions.DiagGaussian(self.gen_model.flat_dim, trainable=False).to(DEVICE)
        z = base.sample(n_samples).detach().requires_grad_(True)

        optimizer = torch.optim.Adam([z], lr=lr)

        scheduler = SCHEDULER_REGISTRY[scheduler_name](steps=steps, lam_hi=lambda_hi, lam_lo=lambda_lo)

        best_loss = float("inf")
        best_z = z.clone()
        losses = []

        pbar = tqdm(range(steps), desc="QED Maximization", unit="step")
        for s in pbar:
            hdc = self.gen_model.decode_from_latent(z)
            y_pred = self.qed_regressor.gen_forward(hdc)

            # Maximize QED (minimize negative QED)
            lam = scheduler(s)
            qed_loss = -y_pred.mean()
            prior_loss = z.pow(2).mean()
            loss = qed_loss + lam * prior_loss

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_z = z.clone()

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([z], grad_clip)
            optimizer.step()

            mean_qed = y_pred.mean().item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "mean_QED": f"{mean_qed:.4f}", "λ": f"{lam:.2e}"})

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
    ) -> tuple[QEDMaximizationResults, list[float], list]:
        """
        Perform final evaluation with GuacaMol metrics and top 100 analysis.

        Returns:
            (QEDMaximizationResults, qed_list, valid_molecules)
        """
        if not molecules:
            return self._empty_results(), [], []

        # Filter for chemically valid molecules
        valid_molecules = []
        valid_similarities = []
        valid_correction_levels = []
        valid_final_flags = []

        for i, g in enumerate(molecules):
            mol = reconstruct_for_eval_v2(g, dataset=self.evaluator.base_dataset)
            if mol and is_valid_molecule(mol):
                valid_molecules.append(g)
                valid_similarities.append(similarities[i])
                valid_correction_levels.append(correction_levels[i])
                valid_final_flags.append(final_flags[i])

        if not valid_molecules:
            return self._empty_results(), [], []

        # Evaluate this final set
        eval_results_dict, qed_list = self._evaluate_sample_set(
            valid_molecules,
            valid_similarities,
            valid_correction_levels,
            valid_final_flags,
        )

        # Calculate GuacaMol score
        guacamol_metrics = calculate_guacamol_score(qed_list)

        # Analyze top 100 molecules
        top100_metrics = self._analyze_top_molecules(valid_molecules, qed_list, n_top=100)

        # Combine results into the final data class
        combined_results = self._combine_results(eval_results_dict, guacamol_metrics, top100_metrics)

        return combined_results, qed_list, valid_molecules

    def _evaluate_sample_set(
        self,
        molecules: list,
        similarities: list,
        correction_levels: list,
        final_flags: list,
    ) -> tuple[dict[str, Any], list[float]]:
        """
        Evaluate a set of molecules.
        Returns (results_dict, qed_list_for_valid_molecules)
        """
        # Use GenerationEvaluator for standard metrics
        eval_results = self.evaluator.evaluate(
            n_samples=len(molecules),
            samples=molecules,
            final_flags=final_flags,
            sims=similarities,
            correction_levels=correction_levels,
        )

        # Get the actual list of QEDs for valid molecules
        prop_fn = PROPERTY_FUNCTIONS["qed"]

        mols, valid_flags, _, _ = self.evaluator.get_mols_valid_flags_sims_and_correction_levels()
        valid_molecules = [m for m, f in zip(mols, valid_flags, strict=False) if f]

        qed_list = [prop_fn(m) for m in valid_molecules]

        # Re-calculate QED stats from the actual list
        qed_mean = np.mean(qed_list) if qed_list else 0
        qed_std = np.std(qed_list) if qed_list else 0
        qed_min = np.min(qed_list) if qed_list else 0
        qed_max = np.max(qed_list) if qed_list else 0

        correction_stats = self._analyze_correction_levels(correction_levels)

        results_dict = {
            "n_samples": len(valid_molecules),
            "validity": eval_results["validity"],
            "uniqueness": eval_results["uniqueness"],
            "novelty": eval_results["novelty"],
            "diversity_p1": eval_results["internal_diversity_p1"],
            "diversity_p2": eval_results["internal_diversity_p2"],
            "qed_mean": qed_mean,
            "qed_std": qed_std,
            "qed_min": qed_min,
            "qed_max": qed_max,
            "property_stats": {
                "qed": {"mean": qed_mean, "std": qed_std},
                "logp": {"mean": eval_results.get("logp_mean", 0), "std": eval_results.get("logp_std", 0)},
                "sa_score": {"mean": eval_results.get("sa_score_mean", 0), "std": eval_results.get("sa_score_std", 0)},
            },
            "correction_levels": correction_stats,
            "cos_sim": eval_results["cos_sim"],
            "final_flags_pct": eval_results["final_flags"],
        }

        return results_dict, qed_list

    def _analyze_top_molecules(
        self, valid_molecules: list, qed_list: list[float], n_top: int = 100
    ) -> dict[str, float]:
        """
        Analyze top N molecules for GuacaMol metrics.

        Returns dict with:
        - novelty (% not in training set)
        - diversity_p1, diversity_p2
        - compound_quality_pass_rate
        """
        if not qed_list or len(qed_list) == 0:
            return {
                "novelty": 0.0,
                "diversity_p1": 0.0,
                "diversity_p2": 0.0,
                "compound_quality_pass_rate": 0.0,
            }

        # Sort molecules by QED descending
        sorted_indices = np.argsort(qed_list)[::-1]
        n_actual = min(n_top, len(sorted_indices))
        top_indices = sorted_indices[:n_actual]

        top_molecules = [valid_molecules[i] for i in top_indices]

        # Calculate novelty for top molecules
        from src.utils.chem import canonical_key

        # Convert NetworkX graphs to RDKit Mol objects before canonical_key
        top_keys = []
        for m in top_molecules:
            try:
                mol, _ = DataTransformer.nx_to_mol(m)  # nx_to_mol returns (mol, mapping)
                key = canonical_key(mol)
                if key is not None:
                    top_keys.append(key)
            except Exception:
                continue
        novel_keys = set(top_keys) - self.evaluator.T
        novelty = 100.0 * len(novel_keys) / len(top_keys) if top_keys else 0.0

        # Calculate diversity for top molecules
        # Convert to RDKit mols for Tanimoto calculation
        try:
            from rdkit.Chem import AllChem
            from rdkit.DataStructs import TanimotoSimilarity

            top_mols = []
            for g in top_molecules:
                try:
                    from src.utils.chem import nx_to_mol

                    mol = nx_to_mol(g)
                    if mol is not None:
                        top_mols.append(mol)
                except Exception:
                    continue

            if len(top_mols) >= 2:
                # Calculate internal diversity p1 (radius=2) and p2 (radius=3)
                fps_p1 = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in top_mols]
                fps_p2 = [AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048) for m in top_mols]

                sims_p1 = []
                sims_p2 = []
                n = len(top_mols)
                for i in range(n):
                    for j in range(i + 1, n):
                        sims_p1.append(TanimotoSimilarity(fps_p1[i], fps_p1[j]))
                        sims_p2.append(TanimotoSimilarity(fps_p2[i], fps_p2[j]))

                diversity_p1 = 100.0 * (1.0 - np.mean(sims_p1)) if sims_p1 else 0.0
                diversity_p2 = 100.0 * (1.0 - np.mean(sims_p2)) if sims_p2 else 0.0
            else:
                diversity_p1 = 0.0
                diversity_p2 = 0.0

            # Calculate compound quality (medicinal chemistry filters)
            quality_results = [apply_medicinal_chemistry_filters(m) for m in top_mols]
            pass_count = sum(1 for r in quality_results if r["ro5_pass"])
            compound_quality_pass_rate = 100.0 * pass_count / len(quality_results) if quality_results else 0.0

        except Exception as e:
            print(f"Error calculating top molecule metrics: {e}")
            diversity_p1 = 0.0
            diversity_p2 = 0.0
            compound_quality_pass_rate = 0.0

        return {
            "novelty": novelty,
            "diversity_p1": diversity_p1,
            "diversity_p2": diversity_p2,
            "compound_quality_pass_rate": compound_quality_pass_rate,
        }

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

    def _empty_results(self) -> QEDMaximizationResults:
        """Return empty results structure."""
        return QEDMaximizationResults(
            validity=0,
            uniqueness=0,
            novelty=0,
            diversity_p1=0,
            diversity_p2=0,
            qed_mean=0,
            qed_std=0,
            qed_min=0,
            qed_max=0,
            n_samples=0,
            property_stats={},
            correction_levels={},
            cos_sim_mean=0,
            cos_sim_std=0,
            guacamol_score=0,
            top1_qed=0,
            top10_mean_qed=0,
            top100_mean_qed=0,
            top100_novelty=0,
            top100_diversity_p1=0,
            top100_diversity_p2=0,
            top100_compound_quality_pass_rate=0,
            optimization_time=0,
            decoding_time=0,
            total_time=0,
        )

    def _combine_results(
        self, eval_results: dict[str, Any], guacamol_metrics: dict, top100_metrics: dict
    ) -> QEDMaximizationResults:
        """Combine evaluation results into final structure."""
        return QEDMaximizationResults(
            validity=eval_results["validity"],
            uniqueness=eval_results["uniqueness"],
            novelty=eval_results["novelty"],
            diversity_p1=eval_results["diversity_p1"],
            diversity_p2=eval_results["diversity_p2"],
            qed_mean=eval_results["qed_mean"],
            qed_std=eval_results["qed_std"],
            qed_min=eval_results["qed_min"],
            qed_max=eval_results["qed_max"],
            n_samples=eval_results["n_samples"],
            property_stats=eval_results["property_stats"],
            correction_levels=eval_results["correction_levels"],
            cos_sim_mean=eval_results["cos_sim"].get("final_sim_mean", 0),
            cos_sim_std=eval_results["cos_sim"].get("final_sim_std", 0),
            guacamol_score=guacamol_metrics["guacamol_score"],
            top1_qed=guacamol_metrics["top1_qed"],
            top10_mean_qed=guacamol_metrics["top10_mean_qed"],
            top100_mean_qed=guacamol_metrics["top100_mean_qed"],
            top100_novelty=top100_metrics["novelty"],
            top100_diversity_p1=top100_metrics["diversity_p1"],
            top100_diversity_p2=top100_metrics["diversity_p2"],
            top100_compound_quality_pass_rate=top100_metrics["compound_quality_pass_rate"],
            optimization_time=0,
            decoding_time=0,
            total_time=0,
        )


# ===== Plotting Functions =====
def plot_qed_distribution(
    generated_qed: list[float],
    dataset_stats: dict,
    save_dir: pathlib.Path,
):
    """Plot QED distributions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if generated_qed:
        ax.hist(
            generated_qed, bins=50, alpha=0.7, label=f"Generated (n={len(generated_qed)})", color="blue", density=True
        )

    # Add dataset reference
    ref_mean = dataset_stats.get("qed", {}).get("mean", 0.732)
    ref_std = dataset_stats.get("qed", {}).get("std", 0.107)
    x = np.linspace(0, 1, 200)
    y = stats.norm.pdf(x, ref_mean, ref_std)
    ax.plot(x, y, "r--", label="Dataset", linewidth=2)

    ax.set_xlabel("QED", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("QED Maximization Distribution", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(save_dir / "qed_distribution.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_optimization_history(losses: list[float], save_dir: pathlib.Path):
    """Plot optimization loss history."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses, linewidth=2)
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Loss")
    ax.set_title("QED Maximization Optimization History")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(save_dir / "optimization_history.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_guacamol_components(guacamol_metrics: dict, save_dir: pathlib.Path):
    """Plot GuacaMol score components."""
    fig, ax = plt.subplots(figsize=(8, 6))

    components = ["Top-1", "Top-10 Mean", "Top-100 Mean", "GuacaMol Score"]
    values = [
        guacamol_metrics["top1_qed"],
        guacamol_metrics["top10_mean_qed"],
        guacamol_metrics["top100_mean_qed"],
        guacamol_metrics["guacamol_score"],
    ]

    bars = ax.bar(components, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    ax.set_ylabel("QED Score", fontsize=12)
    ax.set_title("GuacaMol QED Optimization Components", fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, values, strict=False):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_dir / "guacamol_components.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def save_top_molecules(valid_molecules: list, qed_list: list[float], save_path: pathlib.Path, n_top: int = 100):
    """Save top N molecules to CSV with properties."""
    if not qed_list:
        return

    # Sort by QED descending
    sorted_indices = np.argsort(qed_list)[::-1]
    n_actual = min(n_top, len(sorted_indices))
    top_indices = sorted_indices[:n_actual]

    rows = []
    for rank, idx in enumerate(top_indices, 1):
        g = valid_molecules[idx]
        qed = qed_list[idx]

        try:
            mol, _ = DataTransformer.nx_to_mol(g)  # nx_to_mol returns (mol, mapping)
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                med_chem = apply_medicinal_chemistry_filters(mol)

                rows.append(
                    {
                        "rank": rank,
                        "qed": qed,
                        "smiles": smiles,
                        "ro5_pass": med_chem["ro5_pass"],
                        "mw": med_chem["mw"],
                        "logp": med_chem["logp"],
                        "hbd": med_chem["hbd"],
                        "hba": med_chem["hba"],
                        "rotatable_bonds": med_chem["rotatable_bonds"],
                        "tpsa": med_chem["tpsa"],
                    }
                )
        except Exception as e:
            print(f"Error processing molecule {idx}: {e}")
            continue

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Saved top {len(df)} molecules to {save_path}")


def draw_molecules_with_metadata(
    valid_molecules: list,
    qed_list: list[float],
    correction_levels: list[CorrectionLevel],
    training_smiles: set[str],
    save_dir: pathlib.Path,
    max_draw: int = 100,
    fmt: str = "svg",
):
    """
    Draw molecules with QED values and metadata in filenames.

    Args:
        valid_molecules: List of NetworkX graphs
        qed_list: List of QED values (same length as valid_molecules)
        correction_levels: List of correction levels (same length as valid_molecules)
        training_smiles: Set of SMILES from training set for novelty check
        save_dir: Directory to save molecule images
        max_draw: Maximum number of molecules to draw
        fmt: Image format (svg or png)
    """
    if not valid_molecules:
        print("No valid molecules to draw")
        return

    # Create molecules subdirectory
    mol_dir = save_dir / "molecules"
    mol_dir.mkdir(exist_ok=True)

    # Sort by QED descending to get top molecules
    sorted_indices = np.argsort(qed_list)[::-1]
    n_draw = min(max_draw, len(sorted_indices))
    top_indices = sorted_indices[:n_draw]

    print(f"\nDrawing top {n_draw} molecules to {mol_dir}/")

    drawn_count = 0
    for rank, idx in enumerate(tqdm(top_indices, desc="Drawing molecules"), start=1):
        g = valid_molecules[idx]
        qed = qed_list[idx]
        correction_level = correction_levels[idx]

        try:
            # Convert graph to mol
            mol, _ = DataTransformer.nx_to_mol(g)
            if mol is None:
                continue

            # Get SMILES for novelty check
            smiles = Chem.MolToSmiles(mol)
            is_novel = smiles not in training_smiles
            novelty = "novel" if is_novel else "known"

            # Apply medicinal chemistry filters
            med_chem = apply_medicinal_chemistry_filters(mol)
            ro5 = "ro5pass" if med_chem["ro5_pass"] else "ro5fail"

            # Format correction level
            if correction_level == CorrectionLevel.FAIL:
                corr_str = "FAIL"
            else:
                corr_str = f"L{correction_level.value}"

            # Create filename with 4 decimal places for QED
            filename = f"mol_{rank:03d}_qed{qed:.4f}_{novelty}_{ro5}_{corr_str}.{fmt}"
            save_path = mol_dir / filename

            # Draw molecule using src.utils.chem.draw_mol
            draw_mol(mol, save_path=str(save_path), fmt=fmt)
            drawn_count += 1

        except Exception as e:
            print(f"\nWarning: Failed to draw molecule {rank} (idx {idx}): {e}")
            continue

    print(f"Successfully drew {drawn_count}/{n_draw} molecules")


# ===== Main Execution =====
def run_final_evaluation(
    hpo_dir: pathlib.Path,
    dataset: SupportedDataset,
    gen_model_idx: int,
    n_samples: int,
    output_dir: pathlib.Path,
    selection_criterion: str = "mean_qed",
    draw: bool = True,
    max_draw: int = 100,
    draw_format: str = "svg",
):
    """Run final evaluation with best HPO configuration."""
    device = pick_device()

    # Load best trial from CSV
    csv_files = list(hpo_dir.glob("trials_*.csv"))
    if not csv_files:
        raise ValueError(f"No trials CSV found in {hpo_dir}")

    csv_path = csv_files[0]
    print(f"\nLoading best trial from: {csv_path}")
    print(f"Selection criterion: {selection_criterion}")
    best_trial = load_best_trial_from_csv(csv_path, selection_criterion=selection_criterion)

    print("\n" + "=" * 60)
    print("Best Trial Configuration:")
    print(f"  Selection Criterion: {selection_criterion}")
    print(f"  Trial Number: {best_trial['trial_number']}")
    print(f"  Objective Value (HPO): {best_trial['objective_value']:.6f}")
    print(f"  Selection Value ({selection_criterion}): {best_trial['selection_value']:.6f}")
    print(f"  QED Mean: {best_trial['qed_mean']:.4f}")
    print(f"  QED Max: {best_trial['qed_max']:.6f}")
    print(f"  Learning Rate: {best_trial['lr']:.6f}")
    print(f"  Steps: {best_trial['steps']}")
    print(f"  Scheduler: {best_trial['scheduler']}")
    print(f"  Lambda Lo: {best_trial['lambda_lo']:.6f}")
    print(f"  Lambda Hi: {best_trial['lambda_hi']:.6f}")
    print(f"  Grad Clip: {best_trial['grad_clip']:.4f}")
    print("=" * 60 + "\n")

    # Load models (EXACT same as HPO)
    gen_model_hint = GENERATOR_REGISTRY[dataset][gen_model_idx]
    generator = HDCGenerator(
        gen_model_hint=gen_model_hint,
        ds_config=dataset.default_cfg,
        device=device,
        decoder_settings=DecoderSettings.get_default_for(dataset.default_cfg.base_dataset),
    )

    regressor_hints = REGRESSOR_REGISTRY[dataset].get("qed", [])
    if not regressor_hints:
        raise ValueError(f"No QED regressor available for {dataset.value}")
    pr_path = get_pr_path(hint=regressor_hints[0])
    qed_regressor = retrieve_model(name="PR").load_from_checkpoint(pr_path).to(device).eval()

    base_dataset = dataset.default_cfg.base_dataset
    if base_dataset == "qm9":
        ds = QM9Smiles()
    elif base_dataset == "zinc":
        ds = ZincSmiles()
    else:
        raise ValueError(f"Unknown dataset: {base_dataset}")

    print(f"Dataset QED stats: mean={ds.qed.mean():.4f}, std={ds.qed.std():.4f}, max={ds.qed.max():.6f}\n")
    dataset_stats_dict = DATASET_STATS.get(base_dataset, {})

    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_dir / f"{dataset.value}_qed_final_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Save configuration
    config_dict = {
        "dataset": dataset.value,
        "property": "qed_maximization",
        "gen_model_hint": gen_model_hint,
        "gen_model_idx": gen_model_idx,
        "regressor_hint": regressor_hints[0],
        "n_samples": n_samples,
        "selection_criterion": selection_criterion,
        "timestamp": timestamp,
        "device": str(device),
        "hpo_dir": str(hpo_dir),
        "best_trial": best_trial,
    }
    (experiment_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

    # Run optimization with best hyperparameters
    config = QEDMaximizationConfig(
        dataset=dataset, n_samples=n_samples, gen_model_idx=gen_model_idx, device=str(device)
    )

    optimizer_obj = QEDMaximizationOptimizer(generator=generator, qed_regressor=qed_regressor, config=config)

    print(f"Starting optimization with {n_samples} samples...")
    start_time = time.time()
    opt_results = optimizer_obj.optimize_latent(
        lr=best_trial["lr"],
        steps=best_trial["steps"],
        scheduler_name=best_trial["scheduler"],
        lambda_lo=best_trial["lambda_lo"],
        lambda_hi=best_trial["lambda_hi"],
        grad_clip=best_trial["grad_clip"],
    )
    optimization_time = time.time() - start_time

    decode_start = time.time()
    results, qed_list, valid_molecules = optimizer_obj.evaluate_and_get_results(
        molecules=opt_results["molecules"],
        similarities=opt_results["similarities"],
        correction_levels=opt_results["correction_levels"],
        final_flags=opt_results["final_flags"],
    )
    decoding_time = time.time() - decode_start

    results.optimization_time = optimization_time
    results.decoding_time = decoding_time
    results.total_time = optimization_time + decoding_time

    # Save metrics
    metrics_dict = {
        "validity": results.validity,
        "uniqueness": results.uniqueness,
        "novelty": results.novelty,
        "diversity_p1": results.diversity_p1,
        "diversity_p2": results.diversity_p2,
        "qed_mean": results.qed_mean,
        "qed_std": results.qed_std,
        "qed_min": results.qed_min,
        "qed_max": results.qed_max,
        "n_valid": results.n_samples,
        "guacamol_score": results.guacamol_score,
        "top1_qed": results.top1_qed,
        "top10_mean_qed": results.top10_mean_qed,
        "top100_mean_qed": results.top100_mean_qed,
        "top100_novelty": results.top100_novelty,
        "top100_diversity_p1": results.top100_diversity_p1,
        "top100_diversity_p2": results.top100_diversity_p2,
        "top100_compound_quality_pass_rate": results.top100_compound_quality_pass_rate,
        "optimization_time": results.optimization_time,
        "decoding_time": results.decoding_time,
        "total_time": results.total_time,
        "property_stats": results.property_stats,
        "correction_levels": results.correction_levels,
    }
    (experiment_dir / "metrics.json").write_text(json.dumps(metrics_dict, indent=2, default=float))

    # Save GuacaMol-specific scores
    guacamol_scores = {
        "guacamol_score": results.guacamol_score,
        "top1_qed": results.top1_qed,
        "top10_mean_qed": results.top10_mean_qed,
        "top100_mean_qed": results.top100_mean_qed,
    }
    (experiment_dir / "guacamol_scores.json").write_text(json.dumps(guacamol_scores, indent=2))

    # Save QED values
    np.save(experiment_dir / "qed_values.npy", np.array(qed_list))

    # Save top 100 molecules
    save_top_molecules(valid_molecules, qed_list, experiment_dir / "top100_molecules.csv", n_top=100)

    # Draw molecules with metadata
    if draw:
        print(f"\n{'='*60}")
        print("Drawing molecules with metadata...")
        print(f"{'='*60}")
        draw_molecules_with_metadata(
            valid_molecules=valid_molecules,
            qed_list=qed_list,
            correction_levels=optimizer_obj.evaluator.correction_levels,
            training_smiles=optimizer_obj.evaluator.T,
            save_dir=experiment_dir,
            max_draw=max_draw,
            fmt=draw_format,
        )

    # Generate plots
    print("\nGenerating plots...")
    plot_optimization_history(opt_results["optimization_losses"], plots_dir)
    plot_qed_distribution(generated_qed=qed_list, dataset_stats=dataset_stats_dict, save_dir=plots_dir)
    plot_guacamol_components(guacamol_scores, plots_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Number of valid molecules: {results.n_samples}")
    print(f"Validity: {results.validity:.2f}%")
    print(f"Uniqueness: {results.uniqueness:.2f}%")
    print(f"Novelty: {results.novelty:.2f}%")
    print(f"Diversity (p=1): {results.diversity_p1:.2f}%")
    print(f"Diversity (p=2): {results.diversity_p2:.2f}%")
    print("\nQED Statistics:")
    print(f"  Mean: {results.qed_mean:.4f} ± {results.qed_std:.4f}")
    print(f"  Min: {results.qed_min:.4f}")
    print(f"  Max: {results.qed_max:.6f}")
    print("\nGuacaMol Metrics:")
    print(f"  GuacaMol Score: {results.guacamol_score:.6f}")
    print(f"  Top-1 QED: {results.top1_qed:.6f}")
    print(f"  Top-10 Mean QED: {results.top10_mean_qed:.6f}")
    print(f"  Top-100 Mean QED: {results.top100_mean_qed:.6f}")
    print("\nTop 100 Analysis:")
    print(f"  Novelty: {results.top100_novelty:.2f}%")
    print(f"  Diversity (p=1): {results.top100_diversity_p1:.2f}%")
    print(f"  Diversity (p=2): {results.top100_diversity_p2:.2f}%")
    print(f"  Compound Quality (Ro5 pass): {results.top100_compound_quality_pass_rate:.2f}%")
    print("\nTiming:")
    print(f"  Optimization: {results.optimization_time:.2f}s")
    print(f"  Decoding: {results.decoding_time:.2f}s")
    print(f"  Total: {results.total_time:.2f}s")
    print("=" * 60)
    print(f"\nResults saved to: {experiment_dir}")


def main():
    parser = argparse.ArgumentParser(description="Final evaluation for QED maximization")
    parser.add_argument("--hpo_dir", type=pathlib.Path, required=True, help="HPO results directory")
    parser.add_argument(
        "--dataset",
        type=str,
        default=SupportedDataset.ZINC_SMILES_HRR_256_F64_5G1NG4.value,
        choices=[d.value for d in SupportedDataset],
        help="Dataset to use",
    )
    parser.add_argument("--model_idx", type=int, default=0, help="Index of generator model in registry")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="final_results", help="Output directory")
    parser.add_argument(
        "--selection_criterion",
        type=str,
        default="mean_qed",
        choices=["mean_qed", "max_qed"],
        help="Criterion for selecting best HPO trial: 'mean_qed' (highest average QED) or 'max_qed' (highest peak QED)",
    )
    parser.add_argument("--draw", action="store_true", default=True, help="Draw molecules (enabled by default)")
    parser.add_argument("--no_draw", action="store_false", dest="draw", help="Disable molecule drawing")
    parser.add_argument("--max_draw", type=int, default=100, help="Maximum number of molecules to draw")
    parser.add_argument("--draw_format", type=str, default="svg", choices=["svg", "png"], help="Format for molecule drawings")

    args = parser.parse_args()
    dataset = SupportedDataset(args.dataset)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_final_evaluation(
        hpo_dir=args.hpo_dir,
        dataset=dataset,
        gen_model_idx=args.model_idx,
        n_samples=args.n_samples,
        output_dir=output_dir,
        selection_criterion=args.selection_criterion,
        draw=args.draw,
        max_draw=args.max_draw,
        draw_format=args.draw_format,
    )


if __name__ == "__main__":
    main()
