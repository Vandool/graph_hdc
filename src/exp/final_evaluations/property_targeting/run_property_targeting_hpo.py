#!/usr/bin/env python
"""
Property Targeting HPO for Conditional Generation (MG-DIFF Protocol)
=====================================================================

This script performs hyperparameter optimization for conditional molecular generation
targeting specific property values. Follows MG-DIFF evaluation protocol.

Key Features:
1. Absolute target values (not mean + multiplier * std)
2. Dual evaluation sets:
   - All Valid: All valid molecules (primary metric)
   - Filter 2: Molecules passing latent epsilon filter (secondary)
3. Primary metric: MAD (Mean Absolute Deviation)
4. Exports: SMILES, property values, correction levels, cosine similarities
5. Tracks: VUN metrics, internal diversity, correction statistics

Target Values (MG-DIFF Protocol):
- LogP: [2.0, 4.0, 6.0]
- QED: [0.6, 0.75, 0.9]
- SA Score: [2.0, 3.0, 4.0]
- TPSA: [30.0, 60.0, 90.0]

Usage:
    # Single target HPO
    python run_property_targeting_hpo.py \
        --property logp \
        --dataset ZINC_SMILES_HRR_256_F64_5G1NG4 \
        --n_trials 50 \
        --n_samples 100

    # Run
    pixi run -e local run python run_property_targeting_hpo.py \
        --property logp

    # Quick test
    python run_property_targeting_hpo.py \
        --property logp \
        --targets 2.0 \
        --n_trials 5 \
        --n_samples 100
"""

import argparse
import json
import math
import pathlib
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import normflows as nf
import numpy as np
import optuna
import pandas as pd
import torch
from lightning_fabric import seed_everything
from optuna_integration import BoTorchSampler
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from rdkit.Contrib.SA_Score import sascorer
from torch import nn
from tqdm.auto import tqdm

from src.encoding.configs_and_constants import DecoderSettings, SupportedDataset
from src.encoding.graph_encoders import CorrectionLevel
from src.exp.final_evaluations.models_configs_constants import (
    DATASET_STATS,
    GENERATOR_REGISTRY,
    REGRESSOR_REGISTRY,
    get_pr_path,
)
from src.generation.evaluator import GenerationEvaluator, rdkit_logp, rdkit_max_ring_size, rdkit_qed, rdkit_sa_score
from src.generation.generation import HDCGenerator
from src.utils.chem import reconstruct_for_eval
from src.utils.registery import retrieve_model
from src.utils.utils import pick_device

# Default dtype and device
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)
DEVICE = pick_device()

seed = 42
seed_everything(seed)

# ===== MG-DIFF Target Values =====
MGDIFF_TARGETS = {
    "zinc": {"logp": [2.0, 4.0, 6.0], "qed": [0.6, 0.75, 0.9], "sa_score": [2.0, 3.0, 4.0], "tpsa": [30.0, 60.0, 90.0]},
    "qm9": {"logp": [-1, 0.5, 2.0], "qed": [0.3, 0.45, 0.6], "sa_score": [3.0, 4.5, 6.0], "tpsa": [30.0, 60.0, 90.0]},
}

# ===== Property Functions =====
PROPERTY_FUNCTIONS = {
    "logp": rdkit_logp,
    "qed": rdkit_qed,
    "sa_score": rdkit_sa_score,
    "max_ring_size": rdkit_max_ring_size,
}


# ===== HPO Search Space =====
def get_search_space() -> dict[str, optuna.distributions.BaseDistribution]:
    """
    Simplified HPO search space with fixed standards.

    Tuned (4 parameters):
    - lr: Learning rate
    - steps: Number of optimization steps
    - lambda_prior: Single regularization weight (replaces lambda_lo/lambda_hi)
    - grad_clip: Gradient clipping threshold
    """
    return {
        "lr": optuna.distributions.FloatDistribution(5e-4, 5e-3, log=True),
        "steps": optuna.distributions.IntDistribution(300, 1000),
        "lambda_prior": optuna.distributions.FloatDistribution(5e-4, 5e-3, log=True),
        "grad_clip": optuna.distributions.FloatDistribution(1.0, 10.0),
    }


# ===== Scheduler Classes =====
class CosineScheduler:
    """Cosine annealing scheduler with 5% warmup."""

    def __init__(self, steps: int, lambda_prior: float, warmup_ratio: float = 0.05):
        self.steps = steps
        self.lambda_prior = lambda_prior
        self.warmup_steps = int(steps * warmup_ratio)
        self.cosine_steps = steps - self.warmup_steps

    def __call__(self, step: int) -> float:
        # Warmup phase: linearly increase from 0 to lambda_prior
        if step < self.warmup_steps:
            return self.lambda_prior * (step / self.warmup_steps)

        # Cosine annealing phase: decay from lambda_prior to 0
        cosine_step = step - self.warmup_steps
        if cosine_step >= self.cosine_steps:
            return 0.0

        alpha = 0.5 * (1 + math.cos(math.pi * cosine_step / self.cosine_steps))
        return self.lambda_prior * alpha


# Removed unused scheduler classes (TwoPhase, Linear, Constant)
# Only CosineScheduler with 5% warmup is used in simplified HPO


# ===== Data Classes =====
@dataclass
class PropertyTargetingConfig:
    """Configuration for property targeting experiment."""

    dataset: SupportedDataset
    property_name: str
    target_value: float
    n_samples: int
    gen_model_idx: int
    epsilon_multiplier: float = 0.2  # Smaller epsilon for latent filtering
    device: str = "cuda"


@dataclass
class DualEvaluationResults:
    """Results from dual evaluation (All Valid + Filter 2)."""

    # All Valid results
    all_valid_validity: float
    all_valid_uniqueness: float
    all_valid_novelty: float
    all_valid_diversity_p1: float
    all_valid_diversity_p2: float
    all_valid_mad: float
    all_valid_n_samples: int
    all_valid_property_stats: dict[str, dict[str, float]]
    all_valid_property_values: dict[str, list[float]]  # Individual values for plotting
    all_valid_correction_levels: dict[str, float]
    all_valid_cos_sim_mean: float
    all_valid_cos_sim_std: float

    # Filter 2 results
    filter2_validity: float
    filter2_uniqueness: float
    filter2_novelty: float
    filter2_diversity_p1: float
    filter2_diversity_p2: float
    filter2_mad: float
    filter2_n_samples: int
    filter2_property_stats: dict[str, dict[str, float]]
    filter2_property_values: dict[str, list[float]]  # Individual values for plotting
    filter2_correction_levels: dict[str, float]
    filter2_cos_sim_mean: float
    filter2_cos_sim_std: float

    # Metadata
    optimization_time: float
    decoding_time: float
    total_time: float
    target: float
    epsilon: float
    n_passed_latent_filter: int

    # Molecule data for export
    all_valid_smiles: list[str] = None
    all_valid_rdkit_mols: list = None
    all_valid_latent_flags: list[bool] = None
    all_valid_similarities: list[float] = None
    all_valid_correction_levels_list: list = None
    filter2_smiles: list[str] = None
    filter2_rdkit_mols: list = None


# ===== Helper Functions =====
def get_adaptive_epsilon(property_name: str, dataset: SupportedDataset, epsilon_multiplier: float = 0.2) -> float:
    """
    Compute adaptive epsilon for latent filtering.

    Args:
        property_name: Property to optimize
        dataset: Dataset configuration
        epsilon_multiplier: Multiplier for dataset std (default: 0.2)

    Returns:
        Epsilon value for latent filtering
    """
    base_dataset = dataset.default_cfg.base_dataset
    stats = DATASET_STATS[base_dataset]
    property_stats = stats.get(property_name, stats.get("logp"))
    return epsilon_multiplier * property_stats["std"]


# ===== Property Targeting Implementation =====
class PropertyTargetingOptimizer:
    """Handles gradient-based property optimization in latent space."""

    def __init__(
        self,
        generator: HDCGenerator,
        property_regressor: nn.Module,
        config: PropertyTargetingConfig,
    ):
        self.generator = generator
        self.gen_model = generator.gen_model
        self.property_regressor = property_regressor
        self.hypernet = generator.hypernet
        self.config = config

        # Move models to device
        self.gen_model.to(DEVICE).eval()
        self.property_regressor.to(DEVICE).eval()
        self.hypernet.to(DEVICE).eval()

        # Get dataset info
        base_dataset = config.dataset.default_cfg.base_dataset
        self.dataset_stats = DATASET_STATS[base_dataset]
        self.decoder_settings = DecoderSettings.get_default_for(base_dataset)
        self.base_dataset = base_dataset

        # Create evaluator
        self.evaluator = GenerationEvaluator(base_dataset=base_dataset, device=DEVICE)

    def optimize_latent(
        self,
        target: float,
        epsilon: float,
        lr: float,
        steps: int,
        lambda_prior: float,
        grad_clip: float,
    ) -> dict[str, Any]:
        """
        Optimize latent codes to target property value.

        Uses fixed Adam optimizer and cosine scheduler with 5% warmup.

        Returns:
            Dictionary with optimization results including latent filter flags
        """
        n_samples = self.config.n_samples

        # Initialize latents
        base = nf.distributions.DiagGaussian(self.gen_model.flat_dim, trainable=False).to(DEVICE)
        z = base.sample(n_samples)
        z = z.detach().requires_grad_(True)

        # Setup optimizer (fixed: Adam)
        optimizer = torch.optim.Adam([z], lr=lr)

        # Setup scheduler (fixed: Cosine with 5% warmup)
        scheduler = CosineScheduler(steps=steps, lambda_prior=lambda_prior)

        # Optimization loop
        best_loss = float("inf")
        best_z = z.clone()
        losses = []

        pbar = tqdm(range(steps), desc="Optimization", unit="step")
        for s in pbar:
            # Decode to HDC
            hdc = self.gen_model.decode_from_latent(z)

            # Predict property
            y_pred = self.property_regressor.gen_forward(hdc)

            # Compute loss
            lam = scheduler(s)
            mse_loss = ((y_pred - target) ** 2).mean()
            prior_loss = z.pow(2).mean()
            loss = mse_loss + lam * prior_loss

            # Track best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_z = z.clone()

            losses.append(loss.item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([z], grad_clip)

            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "λ": f"{lam:.2e}"})

        # Use best z for final generation
        z = best_z

        # Final predictions in latent space
        with torch.no_grad():
            x = self.gen_model.decode_from_latent(z)
            edge_terms, graph_terms = self.gen_model.split(x)
            y_pred_latent = self.property_regressor.gen_forward(x)

        # Latent filtering (for Filter 2 set)
        hits_latent = (y_pred_latent - target).abs() <= epsilon
        n_passed_latent = hits_latent.sum().item()

        # Decode ALL samples (not just hits)
        decoded = self.generator.decode(edge_terms=edge_terms, graph_terms=graph_terms)

        molecules = decoded["graphs"]
        similarities = decoded["similarities"]
        correction_levels = decoded["correction_levels"]
        final_flags = decoded["final_flags"]

        return {
            "molecules": molecules,
            "similarities": similarities,
            "correction_levels": correction_levels,
            "final_flags": final_flags,
            "optimization_losses": losses,
            "latent_filter_flags": hits_latent.cpu().numpy(),  # NEW: flags for Filter 2
            "n_passed_latent_filter": n_passed_latent,
            "edge_terms": edge_terms.cpu(),
            "graph_terms": graph_terms.cpu(),
        }

    def evaluate_with_dual_sets(
        self,
        molecules: list,
        similarities: list,
        correction_levels: list,
        final_flags: list,
        latent_filter_flags: np.ndarray,
        target: float,
        epsilon: float,
    ) -> DualEvaluationResults:
        """
        Perform dual evaluation: All Valid + Filter 2.

        Args:
            molecules: All decoded molecules (NetworkX graphs)
            similarities: Cosine similarities
            correction_levels: Correction levels
            final_flags: Final flags from decoder
            latent_filter_flags: Boolean flags indicating which samples passed latent filter
            target: Target property value
            epsilon: Epsilon used for latent filtering

        Returns:
            DualEvaluationResults with metrics for both sets
        """
        if not molecules:
            return self._empty_results(target, epsilon, 0)

        # Convert nx graphs to RDKit mols and compute properties
        valid_molecules = []
        valid_rdkit_mols = []
        valid_smiles = []
        valid_properties = []
        valid_similarities = []
        valid_correction_levels = []
        valid_final_flags = []
        valid_latent_flags = []

        prop_fn = PROPERTY_FUNCTIONS[self.config.property_name]

        for i, g in enumerate(molecules):
            if g is None:
                continue

            try:
                # Convert to RDKit mol
                mol = reconstruct_for_eval(g, dataset=self.base_dataset)
                if mol is None:
                    continue

                # Compute property
                prop_value = prop_fn(mol)

                # Get canonical SMILES
                smiles = Chem.MolToSmiles(mol)

                # Store valid data
                valid_molecules.append(g)  # Keep nx graph for evaluator
                valid_rdkit_mols.append(mol)
                valid_smiles.append(smiles)
                valid_properties.append(prop_value)
                valid_similarities.append(similarities[i])
                valid_correction_levels.append(correction_levels[i])
                valid_final_flags.append(final_flags[i])
                valid_latent_flags.append(latent_filter_flags[i])

            except Exception:
                continue

        if not valid_molecules:
            return self._empty_results(target, epsilon, 0)

        # Convert to numpy for easier manipulation
        valid_properties = np.array(valid_properties)
        valid_latent_flags = np.array(valid_latent_flags)

        # ===== Evaluation Set 1: All Valid =====
        all_valid_results = self._evaluate_sample_set(
            molecules=valid_molecules,
            rdkit_mols=valid_rdkit_mols,
            smiles=valid_smiles,
            properties=valid_properties,
            similarities=valid_similarities,
            correction_levels=valid_correction_levels,
            final_flags=valid_final_flags,
            target=target,
            eval_type="all_valid",
        )

        # ===== Evaluation Set 2: Filter 2 (latent-filtered) =====
        filter2_indices = np.where(valid_latent_flags)[0]

        if len(filter2_indices) > 0:
            filter2_molecules = [valid_molecules[i] for i in filter2_indices]
            filter2_rdkit_mols = [valid_rdkit_mols[i] for i in filter2_indices]
            filter2_smiles = [valid_smiles[i] for i in filter2_indices]
            filter2_properties = valid_properties[filter2_indices]
            filter2_similarities = [valid_similarities[i] for i in filter2_indices]
            filter2_correction_levels = [valid_correction_levels[i] for i in filter2_indices]
            filter2_final_flags = [valid_final_flags[i] for i in filter2_indices]

            filter2_results = self._evaluate_sample_set(
                molecules=filter2_molecules,
                rdkit_mols=filter2_rdkit_mols,
                smiles=filter2_smiles,
                properties=filter2_properties,
                similarities=filter2_similarities,
                correction_levels=filter2_correction_levels,
                final_flags=filter2_final_flags,
                target=target,
                eval_type="filter2",
            )
        else:
            # No molecules passed latent filter
            filter2_results = {
                "evaluation_type": "filter2",
                "n_samples": 0,
                "validity": 0,
                "uniqueness": 0,
                "novelty": 0,
                "diversity_p1": 0,
                "diversity_p2": 0,
                "mad": float("inf"),
                "property_stats": {},
                "property_values": {},  # Empty for plotting
                "correction_levels": {},
                "cos_sim_mean": 0,
                "cos_sim_std": 0,
                "smiles": [],
                "properties": [],
            }

        # Combine results
        return self._combine_dual_results(
            all_valid_results,
            filter2_results,
            target,
            epsilon,
            len(filter2_indices),
            all_valid_smiles=valid_smiles,
            all_valid_rdkit_mols=valid_rdkit_mols,
            all_valid_latent_flags=valid_latent_flags.tolist(),
            all_valid_similarities=valid_similarities,
            all_valid_correction_levels_list=valid_correction_levels,
            filter2_smiles=filter2_smiles if len(filter2_indices) > 0 else [],
            filter2_rdkit_mols=filter2_rdkit_mols if len(filter2_indices) > 0 else [],
        )

    def _evaluate_sample_set(
        self,
        molecules: list,
        rdkit_mols: list,
        smiles: list,
        properties: np.ndarray,
        similarities: list,
        correction_levels: list,
        final_flags: list,
        target: float,
        eval_type: str,
    ) -> dict[str, Any]:
        """Evaluate a set of molecules."""
        # Use GenerationEvaluator for standard metrics (pass nx graphs)
        eval_results = self.evaluator.evaluate(
            n_samples=len(molecules),
            samples=molecules,
            final_flags=final_flags,
            sims=similarities,
            correction_levels=correction_levels,
        )

        # Compute MAD
        mad = np.abs(properties - target).mean()

        # Analyze correction levels
        correction_stats = self._analyze_correction_levels(correction_levels)

        # Compute all properties for individual values (for plotting)
        property_values = {
            "logp": [],
            "qed": [],
            "sa_score": [],
            "max_ring_size": [],
        }

        for mol in rdkit_mols:
            property_values["logp"].append(float(Descriptors.MolLogP(mol)))
            property_values["qed"].append(float(QED.qed(mol)))
            property_values["sa_score"].append(float(sascorer.calculateScore(mol)))
            # Max ring size
            ri = mol.GetRingInfo()
            ring_sizes = [len(ring) for ring in ri.AtomRings()]
            property_values["max_ring_size"].append(float(max(ring_sizes)) if ring_sizes else 0.0)

        # Combine all metrics
        return {
            "evaluation_type": eval_type,
            "n_samples": len(molecules),
            "validity": eval_results["validity"],
            "uniqueness": eval_results["uniqueness"],
            "novelty": eval_results["novelty"],
            "diversity_p1": eval_results["internal_diversity_p1"],
            "diversity_p2": eval_results["internal_diversity_p2"],
            "mad": float(mad),
            "property_stats": {
                "logp": {"mean": eval_results.get("logp_mean", 0), "std": eval_results.get("logp_std", 0)},
                "qed": {"mean": eval_results.get("qed_mean", 0), "std": eval_results.get("qed_std", 0)},
                "sa_score": {"mean": eval_results.get("sa_score_mean", 0), "std": eval_results.get("sa_score_std", 0)},
                "max_ring_size": {
                    "mean": eval_results.get("max_ring_size_mean", 0),
                    "std": eval_results.get("max_ring_size_std", 0),
                },
            },
            "property_values": property_values,  # Individual values for plotting
            "correction_levels": correction_stats,
            "cos_sim_mean": eval_results["cos_sim"].get("final_sim_mean", 0),
            "cos_sim_std": eval_results["cos_sim"].get("final_sim_std", 0),
            "smiles": smiles,
            "properties": properties.tolist(),
        }

    def _analyze_correction_levels(self, correction_levels: list[CorrectionLevel]) -> dict[str, float]:
        """Analyze distribution of correction levels."""
        if not correction_levels:
            return {
                "level_0_pct": 0.0,
                "level_1_pct": 0.0,
                "level_2_pct": 0.0,
                "level_3_pct": 0.0,
                "fail_pct": 0.0,
            }

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

    def _empty_results(self, target: float, epsilon: float, n_passed_latent: int) -> DualEvaluationResults:
        """Return empty results structure."""
        return DualEvaluationResults(
            all_valid_validity=0,
            all_valid_uniqueness=0,
            all_valid_novelty=0,
            all_valid_diversity_p1=0,
            all_valid_diversity_p2=0,
            all_valid_mad=float("inf"),
            all_valid_n_samples=0,
            all_valid_property_stats={},
            all_valid_correction_levels={},
            all_valid_cos_sim_mean=0,
            all_valid_cos_sim_std=0,
            filter2_validity=0,
            filter2_uniqueness=0,
            filter2_novelty=0,
            filter2_diversity_p1=0,
            filter2_diversity_p2=0,
            filter2_mad=float("inf"),
            filter2_n_samples=0,
            filter2_property_stats={},
            filter2_correction_levels={},
            filter2_cos_sim_mean=0,
            filter2_cos_sim_std=0,
            optimization_time=0,
            decoding_time=0,
            total_time=0,
            target=target,
            epsilon=epsilon,
            n_passed_latent_filter=n_passed_latent,
        )

    def _combine_dual_results(
        self,
        all_valid_results: dict[str, Any],
        filter2_results: dict[str, Any],
        target: float,
        epsilon: float,
        n_passed_latent: int,
        all_valid_smiles: list[str] | None = None,
        all_valid_rdkit_mols: list | None = None,
        all_valid_latent_flags: list[bool] = None,
        all_valid_similarities: list[float] = None,
        all_valid_correction_levels_list: list = None,
        filter2_smiles: list[str] | None = None,
        filter2_rdkit_mols: list = None,
    ) -> DualEvaluationResults:
        """Combine All Valid and Filter 2 results into final structure."""
        return DualEvaluationResults(
            # All Valid results
            all_valid_validity=all_valid_results["validity"],
            all_valid_uniqueness=all_valid_results["uniqueness"],
            all_valid_novelty=all_valid_results["novelty"],
            all_valid_diversity_p1=all_valid_results["diversity_p1"],
            all_valid_diversity_p2=all_valid_results["diversity_p2"],
            all_valid_mad=all_valid_results["mad"],
            all_valid_n_samples=all_valid_results["n_samples"],
            all_valid_property_stats=all_valid_results["property_stats"],
            all_valid_property_values=all_valid_results.get("property_values", {}),  # Individual values
            all_valid_correction_levels=all_valid_results["correction_levels"],
            all_valid_cos_sim_mean=all_valid_results["cos_sim_mean"],
            all_valid_cos_sim_std=all_valid_results["cos_sim_std"],
            # Filter 2 results
            filter2_validity=filter2_results["validity"],
            filter2_uniqueness=filter2_results["uniqueness"],
            filter2_novelty=filter2_results["novelty"],
            filter2_diversity_p1=filter2_results["diversity_p1"],
            filter2_diversity_p2=filter2_results["diversity_p2"],
            filter2_mad=filter2_results["mad"],
            filter2_n_samples=filter2_results["n_samples"],
            filter2_property_stats=filter2_results["property_stats"],
            filter2_property_values=filter2_results.get("property_values", {}),  # Individual values
            filter2_correction_levels=filter2_results["correction_levels"],
            filter2_cos_sim_mean=filter2_results["cos_sim_mean"],
            filter2_cos_sim_std=filter2_results["cos_sim_std"],
            # Metadata
            optimization_time=0,
            decoding_time=0,
            total_time=0,
            target=target,
            epsilon=epsilon,
            n_passed_latent_filter=n_passed_latent,
            # Molecule data
            all_valid_smiles=all_valid_smiles,
            all_valid_rdkit_mols=all_valid_rdkit_mols,
            all_valid_latent_flags=all_valid_latent_flags,
            all_valid_similarities=all_valid_similarities,
            all_valid_correction_levels_list=all_valid_correction_levels_list,
            filter2_smiles=filter2_smiles,
            filter2_rdkit_mols=filter2_rdkit_mols,
        )


# ===== HPO Management =====
def load_or_create_study(study_name: str, db_path: pathlib.Path, search_space: dict) -> optuna.Study:
    """Load existing study or create new one."""
    storage = f"sqlite:///{db_path}"

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"Loaded existing study: {study_name} with {len(study.trials)} trials")
    except KeyError:
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=storage,
            sampler=BoTorchSampler(seed=42),
        )
        print(f"Created new study: {study_name}")

    return study


def export_trials_to_csv(study: optuna.Study, csv_path: pathlib.Path):
    """Export all trials to CSV."""
    rows = []
    for trial in study.get_trials(deepcopy=False):
        row = {
            "number": trial.number,
            "value": trial.value,
            "state": trial.state.name,
            **trial.params,
            **trial.user_attrs,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Exported {len(df)} trials to {csv_path}")


# ===== Main Execution =====
def run_property_targeting_trial(
    trial: optuna.Trial,
    config: PropertyTargetingConfig,
    generator: HDCGenerator,
    property_regressor: nn.Module,
) -> DualEvaluationResults:
    """
    Run a single HPO trial with simplified 4-parameter search space.

    Fixed:
    - Optimizer: Adam
    - Scheduler: Cosine with 5% warmup
    - Epsilon: 0.2 (fixed multiplier)

    Tuned:
    - lr: Learning rate
    - steps: Number of optimization steps
    - lambda_prior: Regularization weight
    - grad_clip: Gradient clipping threshold
    """
    # Sample hyperparameters (4 parameters only)
    lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    steps = trial.suggest_int("steps", 300, 800)
    lambda_prior = trial.suggest_float("lambda_prior", 5e-4, 5e-3, log=True)
    grad_clip = trial.suggest_float("grad_clip", 1.0, 10.0)

    # Fixed epsilon multiplier
    epsilon_multiplier = 0.2
    epsilon = get_adaptive_epsilon(config.property_name, config.dataset, epsilon_multiplier)
    target = config.target_value

    # Create optimizer
    optimizer_obj = PropertyTargetingOptimizer(
        generator=generator, property_regressor=property_regressor, config=config
    )

    # Run optimization
    start_time = time.time()
    opt_results = optimizer_obj.optimize_latent(
        target=target,
        epsilon=epsilon,
        lr=lr,
        steps=steps,
        lambda_prior=lambda_prior,
        grad_clip=grad_clip,
    )
    optimization_time = time.time() - start_time

    # Evaluate with dual sets
    decode_start = time.time()
    results = optimizer_obj.evaluate_with_dual_sets(
        molecules=opt_results["molecules"],
        similarities=opt_results["similarities"],
        correction_levels=opt_results["correction_levels"],
        final_flags=opt_results["final_flags"],
        latent_filter_flags=opt_results["latent_filter_flags"],
        target=target,
        epsilon=epsilon,
    )
    decoding_time = time.time() - decode_start

    # Update timing
    results.optimization_time = optimization_time
    results.decoding_time = decoding_time
    results.total_time = optimization_time + decoding_time

    return results


def run_hpo(
    dataset: SupportedDataset,
    property_name: str,
    target_values: list[float],
    gen_model_idx: int,
    n_trials: int,
    n_samples: int,
    output_dir: pathlib.Path,
):
    """Run HPO for property targeting."""
    # Load models
    # Get generator

    print(f"Running on DEVICE: {DEVICE}")

    gen_model_hint = GENERATOR_REGISTRY[dataset][gen_model_idx]
    generator = HDCGenerator(
        gen_model_hint=gen_model_hint,
        ds_config=dataset.default_cfg,
        device=DEVICE,
        decoder_settings=DecoderSettings.get_default_for(dataset.default_cfg.base_dataset),
    )

    # Get property regressor
    regressor_hints = REGRESSOR_REGISTRY[dataset].get(property_name, [])
    if not regressor_hints:
        raise ValueError(f"No regressor available for {property_name} on {dataset.value}")

    pr_path = get_pr_path(hint=regressor_hints[0])
    property_regressor = retrieve_model(name="PR").load_from_checkpoint(pr_path).to(DEVICE).eval()

    # Get search space
    search_space = get_search_space()

    # Create organized directory structure
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_dir / f"{property_name}_{dataset.value}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment metadata
    experiment_metadata = {
        "dataset": dataset.value,
        "property": property_name,
        "target_values": target_values,
        "gen_model_hint": gen_model_hint,
        "gen_model_idx": gen_model_idx,
        "regressor_hint": regressor_hints[0],
        "n_trials": n_trials,
        "n_samples": n_samples,
        "timestamp": timestamp,
        "device": str(DEVICE),
        "protocol": "MG-DIFF Conditional Generation",
    }

    metadata_path = experiment_dir / "experiment_metadata.json"
    metadata_path.write_text(json.dumps(experiment_metadata, indent=2))

    # Run HPO for each target value
    for target in target_values:
        print(f"\n{'=' * 60}")
        print(f"Running HPO for target: {target}")
        print(f"{'=' * 60}")

        # Create unique study name and paths
        study_name = f"{property_name}_{dataset.value}_{gen_model_idx}_target{target:.2f}"
        target_dir = experiment_dir / f"target_{target:.2f}"
        target_dir.mkdir(exist_ok=True)

        db_path = target_dir / f"hpo_{study_name}.db"
        csv_path = target_dir / f"trials_{study_name}.csv"

        # Load or create study
        study = load_or_create_study(study_name, db_path, search_space)

        # Define objective
        def objective(trial: optuna.Trial) -> float:
            try:
                # Create config for this specific target
                specific_config = PropertyTargetingConfig(
                    dataset=dataset,
                    property_name=property_name,
                    target_value=target,
                    n_samples=n_samples,
                    gen_model_idx=gen_model_idx,
                    device=str(DEVICE),
                )

                results = run_property_targeting_trial(
                    trial=trial, config=specific_config, generator=generator, property_regressor=property_regressor
                )

                # Store all metrics as user attributes
                trial.set_user_attr("all_valid_validity", results.all_valid_validity)
                trial.set_user_attr("all_valid_mad", results.all_valid_mad)
                trial.set_user_attr("all_valid_n_samples", results.all_valid_n_samples)
                trial.set_user_attr("all_valid_uniqueness", results.all_valid_uniqueness)
                trial.set_user_attr("all_valid_novelty", results.all_valid_novelty)
                trial.set_user_attr("all_valid_diversity_p1", results.all_valid_diversity_p1)
                trial.set_user_attr("all_valid_cos_sim_mean", results.all_valid_cos_sim_mean)

                trial.set_user_attr("filter2_validity", results.filter2_validity)
                trial.set_user_attr("filter2_mad", results.filter2_mad)
                trial.set_user_attr("filter2_n_samples", results.filter2_n_samples)

                trial.set_user_attr("optimization_time", results.optimization_time)
                trial.set_user_attr("target", results.target)
                trial.set_user_attr("epsilon", results.epsilon)
                trial.set_user_attr("n_passed_latent_filter", results.n_passed_latent_filter)

                # Objective: Maximize valid count while minimizing MAD
                # Use All Valid set for objective (primary metric)
                n_valid = float(results.all_valid_n_samples)
                mad = results.all_valid_mad

                # Minimum viable sample threshold
                MIN_VALID = n_samples // 10
                if n_valid < MIN_VALID:
                    return -1000.0 + n_valid

                # Normalize MAD by epsilon for scale-invariance
                epsilon = results.epsilon
                normalized_mad = mad / epsilon if epsilon > 0 else mad

                # Composite objective
                alpha = 1.0  # Balance throughput vs accuracy
                objective_value = n_valid - alpha * normalized_mad

                return objective_value

            except Exception as e:
                print(f"Trial {trial.number} failed: {e}")
                traceback.print_exc()
                return -1000.0

        # Run optimization
        study.optimize(objective, n_trials=n_trials)

        # Export results
        export_trials_to_csv(study, csv_path)

        # Save best config
        best_trial = study.best_trial
        best_config = {
            "study_name": study_name,
            "best_value": best_trial.value,
            "best_params": best_trial.params,
            "best_user_attrs": best_trial.user_attrs,
            "n_trials": len(study.trials),
            "dataset": dataset.value,
            "property": property_name,
            "target_value": target,
            "gen_model_idx": gen_model_idx,
        }

        best_config_path = target_dir / f"best_config_{study_name}.json"
        best_config_path.write_text(json.dumps(best_config, indent=2, default=float))

        print(f"\nBest trial: {best_trial.number}")
        print(f"Best value: {best_trial.value:.4f}")
        print(f"Best params: {best_trial.params}")

    print(f"\n{'=' * 60}")
    print(f"HPO Complete! Results saved to: {experiment_dir}")
    print(f"{'=' * 60}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="HPO for property targeting (MG-DIFF protocol)")
    parser.add_argument(
        "--dataset",
        type=str,
        default=SupportedDataset.ZINC_SMILES_HRR_256_F64_5G1NG4.value,
        choices=[d.value for d in SupportedDataset],
        help="Dataset to use",
    )
    parser.add_argument(
        "--property",
        type=str,
        default="logp",
        choices=["logp", "qed", "sa_score", "tpsa"],
        help="Property to optimize",
    )
    parser.add_argument(
        "--targets",
        type=float,
        nargs="+",
        default=None,
        help="Target values (default: use MGDIFF_TARGETS)",
    )
    parser.add_argument(
        "--model_idx",
        type=int,
        default=0,
        help="Index of generator model in registry",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=1,
        help="Number of HPO trials per target",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=20,
        help="Number of samples per trial",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="hpo_results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Convert dataset string to enum
    dataset = SupportedDataset(args.dataset)

    # Get target values
    if args.targets is None:
        target_values = MGDIFF_TARGETS.get(dataset.default_cfg.base_dataset).get(args.property, [2.0, 4.0, 6.0])
    else:
        target_values = args.targets

    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run HPO
    run_hpo(
        dataset=dataset,
        property_name=args.property,
        target_values=target_values,
        gen_model_idx=args.model_idx,
        n_trials=args.n_trials,
        n_samples=args.n_samples,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
