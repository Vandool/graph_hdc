#!/usr/bin/env python
"""
(Corrected & Simplified) QED Maximization HPO.

This script performs hyperparameter optimization for gradient-based QED maximization.
The goal is to maximize the mean QED of the generated molecules.

Corrections from previous version:
- Removed all 'targeting' logic (target_qed, epsilon).
- Removed the "Gaussian weighted" evaluation, which is
  nonsensical for a pure maximization task.
- HPO objective is robust: thresholds on absolute valid-unique count,
  then maximizes mean QED.
- Plotting uses the *actual* raw QED values from the best trial.
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

import matplotlib.pyplot as plt
import normflows as nf
import numpy as np
import optuna
import pandas as pd
import torch
from lightning_fabric import seed_everything
from optuna_integration import BoTorchSampler
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


# ===== HPO Search Space =====
def get_search_space() -> dict[str, optuna.distributions.BaseDistribution]:
    """Get literature-informed HPO search space for QED maximization."""
    return {
        "lr": optuna.distributions.FloatDistribution(5e-5, 5e-3, log=True),
        "steps": optuna.distributions.IntDistribution(100, 2000, log=True),
        "scheduler": optuna.distributions.CategoricalDistribution(["cosine", "two-phase", "linear", "constant"]),
        "lambda_lo": optuna.distributions.FloatDistribution(1e-5, 5e-3, log=True),
        "lambda_hi": optuna.distributions.FloatDistribution(5e-3, 5e-2, log=True),
        "grad_clip": optuna.distributions.FloatDistribution(0.5, 5.0, log=True),
    }


# ===== Scheduler Classes (Unchanged) =====
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

    # All 'gaussian_' fields have been REMOVED.

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

    # Optimization metrics
    optimization_time: float
    decoding_time: float
    total_time: float


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

        # Decode ALL molecules. No filtering.
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
    ) -> tuple[QEDMaximizationResults, list[float]]:
        """
        Perform final evaluation.
        REMOVED all dual-evaluation logic.
        Returns:
            (QEDMaximizationResults, qed_list)
        """
        if not molecules:
            return self._empty_results(), []

        # Convert nx graphs to PyG batch for re-encoding
        # This step IS the "Round 2" validation (checking for valid re-encodable graphs)
        pyg_graphs = []
        valid_indices = []
        for i, g in enumerate(molecules):
            try:
                pyg_g = DataTransformer.nx_to_pyg(g)
                pyg_graphs.append(pyg_g)
                valid_indices.append(i)
            except Exception:
                continue

        if not pyg_graphs:
            return self._empty_results(), []

        # Filter to only the structurally valid/re-encodable molecules
        valid_molecules = [molecules[i] for i in valid_indices]
        valid_similarities = [similarities[i] for i in valid_indices]
        valid_correction_levels = [correction_levels[i] for i in valid_indices]
        valid_final_flags = [final_flags[i] for i in valid_indices]

        # Evaluate this final set
        eval_results_dict, qed_list = self._evaluate_sample_set(
            valid_molecules,
            valid_similarities,
            valid_correction_levels,
            valid_final_flags,
        )

        # Combine results into the final data class
        combined_results = self._combine_results(eval_results_dict)

        return combined_results, qed_list

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

        # Get the actual list of QEDs for valid molecules for plotting
        prop_fn = PROPERTY_FUNCTIONS["qed"]

        mols, valid_flags, _, _ = self.evaluator.get_mols_valid_flags_sims_and_correction_levels()
        valid_molecules = [m for m, f in zip(mols, valid_flags, strict=False) if f]

        qed_list = [prop_fn(m) for m in valid_molecules]

        # Re-calculate QED stats from the *actual* list
        qed_mean = np.mean(qed_list) if qed_list else 0
        qed_std = np.std(qed_list) if qed_list else 0
        qed_min = np.min(qed_list) if qed_list else 0
        qed_max = np.max(qed_list) if qed_list else 0

        correction_stats = self._analyze_correction_levels(correction_levels)

        results_dict = {
            "n_samples": len(valid_molecules),  # n_samples is NOW n_valid
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
            optimization_time=0,
            decoding_time=0,
            total_time=0,
        )

    def _combine_results(self, eval_results: dict[str, Any]) -> QEDMaximizationResults:
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
    """
    Plot QED distributions.
    REMOVED gaussian_qed.
    """
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


# ===== HPO Management (Unchanged) =====
def load_or_create_study(study_name: str, db_path: pathlib.Path, search_space: dict) -> optuna.Study:
    storage = f"sqlite:///{db_path}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"Loaded existing study: {study_name} with {len(study.trials)} trials")
    except KeyError:
        study = optuna.create_study(
            study_name=study_name, direction="maximize", storage=storage, sampler=BoTorchSampler(seed=42)
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
def run_qed_maximization_trial(
    trial: optuna.Trial,
    config: QEDMaximizationConfig,
    generator: HDCGenerator,
    qed_regressor: nn.Module,
) -> tuple[QEDMaximizationResults, list[float], list[float]]:
    """
    Run a single HPO trial.
    Returns: (Results, qed_list, optimization_losses)
    """
    # Sample hyperparameters
    lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    steps = trial.suggest_int("steps", 100, 2000, log=True)
    scheduler = trial.suggest_categorical("scheduler", ["cosine", "two-phase", "linear", "constant"])
    lambda_lo = trial.suggest_float("lambda_lo", 1e-5, 5e-3, log=True)
    lambda_hi = trial.suggest_float("lambda_hi", 5e-3, 5e-2, log=True)
    grad_clip = trial.suggest_float("grad_clip", 0.5, 5.0, log=True)

    optimizer_obj = QEDMaximizationOptimizer(generator=generator, qed_regressor=qed_regressor, config=config)

    start_time = time.time()
    opt_results = optimizer_obj.optimize_latent(
        lr=lr,
        steps=steps,
        scheduler_name=scheduler,
        lambda_lo=lambda_lo,
        lambda_hi=lambda_hi,
        grad_clip=grad_clip,
    )
    optimization_time = time.time() - start_time

    decode_start = time.time()
    results, qed_list = optimizer_obj.evaluate_and_get_results(
        molecules=opt_results["molecules"],
        similarities=opt_results["similarities"],
        correction_levels=opt_results["correction_levels"],
        final_flags=opt_results["final_flags"],
    )
    decoding_time = time.time() - decode_start

    results.optimization_time = optimization_time
    results.decoding_time = decoding_time
    results.total_time = optimization_time + decoding_time

    return results, qed_list, opt_results["optimization_losses"]


def run_hpo(
    dataset: SupportedDataset,
    gen_model_idx: int,
    n_trials: int,
    n_samples: int,
    output_dir: pathlib.Path,
):
    """Run HPO for QED maximization."""
    device = pick_device()

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

    print(f"\n{'=' * 60}")
    print("Running QED Maximization")
    print(f"Dataset QED stats: mean={ds.qed.mean():.4f}, std={ds.qed.std():.4f}, max={ds.qed.max():.6f}")
    print(f"{'=' * 60}\n")
    dataset_stats_dict = DATASET_STATS.get(base_dataset, {})

    search_space = get_search_space()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_dir / f"{dataset.value}_qed_maximization_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    experiment_metadata = {
        "dataset": dataset.value,
        "property": "qed_maximization",
        "gen_model_hint": gen_model_hint,
        "gen_model_idx": gen_model_idx,
        "regressor_hint": regressor_hints[0],
        "n_trials": n_trials,
        "n_samples": n_samples,
        "timestamp": timestamp,
        "device": str(device),
    }
    (experiment_dir / "experiment_metadata.json").write_text(json.dumps(experiment_metadata, indent=2))

    study_name = f"qed_maximization_{dataset.value}_{gen_model_idx}"
    db_path = experiment_dir / f"hpo_{study_name}.db"
    csv_path = experiment_dir / f"trials_{study_name}.csv"
    study = load_or_create_study(study_name, db_path, search_space)
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Storage for best trial *plotting data*
    trial_plot_storage = {}

    def objective(trial: optuna.Trial) -> float:
        try:
            config = QEDMaximizationConfig(
                dataset=dataset, n_samples=n_samples, gen_model_idx=gen_model_idx, device=str(device)
            )

            results, qed_list, losses = run_qed_maximization_trial(
                trial=trial, config=config, generator=generator, qed_regressor=qed_regressor
            )

            # Store plotting data in memory *only for this trial*
            trial_plot_storage[trial.number] = {
                "losses": losses,
                "generated_qed": qed_list,
            }

            # --- Store all metrics as user attributes ---
            trial.set_user_attr("validity", results.validity)
            trial.set_user_attr("uniqueness", results.uniqueness)
            trial.set_user_attr("novelty", results.novelty)
            trial.set_user_attr("qed_mean", results.qed_mean)
            trial.set_user_attr("qed_std", results.qed_std)
            trial.set_user_attr("qed_max", results.qed_max)
            trial.set_user_attr("n_samples", results.n_samples)  # n_valid
            trial.set_user_attr("optimization_time", results.optimization_time)
            # All 'gaussian_' attributes REMOVED

            # --- ROBUST HPO OBJECTIVE ---

            # 1. Get key metrics
            n_valid = float(results.n_samples)
            uniqueness_frac = results.uniqueness / 100.0
            mean_qed = results.qed_mean

            # 2. Calculate the *absolute number* of valid, unique molecules
            n_valid_unique = n_valid * uniqueness_frac

            # 3. Apply a meaningful "throughput" threshold
            MIN_VALID_UNIQUE_COUNT = n_samples // 10

            if n_valid_unique < MIN_VALID_UNIQUE_COUNT:
                # Penalize, but provide a gradient for the optimizer
                return -1000.0 + n_valid_unique

            # 4. If throughput is met, optimize for the real goal: mean QED
            return mean_qed

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            traceback.print_exc()
            return -1e9

        finally:
            # Clean up memory after the trial is evaluated
            if trial.number in trial_plot_storage:
                if len(study.best_trials) == 0 or trial.number != study.best_trial.number:
                    del trial_plot_storage[trial.number]

    study.optimize(objective, n_trials=n_trials)
    export_trials_to_csv(study, csv_path)

    best_trial = study.best_trial
    best_config = {
        "study_name": study_name,
        "best_value": best_trial.value,
        "best_params": best_trial.params,
        "best_user_attrs": best_trial.user_attrs,
        "n_trials": len(study.trials),
        "dataset": dataset.value,
        "property": "qed_maximization",
        "gen_model_idx": gen_model_idx,
    }

    (experiment_dir / f"best_config_{study_name}.json").write_text(json.dumps(best_config, indent=2, default=float))

    print(f"\nBest trial: {best_trial.number}")
    print(f"Best value (Objective): {best_trial.value:.6f}")
    print(f"Best params: {best_trial.params}")
    print(f"Best QED mean: {best_trial.user_attrs.get('qed_mean', 0):.4f}")
    print(f"Best QED max: {best_trial.user_attrs.get('qed_max', 0):.4f}")
    print(f"Best N Valid: {best_trial.user_attrs.get('n_samples', 0):.0f}")

    # --- CORRECT PLOTTING ---
    if best_trial.number in trial_plot_storage:
        best_data = trial_plot_storage[best_trial.number]

        print("\nGenerating optimization history plot...")
        plot_optimization_history(best_data["losses"], plots_dir)

        print("Generating QED distribution plot...")
        plot_qed_distribution(
            generated_qed=best_data["generated_qed"],
            dataset_stats=dataset_stats_dict,
            save_dir=plots_dir,
        )
        print(f"Plots saved to: {plots_dir}")
    else:
        print(f"Warning: Best trial {best_trial.number} plot data not found.")


def main():
    parser = argparse.ArgumentParser(description="HPO for QED maximization")
    parser.add_argument(
        "--dataset",
        type=str,
        default=SupportedDataset.ZINC_SMILES_HRR_256_F64_5G1NG4.value,
        choices=[d.value for d in SupportedDataset],
        help="Dataset to use",
    )
    parser.add_argument("--model_idx", type=int, default=0, help="Index of generator model in registry")
    parser.add_argument("--n_trials", type=int, default=1, help="Number of HPO trials")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples per trial")
    parser.add_argument("--output_dir", type=str, default="hpo_results", help="Output directory")

    args = parser.parse_args()
    dataset = SupportedDataset(args.dataset)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_hpo(
        dataset=dataset,
        gen_model_idx=args.model_idx,
        n_trials=args.n_trials,
        n_samples=args.n_samples,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
