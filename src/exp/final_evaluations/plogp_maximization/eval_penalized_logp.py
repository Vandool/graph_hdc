#!/usr/bin/env python
"""
(Corrected) HPO for Penalized LogP Maximization.

Optimizes molecular generation for maximum penalized LogP:
pLogP = LogP - SA_score - max(0, ring_size - 6)

Uses gradient-based latent space optimization with separate regressors
and a robust HPO objective.

Corrections:
- Replaced flawed 'targeting' loss with 'maximization' loss (-plogp_pred.mean()).
- Removed all 'target_plogp' and 'epsilon' logic from the optimization process.
- Wrapped in Optuna HPO to optimize lr, steps, regularizer weights, etc.
- HPO objective thresholds on absolute valid-unique count, then maximizes mean pLogP.
- Evaluation is performed on ALL valid generated molecules, not a filtered subset.
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
import torch
import torch.nn.functional as F
from lightning_fabric import seed_everything
from optuna_integration import BoTorchSampler
from torch import nn
from tqdm.auto import tqdm

from src.datasets.utils import get_dataset_props
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
from src.utils.chem import is_valid_molecule, reconstruct_for_eval_v2
from src.utils.registery import retrieve_model
from src.utils.utils import pick_device

# Default float32
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)
DEVICE = pick_device()

seed = 42
seed_everything(seed)


# ===== RDKit Ground Truth =====
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


# ===== HPO Search Space =====
def get_search_space() -> dict[str, optuna.distributions.BaseDistribution]:
    """Get HPO search space for Penalized LogP maximization."""
    return {
        "lr": optuna.distributions.FloatDistribution(5e-5, 5e-3, log=True),
        "steps": optuna.distributions.IntDistribution(100, 2000, log=True),
        "scheduler": optuna.distributions.CategoricalDistribution(["cosine", "two-phase"]),
        "lambda_lo": optuna.distributions.FloatDistribution(1e-5, 5e-3, log=True),
        "lambda_hi": optuna.distributions.FloatDistribution(5e-3, 5e-2, log=True),
        "lambda_diversity": optuna.distributions.FloatDistribution(1e-3, 1.0, log=True),
        "optimizer": optuna.distributions.CategoricalDistribution(["adam", "sgd"]),
        "grad_clip": optuna.distributions.FloatDistribution(0.5, 5.0, log=True),
    }


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
    device: str = "cuda"


@dataclass
class PenalizedLogPResults:
    """Results from pLogP maximization evaluation."""

    validity: float
    uniqueness: float
    novelty: float
    diversity_p1: float
    diversity_p2: float
    plogp_mean: float
    plogp_std: float
    plogp_min: float
    plogp_max: float
    n_samples: int  # This is n_valid
    property_stats: dict[str, dict[str, float]]
    correction_levels: dict[str, float]
    cos_sim_mean: float
    cos_sim_std: float
    optimization_time: float
    decoding_time: float
    total_time: float


# ===== pLogP Maximization Implementation =====
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

        # Move models to device
        self.gen_model.to(DEVICE).eval()
        self.logp_regressor.to(DEVICE).eval()
        self.sa_regressor.to(DEVICE).eval()
        self.ring_regressor.to(DEVICE).eval()
        self.hypernet.to(DEVICE).eval()

        base_dataset = config.dataset.default_cfg.base_dataset
        self.dataset_stats = DATASET_STATS[base_dataset]
        self.decoder_settings = DecoderSettings.get_default_for(base_dataset=base_dataset)
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

            # Calculate predicted penalized logP
            ring_penalty = torch.clamp(ring_pred - 6.0, min=0.0)
            plogp_pred = logp_pred - sa_pred - ring_penalty

            # --- 1. MAXIMIZATION LOSS ---
            # We minimize the negative mean pLogP
            plogp_loss = -plogp_pred.mean()

            # --- 2. PRIOR LOSS ---
            lam = scheduler(s)
            prior_loss = lam * z.pow(2).mean()

            # --- 3. DIVERSITY LOSS ---
            # Minimize pairwise cosine similarity of graph terms
            g_norm = F.normalize(g, p=2, dim=1)
            sim_matrix = torch.mm(g_norm, g_norm.t())
            mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
            similarities = sim_matrix[mask]
            diversity_loss = lambda_diversity * similarities.mean()

            # --- TOTAL LOSS ---
            loss = plogp_loss + prior_loss + diversity_loss

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_z = z.clone()

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([z], grad_clip)
            optimizer.step()

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "mean_pLogP": f"{plogp_pred.mean().item():.3f}", "λ": f"{lam:.2e}"}
            )

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
    ) -> tuple[PenalizedLogPResults, list[float]]:
        """
        Perform final evaluation on all valid molecules.
        Returns: (PenalizedLogPResults, plogp_list)
        """
        if not molecules:
            return self._empty_results(), []

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
            return self._empty_results(), []

        # Evaluate this final set
        eval_results_dict, plogp_list = self._evaluate_sample_set(
            valid_molecules,
            valid_similarities,
            valid_correction_levels,
            valid_final_flags,
        )

        combined_results = self._combine_results(eval_results_dict)
        return combined_results, plogp_list

    def _evaluate_sample_set(
        self,
        molecules: list,
        similarities: list,
        correction_levels: list,
        final_flags: list,
    ) -> tuple[dict[str, Any], list[float]]:
        """
        Evaluate a set of molecules.
        Returns (results_dict, plogp_list_for_valid_molecules)
        """
        eval_results = self.evaluator.evaluate(
            n_samples=len(molecules),
            samples=molecules,
            final_flags=final_flags,
            sims=similarities,
            correction_levels=correction_levels,
        )

        # Get RDKit molecules from evaluator (same as QED script)
        mols, valid_flags, _, _ = self.evaluator.get_mols_valid_flags_sims_and_correction_levels()
        valid_rdkit_mols = [m for m, f in zip(mols, valid_flags, strict=False) if f]

        # Calculate pLogP for all valid RDKit molecules
        plogp_list = [calculate_penalized_logp_rdkit(m) for m in valid_rdkit_mols]
        plogp_list = [p for p in plogp_list if p > -float("inf")]  # Filter out calculation errors

        # Re-calculate stats from the *actual* list
        plogp_mean = np.mean(plogp_list) if plogp_list else 0
        plogp_std = np.std(plogp_list) if plogp_list else 0
        plogp_min = np.min(plogp_list) if plogp_list else 0
        plogp_max = np.max(plogp_list) if plogp_list else 0

        correction_stats = self._analyze_correction_levels(correction_levels)

        results_dict = {
            "n_samples": len(valid_rdkit_mols),  # n_samples is NOW n_valid
            "validity": eval_results["validity"],
            "uniqueness": eval_results["uniqueness"],
            "novelty": eval_results["novelty"],
            "diversity_p1": eval_results["internal_diversity_p1"],
            "diversity_p2": eval_results["internal_diversity_p2"],
            "plogp_mean": plogp_mean,
            "plogp_std": plogp_std,
            "plogp_min": plogp_min,
            "plogp_max": plogp_max,
            "property_stats": {
                "plogp": {"mean": plogp_mean, "std": plogp_std},
                "logp": {"mean": eval_results.get("logp_mean", 0), "std": eval_results.get("logp_std", 0)},
                "sa_score": {"mean": eval_results.get("sa_score_mean", 0), "std": eval_results.get("sa_score_std", 0)},
            },
            "correction_levels": correction_stats,
            "cos_sim": eval_results["cos_sim"],
            "final_flags_pct": eval_results["final_flags"],
        }
        return results_dict, plogp_list

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

    def _empty_results(self) -> PenalizedLogPResults:
        """Return empty results structure."""
        return PenalizedLogPResults(
            validity=0,
            uniqueness=0,
            novelty=0,
            diversity_p1=0,
            diversity_p2=0,
            plogp_mean=0,
            plogp_std=0,
            plogp_min=0,
            plogp_max=0,
            n_samples=0,
            property_stats={},
            correction_levels={},
            cos_sim_mean=0,
            cos_sim_std=0,
            optimization_time=0,
            decoding_time=0,
            total_time=0,
        )

    def _combine_results(self, eval_results: dict[str, Any]) -> PenalizedLogPResults:
        """Combine evaluation results into final structure."""
        return PenalizedLogPResults(
            validity=eval_results["validity"],
            uniqueness=eval_results["uniqueness"],
            novelty=eval_results["novelty"],
            diversity_p1=eval_results["diversity_p1"],
            diversity_p2=eval_results["diversity_p2"],
            plogp_mean=eval_results["plogp_mean"],
            plogp_std=eval_results["plogp_std"],
            plogp_min=eval_results["plogp_min"],
            plogp_max=eval_results["plogp_max"],
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
def plot_plogp_distribution(
    generated_plogp: list[float],
    dataset_plogp: np.ndarray,
    save_dir: pathlib.Path,
):
    """Plot pLogP distributions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if generated_plogp:
        ax.hist(
            generated_plogp,
            bins=50,
            alpha=0.7,
            label=f"Generated (n={len(generated_plogp)})",
            color="blue",
            density=True,
            range=(-10, 20),
        )

    if dataset_plogp is not None and len(dataset_plogp) > 0:
        ax.hist(
            dataset_plogp,
            bins=50,
            alpha=0.5,
            label=f"Dataset (n={len(dataset_plogp)})",
            color="gray",
            density=True,
            range=(-10, 20),
        )

    ax.set_xlabel("Penalized LogP", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Penalized LogP Maximization Distribution", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-10, 20)

    plt.tight_layout()
    plt.savefig(save_dir / "plogp_distribution.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_optimization_history(losses: list[float], save_dir: pathlib.Path):
    """Plot optimization loss history."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses, linewidth=2)
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Loss")
    ax.set_title("pLogP Maximization Optimization History")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")  # Loss is negative, so log scale might fail. Let's use symlog.
    ax.set_yscale("symlog")
    plt.tight_layout()
    plt.savefig(save_dir / "optimization_history.pdf", dpi=300, bbox_inches="tight")
    plt.close()


# ===== HPO Management =====
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
    df = study.trials_dataframe(attrs=("value", "state", "params", "user_attrs"))
    df.to_csv(csv_path, index=False)
    print(f"Exported {len(df)} trials to {csv_path}")


# ===== Main Execution =====
def run_penalized_logp_trial(
    trial: optuna.Trial,
    config: PenalizedLogPConfig,
    generator: HDCGenerator,
    regressors: dict[str, nn.Module],
) -> tuple[PenalizedLogPResults, list[float], list[float]]:
    """
    Run a single HPO trial.
    Returns: (Results, plogp_list, optimization_losses)
    """
    # Sample hyperparameters
    lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    steps = trial.suggest_int("steps", 100, 2000, log=True)
    scheduler = trial.suggest_categorical("scheduler", ["cosine", "two-phase"])
    lambda_lo = trial.suggest_float("lambda_lo", 1e-5, 5e-3, log=True)
    lambda_hi = trial.suggest_float("lambda_hi", 5e-3, 5e-2, log=True)
    lambda_diversity = trial.suggest_float("lambda_diversity", 1e-3, 1.0, log=True)
    optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    grad_clip = trial.suggest_float("grad_clip", 0.5, 5.0, log=True)

    optimizer_obj = PenalizedLogPOptimizer(generator=generator, regressors=regressors, config=config)

    start_time = time.time()
    opt_results = optimizer_obj.optimize_latent(
        lr=lr,
        steps=steps,
        scheduler_name=scheduler,
        lambda_lo=lambda_lo,
        lambda_hi=lambda_hi,
        lambda_diversity=lambda_diversity,
        optimizer_name=optimizer,
        grad_clip=grad_clip,
    )
    optimization_time = time.time() - start_time

    decode_start = time.time()
    results, plogp_list = optimizer_obj.evaluate_and_get_results(
        molecules=opt_results["molecules"],
        similarities=opt_results["similarities"],
        correction_levels=opt_results["correction_levels"],
        final_flags=opt_results["final_flags"],
    )
    decoding_time = time.time() - decode_start

    results.optimization_time = optimization_time
    results.decoding_time = decoding_time
    results.total_time = optimization_time + decoding_time

    return results, plogp_list, opt_results["optimization_losses"]


def run_hpo(
    dataset: SupportedDataset,
    n_trials: int,
    n_samples: int,
    output_dir: pathlib.Path,
):
    """Run HPO for Penalized LogP maximization."""
    device = pick_device()

    # --- 1. Load Generator ---
    gen_model_hint = GENERATOR_REGISTRY[dataset][0]
    generator = HDCGenerator(
        gen_model_hint=gen_model_hint,
        ds_config=dataset.default_cfg,
        device=device,
        decoder_settings=DecoderSettings.get_default_for(dataset.default_cfg.base_dataset),
    )

    # --- 2. Load Regressors (All 3) ---
    regressor_hints = REGRESSOR_REGISTRY[dataset]
    required_props = ["logp", "sa_score", "max_ring_size"]
    if not all(prop in regressor_hints for prop in required_props):
        raise ValueError(f"Missing regressor hints for {dataset.value}. Need: {required_props}")

    regressors = {}
    print("Loading regressors...")
    for prop in required_props:
        hint = regressor_hints[prop][0]  # Use first available
        pr_path = get_pr_path(hint=hint)
        regressors[prop] = retrieve_model(name="PR").load_from_checkpoint(pr_path).to(device).eval()
        print(f"  {prop}: {pr_path.stem}")

    # --- 3. Get Dataset Stats for Plotting ---
    base_dataset = dataset.default_cfg.base_dataset
    props = get_dataset_props(base_dataset=base_dataset, splits=["train"])
    dataset_plogp = np.array(props.pen_logp)

    print(f"\n{'=' * 60}")
    print("Running Penalized LogP Maximization")
    print(
        f"Dataset pLogP stats: mean={dataset_plogp.mean():.4f}, std={dataset_plogp.std():.4f}, max={dataset_plogp.max():.4f}"
    )
    print(f"{'=' * 60}\n")

    # --- 4. Setup HPO ---
    search_space = get_search_space()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_dir / f"{dataset.value}_plogp_maximization_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    experiment_metadata = {
        "dataset": dataset.value,
        "property": "plogp_maximization",
        "gen_model_hint": gen_model_hint,
        "regressor_hints": {k: regressor_hints[k][0] for k in required_props},
        "n_trials": n_trials,
        "n_samples": n_samples,
        "timestamp": timestamp,
        "device": str(device),
    }
    (experiment_dir / "experiment_metadata.json").write_text(json.dumps(experiment_metadata, indent=2))

    study_name = f"plogp_maximization_{dataset.value}"
    db_path = experiment_dir / f"hpo_{study_name}.db"
    csv_path = experiment_dir / f"trials_{study_name}.csv"
    study = load_or_create_study(study_name, db_path, search_space)
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    trial_plot_storage = {}

    # --- 5. Define Objective ---
    def objective(trial: optuna.Trial) -> float:
        try:
            config = PenalizedLogPConfig(dataset=dataset, n_samples=n_samples, device=str(device))

            results, plogp_list, losses = run_penalized_logp_trial(
                trial=trial, config=config, generator=generator, regressors=regressors
            )

            trial_plot_storage[trial.number] = {
                "losses": losses,
                "generated_plogp": plogp_list,
            }

            # Store all metrics as user attributes
            trial.set_user_attr("validity", results.validity)
            trial.set_user_attr("uniqueness", results.uniqueness)
            trial.set_user_attr("novelty", results.novelty)
            trial.set_user_attr("plogp_mean", results.plogp_mean)
            trial.set_user_attr("plogp_std", results.plogp_std)
            trial.set_user_attr("plogp_max", results.plogp_max)
            trial.set_user_attr("n_samples", results.n_samples)  # n_valid
            trial.set_user_attr("optimization_time", results.optimization_time)

            # --- ROBUST HPO OBJECTIVE ---
            n_valid = float(results.n_samples)
            uniqueness_frac = results.uniqueness / 100.0
            mean_plogp = results.plogp_mean

            n_valid_unique = n_valid * uniqueness_frac

            MIN_VALID_UNIQUE_COUNT = n_samples // 10

            if n_valid_unique < MIN_VALID_UNIQUE_COUNT:
                return -1000.0 + n_valid_unique

            # Primary objective: maximize mean pLogP
            return mean_plogp

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            traceback.print_exc()
            return -1e9

        finally:
            # Clean up memory after the trial is evaluated
            if trial.number in trial_plot_storage:
                if len(study.best_trials) == 0 or trial.number != study.best_trial.number:
                    del trial_plot_storage[trial.number]

    # --- 6. Run HPO ---
    study.optimize(objective, n_trials=n_trials)
    export_trials_to_csv(study, csv_path)

    # --- 7. Save Best Trial ---
    best_trial = study.best_trial
    best_config = {
        "study_name": study_name,
        "best_value": best_trial.value,
        "best_params": best_trial.params,
        "best_user_attrs": best_trial.user_attrs,
        "n_trials": len(study.trials),
        "dataset": dataset.value,
        "property": "plogp_maximization",
    }
    (experiment_dir / f"best_config_{study_name}.json").write_text(json.dumps(best_config, indent=2, default=float))

    print(f"\nBest trial: {best_trial.number}")
    print(f"Best value (Objective): {best_trial.value:.6f}")
    print(f"Best params: {best_trial.params}")
    print(f"Best pLogP mean: {best_trial.user_attrs.get('plogp_mean', 0):.4f}")
    print(f"Best pLogP max: {best_trial.user_attrs.get('plogp_max', 0):.4f}")
    print(f"Best N Valid: {best_trial.user_attrs.get('n_samples', 0):.0f}")

    # --- 8. Plot Best Trial ---
    if best_trial.number in trial_plot_storage:
        best_data = trial_plot_storage[best_trial.number]

        print("\nGenerating optimization history plot...")
        plot_optimization_history(best_data["losses"], plots_dir)

        print("Generating pLogP distribution plot...")
        plot_plogp_distribution(
            generated_plogp=best_data["generated_plogp"],
            dataset_plogp=dataset_plogp,
            save_dir=plots_dir,
        )
        print(f"Plots saved to: {plots_dir}")
    else:
        print(f"Warning: Best trial {best_trial.number} plot data not found.")


def main():
    parser = argparse.ArgumentParser(description="HPO for Penalized LogP maximization")
    parser.add_argument(
        "--dataset",
        type=str,
        default=SupportedDataset.ZINC_SMILES_HRR_256_F64_5G1NG4.value,
        choices=[ds.value for ds in SupportedDataset],
        help="Dataset to use",
    )
    parser.add_argument("--n_trials", type=int, default=2, help="Number of HPO trials")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples per trial")
    parser.add_argument("--output_dir", type=str, default="hpo_results", help="Output directory")

    args = parser.parse_args()

    dataset = SupportedDataset(args.dataset)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_hpo(
        dataset=dataset,
        n_trials=args.n_trials,
        n_samples=args.n_samples,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
