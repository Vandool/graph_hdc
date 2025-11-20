#!/usr/bin/env python
"""
Similarity-Constrained pLogP Final Evaluation with Dual Reporting.

This script performs final evaluation with proper aggregation for both:
- Task 2 (GuacaMol/Unconstrained): Global pooling of all 80,000 candidates
- Task 4 (Lead Optimization/Constrained): Per-starting-point metrics

Default configuration: 800 starting molecules × 100 samples = 80,000 total candidates
"""

import argparse
import json
import pathlib
import time
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning_fabric import seed_everything
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from torch import nn
from tqdm.auto import tqdm

from src.datasets.utils import get_split
from src.encoding.configs_and_constants import DecoderSettings, SupportedDataset
from src.exp.final_evaluations.models_configs_constants import (
    GENERATOR_REGISTRY,
    REGRESSOR_REGISTRY,
    get_pr_path,
)
from src.generation.evaluator import (
    rdkit_logp,
    rdkit_max_ring_size,
    rdkit_qed,
    rdkit_sa_score,
)
from src.generation.generation import HDCGenerator
from src.utils.chem import is_valid_molecule, reconstruct_for_eval_v2
from src.utils.nx_utils import reconstruct_for_eval
from src.utils.registery import retrieve_model
from src.utils.utils import pick_device

# Default float32
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)
DEVICE = pick_device()

seed = 42
seed_everything(seed)


# ===== Ground Truth Property Functions =====
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


def calculate_tanimoto_similarity(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """Calculate Tanimoto similarity between two molecules."""
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


PROPERTY_FUNCTIONS = {
    "logp": rdkit_logp,
    "qed": rdkit_qed,
    "sa_score": rdkit_sa_score,
    "penalized_logp": calculate_penalized_logp_rdkit,
}


# ===== Data Classes =====
@dataclass
class ConstrainedOptimizationConfig:
    """Configuration for constrained optimization experiment."""

    dataset: SupportedDataset
    objective: str
    n_samples: int  # Candidates per starting molecule
    gen_model_idx: int
    similarity_threshold: float
    device: str = "cuda"


@dataclass
class DualConstrainedResults:
    """Results with dual reporting for both evaluation protocols."""

    # Task 2: GuacaMol/Unconstrained (Global Pooling)
    guacamol_score: float
    global_top1_plogp: float
    global_top10_mean: float
    global_top100_mean: float
    global_n_candidates: int
    global_n_unique: int

    # Task 4: Lead Optimization/Constrained (Per-Starter)
    mean_improvement: float
    std_improvement: float
    median_improvement: float
    max_improvement: float
    success_rate: float  # % of starters with at least 1 valid improvement
    n_successful_starters: int

    # Metadata
    n_starters: int
    n_samples_per_starter: int
    similarity_threshold: float
    total_time: float


# ===== Best Trial Loading =====
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
        "lr": best_trial["lr"],
        "steps": int(best_trial["steps"]),
        "lambda_prior": best_trial["lambda_prior"],
        "lambda_similarity": best_trial["lambda_similarity"],
        "proxy_similarity_threshold": best_trial["proxy_similarity_threshold"],
        "grad_clip": best_trial["grad_clip"],
        "trial_number": int(best_trial["number"]),
        "objective_value": best_trial["value"],
        "mean_improvement": best_trial.get("mean_improvement", 0),
        "success_rate": best_trial.get("success_rate", 0),
    }


# ===== Starting Molecule Loading =====
def load_starting_molecules(dataset: SupportedDataset, hypernet, n_starters: int) -> list:
    """
    Load starting molecules from test set for final evaluation.

    Returns list of dicts with:
        - mol: RDKit molecule
        - graph_term: PyTorch tensor
        - smiles: SMILES string
        - property: Starting property value
    """
    base_dataset = dataset.default_cfg.base_dataset
    test_split = get_split(base_dataset=base_dataset, which="test")

    # Ensure we have enough molecules
    if len(test_split) < n_starters:
        print(f"Warning: Test set has only {len(test_split)} molecules, using all")
        n_starters = len(test_split)

    # Take first n_starters from test set (for reproducibility)
    starting_mols = []

    for i in range(n_starters):
        data = test_split[i]
        try:
            nx_graph = data.to_networkx().to_undirected()
            mol = reconstruct_for_eval_v2(nx_graph, dataset=base_dataset)

            if not mol or not is_valid_molecule(mol):
                continue

            # Encode to get graph term
            pyg_graph = data.to(DEVICE)
            with torch.no_grad():
                encoding_output = hypernet.forward([pyg_graph])
                graph_term = encoding_output["graph_embedding"][0]

            starting_mols.append(
                {
                    "mol": mol,
                    "graph_term": graph_term,
                    "smiles": Chem.MolToSmiles(mol, canonical=True),
                    "property": calculate_penalized_logp_rdkit(mol),
                }
            )

            if len(starting_mols) >= n_starters:
                break

        except Exception as e:
            print(f"Failed to process test molecule {i}: {e}")
            continue

    print(f"Loaded {len(starting_mols)} starting molecules from test set")
    return starting_mols


# ===== Optimization Implementation =====
class ConstrainedOptimizer:
    """Handles gradient-based constrained optimization."""

    def __init__(
        self,
        generator: HDCGenerator,
        regressors: dict[str, nn.Module],
        config: ConstrainedOptimizationConfig,
    ):
        self.generator = generator
        self.gen_model = generator.gen_model
        self.regressors = regressors
        self.hypernet = generator.hypernet
        self.config = config
        self.objective_fn = PROPERTY_FUNCTIONS[config.objective]

        # Move models to device
        self.gen_model.to(DEVICE).eval()
        for model in self.regressors.values():
            model.to(DEVICE).eval()
        self.hypernet.to(DEVICE).eval()

    def _calculate_objective(self, hdc: torch.Tensor) -> torch.Tensor:
        """Helper to calculate the objective value from regressors."""
        if self.config.objective == "penalized_logp":
            logp_pred = self.regressors["logp"].gen_forward(hdc)
            sa_pred = self.regressors["sa_score"].gen_forward(hdc)
            ring_pred = self.regressors["max_ring_size"].gen_forward(hdc)
            ring_penalty = torch.clamp(ring_pred - 6.0, min=0.0)
            return logp_pred - sa_pred - ring_penalty
        return self.regressors[self.config.objective].gen_forward(hdc).squeeze()

    def optimize_from_start_mol(
        self, start_mol: Chem.Mol, start_graph_term: torch.Tensor, hps: dict[str, Any], base_dataset: str
    ) -> dict[str, Any]:
        """
        Optimize candidates from a single starting molecule.

        Returns:
            dict with:
                - start_smiles: str
                - start_property: float
                - candidates: list of dicts with smiles, property, similarity, improvement, is_valid
        """
        n_candidates = self.config.n_samples
        start_property = self.objective_fn(start_mol)

        latent_dim = self.gen_model.flat_dim
        z = torch.randn(n_candidates, latent_dim, device=DEVICE, dtype=DTYPE, requires_grad=True)

        optimizer = torch.optim.Adam([z], lr=hps["lr"])

        for step in range(hps["steps"]):
            optimizer.zero_grad()

            x = self.gen_model.decode_from_latent(z)
            _, graph_terms = self.gen_model.split(x)

            # 1. Objective Loss (Maximize)
            property_pred = self._calculate_objective(x)
            property_loss = -property_pred.mean()

            # 2. Similarity Loss (Constraint)
            similarities = torch.nn.functional.cosine_similarity(
                graph_terms, start_graph_term.unsqueeze(0).expand(n_candidates, -1), dim=1
            )
            similarity_penalty = torch.mean(torch.relu(hps["proxy_similarity_threshold"] - similarities))

            # 3. Prior Loss
            prior_loss = torch.mean(z**2)

            # Total loss
            loss = property_loss + hps["lambda_similarity"] * similarity_penalty + hps["lambda_prior"] * prior_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_([z], hps["grad_clip"])
            optimizer.step()

        with torch.no_grad():
            x = self.gen_model.decode_from_latent(z)
            edge_terms, graph_terms = self.gen_model.split(x)

        decoded = self.generator.decode(edge_terms=edge_terms, graph_terms=graph_terms)

        candidates = []
        for i, graph in enumerate(decoded["graphs"]):
            if not graph:
                continue
            try:
                mol = reconstruct_for_eval(graph, dataset=base_dataset)
                if not mol or not is_valid_molecule(mol):
                    continue

                actual_similarity = calculate_tanimoto_similarity(mol, start_mol)
                actual_property = self.objective_fn(mol)

                is_valid_candidate = (
                    actual_similarity >= self.config.similarity_threshold and actual_property > start_property
                )

                candidates.append(
                    {
                        "smiles": Chem.MolToSmiles(mol, canonical=True),
                        "property": actual_property,
                        "similarity": actual_similarity,
                        "improvement": actual_property - start_property,
                        "is_valid": is_valid_candidate,
                    }
                )
            except Exception:
                continue

        return {
            "start_smiles": Chem.MolToSmiles(start_mol),
            "start_property": start_property,
            "candidates": candidates,
        }


# ===== Dual Evaluation Functions =====
def calculate_guacamol_metrics(all_candidates: list[dict]) -> dict:
    """
    Task 2: GuacaMol-style global pooling.

    Pool all candidates, sort globally, take top 100.
    """
    if not all_candidates:
        return {
            "guacamol_score": 0.0,
            "top1": 0.0,
            "top10_mean": 0.0,
            "top100_mean": 0.0,
            "n_candidates": 0,
            "n_unique": 0,
        }

    # Get unique molecules by SMILES
    unique_mols = {}
    for c in all_candidates:
        smi = c["smiles"]
        if smi not in unique_mols or c["property"] > unique_mols[smi]["property"]:
            unique_mols[smi] = c

    # Sort globally by property (descending)
    sorted_candidates = sorted(unique_mols.values(), key=lambda x: x["property"], reverse=True)

    # Take top 100
    top100 = sorted_candidates[:100]

    # Calculate GuacaMol score
    top1 = top100[0]["property"] if len(top100) >= 1 else 0.0
    top10_mean = (
        np.mean([c["property"] for c in top100[:10]]) if len(top100) >= 10 else np.mean([c["property"] for c in top100])
    )
    top100_mean = np.mean([c["property"] for c in top100]) if top100 else 0.0
    guacamol_score = (top1 + top10_mean + top100_mean) / 3.0

    return {
        "guacamol_score": guacamol_score,
        "top1": top1,
        "top10_mean": top10_mean,
        "top100_mean": top100_mean,
        "n_candidates": len(all_candidates),
        "n_unique": len(unique_mols),
        "top100_molecules": [{"smiles": c["smiles"], "plogp": c["property"]} for c in top100],
    }


def calculate_lead_opt_metrics(per_starter_results: list[dict]) -> dict:
    """
    Task 4: Lead optimization per-starting-point metrics.

    For each starter, find best valid improvement.
    Report mean, std, success rate across all starters.
    """
    improvements = []
    successful_starters = 0

    for result in per_starter_results:
        valid_candidates = [c for c in result["candidates"] if c["is_valid"]]

        if valid_candidates:
            best_improvement = max(c["improvement"] for c in valid_candidates)
            improvements.append(best_improvement)
            successful_starters += 1
        else:
            improvements.append(0.0)

    if not improvements:
        return {
            "mean_improvement": 0.0,
            "std_improvement": 0.0,
            "median_improvement": 0.0,
            "max_improvement": 0.0,
            "success_rate": 0.0,
            "n_successful": 0,
        }

    return {
        "mean_improvement": float(np.mean(improvements)),
        "std_improvement": float(np.std(improvements)),
        "median_improvement": float(np.median(improvements)),
        "max_improvement": float(np.max(improvements)),
        "success_rate": 100.0 * successful_starters / len(per_starter_results),
        "n_successful": successful_starters,
    }


# ===== Plotting Functions =====
def plot_dual_distributions(guacamol_results: dict, lead_opt_results: dict, save_dir: pathlib.Path):
    """Plot distributions for both evaluation modes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Task 2: Global top 100 distribution
    if guacamol_results["top100_molecules"]:
        top100_plogps = [m["plogp"] for m in guacamol_results["top100_molecules"]]
        ax1.hist(top100_plogps, bins=30, alpha=0.7, color="blue", edgecolor="black")
        ax1.axvline(
            guacamol_results["top1"], color="red", linestyle="--", label=f"Top-1: {guacamol_results['top1']:.2f}"
        )
        ax1.axvline(
            guacamol_results["top10_mean"],
            color="orange",
            linestyle="--",
            label=f"Top-10 Mean: {guacamol_results['top10_mean']:.2f}",
        )
        ax1.set_xlabel("Penalized LogP", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.set_title("Task 2: Global Top 100 (GuacaMol)", fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Task 4: Lead optimization improvements
    # (Would need per-starter improvements data, which we don't save in summary)
    ax2.text(
        0.5,
        0.5,
        f"Mean Improvement: {lead_opt_results['mean_improvement']:.3f}\n"
        f"Success Rate: {lead_opt_results['success_rate']:.1f}%\n"
        f"Max Improvement: {lead_opt_results['max_improvement']:.3f}",
        ha="center",
        va="center",
        fontsize=14,
        transform=ax2.transAxes,
    )
    ax2.set_title("Task 4: Lead Optimization Metrics", fontsize=14)
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / "dual_evaluation_results.pdf", dpi=300, bbox_inches="tight")
    plt.close()


# ===== Main Evaluation =====
def run_final_evaluation(
    hpo_dir: pathlib.Path,
    dataset: SupportedDataset,
    gen_model_idx: int,
    n_starters: int,
    n_samples: int,
    similarity_threshold: float,
    output_dir: pathlib.Path,
):
    """Run final evaluation with dual reporting."""
    device = pick_device()
    objective_name = "penalized_logp"

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
    print(f"  Mean Improvement: {best_trial['mean_improvement']:.4f}")
    print(f"  Success Rate: {best_trial['success_rate']:.2%}")
    print(f"  Learning Rate: {best_trial['lr']:.6f}")
    print(f"  Steps: {best_trial['steps']}")
    print(f"  Lambda Similarity: {best_trial['lambda_similarity']:.6f}")
    print("=" * 80 + "\n")

    # Load models
    gen_model_hint = GENERATOR_REGISTRY[dataset][gen_model_idx]
    decoder_settings = DecoderSettings.get_default_for(base_dataset=dataset.default_cfg.base_dataset)
    generator = HDCGenerator(
        gen_model_hint=gen_model_hint,
        ds_config=dataset.default_cfg,
        device=device,
        decoder_settings=decoder_settings,
    )

    # Load regressors
    regressor_hints = REGRESSOR_REGISTRY[dataset]
    required_props = ["logp", "sa_score", "max_ring_size"]
    regressors = {}
    print("Loading regressors...")
    for prop in required_props:
        hint = regressor_hints[prop][0]
        pr_path = get_pr_path(hint=hint)
        regressors[prop] = retrieve_model(name="PR").load_from_checkpoint(pr_path).to(device).eval()
        print(f"  {prop}: {pr_path.stem}")

    # Load starting molecules
    print(f"\nLoading {n_starters} starting molecules from test set...")
    starting_mols = load_starting_molecules(dataset, generator.hypernet, n_starters)

    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_dir / f"{dataset.value}_constrained_final_sim{similarity_threshold}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Save configuration
    config_dict = {
        "dataset": dataset.value,
        "property": objective_name,
        "task": "dual_evaluation",
        "gen_model_hint": gen_model_hint,
        "n_starters": n_starters,
        "n_samples_per_starter": n_samples,
        "similarity_threshold": similarity_threshold,
        "timestamp": timestamp,
        "device": str(device),
        "hpo_dir": str(hpo_dir),
        "best_trial": best_trial,
    }
    (experiment_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

    # Run optimization
    config = ConstrainedOptimizationConfig(
        dataset=dataset,
        objective=objective_name,
        n_samples=n_samples,
        gen_model_idx=gen_model_idx,
        similarity_threshold=similarity_threshold,
        device=str(device),
    )

    optimizer = ConstrainedOptimizer(generator=generator, regressors=regressors, config=config)

    print(
        f"\nStarting optimization: {n_starters} starters × {n_samples} samples = {n_starters * n_samples} total candidates"
    )
    start_time = time.time()

    all_candidates = []  # For Task 2 (global pooling)
    per_starter_results = []  # For Task 4 (per-starter metrics)

    for start_data in tqdm(starting_mols, desc="Optimizing from starting molecules"):
        result = optimizer.optimize_from_start_mol(
            start_mol=start_data["mol"],
            start_graph_term=start_data["graph_term"],
            hps=best_trial,
            base_dataset=dataset.default_cfg.base_dataset,
        )

        # Collect for Task 2 (global pooling)
        all_candidates.extend(result["candidates"])

        # Collect for Task 4 (per-starter)
        per_starter_results.append(result)

    total_time = time.time() - start_time

    # Calculate Task 2 metrics (GuacaMol)
    print("\nCalculating Task 2 (GuacaMol) metrics...")
    guacamol_metrics = calculate_guacamol_metrics(all_candidates)

    # Calculate Task 4 metrics (Lead Optimization)
    print("Calculating Task 4 (Lead Optimization) metrics...")
    lead_opt_metrics = calculate_lead_opt_metrics(per_starter_results)

    # Create combined results
    results = DualConstrainedResults(
        guacamol_score=guacamol_metrics["guacamol_score"],
        global_top1_plogp=guacamol_metrics["top1"],
        global_top10_mean=guacamol_metrics["top10_mean"],
        global_top100_mean=guacamol_metrics["top100_mean"],
        global_n_candidates=guacamol_metrics["n_candidates"],
        global_n_unique=guacamol_metrics["n_unique"],
        mean_improvement=lead_opt_metrics["mean_improvement"],
        std_improvement=lead_opt_metrics["std_improvement"],
        median_improvement=lead_opt_metrics["median_improvement"],
        max_improvement=lead_opt_metrics["max_improvement"],
        success_rate=lead_opt_metrics["success_rate"],
        n_successful_starters=lead_opt_metrics["n_successful"],
        n_starters=n_starters,
        n_samples_per_starter=n_samples,
        similarity_threshold=similarity_threshold,
        total_time=total_time,
    )

    # Save metrics
    metrics_dict = {
        "task2_guacamol": {
            "guacamol_score": results.guacamol_score,
            "global_top1": results.global_top1_plogp,
            "global_top10_mean": results.global_top10_mean,
            "global_top100_mean": results.global_top100_mean,
            "n_candidates": results.global_n_candidates,
            "n_unique": results.global_n_unique,
        },
        "task4_lead_optimization": {
            "mean_improvement": results.mean_improvement,
            "std_improvement": results.std_improvement,
            "median_improvement": results.median_improvement,
            "max_improvement": results.max_improvement,
            "success_rate": results.success_rate,
            "n_successful_starters": results.n_successful_starters,
        },
        "metadata": {
            "n_starters": results.n_starters,
            "n_samples_per_starter": results.n_samples_per_starter,
            "similarity_threshold": results.similarity_threshold,
            "total_time": results.total_time,
        },
    }
    (experiment_dir / "dual_metrics.json").write_text(json.dumps(metrics_dict, indent=2, default=float))

    # Save Task 2 results
    (experiment_dir / "task2_guacamol_results.json").write_text(
        json.dumps(metrics_dict["task2_guacamol"], indent=2, default=float)
    )

    # Save Task 4 results
    (experiment_dir / "task4_lead_opt_results.json").write_text(
        json.dumps(metrics_dict["task4_lead_optimization"], indent=2, default=float)
    )

    # Save global top 100 molecules
    if guacamol_metrics["top100_molecules"]:
        df_top100 = pd.DataFrame(guacamol_metrics["top100_molecules"])
        df_top100.to_csv(experiment_dir / "global_top100_molecules.csv", index=False)

    # Save per-starter improvements
    per_starter_data = []
    for i, result in enumerate(per_starter_results):
        valid_candidates = [c for c in result["candidates"] if c["is_valid"]]
        best_improvement = max((c["improvement"] for c in valid_candidates), default=0.0)
        per_starter_data.append(
            {
                "starter_idx": i,
                "start_smiles": result["start_smiles"],
                "start_plogp": result["start_property"],
                "best_improvement": best_improvement,
                "n_valid_candidates": len(valid_candidates),
                "success": best_improvement > 0,
            }
        )
    df_starters = pd.DataFrame(per_starter_data)
    df_starters.to_csv(experiment_dir / "per_starter_improvements.csv", index=False)

    # Generate plots
    print("\nGenerating plots...")
    plot_dual_distributions(guacamol_metrics, lead_opt_metrics, plots_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("FINAL EVALUATION RESULTS - DUAL REPORTING")
    print("=" * 80)
    print(f"\n{'=' * 80}")
    print("TASK 2: UNCONSTRAINED (GuacaMol Protocol)")
    print("=" * 80)
    print(f"Pooled {results.global_n_candidates} candidates from {results.n_starters} starting molecules")
    print(f"Global Top-1 pLogP:     {results.global_top1_plogp:.4f} (cf. GP-MoLFormer: 19.59)")
    print(f"Global Top-10 Mean:     {results.global_top10_mean:.4f}")
    print(f"Global Top-100 Mean:    {results.global_top100_mean:.4f}")
    print(f"GuacaMol Score:         {results.guacamol_score:.4f}")
    print(f"Unique molecules:       {results.global_n_unique}")
    print(f"\n{'=' * 80}")
    print("TASK 4: CONSTRAINED (Lead Optimization)")
    print("=" * 80)
    print(f"{results.n_starters} starting molecules, {results.n_samples_per_starter} samples each")
    print(f"Similarity threshold:   {results.similarity_threshold}")
    print(f"Mean Improvement:       {results.mean_improvement:.4f} ± {results.std_improvement:.4f}")
    print(f"Median Improvement:     {results.median_improvement:.4f}")
    print(f"Max Improvement:        {results.max_improvement:.4f}")
    print(
        f"Success Rate:           {results.success_rate:.1f}% ({results.n_successful_starters}/{results.n_starters} improved)"
    )
    print(f"\nTotal Time:             {results.total_time:.2f}s")
    print("=" * 80)
    print(f"\nResults saved to: {experiment_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Final evaluation for similarity-constrained pLogP with dual reporting"
    )
    parser.add_argument(
        "--hpo_dir",
        type=pathlib.Path,
        default="/home/akaveh/Projects/kit/graph_hdc/src/exp/final_evaluations/constrained_plogp/hpo_results/ZINC_SMILES_HRR_256_F64_5G1NG4_constrained_plogp_sim0.4_20251117_180019",
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
    parser.add_argument("--n_starters", type=int, default=1, help="Number of starting molecules")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples per starting molecule")
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.4,
        help="Tanimoto similarity threshold for valid candidates",
    )
    parser.add_argument("--output_dir", type=str, default="final_results", help="Output directory")

    args = parser.parse_args()
    dataset = SupportedDataset(args.dataset)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_final_evaluation(
        hpo_dir=args.hpo_dir,
        dataset=dataset,
        gen_model_idx=args.model_idx,
        n_starters=args.n_starters,
        n_samples=args.n_samples,
        similarity_threshold=args.similarity_threshold,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
