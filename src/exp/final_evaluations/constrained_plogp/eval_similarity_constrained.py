#!/usr/bin/env python
"""
(Corrected) HPO for Similarity-Constrained Penalized LogP Maximization.

This script performs HPO for optimizing pLogP while maintaining structural
similarity to a set of starting molecules.

It now saves a drawing of the best-performing molecule from EACH trial
into the 'trial_molecules' directory for qualitative analysis.
"""

import argparse
import json
import pathlib
import time
import traceback
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from lightning_fabric import seed_everything
from optuna_integration import BoTorchSampler

# Imports for drawing
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from torch import nn
from torch_geometric.data import Batch
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
from src.utils.chem import draw_mol, is_valid_molecule, reconstruct_for_eval_v2
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


PROPERTY_FUNCTIONS = {
    "logp": rdkit_logp,
    "qed": rdkit_qed,
    "sa_score": rdkit_sa_score,
    "penalized_logp": calculate_penalized_logp_rdkit,
}


# Real Tanimoto Similarity (for final evaluation)
def calculate_tanimoto_similarity(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# ===== HPO Search Space =====
def get_search_space() -> dict[str, optuna.distributions.BaseDistribution]:
    return {
        # 1. INCREASE LR: You need larger steps to jump out of the "Identity" basin
        "lr": optuna.distributions.FloatDistribution(1e-3, 5e-2, log=True),
        # 2. REDUCE STEPS: Long optimization often leads to drifting too far.
        # 100-500 is usually enough for local optimization.
        "steps": optuna.distributions.IntDistribution(50, 500, log=True),
        # 3. REDUCE PRIOR: High prior pulls z to 0 (origin), not z_start.
        # This fights against similarity. Lower it significantly.
        "lambda_prior": optuna.distributions.FloatDistribution(1e-6, 1e-4, log=True),
        # 4. TIGHTEN THRESHOLD: Based on your plot, 0.4 is useless.
        # Start searching at 0.6.
        "proxy_similarity_threshold": optuna.distributions.FloatDistribution(0.6, 0.95),
        "lambda_similarity": optuna.distributions.FloatDistribution(1e-1, 1e2, log=True),
        "grad_clip": optuna.distributions.FloatDistribution(0.5, 5.0, log=True),
    }


# ===== Data Classes =====
@dataclass
class ConstrainedOptimizationConfig:
    """Configuration for constrained optimization experiment."""

    dataset: SupportedDataset
    objective: str
    n_samples: int  # Candidates per starting molecule
    gen_model_idx: int
    real_similarity_threshold: float  # The *actual* Tanimoto threshold
    device: str = "cuda"


@dataclass
class TrialResults:
    """Aggregated results for a single HPO trial."""

    mean_best_improvement: float
    std_best_improvement: float
    max_best_improvement: float
    success_rate: float  # Fraction of starting mols with at least 1 valid improvement
    mean_candidate_success_rate: float  # Mean fraction of candidates that were valid
    n_valid_unique_molecules: int
    total_time: float
    best_molecules: list  # List of (start_smiles, best_smiles, improvement)


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
        """
        n_candidates = self.config.n_samples
        start_property = self.objective_fn(start_mol)

        latent_dim = self.gen_model.flat_dim
        z = torch.randn(n_candidates, latent_dim, device=DEVICE, dtype=DTYPE, requires_grad=True)

        optimizer = torch.optim.Adam([z], lr=hps["lr"])

        pbar = tqdm(range(hps["steps"]), desc=f"Optimizing (start pLogP={start_property:.2f})", leave=False)
        for step in pbar:
            optimizer.zero_grad()

            x = self.gen_model.decode_from_latent(z)
            _, graph_terms = self.gen_model.split(x)

            # 1. Objective Loss (Maximize)
            property_pred = self._calculate_objective(x)
            property_loss = -property_pred.mean()

            # 2. Similarity Loss (Constraint)
            similarities = F.cosine_similarity(
                graph_terms, start_graph_term.unsqueeze(0).expand(n_candidates, -1), dim=1
            )

            if step % 50 == 0:
                print(
                    f"Step {step}: Mean Sim: {similarities.mean().item():.4f}, Threshold: {hps['proxy_similarity_threshold']:.4f}"
                )

            similarity_penalty = torch.mean(torch.relu(hps["proxy_similarity_threshold"] - similarities))

            # 3. Prior Loss
            prior_loss = torch.mean(z**2)

            # Total loss
            loss = property_loss + hps["lambda_similarity"] * similarity_penalty + hps["lambda_prior"] * prior_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_([z], hps["grad_clip"])
            optimizer.step()

            pbar.set_postfix(
                {"prop_loss": f"{property_loss.item():.2f}", "sim_pen": f"{similarity_penalty.item():.2f}"}
            )

        with torch.no_grad():
            x = self.gen_model.decode_from_latent(z)
            edge_terms, graph_terms = self.gen_model.split(x)

        decoded = self.generator.decode(edge_terms=edge_terms, graph_terms=graph_terms)

        candidates = []
        n_decoded = len(decoded["graphs"])
        n_valid_mols = 0
        n_meets_similarity = 0
        n_meets_improvement = 0
        n_meets_both = 0

        unique_smiles_count = len({c["smiles"] for c in candidates})
        print(f"    Diagnostic: {n_decoded} decoded. {n_valid_mols} valid. {unique_smiles_count} unique.")
        print(
            f"    Fail Reasons: {n_valid_mols - n_meets_similarity} dissimilar, {n_valid_mols - n_meets_improvement} no improvement."
        )

        for i, graph in enumerate(decoded["graphs"]):
            if not graph:
                continue

            mol = reconstruct_for_eval_v2(graph, dataset=base_dataset)
            if not mol or not is_valid_molecule(mol):
                continue

            n_valid_mols += 1

            actual_similarity = calculate_tanimoto_similarity(mol, start_mol)
            actual_property = self.objective_fn(mol)
            improvement = actual_property - start_property

            meets_similarity = actual_similarity >= self.config.real_similarity_threshold
            meets_improvement = improvement > 0

            if meets_similarity:
                n_meets_similarity += 1
            if meets_improvement:
                n_meets_improvement += 1
            if meets_similarity and meets_improvement:
                n_meets_both += 1

            is_valid_candidate = meets_similarity and meets_improvement

            candidates.append(
                {
                    "smiles": Chem.MolToSmiles(mol, canonical=True),
                    "property": actual_property,
                    "similarity": actual_similarity,
                    "improvement": improvement,
                    "is_valid": is_valid_candidate,
                }
            )

        # Debug output
        if n_valid_mols > 0:
            valid_candidates = [c for c in candidates if c["is_valid"]]
            print(
                f"  Start pLogP: {start_property:.2f} | Decoded: {n_decoded} | Valid mols: {n_valid_mols} | "
                f"Similarity≥{self.config.real_similarity_threshold}: {n_meets_similarity} | "
                f"Improvement>0: {n_meets_improvement} | Both: {n_meets_both}"
            )
            if valid_candidates:
                best_improvement = max(c["improvement"] for c in valid_candidates)
                print(f"    → Best valid improvement: {best_improvement:.3f}")

        return {
            "start_smiles": Chem.MolToSmiles(start_mol),
            "start_property": start_property,
            "candidates": candidates,
        }


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


# ===== Plotting (Simplified) =====
def plot_best_trial_results(results: TrialResults, save_dir: pathlib.Path):
    """Plots the results for the best HPO trial."""
    improvements = [b[2] for b in results.best_molecules if b[2] > -float("inf")]

    if not improvements:
        print("No valid improvements to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(improvements, bins=20, edgecolor="black", alpha=0.7)
    plt.axvline(np.mean(improvements), color="red", linestyle="--", label=f"Mean Improv.: {np.mean(improvements):.3f}")
    plt.xlabel("pLogP Improvement")
    plt.ylabel("Count")
    plt.title("Distribution of Best Improvements (Best Trial)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "best_trial_improvements.pdf", dpi=300, bbox_inches="tight")
    plt.close()


# ===== Main Execution =====
def load_validation_set(dataset: SupportedDataset, hypernet, n_starters: int) -> list:
    """Loads a fixed set of starting molecules for HPO validation."""
    print(f"Loading {n_starters} starting molecules for HPO validation set...")
    base_dataset = dataset.default_cfg.base_dataset
    ds = get_split(base_dataset=base_dataset, split="test")

    indices = np.random.choice(len(ds), n_starters * 2, replace=False)  # Get more to filter

    starting_molecules = []
    for idx in tqdm(indices, desc="Loading validation set"):
        if len(starting_molecules) >= n_starters:
            break

        # d is ALREADY a PyG Data object
        d = ds[int(idx)]

        # We still create the mol object for filtering and later use
        mol = Chem.MolFromSmiles(d.smiles)
        if not mol or mol.GetNumAtoms() <= 5:
            continue

        batch = Batch.from_data_list([d]).to(DEVICE)

        with torch.no_grad():
            # Pass the 'normalize' flag from the config
            encoding = hypernet.forward(batch, normalize=dataset.default_cfg.normalize)
            graph_term = encoding["graph_embedding"][0]

        starting_molecules.append(
            {
                "mol": mol,
                "graph_term": graph_term,
            }
        )

    if len(starting_molecules) < n_starters:
        raise ValueError(f"Could not load {n_starters} valid starting molecules.")

    print(f"Loaded {len(starting_molecules)} molecules.")
    return starting_molecules


def run_hpo(
    dataset: SupportedDataset,
    gen_model_idx: int,
    n_trials: int,
    n_samples: int,
    n_starters_hpo: int,
    real_similarity_threshold: float,
    output_dir: pathlib.Path,
):
    """Run HPO for constrained pLogP maximization."""
    device = pick_device()
    objective_name = "penalized_logp"  # Hard-coding for this task

    # --- 1. Load Generator ---
    gen_model_hint = GENERATOR_REGISTRY[dataset][gen_model_idx]
    decoder_settings = DecoderSettings.get_default_for(base_dataset=dataset.default_cfg.base_dataset)
    generator = HDCGenerator(
        gen_model_hint=gen_model_hint,
        ds_config=dataset.default_cfg,
        device=device,
        decoder_settings=decoder_settings,
    )

    # --- 2. Load Regressors (All 3) ---
    regressor_hints = REGRESSOR_REGISTRY[dataset]
    required_props = ["logp", "sa_score", "max_ring_size"]
    if not all(prop in regressor_hints for prop in required_props):
        raise ValueError(f"Missing regressor hints for {dataset.value}. Need: {required_props}")

    regressors = {}
    print("Loading regressors...")
    for prop in required_props:
        hint = regressor_hints[prop][0]
        pr_path = get_pr_path(hint=hint)
        regressors[prop] = retrieve_model(name="PR").load_from_checkpoint(pr_path).to(device).eval()
        print(f"  {prop}: {pr_path.stem}")

    # --- 3. Load Fixed Validation Set for HPO ---
    validation_set = load_validation_set(dataset, generator.hypernet, n_starters_hpo)

    # --- 4. Setup HPO ---
    search_space = get_search_space()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_dir / f"{dataset.value}_constrained_plogp_sim{real_similarity_threshold}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "dataset": dataset.value,
        "property": objective_name,
        "task": "constrained_maximization",
        "gen_model_hint": gen_model_hint,
        "n_trials": n_trials,
        "n_samples_per_starter": n_samples,
        "n_starters_hpo": n_starters_hpo,
        "real_similarity_threshold": real_similarity_threshold,
    }
    (experiment_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2))

    study_name = f"constrained_plogp_{dataset.value}"
    db_path = experiment_dir / f"hpo_{study_name}.db"
    csv_path = experiment_dir / f"trials_{study_name}.csv"
    study = load_or_create_study(study_name, db_path, search_space)

    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # *** NEW: Create directory for trial molecules ***
    trial_mols_dir = experiment_dir / "trial_molecules"
    trial_mols_dir.mkdir(exist_ok=True)

    trial_plot_storage = {}

    # --- 5. Define Objective ---
    def objective(trial: optuna.Trial) -> float:
        try:
            config = ConstrainedOptimizationConfig(
                dataset=dataset,
                objective=objective_name,
                n_samples=n_samples,
                gen_model_idx=gen_model_idx,
                device=str(device),
                real_similarity_threshold=real_similarity_threshold,
            )
            hps = {
                hp: trial.suggest_categorical(hp, val.choices)
                if isinstance(val, optuna.distributions.CategoricalDistribution)
                else trial.suggest_float(hp, val.low, val.high, log=val.log)
                if isinstance(val, optuna.distributions.FloatDistribution)
                else trial.suggest_int(hp, val.low, val.high, log=val.log)
                for hp, val in search_space.items()
            }

            optimizer = ConstrainedOptimizer(generator=generator, regressors=regressors, config=config)

            start_time = time.time()
            all_improvements = []
            all_candidate_success_rates = []
            valid_unique_molecules = set()
            best_molecules_for_trial = []

            for start_data in validation_set:
                result = optimizer.optimize_from_start_mol(
                    start_mol=start_data["mol"],
                    start_graph_term=start_data["graph_term"],
                    hps=hps,
                    base_dataset=dataset.default_cfg.base_dataset,
                )

                valid_candidates = [c for c in result["candidates"] if c["is_valid"]]

                if valid_candidates:
                    best_candidate = max(valid_candidates, key=lambda c: c["improvement"])
                    all_improvements.append(best_candidate["improvement"])
                    best_molecules_for_trial.append(
                        (result["start_smiles"], best_candidate["smiles"], best_candidate["improvement"])
                    )
                    valid_unique_molecules.update([c["smiles"] for c in valid_candidates])
                else:
                    all_improvements.append(0.0)  # No valid improvement

                all_candidate_success_rates.append(len(valid_candidates) / max(1, len(result["candidates"])))

            total_time = time.time() - start_time

            # --- Calculate Aggregate Metrics for this Trial ---
            mean_best_improvement = np.mean(all_improvements)
            std_best_improvement = np.std(all_improvements)
            max_best_improvement = np.max(all_improvements) if all_improvements else 0.0
            success_rate = np.mean([1.0 if imp > 0 else 0.0 for imp in all_improvements])
            mean_candidate_success_rate = np.mean(all_candidate_success_rates)

            results = TrialResults(
                mean_best_improvement=mean_best_improvement,
                std_best_improvement=std_best_improvement,
                max_best_improvement=max_best_improvement,
                success_rate=success_rate,
                mean_candidate_success_rate=mean_candidate_success_rate,
                n_valid_unique_molecules=len(valid_unique_molecules),
                total_time=total_time,
                best_molecules=best_molecules_for_trial,
            )

            trial_plot_storage[trial.number] = results

            # Store user attributes
            trial.set_user_attr("mean_improvement", results.mean_best_improvement)
            trial.set_user_attr("std_improvement", results.std_best_improvement)
            trial.set_user_attr("max_improvement", results.max_best_improvement)
            trial.set_user_attr("success_rate", results.success_rate)
            trial.set_user_attr("candidate_success_rate", results.mean_candidate_success_rate)
            trial.set_user_attr("n_valid_unique", results.n_valid_unique_molecules)

            # --- *** NEW: Save Best Molecule Image *** ---
            if results.best_molecules:
                # Find the single best molecule from this trial
                sorted_mols = sorted(results.best_molecules, key=lambda x: x[2], reverse=True)
                best_start_smi, best_opt_smi, best_improv = sorted_mols[0]

                if best_improv > 0:
                    mol = Chem.MolFromSmiles(best_opt_smi)
                    if mol:
                        # Save image to the trial_mols_dir
                        save_path = trial_mols_dir / f"trial_{trial.number:04d}_improv_{best_improv:.3f}.png"
                        draw_mol(mol, save_path=str(save_path), fmt="png")

            # --- HPO Objective ---
            return results.mean_best_improvement + results.success_rate

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            traceback.print_exc()
            return -1e9

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
        **metadata,
    }
    (experiment_dir / f"best_config_{study_name}.json").write_text(json.dumps(best_config, indent=2, default=float))

    print(f"\nBest trial: {best_trial.number}")
    print(f"Best value (Objective): {best_trial.value:.6f}")
    print(f"Best params: {best_trial.params}")
    print(f"Best Mean Improvement: {best_trial.user_attrs.get('mean_improvement', 0):.4f}")
    print(f"Best Max Improvement: {best_trial.user_attrs.get('max_improvement', 0):.4f}")
    print(f"Best Success Rate: {best_trial.user_attrs.get('success_rate', 0):.1%}")

    # --- 8. Plot Best Trial ---
    if best_trial.number in trial_plot_storage:
        best_data = trial_plot_storage[best_trial.number]
        print("\nGenerating plots for best trial...")
        plot_best_trial_results(best_data, plots_dir)

        df_best_mols = pd.DataFrame(
            best_data.best_molecules, columns=["start_smiles", "optimized_smiles", "improvement"]
        )
        df_best_mols.to_csv(plots_dir / "best_trial_molecules.csv", index=False)
        print(f"Plots and best molecules saved to: {plots_dir}")
    else:
        print(f"Warning: Best trial {best_trial.number} plot data not found.")


def main():
    parser = argparse.ArgumentParser(description="HPO for Similarity-Constrained pLogP maximization")
    parser.add_argument(
        "--dataset",
        type=str,
        default=SupportedDataset.ZINC_SMILES_HRR_256_F64_5G1NG4.value,
        choices=[ds.value for ds in SupportedDataset],
        help="Dataset to use (e.g., ZINC_SMILES...)",
    )
    parser.add_argument("--model_idx", type=int, default=0, help="Index of generator model in registry")
    parser.add_argument("--n_trials", type=int, default=1, help="Number of HPO trials")
    parser.add_argument("--n_samples", type=int, default=30, help="Number of candidates per starting molecule")
    parser.add_argument(
        "--n_starters_hpo", type=int, default=2, help="Number of fixed starting molecules for HPO validation"
    )
    parser.add_argument("--similarity_threshold", type=float, default=0.4, help="*Real* Tanimoto similarity threshold")
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
        n_starters_hpo=args.n_starters_hpo,
        real_similarity_threshold=args.similarity_threshold,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
