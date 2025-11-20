#!/usr/bin/env python
"""
Final Evaluation for Property Targeting (MG-DIFF Protocol)
===========================================================

This script runs final evaluation using best hyperparameters from HPO.
Generates 10,000 molecules per target and computes aggregate metrics.

Features:
1. Loads best hyperparameters from HPO results
2. Generates 10,000 molecules per target
3. Dual evaluation (All Valid + Filter 2)
4. Optional molecule drawings (--draw flag)
5. Exports SMILES with all metadata
6. Computes aggregate MAD across targets
7. Generates publication-quality plots

Usage:
    # Standard final evaluation
    python run_property_targeting_final_eval.py \
        --hpo_dir hpo_results/logp_QM9_SMILES_HRR_1600_F64_G1NG3_20251113_120000 \
        --n_samples 10000 \
        --output_dir final_results

    # With molecule drawings
    python run_property_targeting_final_eval.py \
        --hpo_dir hpo_results/logp_QM9_SMILES_HRR_1600_F64_G1NG3_20251113_120000 \
        --n_samples 10000 \
        --draw \
        --max_draw 200 \
        --output_dir final_results

    # Quick test
    python run_property_targeting_final_eval.py \
        --hpo_dir <hpo_dir> \
        --n_samples 500 \
        --draw \
        --max_draw 10
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from lightning_fabric import seed_everything
from rdkit import Chem
from scipy.stats import gaussian_kde
from tqdm.auto import tqdm

# Import from HPO script
from src.encoding.configs_and_constants import DecoderSettings, SupportedDataset
from src.exp.final_evaluations.models_configs_constants import (
    GENERATOR_REGISTRY,
    REGRESSOR_REGISTRY,
    get_pr_path,
)
from src.exp.final_evaluations.property_targeting.run_property_targeting_hpo import (
    PROPERTY_FUNCTIONS,
    PropertyTargetingConfig,
    PropertyTargetingOptimizer,
    get_adaptive_epsilon,
)
from src.generation.generation import HDCGenerator
from src.utils.chem import draw_mol, is_valid_molecule, reconstruct_for_eval_v2
from src.utils.registery import retrieve_model
from src.utils.utils import pick_device

# Configuration
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)
DEVICE = pick_device()

SEED = 42
seed_everything(SEED)

# Plotting style
plt.style.use("seaborn-v0_8-paper")
sns.set_palette("husl")
plt.rcParams.update(
    {
        "font.size": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
    }
)


# ===== HPO Directory Parsing =====
def parse_hpo_directory(hpo_dir: Path) -> dict:
    """Parse HPO experiment directory and extract metadata and best configs."""
    metadata_path = hpo_dir / "experiment_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No experiment_metadata.json found in {hpo_dir}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Find all target directories
    target_dirs = sorted(hpo_dir.glob("target_*"))
    if not target_dirs:
        raise ValueError(f"No target directories found in {hpo_dir}")

    targets_info = []
    for target_dir in target_dirs:
        # Extract target value from directory name
        target_str = target_dir.name.replace("target_", "")
        target_value = float(target_str)

        # Find best config file
        best_config_files = list(target_dir.glob("best_config_*.json"))
        if not best_config_files:
            print(f"Warning: No best_config found in {target_dir}, skipping")
            continue

        with open(best_config_files[0]) as f:
            best_config = json.load(f)

        targets_info.append(
            {
                "value": target_value,
                "dir": target_dir,
                "best_config": best_config,
                "best_params": best_config["best_params"],
            }
        )

    return {
        "metadata": metadata,
        "targets": sorted(targets_info, key=lambda x: x["value"]),
        "hpo_dir": hpo_dir,
    }


# ===== Molecule Drawing =====
def save_molecule_drawings(
    rdkit_mols: list,
    smiles: list,
    properties: list,
    target: float,
    output_dir: Path,
    max_draw: int = 100,
    fmt: str = "svg",
    metadata: list[dict] | None = None,
    property_name: str = "",
):
    """
    Draw molecules and save to output_dir/drawings/.

    Args:
        rdkit_mols: RDKit molecules
        smiles: SMILES strings
        properties: Property values
        target: Target value
        output_dir: Base directory
        max_draw: Maximum number to draw
        fmt: 'svg' or 'png'
        metadata: Optional metadata list (for Set 4 naming with original_sample, decoder_rank, similarity)
        property_name: Property name (for Set 4 naming)
    """
    drawings_dir = output_dir / "drawings"
    drawings_dir.mkdir(parents=True, exist_ok=True)

    n_draw = min(len(rdkit_mols), max_draw)

    print(f"    Drawing {n_draw} molecules...")

    for i in tqdm(range(n_draw), desc="Drawing", unit="mol"):
        if rdkit_mols[i] is not None:
            # Set 4 naming: top{original_sample}-{decoder_rank}_sim{similarity}_{property}{value}_target{target}
            if metadata is not None and i < len(metadata):
                meta = metadata[i]
                filename = (
                    f"top{meta['original_sample']:03d}-{meta['decoder_rank']:02d}_"
                    f"sim{meta['similarity']:.3f}_{property_name}{properties[i]:.2f}_"
                    f"target{target:.2f}.{fmt}"
                )
            else:
                # Default naming for other sets
                filename = f"mol_{i:04d}_{properties[i]:.2f}.{fmt}"

            save_path = drawings_dir / filename

            try:
                draw_mol(rdkit_mols[i], save_path=str(save_path), fmt=fmt)
            except Exception as e:
                print(f"    Warning: Failed to draw molecule {i}: {e}")

    print(f"    ✓ Saved {n_draw} molecule drawings to {drawings_dir}")


# ===== Export Functions =====
def export_molecules_metadata(
    smiles: list,
    properties: list,
    latent_flags: list,
    cos_similarities: list,
    correction_levels: list,
    target: float,
    output_path: Path,
):
    """Export molecules metadata to CSV."""
    df = pd.DataFrame(
        {
            "smiles": smiles,
            "property_value": properties,
            "target": [target] * len(smiles),
            "passed_filter2": latent_flags,
            "cos_similarity": cos_similarities,
            "correction_level": [str(level) for level in correction_levels],
        }
    )

    df.to_csv(output_path, index=False)
    print(f"    ✓ Saved molecules metadata to {output_path}")


# ===== Evaluation =====
def run_final_evaluation_for_target(
    target_info: dict,
    experiment_metadata: dict,
    generator: HDCGenerator,
    property_regressor: torch.nn.Module,
    dataset_props: dict[str, list[float]],
    n_samples: int = 10000,
    draw: bool = False,
    max_draw: int = 100,
    output_dir: Path = None,
    top_k_property: int = 100,
    top_n_best: int = 10,
    decoder_k: int = 10,
) -> dict:
    """
    Run final evaluation for a single target using best hyperparameters.

    Args:
        target_info: Dictionary with target value and best parameters
        experiment_metadata: Experiment metadata
        generator: Generator model
        property_regressor: Property regressor model
        dataset_props: Dictionary of training dataset property distributions
        n_samples: Number of samples to generate
        draw: Whether to draw molecules
        max_draw: Maximum number of molecules to draw
        output_dir: Output directory
        top_k_property: Number of top molecules by property distance for evaluation set 3
        top_n_best: Number of best molecules to expand with all decoder outputs for evaluation set 4
        decoder_k: Number of decoder outputs per molecule for evaluation set 4

    Returns:
        Dictionary with all evaluation results
    """
    best_params = target_info["best_params"]
    target_value = target_info["value"]

    dataset = SupportedDataset(experiment_metadata["dataset"])
    property_name = experiment_metadata["property"]

    print(f"\n{'=' * 60}")
    print(f"Final Evaluation: Target = {target_value}")
    print(f"Best Hyperparameters: {best_params}")
    print(f"{'=' * 60}\n")

    # Extract epsilon_multiplier from best params
    epsilon_multiplier = best_params.get("epsilon_multiplier", 0.2)

    # Create config
    config = PropertyTargetingConfig(
        dataset=dataset,
        property_name=property_name,
        target_value=target_value,
        n_samples=n_samples,
        gen_model_idx=experiment_metadata["gen_model_idx"],
        epsilon_multiplier=epsilon_multiplier,
        device=str(DEVICE),
    )

    # Get epsilon
    epsilon = get_adaptive_epsilon(property_name, dataset, epsilon_multiplier)

    # Create optimizer
    optimizer = PropertyTargetingOptimizer(generator=generator, property_regressor=property_regressor, config=config)

    # Run optimization
    print(f"Running optimization with {n_samples} samples...")
    start_time = time.time()
    opt_results = optimizer.optimize_latent(
        target=target_value,
        epsilon=epsilon,
        lr=best_params["lr"],
        steps=best_params["steps"],
        lambda_prior=best_params["lambda_prior"],
        grad_clip=best_params["grad_clip"],
    )
    optimization_time = time.time() - start_time

    # Evaluate with dual sets (Sets 1 & 2)
    print("Evaluating molecules (Sets 1 & 2: All Valid + Filter 2)...")
    decode_start = time.time()
    eval_results = optimizer.evaluate_with_dual_sets(
        molecules=opt_results["molecules"],
        similarities=opt_results["similarities"],
        correction_levels=opt_results["correction_levels"],
        final_flags=opt_results["final_flags"],
        latent_filter_flags=opt_results["latent_filter_flags"],
        target=target_value,
        epsilon=epsilon,
    )

    # ===== EVALUATION SET 3: Top-k Property =====
    print(f"Evaluating Set 3: Top-{top_k_property} by property distance...")

    # Get valid molecules with properties from Set 1 (All Valid)
    # Need to get the original nx graphs for the evaluator
    all_valid_nx_graphs = []
    all_valid_sims = []
    all_valid_corr_levels = []
    all_valid_final_flags = []

    # Reconstruct data from the original opt_results by matching SMILES
    valid_smiles_set = set(eval_results.all_valid_smiles)
    for i, mol in enumerate(opt_results["molecules"]):
        if mol is None:
            continue
        try:
            rdkit_mol = reconstruct_for_eval_v2(mol, dataset=optimizer.base_dataset)
            if rdkit_mol and is_valid_molecule(rdkit_mol):
                smi = Chem.MolToSmiles(rdkit_mol)
                if smi in valid_smiles_set:
                    all_valid_nx_graphs.append(mol)
                    all_valid_sims.append(opt_results["similarities"][i])
                    all_valid_corr_levels.append(opt_results["correction_levels"][i])
                    all_valid_final_flags.append(opt_results["final_flags"][i])
        except:
            pass

    all_valid_props = eval_results.all_valid_property_values.get(property_name, [])

    # Compute top-k by property distance
    if len(all_valid_props) > 0:
        abs_diffs = np.abs(np.array(all_valid_props) - target_value)
        sorted_indices = np.argsort(abs_diffs)
        k_actual = min(top_k_property, len(sorted_indices))
        top_k_indices = sorted_indices[:k_actual]

        top_k_nx_graphs = [all_valid_nx_graphs[i] for i in top_k_indices]
        top_k_sims = [all_valid_sims[i] for i in top_k_indices]
        top_k_corr_levels = [all_valid_corr_levels[i] for i in top_k_indices]
        top_k_final_flags = [all_valid_final_flags[i] for i in top_k_indices]
        top_k_props = [all_valid_props[i] for i in top_k_indices]
        top_k_smiles = [eval_results.all_valid_smiles[i] for i in top_k_indices]
        top_k_rdkit_mols = [eval_results.all_valid_rdkit_mols[i] for i in top_k_indices]

        # Compute metrics for Set 3 using the evaluator
        top_k_eval = optimizer.evaluator.evaluate(
            n_samples=len(top_k_nx_graphs),
            samples=top_k_nx_graphs,
            final_flags=top_k_final_flags,
            sims=top_k_sims,
            correction_levels=top_k_corr_levels,
        )
        top_k_mad = np.mean(abs_diffs[top_k_indices])

        top_k_property_results = {
            "validity": top_k_eval["validity"],
            "uniqueness": top_k_eval["uniqueness"],
            "novelty": top_k_eval["novelty"],
            "diversity_p1": top_k_eval["internal_diversity_p1"],
            "diversity_p2": top_k_eval["internal_diversity_p2"],
            "mad": float(top_k_mad),
            "n_samples": k_actual,
            "property_values": top_k_props,
            "smiles": top_k_smiles,
            "rdkit_mols": top_k_rdkit_mols,
        }
        print(f"  ✓ Set 3: N={k_actual}, MAD={top_k_mad:.4f}")
    else:
        top_k_property_results = {"n_samples": 0, "mad": None}
        print("  ✗ Set 3: No valid molecules")

    # ===== EVALUATION SET 4: Top-n All Decoder Outputs =====
    print(f"Evaluating Set 4: All decoder outputs from top-{top_n_best} best molecules...")
    print(f"  Target: {top_n_best * decoder_k} graphs ({top_n_best} samples × {decoder_k} decoder outputs)")

    if len(all_valid_props) > 0:
        # Adaptive sampling: Keep pulling more candidates until we reach n*k graphs
        target_graph_count = top_n_best * decoder_k

        # Start with a larger candidate pool (3x to ensure we can reach target)
        candidate_pool_size = min(top_n_best * 3, len(sorted_indices))

        # Map ALL candidate indices back to original sample indices
        all_original_indices = []
        valid_count = 0
        for i, mol in enumerate(opt_results["molecules"]):
            if mol is not None:
                try:
                    rdkit_mol = reconstruct_for_eval_v2(mol, dataset=optimizer.base_dataset)
                    if rdkit_mol and is_valid_molecule(rdkit_mol):
                        if valid_count < candidate_pool_size:
                            all_original_indices.append(i)
                        valid_count += 1
                except:
                    pass

        print(f"  Candidate pool: {len(all_original_indices)} samples (from top-{candidate_pool_size})")

        # Extract all decoder outputs for the entire candidate pool
        edge_terms = opt_results["edge_terms"].to(DEVICE)
        graph_terms = opt_results["graph_terms"].to(DEVICE)

        expanded_results = generator.extract_topn_all_decoder_outputs(
            edge_terms=edge_terms,
            graph_terms=graph_terms,
            sample_indices=all_original_indices,
            decoder_k=decoder_k,
        )

        # Prepare data for evaluator (nx graphs)
        expanded_nx_graphs = expanded_results["graphs"]
        expanded_sims = expanded_results["similarities"]
        expanded_final_flags = expanded_results["final_flags"]

        # For correction levels, use placeholder since we're just decoding
        from src.encoding.graph_encoders import CorrectionLevel

        expanded_corr_levels = [CorrectionLevel.ZERO] * len(expanded_nx_graphs)

        # Convert to RDKit mols and compute properties
        # Process until we reach target_graph_count or exhaust all candidates
        expanded_rdkit_mols = []
        expanded_props = []
        expanded_smiles = []
        expanded_metadata = []
        valid_indices = []  # Track which indices are valid

        prop_fn = PROPERTY_FUNCTIONS[property_name]

        for i, graph in enumerate(expanded_nx_graphs):
            # Stop if we've reached the target count
            if len(expanded_rdkit_mols) >= target_graph_count:
                break

            if graph is None:
                continue
            try:
                mol = reconstruct_for_eval_v2(graph, dataset=optimizer.base_dataset)
                if not mol or not is_valid_molecule(mol):
                    continue

                prop_val = prop_fn(mol)
                smi = Chem.MolToSmiles(mol)

                expanded_rdkit_mols.append(mol)
                expanded_props.append(prop_val)
                expanded_smiles.append(smi)
                expanded_metadata.append(
                    {
                        "original_sample": expanded_results["sample_origins"][i],
                        "decoder_rank": expanded_results["decoder_ranks"][i],
                        "similarity": expanded_results["similarities"][i],
                    }
                )
                valid_indices.append(i)
            except:
                continue

        print(f"  Collected {len(expanded_rdkit_mols)} valid graphs from candidate pool")

        # Compute metrics for Set 4
        if len(expanded_rdkit_mols) > 0:
            # Filter nx graphs to only valid ones
            valid_nx_graphs = [expanded_nx_graphs[i] for i in valid_indices]
            valid_sims = [expanded_sims[i] for i in valid_indices]
            valid_final_flags = [expanded_final_flags[i] for i in valid_indices]
            valid_corr_levels = [expanded_corr_levels[i] for i in valid_indices]

            # Use evaluator for VUN and diversity metrics
            expanded_eval = optimizer.evaluator.evaluate(
                n_samples=len(valid_nx_graphs),
                samples=valid_nx_graphs,
                final_flags=valid_final_flags,
                sims=valid_sims,
                correction_levels=valid_corr_levels,
            )

            expanded_abs_diffs = np.abs(np.array(expanded_props) - target_value)
            expanded_mad = np.mean(expanded_abs_diffs)

            top_n_all_decoder_results = {
                "validity": expanded_eval["validity"],
                "uniqueness": expanded_eval["uniqueness"],
                "novelty": expanded_eval["novelty"],
                "diversity_p1": expanded_eval["internal_diversity_p1"],
                "diversity_p2": expanded_eval["internal_diversity_p2"],
                "mad": float(expanded_mad),
                "n_requested": target_graph_count,
                "n_actual": len(expanded_rdkit_mols),
                "counts_per_sample": expanded_results["counts_per_sample"],
                "property_values": expanded_props,
                "smiles": expanded_smiles,
                "metadata": expanded_metadata,
                "rdkit_mols": expanded_rdkit_mols,
            }
            print(f"  ✓ Set 4: N={len(expanded_rdkit_mols)}/{target_graph_count}, MAD={expanded_mad:.4f}")
        else:
            top_n_all_decoder_results = {"n_requested": target_graph_count, "n_actual": 0, "mad": None}
            print("  ✗ Set 4: No valid molecules")
    else:
        top_n_all_decoder_results = {"n_requested": 0, "n_actual": 0, "mad": None}
        print("  ✗ Set 4: No valid molecules in Set 1")

    decoding_time = time.time() - decode_start

    # Update timing
    eval_results.optimization_time = optimization_time
    eval_results.decoding_time = decoding_time
    eval_results.total_time = optimization_time + decoding_time

    # Create target-specific output directory
    target_output_dir = output_dir / f"target_{target_value:.2f}"
    target_output_dir.mkdir(parents=True, exist_ok=True)

    # Clean results for JSON serialization (remove rdkit_mols)
    top_k_clean = {k: v for k, v in top_k_property_results.items() if k != "rdkit_mols"}
    top_n_clean = {k: v for k, v in top_n_all_decoder_results.items() if k != "rdkit_mols"}

    # Save results
    results_dict = {
        "config": {
            "dataset": dataset.value,
            "property": property_name,
            "target_value": target_value,
            "epsilon": epsilon,
            "n_samples": n_samples,
            "top_k_property": top_k_property,
            "top_n_best": top_n_best,
            "decoder_k": decoder_k,
        },
        "best_params": best_params,
        "all_valid_results": {
            "validity": eval_results.all_valid_validity,
            "uniqueness": eval_results.all_valid_uniqueness,
            "novelty": eval_results.all_valid_novelty,
            "diversity_p1": eval_results.all_valid_diversity_p1,
            "diversity_p2": eval_results.all_valid_diversity_p2,
            "mad": eval_results.all_valid_mad,
            "n_samples": eval_results.all_valid_n_samples,
            "property_stats": eval_results.all_valid_property_stats,
            "property_values": eval_results.all_valid_property_values,  # Individual values for plotting
            "correction_levels": eval_results.all_valid_correction_levels,
            "cos_sim_mean": eval_results.all_valid_cos_sim_mean,
            "cos_sim_std": eval_results.all_valid_cos_sim_std,
        },
        "filter2_results": {
            "validity": eval_results.filter2_validity,
            "uniqueness": eval_results.filter2_uniqueness,
            "novelty": eval_results.filter2_novelty,
            "diversity_p1": eval_results.filter2_diversity_p1,
            "diversity_p2": eval_results.filter2_diversity_p2,
            "mad": eval_results.filter2_mad,
            "n_samples": eval_results.filter2_n_samples,
            "property_stats": eval_results.filter2_property_stats,
            "property_values": eval_results.filter2_property_values,  # Individual values for plotting
            "correction_levels": eval_results.filter2_correction_levels,
            "cos_sim_mean": eval_results.filter2_cos_sim_mean,
            "cos_sim_std": eval_results.filter2_cos_sim_std,
        },
        "top_k_property_results": top_k_clean,
        "top_n_all_decoder_results": top_n_clean,
        "timing": {
            "optimization_time": optimization_time,
            "decoding_time": decoding_time,
            "total_time": optimization_time + decoding_time,
        },
        "n_passed_latent_filter": eval_results.n_passed_latent_filter,
    }

    # Save JSON results
    with open(target_output_dir / "results.json", "w") as f:
        json.dump(results_dict, f, indent=2, default=float)

    # Save molecules as pickle
    with open(target_output_dir / "molecules.pkl", "wb") as f:
        pickle.dump(opt_results["molecules"], f)

    # Create plots subdirectory
    plots_dir = target_output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Generate distribution plots
    print("\n[Plotting] Generating property distribution plots...")

    # Plot for All Valid set
    if len(eval_results.all_valid_property_values.get(property_name, [])) > 0:
        # Convert to dict format for plotting (only the target property)
        all_valid_props_dict = {property_name: eval_results.all_valid_property_values.get(property_name, [])}
        plot_property_distribution(
            dataset_props=dataset_props,
            generated_props=all_valid_props_dict,
            target_value=target_value,
            property_name=property_name,
            output_dir=plots_dir,
            set_name="All Valid",
        )

    # Plot for Filter 2 set
    if len(eval_results.filter2_property_values.get(property_name, [])) > 0:
        # Convert to dict format for plotting (only the target property)
        filter2_props_dict = {property_name: eval_results.filter2_property_values.get(property_name, [])}
        plot_property_distribution(
            dataset_props=dataset_props,
            generated_props=filter2_props_dict,
            target_value=target_value,
            property_name=property_name,
            output_dir=plots_dir,
            set_name="Filter 2",
        )

    # Plot for Set 3 (Top-k Property)
    if top_k_property_results.get("n_samples", 0) > 0:
        # Convert to dict format for plotting
        top_k_props_dict = {property_name: top_k_property_results["property_values"]}
        plot_property_distribution(
            dataset_props=dataset_props,
            generated_props=top_k_props_dict,
            target_value=target_value,
            property_name=property_name,
            output_dir=plots_dir,
            set_name="Top-k Property",
        )

    # Plot for Set 4 (Top-n All Decoder)
    if top_n_all_decoder_results.get("n_actual", 0) > 0:
        # Convert to dict format for plotting
        top_n_props_dict = {property_name: top_n_all_decoder_results["property_values"]}
        plot_property_distribution(
            dataset_props=dataset_props,
            generated_props=top_n_props_dict,
            target_value=target_value,
            property_name=property_name,
            output_dir=plots_dir,
            set_name="Top-n All Decoder",
        )

    # Plot multi-property panel
    if (
        len(eval_results.all_valid_property_values.get("logp", [])) > 0
        and len(eval_results.filter2_property_values.get("logp", [])) > 0
    ):
        plot_multi_property_panel(
            dataset_props=dataset_props,
            all_valid_props=eval_results.all_valid_property_values,
            filter2_props=eval_results.filter2_property_values,
            target_property=property_name,
            target_value=target_value,
            output_dir=plots_dir,
        )

    # Export molecules metadata (SMILES with properties and correction levels)
    if eval_results.all_valid_smiles:
        print("\n[Export] Saving molecules metadata...")
        export_molecules_metadata(
            smiles=eval_results.all_valid_smiles,
            properties=eval_results.all_valid_property_values.get(property_name, []),
            latent_flags=eval_results.all_valid_latent_flags,
            cos_similarities=eval_results.all_valid_similarities,
            correction_levels=eval_results.all_valid_correction_levels_list,
            target=target_value,
            output_path=target_output_dir / "molecules_metadata.csv",
        )

    # Draw molecules if requested
    if draw:
        print("\n[Drawing] Generating molecule drawings...")

        # Draw All Valid set
        if eval_results.all_valid_rdkit_mols:
            save_molecule_drawings(
                rdkit_mols=eval_results.all_valid_rdkit_mols,
                smiles=eval_results.all_valid_smiles,
                properties=eval_results.all_valid_property_values.get(property_name, []),
                target=target_value,
                output_dir=target_output_dir / "drawings_all_valid",
                max_draw=max_draw,
                fmt="svg",
            )

        # Draw Top-k Property set (Set 3)
        if top_k_property_results.get("rdkit_mols"):
            save_molecule_drawings(
                rdkit_mols=top_k_property_results["rdkit_mols"],
                smiles=top_k_property_results["smiles"],
                properties=top_k_property_results["property_values"],
                target=target_value,
                output_dir=target_output_dir / "drawings_top_k_property",
                max_draw=max_draw,
                fmt="svg",
            )

        # Draw Top-n All Decoder set (Set 4)
        if top_n_all_decoder_results.get("rdkit_mols"):
            save_molecule_drawings(
                rdkit_mols=top_n_all_decoder_results["rdkit_mols"],
                smiles=top_n_all_decoder_results["smiles"],
                properties=top_n_all_decoder_results["property_values"],
                target=target_value,
                output_dir=target_output_dir / "drawings_top_n_decoder",
                max_draw=max_draw,
                fmt="svg",
                metadata=top_n_all_decoder_results.get("metadata"),
                property_name=property_name,
            )

    print(f"\n  ✓ Results saved to {target_output_dir}")
    print(f"  Set 1 (All Valid):      MAD = {eval_results.all_valid_mad:.4f}, N = {eval_results.all_valid_n_samples}")
    print(f"  Set 2 (Filter 2):       MAD = {eval_results.filter2_mad:.4f}, N = {eval_results.filter2_n_samples}")
    if top_k_property_results.get("mad") is not None:
        print(
            f"  Set 3 (Top-k Property): MAD = {top_k_property_results['mad']:.4f}, N = {top_k_property_results['n_samples']}"
        )
    if top_n_all_decoder_results.get("mad") is not None:
        print(
            f"  Set 4 (Top-n Decoder):  MAD = {top_n_all_decoder_results['mad']:.4f}, N = {top_n_all_decoder_results['n_actual']}/{top_n_all_decoder_results['n_requested']}"
        )

    return results_dict


# ===== Dataset Property Loading =====
def load_dataset_properties(dataset_config: SupportedDataset) -> dict[str, list[float]]:
    """Load property distributions from training dataset using cached properties.

    Args:
        dataset_config: Dataset configuration

    Returns:
        Dictionary mapping property names to lists of values
    """
    print("\n[Dataset Properties] Loading training dataset properties from cache...")

    from src.datasets.utils import get_dataset_props

    base_dataset = "qm9" if "QM9" in dataset_config.name else "zinc"
    dataset_props_obj = get_dataset_props(base_dataset=base_dataset, splits=["train"])

    dataset_props = {
        "logp": dataset_props_obj.logp,
        "qed": dataset_props_obj.qed,
        "sa_score": dataset_props_obj.sa_score,
        "max_ring_size": dataset_props_obj.max_ring_size_data,
    }

    print(f"  ✓ Loaded {len(dataset_props['logp'])} samples from training dataset")
    for prop, values in dataset_props.items():
        print(f"    {prop}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")

    return dataset_props


# ===== Distribution Plotting =====
def plot_property_distribution(
    dataset_props: dict[str, list[float]],
    generated_props: dict[str, list[float]],
    target_value: float,
    property_name: str,
    output_dir: Path,
    set_name: str = "All Valid",
):
    """Create overlay distribution plot for a specific property.

    Args:
        dataset_props: Dictionary of dataset property values
        generated_props: Dictionary of generated property values
        target_value: Target value to mark with dotted line
        property_name: Name of the property to plot
        output_dir: Directory to save plot
        set_name: Name of the evaluation set (for title)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Get property values
    dataset_vals = np.array(dataset_props[property_name])
    generated_vals = np.array(generated_props[property_name])

    # Calculate ranges
    all_values = np.concatenate([dataset_vals, generated_vals])
    min_val = np.min(all_values) - 0.5
    max_val = np.max(all_values) + 0.5
    x_range = np.linspace(min_val, max_val, 300)

    # Plot dataset distribution (KDE)
    if len(dataset_vals) > 1:
        dataset_kde = gaussian_kde(dataset_vals)
        ax.plot(
            x_range,
            dataset_kde(x_range),
            label="Training Dataset",
            color="blue",
            alpha=0.7,
            linewidth=2.5,
        )
        ax.fill_between(x_range, dataset_kde(x_range), alpha=0.2, color="blue")

    # Plot generated distribution (KDE)
    if len(generated_vals) > 1:
        generated_kde = gaussian_kde(generated_vals)
        ax.plot(
            x_range,
            generated_kde(x_range),
            label=f"Generated ({set_name})",
            color="red",
            alpha=0.7,
            linewidth=2.5,
        )
        ax.fill_between(x_range, generated_kde(x_range), alpha=0.2, color="red")

    # Add target line
    ax.axvline(
        x=target_value,
        color="green",
        linestyle="--",
        linewidth=2.5,
        label=f"Target = {target_value}",
        zorder=10,
    )

    # Styling
    ax.set_xlabel(f"{property_name.upper()}", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(
        f"{property_name.upper()} Distribution: Dataset vs Generated ({set_name})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()
    filename = f"{property_name}_distribution_{set_name.lower().replace(' ', '_')}.pdf"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved {filename}")


def plot_multi_property_panel(
    dataset_props: dict[str, list[float]],
    all_valid_props: dict[str, list[float]],
    filter2_props: dict[str, list[float]],
    target_property: str,
    target_value: float,
    output_dir: Path,
):
    """Create 2x2 panel with distributions for key properties.

    Args:
        dataset_props: Dictionary of dataset property values
        all_valid_props: Dictionary of All Valid set property values
        filter2_props: Dictionary of Filter 2 set property values
        target_property: Name of the targeted property
        target_value: Target value to mark with dotted line
        output_dir: Directory to save plot
    """
    # Minimum samples required for KDE (avoid singular covariance matrix)
    MIN_SAMPLES_FOR_KDE = 5

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    properties = ["logp", "qed", "sa_score", "max_ring_size"]

    for idx, (ax, prop) in enumerate(zip(axes.flat, properties, strict=False)):
        # Get property values
        dataset_vals = np.array(dataset_props[prop])
        all_valid_vals = np.array(all_valid_props[prop])
        filter2_vals = np.array(filter2_props[prop])

        # Calculate ranges
        all_values = np.concatenate([dataset_vals, all_valid_vals, filter2_vals])
        min_val = np.min(all_values) - 0.5
        max_val = np.max(all_values) + 0.5
        x_range = np.linspace(min_val, max_val, 300)

        # Plot dataset distribution
        if len(dataset_vals) >= MIN_SAMPLES_FOR_KDE:
            try:
                dataset_kde = gaussian_kde(dataset_vals)
                ax.plot(
                    x_range,
                    dataset_kde(x_range),
                    label="Dataset",
                    color="blue",
                    alpha=0.6,
                    linewidth=2,
                )
            except np.linalg.LinAlgError:
                # KDE failed due to singular covariance - skip this distribution
                print(f"  ⚠ Skipping Dataset KDE for {prop} (singular covariance)")

        # Plot All Valid distribution
        if len(all_valid_vals) >= MIN_SAMPLES_FOR_KDE:
            try:
                all_valid_kde = gaussian_kde(all_valid_vals)
                ax.plot(
                    x_range,
                    all_valid_kde(x_range),
                    label="All Valid",
                    color="red",
                    alpha=0.6,
                    linewidth=2,
                )
            except np.linalg.LinAlgError:
                # KDE failed due to singular covariance - skip this distribution
                print(f"  ⚠ Skipping All Valid KDE for {prop} (singular covariance)")

        # Plot Filter 2 distribution
        if len(filter2_vals) >= MIN_SAMPLES_FOR_KDE:
            try:
                filter2_kde = gaussian_kde(filter2_vals)
                ax.plot(
                    x_range,
                    filter2_kde(x_range),
                    label="Filter 2",
                    color="purple",
                    alpha=0.6,
                    linewidth=2,
                )
            except np.linalg.LinAlgError:
                # KDE failed due to singular covariance - skip this distribution
                print(f"  ⚠ Skipping Filter 2 KDE for {prop} (singular covariance)")

        # Highlight targeted property with target line
        if prop == target_property:
            ax.axvline(
                x=target_value,
                color="green",
                linestyle="--",
                linewidth=2.5,
                label=f"Target = {target_value}",
                zorder=10,
            )
            ax.set_facecolor("#f9f9f9")  # Light background for target

        ax.set_xlabel(prop.upper(), fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"{prop.upper()} Distribution", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, framealpha=0.9, loc="best")
        ax.grid(True, alpha=0.3, linestyle=":")

    plt.suptitle(
        f"Property Distributions: Dataset vs Generated\\nTarget: {target_property.upper()} = {target_value}",
        fontsize=16,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "property_distributions_panel.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print("  ✓ Saved property_distributions_panel.pdf")


# ===== Plotting =====
def plot_mad_comparison(all_results: list[dict], output_dir: Path, property_name: str):
    """Plot MAD comparison across targets."""
    targets = [r["config"]["target_value"] for r in all_results]
    all_valid_mad = [r["all_valid_results"]["mad"] for r in all_results]
    filter2_mad = [r["filter2_results"]["mad"] for r in all_results]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(targets))
    width = 0.35

    ax.bar(x - width / 2, all_valid_mad, width, label="All Valid", color="#1f77b4", alpha=0.8)
    ax.bar(x + width / 2, filter2_mad, width, label="Filter 2", color="#ff7f0e", alpha=0.8)

    ax.set_xlabel("Target Value", fontsize=12, fontweight="bold")
    ax.set_ylabel("MAD (Mean Absolute Deviation)", fontsize=12, fontweight="bold")
    ax.set_title(f"MAD Comparison: {property_name.upper()}", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in targets])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "mad_comparison.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print("  ✓ Saved MAD comparison plot")


def plot_vun_metrics(all_results: list[dict], output_dir: Path, property_name: str):
    """Plot VUN (Validity, Uniqueness, Novelty) metrics."""
    targets = [r["config"]["target_value"] for r in all_results]

    validity = [r["all_valid_results"]["validity"] for r in all_results]
    uniqueness = [r["all_valid_results"]["uniqueness"] for r in all_results]
    novelty = [r["all_valid_results"]["novelty"] for r in all_results]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(targets))
    width = 0.25

    ax.bar(x - width, validity, width, label="Validity", color="#2ecc71", alpha=0.8)
    ax.bar(x, uniqueness, width, label="Uniqueness", color="#3498db", alpha=0.8)
    ax.bar(x + width, novelty, width, label="Novelty", color="#e74c3c", alpha=0.8)

    ax.set_xlabel("Target Value", fontsize=12, fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
    ax.set_title(f"VUN Metrics: {property_name.upper()}", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in targets])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(output_dir / "vun_metrics.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print("  ✓ Saved VUN metrics plot")


def plot_sample_counts(all_results: list[dict], output_dir: Path, property_name: str):
    """Plot sample counts for All Valid and Filter 2."""
    targets = [r["config"]["target_value"] for r in all_results]
    all_valid_counts = [r["all_valid_results"]["n_samples"] for r in all_results]
    filter2_counts = [r["filter2_results"]["n_samples"] for r in all_results]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(targets))
    width = 0.35

    ax.bar(x - width / 2, all_valid_counts, width, label="All Valid", color="#1f77b4", alpha=0.8)
    ax.bar(x + width / 2, filter2_counts, width, label="Filter 2", color="#ff7f0e", alpha=0.8)

    ax.set_xlabel("Target Value", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Samples", fontsize=12, fontweight="bold")
    ax.set_title(f"Valid Sample Counts: {property_name.upper()}", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in targets])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "sample_counts.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print("  ✓ Saved sample counts plot")


# ===== Aggregate Results =====
def compute_aggregate_results(all_results: list[dict], property_name: str) -> dict:
    """Compute aggregate metrics across all targets."""
    # Aggregate MAD (average across targets)
    all_valid_mad_values = [r["all_valid_results"]["mad"] for r in all_results]
    filter2_mad_values = [r["filter2_results"]["mad"] for r in all_results]

    aggregate_all_valid_mad = np.mean(all_valid_mad_values)
    aggregate_filter2_mad = np.mean(filter2_mad_values)

    # Aggregate VUN (average across targets)
    avg_validity = np.mean([r["all_valid_results"]["validity"] for r in all_results])
    avg_uniqueness = np.mean([r["all_valid_results"]["uniqueness"] for r in all_results])
    avg_novelty = np.mean([r["all_valid_results"]["novelty"] for r in all_results])

    # Aggregate sample counts
    total_all_valid = sum([r["all_valid_results"]["n_samples"] for r in all_results])
    total_filter2 = sum([r["filter2_results"]["n_samples"] for r in all_results])

    return {
        "property": property_name,
        "n_targets": len(all_results),
        "targets": [r["config"]["target_value"] for r in all_results],
        "aggregate_metrics": {
            "all_valid_mad_mean": aggregate_all_valid_mad,
            "all_valid_mad_std": np.std(all_valid_mad_values),
            "filter2_mad_mean": aggregate_filter2_mad,
            "filter2_mad_std": np.std(filter2_mad_values),
            "avg_validity": avg_validity,
            "avg_uniqueness": avg_uniqueness,
            "avg_novelty": avg_novelty,
            "total_all_valid_samples": total_all_valid,
            "total_filter2_samples": total_filter2,
        },
        "per_target_metrics": {
            "all_valid_mad": all_valid_mad_values,
            "filter2_mad": filter2_mad_values,
        },
    }


# ===== Main Execution =====
def run_final_evaluation(
    hpo_dir: Path,
    n_samples: int = 10000,
    draw: bool = False,
    max_draw: int = 100,
    output_dir: Path = None,
    top_k_property: int = 100,
    top_n_best: int = 10,
    decoder_k: int = 10,
    target_value: float = None,
):
    """
    Run final evaluation for all targets.

    Args:
        hpo_dir: Path to HPO results directory
        n_samples: Number of samples to generate per target
        draw: Whether to draw molecules
        max_draw: Maximum number of molecules to draw
        output_dir: Output directory for results
        top_k_property: Number of top molecules by property distance for evaluation set 3
        top_n_best: Number of best molecules to expand with all decoder outputs for evaluation set 4
        decoder_k: Number of decoder outputs per molecule for evaluation set 4
    """
    print("=" * 80)
    print("Property Targeting Final Evaluation (MG-DIFF Protocol)")
    print("=" * 80)

    # Parse HPO directory
    print("\n[1/5] Parsing HPO experiment directory...")
    experiment_info = parse_hpo_directory(hpo_dir)

    metadata = experiment_info["metadata"]
    property_name = metadata["property"]
    dataset = SupportedDataset(metadata["dataset"])

    print(f"  Property: {property_name}")
    print(f"  Dataset: {dataset.value}")
    print(f"  Targets: {len(experiment_info['targets'])}")

    # Create output directory
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"final_results/{property_name}_{dataset.value}_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print("\n[2/5] Loading models...")
    gen_model_hint = GENERATOR_REGISTRY[dataset][metadata["gen_model_idx"]]
    decoder_settings = DecoderSettings.get_default_for(base_dataset=dataset.default_cfg.base_dataset)
    generator = HDCGenerator(
        gen_model_hint=gen_model_hint,
        ds_config=dataset.default_cfg,
        device=DEVICE,
        decoder_settings=decoder_settings,
    )

    regressor_hints = REGRESSOR_REGISTRY[dataset].get(property_name, [])
    if not regressor_hints:
        raise ValueError(f"No regressor available for {property_name} on {dataset.value}")

    pr_path = get_pr_path(hint=regressor_hints[0])
    property_regressor = retrieve_model(name="PR").load_from_checkpoint(pr_path).to(DEVICE).eval()

    print("  ✓ Models loaded successfully!")

    # Load dataset properties for distribution plots
    print("\n[3/5] Loading dataset properties for distribution plots...")
    dataset_props = load_dataset_properties(dataset)
    print(f"  ✓ Loaded {len(dataset_props['logp'])} samples from training dataset")

    # Filter targets if specific target requested
    targets_to_process = experiment_info["targets"]
    if target_value is not None:
        targets_to_process = [t for t in targets_to_process if abs(t["value"] - target_value) < 1e-6]
        if not targets_to_process:
            raise ValueError(
                f"Target value {target_value} not found in HPO results. Available targets: {[t['value'] for t in experiment_info['targets']]}"
            )
        print(f"\n[4/5] Running final evaluation for specific target: {target_value}")
    else:
        print(f"\n[4/5] Running final evaluation for {len(targets_to_process)} targets...")

    all_results = []
    for target_info in targets_to_process:
        result = run_final_evaluation_for_target(
            target_info=target_info,
            experiment_metadata=metadata,
            generator=generator,
            property_regressor=property_regressor,
            n_samples=n_samples,
            draw=draw,
            max_draw=max_draw,
            output_dir=output_dir,
            dataset_props=dataset_props,
            top_k_property=top_k_property,
            top_n_best=top_n_best,
            decoder_k=decoder_k,
        )
        all_results.append(result)

    # Compute aggregate results
    print("\n[5/6] Computing aggregate results...")
    aggregate_results = compute_aggregate_results(all_results, property_name)

    # Save aggregate results
    with open(output_dir / "aggregate_results.json", "w") as f:
        json.dump(aggregate_results, f, indent=2, default=float)

    print(f"\n  Aggregate MAD (All Valid): {aggregate_results['aggregate_metrics']['all_valid_mad_mean']:.4f}")
    print(f"  Aggregate MAD (Filter 2):  {aggregate_results['aggregate_metrics']['filter2_mad_mean']:.4f}")

    # Generate plots
    print("\n[6/6] Generating plots...")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_mad_comparison(all_results, plots_dir, property_name)
    plot_vun_metrics(all_results, plots_dir, property_name)
    plot_sample_counts(all_results, plots_dir, property_name)

    print("\n" + "=" * 80)
    print("Final Evaluation Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("  - Per-target results: target_*/")
    print("  - Aggregate results: aggregate_results.json")
    print("  - Plots: plots/")
    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Final evaluation for property targeting (MG-DIFF protocol)")

    parser.add_argument(
        "--hpo_dir",
        type=str,
        default="/home/akaveh/Projects/kit/graph_hdc/src/exp/final_evaluations/property_targeting/hpo_results/logp_QM9_SMILES_HRR_256_F64_G1NG3_20251114_213822",
        help="Path to HPO experiment directory",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=200,
        help="Number of samples to generate per target (default: 10000)",
    )
    parser.add_argument(
        "--draw",
        action="store_true",
        help="Draw molecules and save to <target_dir>/drawings/",
    )
    parser.add_argument(
        "--max_draw",
        type=int,
        default=100,
        help="Maximum number of molecules to draw per target (default: 100)",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=None,
        help="Specific target value to process (if not provided, processes all targets)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: final_results/<property>_<dataset>_<timestamp>)",
    )
    parser.add_argument(
        "--top_k_property",
        type=int,
        default=100,
        help="Number of top molecules by property distance for evaluation set 3 (default: 100)",
    )
    parser.add_argument(
        "--top_n_best",
        type=int,
        default=10,
        help="Number of best molecules to expand with all decoder outputs for evaluation set 4 (default: 10)",
    )
    parser.add_argument(
        "--decoder_k",
        type=int,
        default=10,
        help="Number of decoder outputs per molecule for evaluation set 4 (default: 10)",
    )

    args = parser.parse_args()

    hpo_dir = Path(args.hpo_dir)
    if not hpo_dir.exists():
        raise FileNotFoundError(f"HPO directory not found: {hpo_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else None

    run_final_evaluation(
        hpo_dir=hpo_dir,
        n_samples=args.n_samples,
        draw=args.draw,
        max_draw=args.max_draw,
        output_dir=output_dir,
        top_k_property=args.top_k_property,
        top_n_best=args.top_n_best,
        decoder_k=args.decoder_k,
        target_value=args.target,
    )


if __name__ == "__main__":
    main()
