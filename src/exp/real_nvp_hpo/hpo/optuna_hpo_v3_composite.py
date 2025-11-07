"""
Optuna HPO Script for Real NVP-V3 with Multi-Objective Optimization

This script uses the NVP-V3 model (semantic masking architecture) with multi-objective
optimization to find the Pareto front between:
  1. Minimizing negative log-likelihood (NLL)
  2. Maximizing CorrectionLevel.ZERO percentage (decoder quality)

NVP-V3 Architecture Differences:
- Semantic masking: Alternates between transforming edge_terms and graph_terms
- Configurable MLP: hidden_dim and num_hidden_layers instead of num_hidden_channels
- ActNorm enabled by default

Multi-Objective Optimization:
Instead of optimizing a fixed weighted combination, Optuna explores the Pareto front,
allowing post-hoc selection of solutions based on preferred NLL/quality tradeoff.

Usage:
======
# QM9 dataset (1000 decodes per trial)
python optuna_hpo_v3_composite.py --dataset QM9_SMILES_HRR_1600_F64_G1NG3 --n_trials 50

# ZINC dataset (100 decodes per trial for efficiency)
python optuna_hpo_v3_composite.py --dataset ZINC_SMILES_HRR_1024_F64_5G1NG4 --n_trials 100
python optuna_hpo_v3_composite.py --dataset ZINC_SMILES_HRR_2048_F64_5G1NG4 --n_trials 100

# 256-dim datasets
python optuna_hpo_v3_composite.py --dataset QM9_SMILES_HRR_256_F64_G1NG3 --n_trials 50
python optuna_hpo_v3_composite.py --dataset ZINC_SMILES_HRR_256_F64_5G1NG4 --n_trials 100
"""

import argparse
import json
import math
import pathlib
from pathlib import Path

import optuna
import pandas as pd
import torch
from optuna_integration import BoTorchSampler

from src.encoding.configs_and_constants import SupportedDataset

# Import from real_nvp_v3_composite instead of real_nvp_composite
from src.exp.real_nvp_hpo.real_nvp_v3_composite import get_hidden_dim_dist, run_qm9_trial, run_zinc_trial

DIRECTION = "minimize"


def get_space(dataset: SupportedDataset):
    """
    Define the hyperparameter search space for NVP-V3.

    Key differences from NVP:
    - hidden_dim instead of num_hidden_channels (MLP hidden layer width)
    - num_hidden_layers (new parameter, 2-5 layers)
    """
    low, high, step = get_hidden_dim_dist(dataset)
    return {
        "batch_size": optuna.distributions.IntDistribution(32, 512, step=32),
        "lr": optuna.distributions.FloatDistribution(5e-5, 1e-3, log=True),
        "weight_decay": optuna.distributions.FloatDistribution(1e-6, 1e-3, log=True),
        "num_flows": optuna.distributions.IntDistribution(4, 16),
        "hidden_dim": optuna.distributions.IntDistribution(low, high, step=step),
        "num_hidden_layers": optuna.distributions.IntDistribution(2, 4),
    }


def load_study(study_name: str, sqlite_path: str) -> optuna.Study:
    """Create or load an Optuna study bound to a local SQLite file."""
    return optuna.create_study(
        study_name=study_name,
        directions=[DIRECTION, DIRECTION],
        storage=f"sqlite:///{sqlite_path}",
        load_if_exists=True,
        sampler=BoTorchSampler(seed=42),
    )


def rebuild_study_from_csv(
    study_name: str, dataset: SupportedDataset, csv: pathlib.Path, db_path: pathlib.Path
) -> optuna.Study | None:
    study = load_study(study_name, str(db_path))

    if not csv.exists():
        print("No CSV; nothing to rebuild.")
        return study

    df = pd.read_csv(csv)
    if df.empty:
        print(f"Empty CSV@{csv}; nothing to rebuild.")
        return study
    space = get_space(dataset)
    added = 0
    for _, r in df.iterrows():
        params = {k: r[k] for k in space if k in r and pd.notna(r[k])}
        # Multi-objective: read both objectives
        obj0 = float(r["obj0_nll"]) if "obj0_nll" in r and pd.notna(r["obj0_nll"]) else None
        obj1 = float(r["obj1_zero_correction_error"]) if "obj1_zero_correction_error" in r and pd.notna(r["obj1_zero_correction_error"]) else None
        values = [obj0, obj1] if obj0 is not None and obj1 is not None else None

        # Restore all user attributes from CSV (not just exp_dir_name)
        # Skip standard columns: dataset, number, objectives, state, and hyperparameters
        standard_cols = {"dataset", "number", "obj0_nll", "obj1_zero_correction_error", "state"}
        user_attrs = {}
        for col in r.index:
            if col not in standard_cols and col not in space and pd.notna(r[col]):
                # Convert numpy types to Python types for JSON serialization
                val = r[col]
                if hasattr(val, 'item'):  # numpy scalar
                    val = val.item()
                user_attrs[col] = val

        if values:
            t = optuna.trial.create_trial(
                params=params,
                distributions=space,
                values=values,
                state=optuna.trial.TrialState.COMPLETE,
                user_attrs=user_attrs,  # <-- persist exp_dir_name into DB if present
            )
            study.add_trial(t)
            added += 1

    print(f"Rebuilt {added} trials into {db_path} for dataset='{dataset.name}'.")
    return study


def export_trials(study_name: str, db_path: pathlib.Path, dataset: SupportedDataset, csv: pathlib.Path):
    """
    Export trials to CSV for rebuilding the study on other machines.
    Overwrites the dataset's CSV (no append/mix), includes exp_dir_name.
    """
    study = load_study(study_name, str(db_path))

    rows = []
    space = get_space(dataset)
    for t in study.get_trials(deepcopy=False):
        row = {
            "dataset": dataset.default_cfg.name,
            "number": t.number,
            # Multi-objective: store both objectives separately
            "obj0_nll": t.values[0] if t.values else None,
            "obj1_zero_correction_error": t.values[1] if t.values else None,
            "state": t.state.name if hasattr(t.state, "name") else str(t.state),
        }
        # Add all hyperparameters
        for k in space:
            row[k] = t.params.get(k, None)
        # Add all user attributes
        for attr_name, attr_value in t.user_attrs.items():
            row[attr_name] = attr_value
        rows.append(row)

    tidy = pd.DataFrame(rows)
    # Since you keep separate CSV per dataset, just overwrite it:
    tidy.to_csv(csv, index=False)
    print(f"Wrote {len(tidy)} rows for dataset='{dataset}' to {csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Real NVP-V3 - HPO with Composite Metric (NLL + CorrectionLevel.ZERO)")
    p.add_argument(
        "--dataset",
        type=str,
        default=SupportedDataset.ZINC_SMILES_HRR_1024_F64_5G1NG4.value,
        choices=[ds.value for ds in SupportedDataset],
    )
    p.add_argument("--n_trials", type=int, default=1)
    p.add_argument("--norm_per", type=str, default="term", choices=["term", "dim"])
    args = p.parse_args()

    ds = SupportedDataset(args.dataset)

    # Paths (per-dataset DB + CSV) - use "_v3_comp" suffix to distinguish from V2 and standard HPO
    here = pathlib.Path(__file__).parent
    study_base = here.parent.name
    study_name = f"{study_base}_v3_comp_{ds.default_cfg.name}_{args.norm_per}"
    db_path = here / f"real_nvp_hpo_v3_comp_{ds.default_cfg.name}_{args.norm_per}.db"
    csv = here / f"trials_v3_comp_{ds.default_cfg.name}_{args.norm_per}.csv"

    print(f"Study name: {study_name}")
    print(f"Database: {db_path}")
    print(f"CSV export: {csv}")
    print("Model: NVP-V3 (semantic masking architecture)")
    print("Multi-objective optimization: (1) minimize NLL, (2) minimize incorrect percentage (non-zero corrections)")
    print(f"  Dataset: {ds.default_cfg.name}")
    print(f"  Base dataset: {ds.default_cfg.base_dataset}")
    print(f"  Decode samples: {1000 if ds.default_cfg.base_dataset == 'qm9' else 100}")
    print(f"  Normalization per: {args.norm_per}")
    print(
        "  HPO space: hidden_dim, num_hidden_layers, num_flows, batch_size, lr, weight_decay, per_term_standardization"
    )
    print()

    # Rebuild if DB missing, else load
    if not db_path.exists():
        study = rebuild_study_from_csv(study_name=study_name, dataset=ds, csv=csv, db_path=db_path)
        if study is None:
            study = load_study(study_name=study_name, sqlite_path=str(db_path))
    else:
        study = load_study(study_name=study_name, sqlite_path=str(db_path))

    # Choose an objective provided by your code (now returns composite metric)
    base_objective = run_qm9_trial if ds.default_cfg.base_dataset == "qm9" else run_zinc_trial

    # Wrapper to read metrics and store in trial attributes with error handling
    def objective(trial: optuna.Trial) -> tuple[float, float]:
        try:
            # Note: base_objective returns (min_nll, 100.0 - zero_correction_pct) = (min_nll, incorrect_pct)
            min_nll, incorrect_pct = base_objective(trial, dataset=ds, norm_per=args.norm_per)

            # Check for NaN/inf from a successful but diverged run
            if not math.isfinite(min_nll) or not math.isfinite(incorrect_pct):
                print(f"Trial {trial.number} resulted in non-finite values (NaN/inf)")
                trial.set_user_attr("failure_reason", "DIVERGED")
                return (float('inf'), float('inf'))

            # Find the most recent metrics file (just created by the training run)
            # Training script saves to: src/exp/real_nvp_hpo/results/real_nvp_v3_composite/{exp_dir_name}/evaluations/hpo_metrics.json
            training_script_dir = here.parent  # Go up from hpo/ to real_nvp_hpo/
            results_dir = training_script_dir / "results" / "real_nvp_v3_composite"

            # Find all hpo_metrics.json files and get the most recent one
            metrics_files = list(results_dir.glob("*/evaluations/hpo_metrics.json"))
            if metrics_files:
                # Sort by modification time, get most recent
                metrics_file = max(metrics_files, key=lambda p: p.stat().st_mtime)

                with open(metrics_file) as f:
                    metrics = json.load(f)

                # Store exp_dir_name from the metrics file (this is the actual folder name used)
                exp_dir_name = metrics.get("exp_dir_name")
                if exp_dir_name:
                    trial.set_user_attr("exp_dir_name", exp_dir_name)

                # Store key metrics for easy filtering/analysis
                trial.set_user_attr("min_val_loss", metrics.get("min_val_loss"))
                trial.set_user_attr("incorrect_pct", metrics.get("incorrect_pct"))
                trial.set_user_attr("zero_correction_pct", metrics.get("zero_correction_pct"))
                trial.set_user_attr("n_decoded", metrics.get("n_decoded"))
                trial.set_user_attr("decode_time_sec", metrics.get("decode_time_sec"))
                trial.set_user_attr("decode_time_per_sample_sec", metrics.get("decode_time_per_sample_sec"))
                trial.set_user_attr("norm_per", args.norm_per)

                # Store correction level distribution
                if "correction_level_distribution" in metrics:
                    for level, count in metrics["correction_level_distribution"].items():
                        # Replace spaces with underscores for attribute names
                        attr_name = f"cl_{level.replace(' ', '_')}"
                        trial.set_user_attr(attr_name, count)
            else:
                print(f"Warning: No metrics files found in {results_dir}")

            return min_nll, incorrect_pct

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Trial {trial.number} failed with CUDA OOM")
                trial.set_user_attr("failure_reason", "CUDA_OOM")
                torch.cuda.empty_cache()
                return (float('inf'), float('inf'))
            else:
                print(f"Trial {trial.number} failed with RuntimeError: {e}")
                trial.set_user_attr("failure_reason", f"RuntimeError: {str(e)[:100]}")
                return (float('inf'), float('inf'))

        except Exception as e:
            print(f"Trial {trial.number} failed with unexpected error: {e}")
            trial.set_user_attr("failure_reason", f"{type(e).__name__}: {str(e)[:100]}")
            return (float('inf'), float('inf'))

    # Run optimization with error handling
    try:
        print(f"Starting optimization with {args.n_trials} trials...")
        study.optimize(objective, n_trials=args.n_trials)
    except KeyboardInterrupt:
        print("\nHPO interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"HPO study error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always export current results, even if interrupted
        export_trials(study_name=study_name, db_path=db_path, dataset=ds, csv=csv)

    print()
    print("=== Pareto Front Summary ===")
    print(f"Total trials: {len(study.trials)}")
    print(f"Pareto-optimal trials: {len(study.best_trials)}")
    if study.best_trials:
        print("\nTop 5 Pareto-optimal solutions:")
        print("(Objectives: minimize NLL, minimize incorrect%; displayed as maximize CorrectionLevel.ZERO%)")
        for i, trial in enumerate(study.best_trials[:5], 1):
            nll = trial.values[0]
            incorrect_pct = trial.values[1]
            zero_pct = 100.0 - incorrect_pct
            print(f"  {i}. NLL: {nll:.4f}, Incorrect%: {incorrect_pct:.2f}%, CorrectionLevel.ZERO%: {zero_pct:.2f}%")
            if "exp_dir_name" in trial.user_attrs:
                print(f"     Experiment dir: {trial.user_attrs['exp_dir_name']}")
