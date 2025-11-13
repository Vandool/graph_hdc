#!/usr/bin/env python
"""
Optuna HPO Script for Spline Flow with Multi-Objective Optimization

This script uses `spline_flow.py` to find the Pareto front between:
  1. Minimizing NLL (min_val_loss)
  2. Maximizing Generation Quality (by minimizing incorrect_pct)
"""

import argparse
import datetime
import json
import math
import pathlib
import traceback

import optuna
import pandas as pd
import torch
from optuna_integration import BoTorchSampler

from src.encoding.configs_and_constants import SupportedDataset
from src.exp.flow_matching.hpo.folder_name import make_run_folder_name
from src.exp.spline_flow.spline_flow import run_experiment
from src.normalizing_flow.models import SFConfig

# Define keys for folder naming
SF_HPO_KEYS = {
    "batch_size",
    "lr",
    "weight_decay",
    "num_flows",
    "num_hidden_channels",
    "num_bins",
    "num_blocks",
    "dropout_probability",
}


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------
# HPO Space Definition (Your correct version)
# ---------------------------------------------------------------------


def get_spline_space(dataset: SupportedDataset) -> dict:
    """Defines the Optuna search space for SplineFlow."""
    input_dim = dataset.default_cfg.hv_dim * dataset.default_cfg.hv_count
    hidden_low = input_dim
    hidden_high = input_dim * 4
    hidden_step = input_dim // 2

    return {
        "batch_size": optuna.distributions.IntDistribution(128, 512, step=128),
        "lr": optuna.distributions.FloatDistribution(1e-5, 5e-4, log=True),
        "weight_decay": optuna.distributions.FloatDistribution(1e-7, 1e-4, log=True),
        "num_flows": optuna.distributions.IntDistribution(4, 12, step=2),
        "num_hidden_channels": optuna.distributions.IntDistribution(hidden_low, hidden_high, step=hidden_step),
        "num_bins": optuna.distributions.IntDistribution(4, 16, step=4),
        "num_blocks": optuna.distributions.IntDistribution(2, 6, step=1),
        "dropout_probability": optuna.distributions.FloatDistribution(0.0, 0.3, step=0.1),
    }


# ---------------------------------------------------------------------
# Optuna Study Management (Cleaned and Corrected)
# ---------------------------------------------------------------------


def load_study(study_name: str, sqlite_path: str) -> optuna.Study:
    """Create or load a multi-objective Optuna study."""
    # *** This is the only load_study function needed ***
    return optuna.create_study(
        study_name=study_name,
        directions=["minimize", "minimize"],  # [0] min_val_loss, [1] incorrect_pct
        storage=f"sqlite:///{sqlite_path}",
        load_if_exists=True,
        sampler=BoTorchSampler(seed=43, consider_running_trials=True, n_startup_trials=5),
    )


def export_trials_to_csv(study: optuna.Study, csv_path: pathlib.Path):
    """Exports all trial data to a CSV file with consistent naming."""
    df = study.trials_dataframe(attrs=("value", "state", "params", "user_attrs"))

    # *** Use consistent column names for objectives ***
    df = df.rename(columns={"values_0": "obj0_min_val_loss", "values_1": "obj1_incorrect_pct"})

    df.to_csv(csv_path, index=False)
    log(f"Exported {len(df)} trials to {csv_path}")


def rebuild_study_from_csv(
    study_name: str, dataset: SupportedDataset, csv: pathlib.Path, db_path: pathlib.Path
) -> optuna.Study:
    """Rebuilds a study from a CSV export."""
    study = load_study(study_name, str(db_path))

    if not csv.exists():
        log("No CSV found; nothing to rebuild.")
        return study

    df = pd.read_csv(csv)
    if df.empty:
        log(f"Empty CSV@{csv}; nothing to rebuild.")
        return study

    space = get_spline_space(dataset)
    added = 0

    # *** Use consistent column names ***
    obj0_col = "obj0_min_val_loss"
    obj1_col = "obj1_incorrect_pct"

    for _, r in df.iterrows():
        params = {k: r[k] for k in space if k in r and pd.notna(r[k])}

        obj0 = float(r[obj0_col]) if obj0_col in r and pd.notna(r[obj0_col]) else None
        obj1 = float(r[obj1_col]) if obj1_col in r and pd.notna(r[obj1_col]) else None
        values = [obj0, obj1] if obj0 is not None and obj1 is not None else None

        standard_cols = {"dataset", "number", obj0_col, obj1_col, "state"}
        user_attrs = {}
        for col in r.index:
            if col not in standard_cols and col not in space and pd.notna(r[col]):
                val = r[col]
                if hasattr(val, "item"):
                    val = val.item()
                user_attrs[col] = val

        if values:
            try:
                study.add_trial(
                    optuna.trial.create_trial(
                        params=params,
                        distributions=space,
                        values=values,
                        state=optuna.trial.TrialState.COMPLETE,
                        user_attrs=user_attrs,
                    )
                )
                added += 1
            except (ValueError, TypeError) as e:
                if "already exists" not in str(e):
                    log(f"Warning: Could not add trial {r.get('number', 'N/A')}: {e}")

    log(f"Rebuilt {added} trials into {db_path} for dataset='{dataset.name}'.")
    return study


# ---------------------------------------------------------------------
# HPO Objective Function (Corrected)
# ---------------------------------------------------------------------


def objective(trial: optuna.Trial, dataset: SupportedDataset, norm_per: str) -> tuple[float, float]:
    """
    The main objective function for Optuna.
    Creates a config, runs the experiment, and returns the two objectives.
    """
    space = get_spline_space(dataset)

    # 1. Create the Config from the trial
    cfg = SFConfig()
    cfg.dataset = dataset
    cfg.hv_dim = dataset.default_cfg.hv_dim
    cfg.hv_count = dataset.default_cfg.hv_count
    cfg.vsa = dataset.default_cfg.vsa.value
    cfg.per_term_standardization = norm_per == "term"

    # Apply HPO params
    cfg.batch_size = trial.suggest_int("batch_size", **space["batch_size"]._asdict())
    cfg.lr = trial.suggest_float("lr", **space["lr"]._asdict())
    cfg.weight_decay = trial.suggest_float("weight_decay", **space["weight_decay"]._asdict())
    cfg.num_flows = trial.suggest_int("num_flows", **space["num_flows"]._asdict())
    cfg.num_hidden_channels = trial.suggest_int("num_hidden_channels", **space["num_hidden_channels"]._asdict())
    cfg.num_blocks = trial.suggest_int("num_blocks", **space["num_blocks"]._asdict())
    cfg.num_bins = trial.suggest_int("num_bins", **space["num_bins"]._asdict())
    cfg.dropout_probability = trial.suggest_float("dropout_probability", **space["dropout_probability"]._asdict())

    hpo_params_for_name = {k: getattr(cfg, k) for k in SF_HPO_KEYS}
    cfg.exp_dir_name = make_run_folder_name(hpo_params_for_name, prefix=f"sf_hpo_{dataset.default_cfg.name}")
    trial.set_user_attr("exp_dir_name", cfg.exp_dir_name)
    trial.set_user_attr("norm_per", norm_per)

    try:
        # 2. Run the experiment
        min_val_loss, incorrect_pct = run_experiment(cfg)

        if not (math.isfinite(min_val_loss) and math.isfinite(incorrect_pct)):
            log(f"Trial {trial.number} resulted in non-finite values (NaN/inf)")
            trial.set_user_attr("failure_reason", "DIVERGED")
            return float("inf"), float("inf")  # Prune this trial

        # (This logic is adapted from your FM HPO script)
        try:
            # Find the metrics file that was just created
            here = pathlib.Path(__file__).parent
            # Assumes training script is at ../spline_flow.py
            results_dir = here.parent / "results" / "spline_flow"
            metrics_file = results_dir / cfg.exp_dir_name / "evaluations" / "hpo_metrics.json"

            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)

                # Log all key metrics
                trial.set_user_attr("min_val_loss", metrics.get("min_val_loss"))
                trial.set_user_attr("incorrect_pct", metrics.get("incorrect_pct"))
                trial.set_user_attr("zero_correction_pct", metrics.get("zero_correction_pct"))
                trial.set_user_attr("n_decoded", metrics.get("n_decoded"))
                trial.set_user_attr("decode_time_sec", metrics.get("decode_time_sec"))
                trial.set_user_attr("training_time_min", metrics.get("training_time_min"))

                if "correction_level_distribution" in metrics:
                    for level, count in metrics["correction_level_distribution"].items():
                        attr_name = f"cl_{level.replace(' ', '_')}"
                        trial.set_user_attr(attr_name, count)
            else:
                log(f"Warning: Could not find metrics file at {metrics_file}")
                # Fallback to logging the returned values
                trial.set_user_attr("min_val_loss", min_val_loss)
                trial.set_user_attr("incorrect_pct", incorrect_pct)
                trial.set_user_attr("zero_correction_pct", 100.0 - incorrect_pct)

        except Exception as log_e:
            log(f"Warning: Failed to log metrics from JSON: {log_e}")

        # 4. Return the objectives
        return min_val_loss, incorrect_pct

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            log(f"Trial {trial.number} failed with CUDA OOM")
            trial.set_user_attr("failure_reason", "CUDA_OOM")
            torch.cuda.empty_cache()
            raise optuna.TrialPruned("CUDA OOM")
        log(f"Trial {trial.number} failed with RuntimeError: {e}")
        trial.set_user_attr("failure_reason", f"RuntimeError: {str(e)[:100]}")
        raise optuna.TrialPruned(f"RuntimeError: {e}")
    except Exception as e:
        log(f"Trial {trial.number} failed with unexpected error: {e}")
        trial.set_user_attr("failure_reason", f"{type(e).__name__}: {str(e)[:100]}")
        raise optuna.TrialPruned(f"Unexpected error: {e}")


# ---------------------------------------------------------------------
# Main HPO Runner
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SplineFlow HPO with Multi-Objective (NLL vs. Quality)")
    parser.add_argument(
        "--dataset",
        type=str,
        default=SupportedDataset.QM9_SMILES_HRR_256_F64_G1NG3.value,
        choices=[ds.value for ds in SupportedDataset],
    )
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--norm_per", type=str, default="term", choices=["term", "dim"])
    args = parser.parse_args()

    ds = SupportedDataset(args.dataset)

    # Paths (per-dataset DB + CSV) - use "_comp" suffix to distinguish from standard HPO
    here = pathlib.Path(__file__).parent
    study_base = here.parent.name
    study_name = f"{study_base}_multi_{ds.default_cfg.name}_{args.norm_per}"
    db_path = here / f"sf_hpo_multi_{ds.default_cfg.name}_{args.norm_per}.db"
    csv = here / f"trials_sf_multi_{ds.default_cfg.name}_{args.norm_per}.csv"

    log(f"Starting HPO study: {study_name}")
    log(f"Database: {db_path}")
    log(f"CSV export: {csv}")
    log("Multi-objective: (1) minimize NLL, (2) minimize incorrect_pct")
    log(f"Normalization per: {args.norm_per}")
    log(f"Dataset: {ds.value}")

    # *** Rebuild logic from analogue script ***
    if not db_path.exists() and csv.exists():
        log(f"Database not found. Rebuilding from {csv}...")
        study = rebuild_study_from_csv(study_name=study_name, dataset=ds, csv=csv, db_path=db_path)
    else:
        log("Loading study from database...")
        study = load_study(study_name=study_name, sqlite_path=str(db_path))

    # Run optimization
    try:
        log(f"Optimizing for {args.n_trials} trials...")
        study.optimize(lambda trial: objective(trial, dataset=ds, norm_per=args.norm_per), n_trials=args.n_trials)
    except KeyboardInterrupt:
        log("\nHPO interrupted by user (Ctrl+C)")
    except Exception as e:
        log(f"HPO study error: {e}")
        traceback.print_exc()
    finally:
        export_trials_to_csv(study, csv)  # Use the correct function

    log("\n=== Pareto Front Summary ===")
    log(f"Total trials: {len(study.trials)}")
    log(f"Pareto-optimal trials: {len(study.best_trials)}")
    if study.best_trials:
        print("\nTop 5 Pareto-optimal solutions:")
        for i, trial in enumerate(study.best_trials[:5], 1):
            try:
                min_val_loss = trial.values[0]
                incorrect_pct = trial.values[1]
                zero_pct = 100.0 - incorrect_pct
                print(
                    f"  {i}. NLL: {min_val_loss:.4f}, Incorrect%: {incorrect_pct:.2f}%, (ZeroCorrection%: {zero_pct:.2f}%)"
                )
                if "exp_dir_name" in trial.user_attrs:
                    print(f"     Experiment dir: {trial.user_attrs['exp_dir_name']}")
            except (TypeError, IndexError):
                print(f"  {i}. Could not display trial {trial.number} (values: {trial.values})")
