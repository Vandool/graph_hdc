"""
Property Regressor HPO - Multi-property support

Supports hyperparameter optimization for:
- logp: Octanol-water partition coefficient
- sa_score: Synthetic Accessibility Score
- qed: Quantitative Estimate of Drug-likeness
- max_ring_size: Maximum ring size in molecule

Usage:
    # LogP regression (default, backward compatible)
    python property_regression_hpo.py --dataset ZINC_SMILES_HRR_6144_F64_G1G3 --n_trials 50

    # SA Score regression
    python property_regression_hpo.py --dataset QM9_SMILES_HRR_1600 --property sa_score --n_trials 50

    # QED regression
    python property_regression_hpo.py --dataset ZINC_SMILES_HRR_6144_F64_G1G3 --property qed --n_trials 50
"""

import argparse
import pathlib

import optuna
import pandas as pd
from optuna.samplers import TPESampler

from src.encoding.configs_and_constants import SupportedDataset
from src.exp.logp_regressor.hpo.folder_name import make_run_folder_name
from src.exp.logp_regressor.pr import run_trial
from src.generation.property_regressor import ACTS, NORMS


def get_space_for_dataset(dataset: SupportedDataset) -> dict:
    """
    Return appropriate hyperparameter space based on input dimensionality.

    This matches the ranges defined in pr.get_cfg() to maintain single source of truth.
    """
    # Get ranges from pr.py logic (single source of truth)

    if dataset.default_cfg.base_dataset == "qm9":
        h1_min, h1_max = 512, 1536
        h2_min, h2_max = 128, 1024
    elif dataset == SupportedDataset.QM9_SMILES_HRR_256_F64_G1NG3:
        h1_min, h1_max = 256, 512
        h2_min, h2_max = 128, 256
    elif dataset == SupportedDataset.ZINC_SMILES_HRR_1024_F64_5G1NG4:
        h1_min, h1_max = 512, 1024
        h2_min, h2_max = 256, 512
    else:
        h1_min, h1_max = 1024, 2048
        h2_min, h2_max = 512, 1024

    h3_min, h3_max = 64, 128
    h4_min, h4_max = 32, 64

    return {
        "batch_size": optuna.distributions.IntDistribution(32, 512, step=32),
        "lr": optuna.distributions.FloatDistribution(5e-5, 1e-3, log=True),
        "weight_decay": optuna.distributions.CategoricalDistribution([0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 5e-4]),
        "depth": optuna.distributions.CategoricalDistribution([2, 3, 4]),
        "h1": optuna.distributions.IntDistribution(h1_min, h1_max, step=256),
        "h2": optuna.distributions.IntDistribution(h2_min, h2_max, step=128),
        "h3": optuna.distributions.IntDistribution(h3_min, h3_max, step=64),
        "h4": optuna.distributions.IntDistribution(h4_min, h4_max, step=32),
        "activation": optuna.distributions.CategoricalDistribution(list(ACTS.keys())),
        "norm": optuna.distributions.CategoricalDistribution(list(NORMS.keys())),
        "dropout": optuna.distributions.FloatDistribution(0.0, 0.2),
    }


DIRECTION = "minimize"  # minimize validation loss


def load_study(study_name: str, sqlite_path: str) -> optuna.Study:
    """Create or load an Optuna study bound to a local SQLite file."""
    return optuna.create_study(
        study_name=study_name,
        direction=DIRECTION,
        storage=f"sqlite:///{sqlite_path}",
        load_if_exists=True,
        sampler=TPESampler(
            seed=42,
            n_startup_trials=12,
            multivariate=True,  # model interactions between params
            group=True,  # handles conditional/discrete structure better
        ),
    )


def rebuild_study_from_csv(
    study_name: str, dataset: str, property_name: str, csv: pathlib.Path, db_path: pathlib.Path, space: dict
) -> optuna.Study | None:
    """Rebuild Optuna study from CSV file (for portability across machines)."""
    study = load_study(study_name, str(db_path))

    if not csv.exists():
        print(f"No CSV at {csv}; nothing to rebuild.")
        return study

    df = pd.read_csv(csv)
    if df.empty:
        print(f"Empty CSV@{csv}; nothing to rebuild.")
        return study

    added = 0
    for _, r in df.iterrows():
        params = {k: r[k] for k in space if k in r and pd.notna(r[k])}
        value = float(r["value"])
        user_attrs = {}
        if "exp_dir_name" in r and pd.notna(r["exp_dir_name"]):
            user_attrs["exp_dir_name"] = str(r["exp_dir_name"])
        if "best_epoch" in r and pd.notna(r["best_epoch"]):
            user_attrs["best_epoch"] = int(r["best_epoch"])

        t = optuna.trial.create_trial(
            params=params,
            distributions=space,
            value=value,
            state=optuna.trial.TrialState.COMPLETE,
            user_attrs=user_attrs,
        )
        study.add_trial(t)
        added += 1

    print(f"Rebuilt {added} trials into {db_path} for dataset='{dataset}', property='{property_name}'.")
    return study


def export_trials(
    study_name: str, db_path: pathlib.Path, dataset: str, property_name: str, csv: pathlib.Path, space: dict
):
    """
    Export trials to CSV for rebuilding the study on other machines.
    Overwrites the CSV (no append), includes exp_dir_name and best_epoch.
    """
    study = load_study(study_name, str(db_path))

    rows = []
    for t in study.get_trials(deepcopy=False):
        exp_dir = t.user_attrs.get("exp_dir_name")
        row = {
            "dataset": dataset,
            "property": property_name,
            "exp_dir_name": exp_dir,
            "best_epoch": t.user_attrs.get("best_epoch", -1),
            "number": t.number,
            "value": t.value,
            "state": t.state.name if hasattr(t.state, "name") else str(t.state),
        }
        for k in space:
            row[k] = t.params.get(k, None)
        rows.append(row)

    tidy = pd.DataFrame(rows)
    tidy.to_csv(csv, index=False)
    print(f"Wrote {len(tidy)} rows for dataset='{dataset}', property='{property_name}' to {csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Property Regressor - HPO")
    p.add_argument(
        "--dataset",
        type=str,
        default=SupportedDataset.QM9_SMILES_HRR_256_F64_G1NG3.value,
        choices=[ds.value for ds in SupportedDataset],
        help="Dataset to use for training",
    )
    p.add_argument(
        "--property",
        type=str,
        default="qed",
        choices=["logp", "sa_score", "qed", "max_ring_size"],
        help="Target molecular property to regress",
    )
    p.add_argument("--n_trials", type=int, default=1, help="Number of HPO trials to run")
    args = p.parse_args()

    ds = SupportedDataset(args.dataset)
    space = get_space_for_dataset(dataset=ds)

    # Property-specific naming (e.g., logp_reg, sa_score_reg, qed_reg, max_ring_size_reg)
    prop_short = args.property  # logp, sa_score, qed, max_ring_size
    here = pathlib.Path(__file__).parent
    study_base = here.parent.name  # "logp_regressor"

    # Study name includes property: logp_regressor_logp_ZINC_..., logp_regressor_qed_ZINC_..., etc.
    study_name = f"{study_base}_{prop_short}_{args.dataset}"

    # DB and CSV paths include property
    db_path = here / f"{prop_short}_reg_hpo_{args.dataset}.db"
    csv = here / f"trials_{prop_short}_{args.dataset}.csv"

    print("HPO Configuration:")
    print(f"  Property: {args.property}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Study name: {study_name}")
    print(f"  DB path: {db_path}")
    print(f"  CSV path: {csv}")
    print(f"  Trials: {args.n_trials}")
    print()

    # Rebuild from CSV if DB missing, else load existing
    if not db_path.exists():
        study = rebuild_study_from_csv(
            study_name=study_name,
            dataset=args.dataset,
            property_name=args.property,
            csv=csv,
            db_path=db_path,
            space=space,
        )
        if study is None:
            study = load_study(study_name=study_name, sqlite_path=str(db_path))
    else:
        study = load_study(study_name=study_name, sqlite_path=str(db_path))

    # Wrapper to set exp_dir_name and best_epoch once params are known
    def objective(trial: optuna.Trial) -> float:
        val, best_epoch = run_trial(trial, dataset=ds, target_property=args.property)

        # After suggestions, params are available
        cfg = dict(trial.params)
        exp_dir = make_run_folder_name(cfg, prefix=f"pr_{args.property}_{ds.default_cfg.name}")
        trial.set_user_attr("exp_dir_name", exp_dir)
        trial.set_user_attr("best_epoch", best_epoch)

        return val

    # Run optimization
    print(f"Starting optimization for {args.property}...")
    study.optimize(objective, n_trials=args.n_trials)

    # Export canonical CSV (with exp_dir_name and best_epoch)
    export_trials(
        study_name=study_name,
        db_path=db_path,
        dataset=args.dataset,
        property_name=args.property,
        csv=csv,
        space=space,
    )
