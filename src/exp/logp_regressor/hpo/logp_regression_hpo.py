import argparse
import pathlib

import optuna
import pandas as pd
from optuna.samplers import TPESampler

from src.encoding.configs_and_constants import SupportedDataset
from src.exp.logp_regressor.hpo.folder_name import make_run_folder_name
from src.exp.logp_regressor.lpr import run_trial
from src.generation import logp_regressor


def get_space_for_dataset(dataset: SupportedDataset) -> dict:
    """Return appropriate SPACE based on input dimensionality."""
    if dataset.default_cfg.base_dataset == "qm9":
        h1_min, h1_max = 512, 1536
        h2_min, h2_max = 128, 1024
    else:
        h1_min, h1_max = 1024, 3072
        h2_min, h2_max = 512, 1536

    h3_min, h3_max = 64, 512
    h4_min, h4_max = 32, 256
    return {
        "batch_size": optuna.distributions.IntDistribution(32, 512, step=32),
        "lr": optuna.distributions.FloatDistribution(5e-5, 1e-3, log=True),
        "weight_decay": optuna.distributions.CategoricalDistribution([0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 5e-4]),
        "depth": optuna.distributions.CategoricalDistribution([2, 3, 4]),
        "h1": optuna.distributions.IntDistribution(h1_min, h1_max, step=256),
        "h2": optuna.distributions.IntDistribution(h2_min, h2_max, step=128),
        "h3": optuna.distributions.IntDistribution(h3_min, h3_max, step=64),
        "h4": optuna.distributions.IntDistribution(h4_min, h4_max, step=32),
        "activation": optuna.distributions.CategoricalDistribution(logp_regressor.ACTS.keys()),
        "norm": optuna.distributions.CategoricalDistribution(logp_regressor.NORMS.keys()),
        "dropout": optuna.distributions.FloatDistribution(0.0, 0.2),
    }


DIRECTION = "minimize"  # e.g., minimize val_mae


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
    study_name: str, dataset: str, csv: pathlib.Path, db_path: pathlib.Path, space: dict
) -> optuna.Study | None:
    study = load_study(study_name, str(db_path))

    if not csv.exists():
        print("No CSV; nothing to rebuild.")
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
        t = optuna.trial.create_trial(
            params=params,
            distributions=space,
            value=value,
            state=optuna.trial.TrialState.COMPLETE,
            user_attrs=user_attrs,  # <-- persist exp_dir_name into DB if present
        )
        study.add_trial(t)
        added += 1

    print(f"Rebuilt {added} trials into {db_path} for dataset='{dataset}'.")
    return study


def export_trials(study_name: str, db_path: pathlib.Path, dataset: str, csv: pathlib.Path, space: dict):
    """
    Export trials to CSV for rebuilding the study on other machines.
    Overwrites the dataset's CSV (no append/mix), includes exp_dir_name.
    """
    study = load_study(study_name, str(db_path))

    rows = []
    for t in study.get_trials(deepcopy=False):
        # Preferred: use stored user_attr; fallback: recompute from params
        exp_dir = t.user_attrs.get("exp_dir_name")
        row = {
            "dataset": dataset,
            "exp_dir_name": exp_dir,
            "best_epoch": t.user_attrs.get("best_epoch") or -1,
            "number": t.number,
            "value": t.value,
            "state": t.state.name if hasattr(t.state, "name") else str(t.state),
        }
        for k in space:
            row[k] = t.params.get(k, None)
        rows.append(row)

    tidy = pd.DataFrame(rows)
    # Since you keep separate CSV per dataset, just overwrite it:
    tidy.to_csv(csv, index=False)
    print(f"Wrote {len(tidy)} rows for dataset='{dataset}' to {csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Real NVP V2 - HPO")
    p.add_argument(
        "--dataset",
        type=str,
        default=SupportedDataset.ZINC_SMILES_HRR_6144_F64_G1G3.value,
        choices=[ds.value for ds in SupportedDataset],
    )
    p.add_argument("--n_trials", type=int, default=1)
    args = p.parse_args()

    ds = SupportedDataset(args.dataset)
    space = get_space_for_dataset(dataset=ds)

    # Paths (per-dataset DB + CSV)
    here = pathlib.Path(__file__).parent
    study_base = here.parent.name
    study_name = f"{study_base}_{args.dataset}_3d"
    db_path = here / f"logp_reg_hpo_{args.dataset}_3d.db"
    csv = here / f"trials_{args.dataset}_3d.csv"

    # Rebuild if DB missing, else load
    if not db_path.exists():
        study = rebuild_study_from_csv(
            study_name=study_name, dataset=args.dataset, csv=csv, db_path=db_path, space=space
        )
        if study is None:
            study = load_study(study_name=study_name, sqlite_path=str(db_path))
    else:
        study = load_study(study_name=study_name, sqlite_path=str(db_path))

    # Wrapper to set exp_dir_name once params are known
    def objective(trial: optuna.Trial) -> float:
        val, best_epoch = run_trial(trial, dataset=ds)
        # After suggestions happened, params are available:
        cfg = dict(trial.params)
        exp_dir = make_run_folder_name(cfg, prefix=f"lpr_{ds.default_cfg.name}")
        trial.set_user_attr("exp_dir_name", exp_dir)
        trial.set_user_attr("best_epoch", best_epoch)
        return val

    # Run optimization
    study.optimize(objective, n_trials=args.n_trials)

    # Export canonical CSV (with exp_dir_name)
    export_trials(study_name=study_name, db_path=db_path, dataset=args.dataset, csv=csv, space=space)
