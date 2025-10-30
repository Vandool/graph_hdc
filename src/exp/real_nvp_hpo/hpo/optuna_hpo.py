import argparse
import pathlib

import optuna
import pandas as pd
from optuna_integration import BoTorchSampler

from src.encoding.configs_and_constants import SupportedDataset
from src.exp.real_nvp_hpo.hpo.folder_name import make_run_folder_name
from src.exp.real_nvp_hpo.real_nvp import get_hidden_channel_dist, run_qm9_trial, run_zinc_trial

DIRECTION = "minimize"


def get_space(dataset: SupportedDataset):
    low, high, step = get_hidden_channel_dist(dataset)
    return {
        "batch_size": optuna.distributions.IntDistribution(32, 512, step=32),
        "lr": optuna.distributions.FloatDistribution(5e-5, 1e-3, log=True),
        "weight_decay": optuna.distributions.CategoricalDistribution([0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 5e-4]),
        "num_flows": optuna.distributions.IntDistribution(4, 16),
        "num_hidden_channels": optuna.distributions.IntDistribution(low, high, step=step),
    }


def load_study(study_name: str, sqlite_path: str) -> optuna.Study:
    """Create or load an Optuna study bound to a local SQLite file."""
    return optuna.create_study(
        study_name=study_name,
        direction=DIRECTION,
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
        # Preferred: use stored user_attr; fallback: recompute from params
        exp_dir = t.user_attrs.get("exp_dir_name")
        row = {
            "dataset": dataset.default_cfg.name,
            "exp_dir_name": exp_dir,  # <-- include in CSV
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
        default=SupportedDataset.QM9_SMILES_HRR_1600_F64_G1NG3.value,
        choices=[ds.value for ds in SupportedDataset],
    )
    p.add_argument("--n_trials", type=int, default=1)
    args = p.parse_args()

    ds = SupportedDataset(args.dataset)

    # Paths (per-dataset DB + CSV)
    here = pathlib.Path(__file__).parent
    study_base = here.parent.name
    study_name = f"{study_base}_{ds.default_cfg.name}"
    db_path = here / f"real_nvp_hpo_{ds.default_cfg.name}.db"
    csv = here / f"trials_{ds.default_cfg.name}.csv"

    # Rebuild if DB missing, else load
    if not db_path.exists():
        study = rebuild_study_from_csv(study_name=study_name, dataset=ds, csv=csv, db_path=db_path)
        if study is None:
            study = load_study(study_name=study_name, sqlite_path=str(db_path))
    else:
        study = load_study(study_name=study_name, sqlite_path=str(db_path))

    # Choose an objective provided by your code
    base_objective = run_qm9_trial if ds.default_cfg.base_dataset == "qm9" else run_zinc_trial

    # Wrapper to set exp_dir_name once params are known
    def objective(trial: optuna.Trial) -> float:
        val = base_objective(trial, dataset=ds)
        # After suggestions happened, params are available:
        cfg = dict(trial.params)
        exp_dir = make_run_folder_name(cfg, prefix=f"nvp_{ds.default_cfg.name}")
        trial.set_user_attr("exp_dir_name", exp_dir)
        return val

    # Run optimization
    study.optimize(objective, n_trials=args.n_trials)

    # Export canonical CSV (with exp_dir_name)
    export_trials(study_name=study_name, db_path=db_path, dataset=ds, csv=csv)
