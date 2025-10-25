import os
import pathlib

import optuna
import pandas as pd
from optuna_integration import BoTorchSampler

from src.encoding.configs_and_constants import SupportedDataset
from src.exp.cond_gen_hpo.cond_generation_hpo import run_qm9_cond_gen, run_zinc_cond_gen

SPACE = {
    "lr": optuna.distributions.FloatDistribution(5e-5, 5e-3, log=True),
    "steps": optuna.distributions.FloatDistribution(50, 2000, log=True),
    "scheduler": optuna.distributions.CategoricalDistribution(["cosine", "two-phase", "linear"]),
    "lambda_lo": optuna.distributions.FloatDistribution(1e-5, 5e-3, log=True),
    "lambda_hi": optuna.distributions.FloatDistribution(5e-3, 5e-2, log=True),
}
DIRECTION = "minimize"


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
    study_name: str, dataset: str, csv: pathlib.Path, db_path: pathlib.Path
) -> optuna.Study | None:
    study = load_study(study_name, str(db_path))

    if not csv.exists():
        print("No CSV; nothing to rebuild.")
        return None

    df = pd.read_csv(csv)
    if df.empty:
        print(f"Empty CSV@{csv}; nothing to rebuild.")
        return None

    added = 0
    for _, r in df.iterrows():
        params = {k: r[k] for k in SPACE if k in r and pd.notna(r[k])}
        value = float(r["value"])
        user_attrs = {}
        if "exp_dir_name" in r and pd.notna(r["exp_dir_name"]):
            user_attrs["exp_dir_name"] = str(r["exp_dir_name"])
        t = optuna.trial.create_trial(
            params=params,
            distributions=SPACE,
            value=value,
            state=optuna.trial.TrialState.COMPLETE,
            user_attrs=user_attrs,  # <-- persist exp_dir_name into DB if present
        )
        study.add_trial(t)
        added += 1

    print(f"Rebuilt {added} trials into {db_path} for dataset='{dataset}'.")
    return study


def export_trials(study_name: str, db_path: pathlib.Path, dataset: str, csv: pathlib.Path):
    """
    Export trials to CSV for rebuilding the study on other machines.
    Overwrites the dataset's CSV (no append/mix), includes exp_dir_name.
    """
    study = load_study(study_name, str(db_path))

    rows = []
    for t in study.get_trials(deepcopy=False):
        row = {
            "dataset": dataset,
            **t.user_attrs,
            "number": t.number,
            "value": t.value,
            "state": t.state.name if hasattr(t.state, "name") else str(t.state),
        }
        for k in SPACE:
            row[k] = t.params.get(k, None)
        rows.append(row)

    tidy = pd.DataFrame(rows)
    # Since you keep separate CSV per dataset, just overwrite it:
    tidy.to_csv(csv, index=False)
    print(f"Wrote {len(tidy)} rows for dataset='{dataset}' to {csv}")


if __name__ == "__main__":
    dataset = SupportedDataset.QM9_SMILES_HRR_1600_F64_G1G3
    base_dataset = dataset.default_cfg.base_dataset
    base_objective = run_qm9_cond_gen if base_dataset == "qm9" else run_zinc_cond_gen
    for gen_model in [
        # "nvp-3d-f64_qm9_f8_hid1536_lr0.000503983_wd1e-5_bs384_smf7.43606_smi1.94892_smw15_an",
        # "nvp-3d-f64_qm9_f8_hid1792_lr0.000747838_wd1e-5_bs384_smf5.9223_smi2.08013_smw16_an",
        # "nvp-3d-f64_qm9_f8_hid800_lr0.000373182_wd1e-5_bs384_smf6.54123_smi2.25695_smw16_an"
    ]:
        for classifier in [
            "HDC-Decoder",
        ]:
            # Paths (per-dataset DB + CSV)
            here = pathlib.Path(__file__).parent
            # study_base = here.parent.name
            study_name = f"{gen_model}_{classifier}_{base_dataset}_mae"
            db_path = here / f"{gen_model}_{classifier}_{base_dataset}_mae.db"
            csv = here / f"trials_{gen_model}_{classifier}_{base_dataset}_mae.csv"

            # Rebuild if DB missing, else load
            if not db_path.exists():
                study = rebuild_study_from_csv(study_name=study_name, dataset=base_dataset, csv=csv, db_path=db_path)
                if study is None:
                    study = load_study(study_name=study_name, sqlite_path=str(db_path))
            else:
                study = load_study(study_name=study_name, sqlite_path=str(db_path))

            os.environ["GEN_MODEL"] = gen_model
            os.environ["CLASSIFIER"] = classifier

            # Choose an objective provided by your code

            # Wrapper to set exp_dir_name once params are known
            def objective(trial: optuna.Trial) -> float:
                res = base_objective(trial, dataset)
                # After suggestions happened, params are available:
                for k, v in res.items():
                    trial.set_user_attr(key=k, value=v)

                return res["eval_success@eps"] * res["eval_validity"] * res["eval_uniqueness_overall"]
                # return res["valid_success_at_eps_pct"] * res["total_validity_pct"]
                # return res["valid_mae_to_target"]

            # Run optimization
            study.optimize(objective, n_trials=1)

            # Export canonical CSV (with exp_dir_name)
            export_trials(study_name=study_name, db_path=db_path, dataset=base_dataset, csv=csv)
