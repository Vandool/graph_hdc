import argparse
import pathlib

import optuna
import pandas as pd

from src.exp.real_nvp_v2.hpo.space import SPACE
from src.exp.real_nvp_v2.hpo.study import load_study

HERE = pathlib.Path(__file__).parent
CSV = HERE / "trials.csv"


def main():
    ap = argparse.ArgumentParser(description="Rebuild local SQLite study from CSV for a dataset.")
    ap.add_argument("--dataset", required=True, choices=["qm9", "zinc"])
    args = ap.parse_args()

    study_base = HERE.parent.name  # folder name as base
    study_name = f"{study_base}_{args.dataset}"
    db_path = HERE / f"local_study_{args.dataset}.db"

    study = load_study(study_name, str(db_path))

    if not CSV.exists():
        print("No CSV; nothing to rebuild.")
        return

    df = pd.read_csv(CSV)

    if "dataset" in df.columns:
        df = df[df["dataset"].str.lower() == args.dataset]
    if df.empty:
        print(f"No rows for dataset='{args.dataset}'.")
        return

    # Public API to recreate exact trials
    added = 0
    for _, r in df.iterrows():
        params = {k: r[k] for k in SPACE if k in r and pd.notna(r[k])}
        value = float(r["value"])
        t = optuna.trial.create_trial(
            params=params,
            distributions=SPACE,
            value=value,
            state=optuna.trial.TrialState.COMPLETE,
        )
        study.add_trial(t)
        added += 1

    print(f"Rebuilt {added} trials into {db_path} for dataset='{args.dataset}'.")


if __name__ == "__main__":
    main()
