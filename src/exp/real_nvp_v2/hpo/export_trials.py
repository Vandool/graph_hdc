import argparse
import pathlib

import pandas as pd

from src.exp.real_nvp_v2.hpo.space import SPACE
from src.exp.real_nvp_v2.hpo.study import load_study

HERE = pathlib.Path(__file__).parent
CSV = HERE / "trials.csv"


def main():
    ap = argparse.ArgumentParser(description="Export local SQLite study to CSV, tagged by dataset.")
    ap.add_argument("--dataset", required=True, choices=["qm9", "zinc"])
    args = ap.parse_args()

    study_base = HERE.parent.name
    study_name = f"{study_base}_{args.dataset}"
    db_path = HERE / f"local_study_{args.dataset}.db"

    study = load_study(study_name, str(db_path))

    # Build rows manually (works across Optuna versions)
    rows = []
    for t in study.get_trials(deepcopy=False):
        row = {
            "dataset": args.dataset,
            "number": t.number,
            "value": t.value,
            "state": t.state.name if hasattr(t.state, "name") else str(t.state),
        }
        # ensure all SPACE keys are present as columns
        for k in SPACE:
            row[k] = t.params.get(k, None)
        rows.append(row)

    tidy = pd.DataFrame(rows)

    # Merge into CSV (preserve other datasets)
    if CSV.exists():
        old = pd.read_csv(CSV)
        other = old[old.get("dataset", "").str.lower() != args.dataset]
        out = pd.concat([other, tidy], ignore_index=True)
    else:
        out = tidy

    out.to_csv(CSV, index=False)
    print(f"Wrote {len(tidy)} rows for dataset='{args.dataset}' to {CSV}")


if __name__ == "__main__":
    main()
