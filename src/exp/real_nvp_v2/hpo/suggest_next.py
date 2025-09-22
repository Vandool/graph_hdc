# Ask one new config for a given dataset (offline mode).
from __future__ import annotations

import argparse
import json
import pathlib

from src.exp.real_nvp_v2.hpo.space import SPACE
from src.exp.real_nvp_v2.hpo.study import load_study

HERE = pathlib.Path(__file__).parent


def main():
    ap = argparse.ArgumentParser(description="Suggest one new config (offline) for a dataset.")
    ap.add_argument("--dataset", required=True, choices=["qm9", "zinc"])
    args = ap.parse_args()

    study_base = HERE.parent.name
    study_name = f"{study_base}_{args.dataset}"
    db_path = HERE / f"local_study_{args.dataset}.db"
    pending_fp = HERE / f"pending_trial_{args.dataset}.json"

    study = load_study(study_name, str(db_path))
    tr = study.ask(SPACE)
    payload = {
        "study_name": study_name,
        "dataset": args.dataset,
        "trial_number": tr.number,
        "trial_id": getattr(tr, "_trial_id", None),  # backward info only
        "params": tr.params,
    }
    pending_fp.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))  # also echo to stdout


if __name__ == "__main__":
    main()
