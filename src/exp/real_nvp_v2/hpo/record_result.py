import argparse
import json
import pathlib

from src.exp.real_nvp_v2.hpo.study import load_study

HERE = pathlib.Path(__file__).parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["qm9", "zinc"])
    ap.add_argument("val_nll", type=float)
    args = ap.parse_args()

    study_base = HERE.parent.name
    study_name = f"{study_base}_{args.dataset}"
    db_path    = HERE / f"local_study_{args.dataset}.db"
    pending_fp = HERE / f"pending_trial_{args.dataset}.json"

    payload = json.loads(pending_fp.read_text())
    assert payload["study_name"] == study_name, "Pending file is for a different study/dataset."

    trial_number = payload.get("trial_number")
    if trial_number is None:
        msg = (
            "Pending file lacks 'trial_number' (you likely created it with the old script). "
            "Delete the file and run suggest_next.py again."
        )
        raise SystemExit(msg)

    study = load_study(study_name, str(db_path))
    study.tell(int(trial_number), args.val_nll)
    print("Recorded. Best value:", study.best_value)
    print("Best params:", study.best_params)

if __name__ == "__main__":
    main()
