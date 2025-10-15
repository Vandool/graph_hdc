import json
from pathlib import Path

import pandas as pd

from src.exp.real_nvp_hpo.hpo.optuna_hpo import SPACE
from src.utils import registery
from src.utils.utils import GLOBAL_ARTEFACTS_PATH, GLOBAL_MODEL_PATH, find_files


def get_model_type(path: Path | str) -> registery.ModelType:
    res: registery.ModelType = "MLP"
    if "bah" in str(path):
        res = "BAH"
    elif "gin-c" in str(path):
        res = "GIN-C"
    elif "gin-f" in str(path):
        res = "GIN-F"
    elif "nvp" in str(path):
        res = "NVP"
    return res


gen_paths = list(
    find_files(
        start_dir=GLOBAL_MODEL_PATH / "0_real_nvp_v2",
        prefixes=("epoch",),
        skip_substrings=("zinc", "nvp_qm9"),
        desired_ending=".ckpt",
    )
)
print(f"Found {len(gen_paths)} generator checkpoints")
metrics = []
for gen_ckpt_path in gen_paths:
    # print(f"loop #{loop}")
    # if loop >= 1:
    #     break
    # loop += 1
    print(f"Generator Checkpoint: {gen_ckpt_path}")
    # Read the metrics from training
    evals_dir = gen_ckpt_path.parent.parent / "evaluations"
    epoch_metrics = pd.read_parquet(evals_dir / "metrics.parquet")
    # Find min val_loss
    idx = epoch_metrics["val_loss"].idxmin()
    min_val_loss = epoch_metrics.loc[idx, "val_loss"]
    best_epoch = epoch_metrics.loc[idx, "epoch"]

    print(f"Best epoch: {best_epoch}, min val_loss: {min_val_loss}")

    # extract lr, wd,
    # nvp_qm9_h1600_f6_hid1024_s42_lr1e-3_wd1e-4_an
    with (evals_dir / "run_config.json").open("r", encoding="utf-8") as f:
        run_config = json.load(f)

    ### Save metrics
    gen_path = "/".join(gen_ckpt_path.parts[-4:])
    metrics.append(
        {
            "gen_path": str(gen_path),
            "best_epoch": best_epoch,
            "dataset": "zinc" if "zinc" in str(gen_ckpt_path) else "qm9",
            **{k: run_config[k] for k in SPACE},
            "val_nll_best": min_val_loss,
        }
    )

asset_dir = GLOBAL_ARTEFACTS_PATH / "generators"
asset_dir.mkdir(parents=True, exist_ok=True)

parquet_path = asset_dir / "qm9_3d_f64.parquet"
csv_path = asset_dir / "qm9_3d_f64.csv"

metrics_df = pd.DataFrame(metrics)

# write back out
metrics_df.to_parquet(parquet_path, index=False)
metrics_df.to_csv(csv_path, index=False)
