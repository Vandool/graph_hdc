from pathlib import Path

import pandas as pd

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
    return res


# Iterate all the checkpoints
metrics = []
best_epochs = list(find_files(start_dir=GLOBAL_MODEL_PATH, prefixes=("epoch",), skip_substring="nvp"))
print(len(best_epochs))
for ckpt_path in best_epochs:
    print(f"File Name: {ckpt_path}")

    # Read the metrics from training
    evals_dir = ckpt_path.parent.parent / "evaluations"
    epoch_metrics = pd.read_parquet(evals_dir / "epoch_metrics.parquet")

    val_loss = "val_loss" if "val_loss" in epoch_metrics.columns else "val_loss_cb"
    best = epoch_metrics.loc[epoch_metrics[val_loss].idxmin()].add_suffix("_best")
    last = epoch_metrics.iloc[-1].add_suffix("_last")

    ## Determine model type
    model_type = get_model_type(ckpt_path)

    ## Determine Dataset
    dataset = "zinc" if "zinc" in str(ckpt_path) else "qm9"

    ### Save metrics
    metrics.append(
        {
            "path": "/".join(ckpt_path.parts[-4:]),
            "model_type": model_type,
            "dataset": dataset,
            **best.to_dict(),
            **last.to_dict(),
        }
    )


asset_dir = GLOBAL_ARTEFACTS_PATH / "classification"
asset_dir.mkdir(parents=True, exist_ok=True)

parquet_path = asset_dir / "classifier_metrics.parquet"
csv_path = asset_dir / "classifier_metrics.csv"

metrics_df = pd.DataFrame(metrics)
# write back out
metrics_df.to_parquet(parquet_path, index=False)
metrics_df.to_csv(csv_path, index=False)
