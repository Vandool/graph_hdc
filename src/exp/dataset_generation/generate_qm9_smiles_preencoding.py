#!/usr/bin/env python3
from pathlib import Path

from pytorch_lightning import seed_everything

from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_smiles_generation import precompute_encodings
from src.encoding.configs_and_constants import QM9_SMILES_HRR_1600_CONFIG, QM9_SMILES_HRR_1600_CONFIG_F64
from src.encoding.graph_encoders import load_or_create_hypernet
from src.utils.utils import GLOBAL_MODEL_PATH, pick_device


def generate():
    seed = 42
    seed_everything(seed)

    device = pick_device()
    print(f"Using device: {device!s}")
    cfg = QM9_SMILES_HRR_1600_CONFIG_F64

    hypernet = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, cfg=cfg).to(device)

    # Precompute and cache encodings for each split
    for split in ["train", "valid", "test"]:
        ds = QM9Smiles(split=split)
        # Writes processed/data_<split>_<out_suffix>.pt (e.g., data_train_HRR1600.pt)
        out_path: Path = precompute_encodings(
            base_ds=ds,
            hypernet=hypernet,
            batch_size=128,
            device=device,
            out_suffix="HRR1600F64",
        )
        print(f"{split}: wrote {out_path}")


if __name__ == "__main__":
    generate()
