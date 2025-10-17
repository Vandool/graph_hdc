#!/usr/bin/env python3
from pathlib import Path

from pytorch_lightning import seed_everything

from src.datasets.qm9_smiles_generation import QM9Smiles, precompute_encodings
from src.encoding.configs_and_constants import (
    QM9_SMILES_HRR_1600_CONFIG_F64_G1G3_CONFIG,
    QM9_SMILES_HRR_1600_CONFIG_F64_G1NG3_CONFIG,
    DSHDCConfig,
)
from src.encoding.graph_encoders import load_or_create_hypernet
from src.utils.utils import GLOBAL_MODEL_PATH, pick_device


def generate(cfg: DSHDCConfig):
    seed = 42
    seed_everything(seed)

    device = pick_device()
    print(f"Using device: {device!s}")

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
            normalize=cfg.normalize,
            out_suffix=cfg.name,
        )
        print(f"{split}: wrote {out_path}")


if __name__ == "__main__":
    generate(cfg=QM9_SMILES_HRR_1600_CONFIG_F64_G1G3_CONFIG)
    generate(cfg=QM9_SMILES_HRR_1600_CONFIG_F64_G1NG3_CONFIG)
