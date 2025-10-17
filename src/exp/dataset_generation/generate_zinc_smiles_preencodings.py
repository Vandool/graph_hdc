#!/usr/bin/env python3
from pathlib import Path

import torch
from pytorch_lightning import seed_everything

from src.datasets.zinc_smiles_generation import ZincSmiles, precompute_encodings
from src.encoding.configs_and_constants import (
    ZINC_SMILES_HRR_6144_G1G4_CONFIG,
    HDCConfig,
)
from src.encoding.graph_encoders import load_or_create_hypernet
from src.utils.utils import GLOBAL_MODEL_PATH, pick_device

torch.set_default_dtype(torch.float64)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"CUDA is available. Detected {count} GPU device{'s' if count != 1 else ''}.")
        return torch.device("cuda")
    print("CUDA is not available.")
    return torch.device("cpu")


def generate(cfg: HDCConfig):
    seed = 42
    seed_everything(seed)

    device = pick_device()

    hypernet = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, cfg=cfg).to(device)
    hypernet.depth = cfg.hypernet_depth

    # Precompute and cache encodings for each split
    for split in ["train", "valid", "test"]:
        ds = ZincSmiles(split=split)
        # Writes processed/data_<split>_<out_suffix>.pt (e.g., data_train_HRR7744.pt)
        print(f"split: {split}")
        out_path: Path = precompute_encodings(
            base_ds=ds,
            hypernet=hypernet,
            batch_size=32,
            device=device,
            normalize=cfg.normalize,
            out_suffix=cfg.name,
        )
        print(f"{split}: wrote {out_path}")


if __name__ == "__main__":
    generate(ZINC_SMILES_HRR_6144_G1G4_CONFIG)
