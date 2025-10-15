#!/usr/bin/env python3
from collections import OrderedDict
from math import prod
from pathlib import Path

import torch
from pytorch_lightning import seed_everything

from src.datasets.zinc_smiles_generation import ZincSmiles, precompute_encodings
from src.encoding.configs_and_constants import DatasetConfig, FeatureConfig, Features, IndexRange, \
    ZINC_SMILES_HRR_5120D5_CONFIG_F64
from src.encoding.feature_encoders import CombinatoricIntegerEncoder
from src.encoding.graph_encoders import load_or_create_hypernet
from src.encoding.the_types import VSAModel
from src.utils.utils import GLOBAL_MODEL_PATH, pick_device

torch.set_default_dtype(torch.float64)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"CUDA is available. Detected {count} GPU device{'s' if count != 1 else ''}.")
        return torch.device("cuda")
    print("CUDA is not available.")
    return torch.device("cpu")


def generate():
    seed = 42
    seed_everything(seed)

    device = pick_device()
    ds_config = ZINC_SMILES_HRR_5120D5_CONFIG_F64

    hypernet = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, cfg=ds_config).to(device)
    hypernet.depth = ds_config.hypernet_depth

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
            out_suffix=ds_config.name,
        )
        print(f"{split}: wrote {out_path}")


if __name__ == "__main__":
    generate()
