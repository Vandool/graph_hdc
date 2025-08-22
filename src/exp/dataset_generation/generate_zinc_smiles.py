#!/usr/bin/env python3
from collections import OrderedDict
from math import prod
from pathlib import Path

import torch
from pytorch_lightning import seed_everything

from src.datasets.zinc_smiles_generation import ZincSmiles, precompute_encodings
from src.encoding.configs_and_constants import DatasetConfig, Features, FeatureConfig, IndexRange
from src.encoding.feature_encoders import CombinatoricIntegerEncoder
from src.encoding.graph_encoders import load_or_create_hypernet
from src.encoding.the_types import VSAModel
from src.utils.utils import GLOBAL_MODEL_PATH


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

    ds_name = "ZincSmilesHRR7744"
    zinc_feature_bins = [9, 6, 3, 4]
    device = get_device()

    dataset_config = DatasetConfig(
        seed=seed,
        name=ds_name,
        vsa=VSAModel.HRR,
        hv_dim=88 * 88,  # 7744
        device=device,
        node_feature_configs=OrderedDict(
            [
                (
                    Features.ATOM_TYPE,
                    FeatureConfig(
                        count=prod(zinc_feature_bins),  # 9 * 6 * 3 * 4
                        encoder_cls=CombinatoricIntegerEncoder,
                        index_range=IndexRange((0, 4)),
                        bins=zinc_feature_bins,
                    ),
                ),
            ]
        ),
    )

    hypernet = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, ds_name=ds_name, cfg=dataset_config)

    # Precompute and cache encodings for each split
    for split in [
            "train",
            "valid",
            "test"
    ]:
        ds = ZincSmiles(split=split)
        # Writes processed/data_<split>_<out_suffix>.pt (e.g., data_train_HRR7744.pt)
        out_path: Path = precompute_encodings(
            base_ds=ds,
            hypernet=hypernet,
            batch_size=1028,
            device=device,
            out_suffix="HRR7744",
        )
        print(f"{split}: wrote {out_path}")


if __name__ == "__main__":
    generate()
