from math import prod

import torch
from pytorch_lightning import seed_everything

from src.datasets.utils import AddHDCEncodings
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import DatasetConfig, Features, FeatureConfig, IndexRange
from src.encoding.feature_encoders import CombinatoricIntegerEncoder
from src.encoding.graph_encoders import HyperNet, load_or_create_hypernet
from src.encoding.the_types import VSAModel
from collections import Counter, OrderedDict

from src.utils.utils import GLOBAL_MODEL_PATH

def get_device():
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
    dataset_config = DatasetConfig(
        seed=seed,
        name=ds_name,
        vsa=VSAModel.HRR,
        hv_dim=88*88,
        device=get_device(),
        node_feature_configs=OrderedDict(
            [
                (
                    Features.ATOM_TYPE,
                    FeatureConfig(
                        # Atom types size: 9
                        # Atom types: ['Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S']
                        # Degrees size: 5
                        # Degrees: {1, 2, 3, 4, 5}
                        # Formal Charges size: 3
                        # Formal Charges: {0, 1, -1}
                        # Explicit Hs size: 4
                        # Explicit Hs: {0, 1, 2, 3}
                        count=prod(
                            zinc_feature_bins
                        ),  # 9 Atom Types, 6 Unique Node Degrees: [0.0 (for ease of indexing), 1.0, 2.0, 3.0, 4.0, 5.0]
                        encoder_cls=CombinatoricIntegerEncoder,
                        index_range=IndexRange((0, 4)),
                        bins=zinc_feature_bins,
                    ),
                ),
            ]
        ),
    )
    hypernet = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, ds_name=ds_name, cfg=dataset_config)
    ZincSmiles(split="train", pre_transform=AddHDCEncodings(encoder=hypernet))
    ZincSmiles(split="valid", pre_transform=AddHDCEncodings(encoder=hypernet))
    ZincSmiles(split="test", pre_transform=AddHDCEncodings(encoder=hypernet))

if __name__ == "__main__":
    generate()