import enum
import math
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

from src.encoding.feature_encoders import (
    AbstractFeatureEncoder,
    CategoricalIntegerEncoder,
    CategoricalLevelEncoder,
    CategoricalOneHotEncoder,
    CombinatoricIntegerEncoder,
    TrueFalseEncoder,
)
from src.encoding.the_types import VSAModel
from src.utils.utils import pick_device_str

IndexRange = tuple[int, int]


@dataclass
class FeatureConfig:
    """
    Configuration for a single feature's hypervector codebook.
    """

    count: int  # number of distinct values or bins
    encoder_cls: type[AbstractFeatureEncoder]
    index_range: IndexRange = (0, 1)  # feature slice indices (start, end)
    idx_offset: int = 0
    bins: list[int] | None = None


class Features(enum.Enum):
    ATOM_TYPE = ("atom_type", 0)
    BOND_TYPE = ("bond_type", 0)
    NODE_DEGREE = ("node_degree", 1)
    ATOMIC_NUMBER = ("atom_number", 5)  # unique values [1.0, 6.0, 7.0, 8.0, 9.0]
    AROMATIC = ("aromatic", 6)
    NHA = ("nha", 3)  # unique values [1.0, 2.0, 3.0]

    # three hybridization flags
    SP = ("todo", 7)
    SP2 = ("todo", 8)
    SP3 = ("todo", 9)

    # Number of bonded hydrogens
    NUM_HS = ("todo", 10)  # unique values [0.0, 1.0, 2.0, 3.0, 4.0]

    def __new__(cls, value, idx):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.idx = idx
        return obj


@dataclass
class DatasetConfig:
    """
    Configuration for hyperdimensional base encoding of a dataset.
    """

    name: str
    hv_dim: int = 10000
    vsa: VSAModel = field(default_factory=lambda: VSAModel.MAP)
    node_feature_configs: dict[Features, FeatureConfig] = field(default_factory=OrderedDict)
    edge_feature_configs: dict[Features, FeatureConfig] | None = field(default_factory=OrderedDict)
    graph_feature_configs: dict[Features, FeatureConfig] | None = field(default_factory=OrderedDict)
    device: str = "mps"
    seed: int | None = None
    nha_bins: int | None = None
    nha_depth: int | None = None
    dtype: str = "float32"
    base_dataset: Literal["zinc", "qm9"] = "qm9"
    hypernet_depth: int = 3


ZINC_CONFIG: DatasetConfig = DatasetConfig(
    name="ZINC",
    base_dataset="zinc",
    hv_dim=10000,
    node_feature_configs=OrderedDict(
        [
            (
                Features.ATOM_TYPE,
                FeatureConfig(
                    count=28,  # Number of distinct atom types in ZINC
                    encoder_cls=CategoricalIntegerEncoder,
                ),
            ),
        ]
    ),
    edge_feature_configs=OrderedDict(
        [
            (
                Features.BOND_TYPE,
                FeatureConfig(
                    count=4,  # zero(for ease of indexing), single, double, triple
                    encoder_cls=CategoricalIntegerEncoder,
                ),
            ),
        ]
    ),
)

# ZINC_ND has added node degrees as an extra feature to the original ZINC located in data.x
ZINC_ND_CONFIG: DatasetConfig = deepcopy(ZINC_CONFIG)
ZINC_ND_CONFIG.name = "ZINC_ND"
ZINC_ND_CONFIG.node_feature_configs[Features.NODE_DEGREE] = FeatureConfig(
    count=6,  # Unique Node Degrees: [0.0 (for ease of indexing), 1.0, 2.0, 3.0, 4.0, 5.0]
    encoder_cls=CategoricalIntegerEncoder,
    index_range=IndexRange((1, 2)),
)

# ZINC_ND has added node degrees as an extra feature to the original ZINC located in data.x
ZINC_ND_COMB_CONFIG: DatasetConfig = deepcopy(ZINC_CONFIG)
ZINC_ND_COMB_CONFIG.name = "ZINC_ND_COMB"
ZINC_ND_COMB_CONFIG.node_feature_configs[Features.ATOM_TYPE] = FeatureConfig(
    count=28 * 6,  # 28 Atom Types, 6 Unique Node Degrees: [0.0 (for ease of indexing), 1.0, 2.0, 3.0, 4.0, 5.0]
    encoder_cls=CombinatoricIntegerEncoder,
    index_range=IndexRange((0, 2)),
)

ZINC_ND_COMB_CONFIG_NHA: DatasetConfig = deepcopy(ZINC_CONFIG)
ZINC_ND_COMB_CONFIG_NHA.name = "ZINC_ND_COMB_NHA"
ZINC_ND_COMB_CONFIG_NHA.node_feature_configs[Features.ATOM_TYPE] = FeatureConfig(
    # Added Neighbourhood awareness encodings (3 distinct values)
    count=28 * 6 * 3,  # 28 Atom Types, 6 Unique Node Degrees: [0.0 (for ease of indexing), 1.0, 2.0, 3.0, 4.0, 5.0]
    encoder_cls=CombinatoricIntegerEncoder,
    index_range=IndexRange((0, 3)),
)

QM9_CONFIG: DatasetConfig = DatasetConfig(
    name="QM9",
    base_dataset="qm9",
    hv_dim=10000,
    node_feature_configs=OrderedDict(
        [
            (
                Features.ATOM_TYPE,
                FeatureConfig(
                    count=5,
                    encoder_cls=CategoricalOneHotEncoder,
                    index_range=IndexRange((0, 5)),
                ),
            ),
            (
                Features.ATOMIC_NUMBER,
                FeatureConfig(
                    count=5,
                    encoder_cls=CategoricalIntegerEncoder,
                    index_range=IndexRange((Features.ATOMIC_NUMBER.idx, Features.ATOMIC_NUMBER.idx + 1)),
                ),
            ),
            (
                Features.ATOMIC_NUMBER,
                FeatureConfig(
                    count=10,
                    encoder_cls=CategoricalLevelEncoder,
                    index_range=IndexRange((Features.ATOMIC_NUMBER.idx, Features.ATOMIC_NUMBER.idx + 1)),
                ),
            ),
            (
                Features.AROMATIC,
                FeatureConfig(
                    count=2,
                    encoder_cls=TrueFalseEncoder,
                    index_range=IndexRange((Features.AROMATIC.idx, Features.AROMATIC.idx + 1)),
                ),
            ),
            (
                Features.SP,
                FeatureConfig(
                    count=2,
                    encoder_cls=TrueFalseEncoder,
                    index_range=IndexRange((Features.SP.idx, Features.SP.idx + 1)),
                ),
            ),
            (
                Features.SP2,
                FeatureConfig(
                    count=2,
                    encoder_cls=TrueFalseEncoder,
                    index_range=IndexRange((Features.SP2.idx, Features.SP2.idx + 1)),
                ),
            ),
            (
                Features.SP3,
                FeatureConfig(
                    count=2,
                    encoder_cls=TrueFalseEncoder,
                    index_range=IndexRange((Features.SP3.idx, Features.SP3.idx + 1)),
                ),
            ),
            (
                Features.NUM_HS,
                FeatureConfig(
                    count=2,
                    encoder_cls=TrueFalseEncoder,
                    index_range=IndexRange((Features.SP3.idx, Features.SP3.idx + 1)),
                ),
            ),
        ]
    ),
    edge_feature_configs=OrderedDict(
        [
            (
                Features.BOND_TYPE,
                FeatureConfig(
                    count=4,  # single, double, triple, aromatic
                    encoder_cls=CategoricalOneHotEncoder,
                    index_range=IndexRange((0, 4)),
                ),
            ),
        ]
    ),
)

ZINC_SMILES_CONFIG: DatasetConfig = DatasetConfig(
    name="ZINC_SMILES",
    node_feature_configs=OrderedDict(
        [
            (
                Features.ATOM_TYPE,
                FeatureConfig(count=1, encoder_cls=CombinatoricIntegerEncoder),  # Place holder
            ),
        ]
    ),
)

ZINC_SMILES_HRR_7744_CONFIG = DatasetConfig(
    seed=42,
    name="ZincSmilesHRR7744",
    base_dataset="zinc",
    vsa=VSAModel.HRR,
    hv_dim=88 * 88,
    device=pick_device_str(),
    node_feature_configs=OrderedDict(
        [
            (
                Features.ATOM_TYPE,
                FeatureConfig(
                    count=math.prod([9, 6, 3, 4]),
                    encoder_cls=CombinatoricIntegerEncoder,
                    index_range=IndexRange((0, 4)),
                    bins=[9, 6, 3, 4],
                ),
            ),
        ]
    ),
)

ZINC_SMILES_HRR_7744_CONFIG_F64: DatasetConfig = deepcopy(ZINC_SMILES_HRR_7744_CONFIG)
ZINC_SMILES_HRR_7744_CONFIG_F64.name = "ZincSmilesHRR7744F64"
ZINC_SMILES_HRR_7744_CONFIG_F64.dtype = "float64"

ZINC_SMILES_HRR_5120D5_CONFIG_F64: DatasetConfig = deepcopy(ZINC_SMILES_HRR_7744_CONFIG_F64)
ZINC_SMILES_HRR_5120D5_CONFIG_F64.name = "ZincSmilesHRR5120D5F64"
ZINC_SMILES_HRR_5120D5_CONFIG_F64.hv_dim = 5120
ZINC_SMILES_HRR_5120D5_CONFIG_F64.hypernet_depth = 5


QM9_SMILES_CONFIG = DatasetConfig(
    seed=42,
    name="QM9Smiles",
    base_dataset="qm9",
    vsa=VSAModel.HRR,
    hv_dim=40 * 40,
    device=pick_device_str(),
    node_feature_configs=OrderedDict(
        [
            (
                Features.ATOM_TYPE,
                FeatureConfig(
                    # Atom types size: 4
                    # Atom types: ['C', 'F', 'N', 'O']
                    # Degrees size: 5
                    # Degrees: {0, 1, 2, 3, 4}
                    # Formal Charges size: 3
                    # Formal Charges: {0, 1, -1}
                    # Explicit Hs size: 5
                    # Explicit Hs: {0, 1, 2, 3, 4}
                    count=math.prod([4, 5, 3, 5]),
                    encoder_cls=CombinatoricIntegerEncoder,
                    index_range=IndexRange((0, 4)),
                    bins=[4, 5, 3, 5],
                ),
            ),
        ]
    ),
)

QM9_SMILES_HRR_1600_CONFIG: DatasetConfig = deepcopy(QM9_SMILES_CONFIG)
QM9_SMILES_HRR_1600_CONFIG.hv_dim = 1600
QM9_SMILES_HRR_1600_CONFIG.name = "QM9SmilesHRR1600"

QM9_SMILES_HRR_1600_CONFIG_F64: DatasetConfig = deepcopy(QM9_SMILES_CONFIG)
QM9_SMILES_HRR_1600_CONFIG_F64.hv_dim = 1600
QM9_SMILES_HRR_1600_CONFIG_F64.name = "QM9SmilesHRR1600F64"
QM9_SMILES_HRR_1600_CONFIG_F64.dtype = "float64"


class SupportedDataset(enum.Enum):
    ZINC = ("ZINC", ZINC_CONFIG)
    ZINC_NODE_DEGREE = ("ZINC_ND", ZINC_ND_CONFIG)
    ZINC_NODE_DEGREE_COMB = ("ZINC_ND_COMB", ZINC_ND_COMB_CONFIG)
    ZINC_NODE_DEGREE_COMB_NHA = ("ZINC_ND_COMB_NHA", ZINC_ND_COMB_CONFIG_NHA)
    ZINC_SMILES = ("ZINC_SMILES", ZINC_SMILES_CONFIG)
    ZINC_SMILES_HRR_7744 = ("ZINC_SMILES_HRR_7744", ZINC_SMILES_HRR_7744_CONFIG)
    ZINC_SMILES_HRR_7744_F64 = ("ZINC_SMILES_HRR_7744_F64", ZINC_SMILES_HRR_7744_CONFIG_F64)
    ZINC_SMILES_HRR_5120D5_F64 = ("ZINC_SMILES_HRR_5120D50_F64", ZINC_SMILES_HRR_5120D5_CONFIG_F64)
    QM9 = ("QM9", QM9_CONFIG)
    QM9_SMILES = ("QM9_SMILES", QM9_SMILES_CONFIG)
    QM9_SMILES_HRR_1600 = ("QM9_SMILES_HRR_1600", QM9_SMILES_HRR_1600_CONFIG)
    QM9_SMILES_HRR_1600_F64 = ("QM9_SMILES_HRR_1600_F64", QM9_SMILES_HRR_1600_CONFIG_F64)

    def __new__(cls, value: str, default_cfg: DatasetConfig):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.default_cfg = default_cfg
        return obj
