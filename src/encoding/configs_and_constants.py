import enum
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from src.encoding.feature_encoders import (
    AbstractFeatureEncoder,
    CategoricalIntegerEncoder,
    CategoricalLevelEncoder,
    CategoricalOneHotEncoder,
    CombinatoricIntegerEncoder,
    TrueFalseEncoder,
)
from src.encoding.the_types import VSAModel

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
    NHA = ("nha", 3) # unique values [1.0, 2.0, 3.0]

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


ZINC_CONFIG: DatasetConfig = DatasetConfig(
    name="ZINC",
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


class SupportedDataset(enum.Enum):
    ZINC = ("ZINC", ZINC_CONFIG)
    ZINC_NODE_DEGREE = ("ZINC_ND", ZINC_ND_CONFIG)
    ZINC_NODE_DEGREE_COMB = ("ZINC_ND_COMB", ZINC_ND_COMB_CONFIG)
    # NHA: Neighbourhood Aware
    ZINC_NODE_DEGREE_COMB_NHA = ("ZINC_ND_COMB_NHA", ZINC_ND_COMB_CONFIG_NHA)
    QM9 = ("QM9", QM9_CONFIG)

    def __new__(cls, value: str, default_cfg: DatasetConfig):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.default_cfg = default_cfg
        return obj


class BasisHVSet(enum.Enum):
    RANDOM = "random"
    EMPTY = "empty"
    IDENTITY = "identity"
    CIRCULAR = "circular"
    LEVEL = "level"
    CUSTOM = "custom"
