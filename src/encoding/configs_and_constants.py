import enum
import math
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from typing import Literal

from src.encoding.feature_encoders import (
    AbstractFeatureEncoder,
    CombinatoricIntegerEncoder,
)
from src.encoding.the_types import VSAModel
from src.utils.utils import pick_device_str

IndexRange = tuple[int, int]
BaseDataset = Literal["qm9", "zinc"]


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
    NODE_FEATURES = ("node_features", 0)

    def __new__(cls, value, idx):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.idx = idx
        return obj


@dataclass
class DSHDCConfig:
    """
    Configuration for hyperdimensional base encoding of a dataset.
    """

    name: str
    hv_dim: int = 10000
    hv_count: int = 2
    vsa: VSAModel = field(default_factory=lambda: VSAModel.MAP)
    node_feature_configs: dict[Features, FeatureConfig] = field(default_factory=OrderedDict)
    edge_feature_configs: dict[Features, FeatureConfig] | None = field(default_factory=OrderedDict)
    graph_feature_configs: dict[Features, FeatureConfig] | None = field(default_factory=OrderedDict)
    device: str = "cpu"
    seed: int | None = None
    nha_bins: int | None = None
    nha_depth: int | None = None
    dtype: str = "float32"
    base_dataset: BaseDataset = "qm9"
    hypernet_depth: int = 3
    normalize: bool = False


ZINC_SMILES_CONFIG: DSHDCConfig = DSHDCConfig(
    name="ZINC_SMILES",
    node_feature_configs=OrderedDict(
        [
            (
                Features.NODE_FEATURES,
                FeatureConfig(count=1, encoder_cls=CombinatoricIntegerEncoder),
            ),
        ]
    ),
)

ZINC_SMILES_HRR_7744_CONFIG = DSHDCConfig(
    seed=42,
    name="ZincSmilesHRR7744",
    base_dataset="zinc",
    vsa=VSAModel.HRR,
    hv_dim=88 * 88,
    device=pick_device_str(),
    node_feature_configs=OrderedDict(
        [
            (
                Features.NODE_FEATURES,
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

ZINC_SMILES_HRR_7744_CONFIG_F64: DSHDCConfig = deepcopy(ZINC_SMILES_HRR_7744_CONFIG)
ZINC_SMILES_HRR_7744_CONFIG_F64.name = "ZincSmilesHRR7744F64"
ZINC_SMILES_HRR_7744_CONFIG_F64.dtype = "float64"

ZINC_SMILES_HRR_6144_G1G4_CONFIG: DSHDCConfig = deepcopy(ZINC_SMILES_HRR_7744_CONFIG_F64)
ZINC_SMILES_HRR_6144_G1G4_CONFIG.name = "ZincSmilesHRR6144F64G1G4"
ZINC_SMILES_HRR_6144_G1G4_CONFIG.hv_dim = 6144
ZINC_SMILES_HRR_6144_G1G4_CONFIG.hypernet_depth = 4
ZINC_SMILES_HRR_6144_G1G4_CONFIG.hv_count = 2

ZINC_SMILES_HRR_5120_G1G4_CONFIG: DSHDCConfig = deepcopy(ZINC_SMILES_HRR_6144_G1G4_CONFIG)
ZINC_SMILES_HRR_5120_G1G4_CONFIG.name = "ZincSmilesHRR5120F64G1G4"
ZINC_SMILES_HRR_5120_G1G4_CONFIG.hv_dim = 5120

ZINC_SMILES_HRR_2048_F64_5G1NG4_CONFIG: DSHDCConfig = DSHDCConfig(
    seed=42,
    name="ZincSmilesHRR2048F645G1NG4",
    base_dataset="zinc",
    vsa=VSAModel.HRR,
    hv_dim=2048,
    device=pick_device_str(),
    node_feature_configs=OrderedDict(
        [
            (
                Features.NODE_FEATURES,
                FeatureConfig(
                    count=math.prod([9, 6, 3, 4, 2]),
                    encoder_cls=CombinatoricIntegerEncoder,
                    index_range=IndexRange((0, 5)),
                    bins=[9, 6, 3, 4, 2],
                ),
            ),
        ]
    ),
    normalize=True,
    hypernet_depth=4,
    dtype="float64",
)

ZINC_SMILES_HRR_1024_F64_5G1NG4_CONFIG = deepcopy(ZINC_SMILES_HRR_2048_F64_5G1NG4_CONFIG)
ZINC_SMILES_HRR_1024_F64_5G1NG4_CONFIG.name = "ZincSmilesHRR1024F645G1NG4"
ZINC_SMILES_HRR_1024_F64_5G1NG4_CONFIG.hv_dim = 1024

ZINC_SMILES_HRR_256_F64_5G1NG4_CONFIG = deepcopy(ZINC_SMILES_HRR_2048_F64_5G1NG4_CONFIG)
ZINC_SMILES_HRR_256_F64_5G1NG4_CONFIG.name = "ZincSmilesHRR256F645G1NG4"
ZINC_SMILES_HRR_256_F64_5G1NG4_CONFIG.hv_dim = 256

QM9_SMILES_CONFIG = DSHDCConfig(
    seed=42,
    name="QM9Smiles",
    base_dataset="qm9",
    vsa=VSAModel.HRR,
    hv_dim=40 * 40,
    device=pick_device_str(),
    node_feature_configs=OrderedDict(
        [
            (
                Features.NODE_FEATURES,
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

QM9_SMILES_HRR_1600_CONFIG: DSHDCConfig = deepcopy(QM9_SMILES_CONFIG)
QM9_SMILES_HRR_1600_CONFIG.hv_dim = 1600
QM9_SMILES_HRR_1600_CONFIG.name = "QM9SmilesHRR1600"

QM9_SMILES_HRR_1600_CONFIG_F64: DSHDCConfig = deepcopy(QM9_SMILES_CONFIG)
QM9_SMILES_HRR_1600_CONFIG_F64.hv_dim = 1600
QM9_SMILES_HRR_1600_CONFIG_F64.name = "QM9SmilesHRR1600F64"
QM9_SMILES_HRR_1600_CONFIG_F64.dtype = "float64"
QM9_SMILES_HRR_1600_CONFIG_F64.hv_count = 3


QM9_SMILES_HRR_1600_CONFIG_F64_G1G3_CONFIG: DSHDCConfig = deepcopy(QM9_SMILES_HRR_1600_CONFIG_F64)
QM9_SMILES_HRR_1600_CONFIG_F64_G1G3_CONFIG.name = "QM9SmilesHRR1600F64G1G3"
QM9_SMILES_HRR_1600_CONFIG_F64_G1G3_CONFIG.hv_count = 2

QM9_SMILES_HRR_1600_CONFIG_F64_G1NG3_CONFIG: DSHDCConfig = deepcopy(QM9_SMILES_HRR_1600_CONFIG_F64_G1G3_CONFIG)
QM9_SMILES_HRR_1600_CONFIG_F64_G1NG3_CONFIG.name = "QM9SmilesHRR1600F64G1NG3"
QM9_SMILES_HRR_1600_CONFIG_F64_G1NG3_CONFIG.normalize = True

QM9_SMILES_HRR_256_CONFIG_F64_G1NG3_CONFIG: DSHDCConfig = deepcopy(QM9_SMILES_HRR_1600_CONFIG_F64_G1NG3_CONFIG)
QM9_SMILES_HRR_256_CONFIG_F64_G1NG3_CONFIG.name = "QM9SmilesHRR256F64G1NG3"
QM9_SMILES_HRR_256_CONFIG_F64_G1NG3_CONFIG.hv_dim = 256


class SupportedDataset(enum.Enum):
    # Currently considered for the final experiments
    ZINC_SMILES_HRR_256_F64_5G1NG4 = ("ZINC_SMILES_HRR_256_F64_5G1NG4", ZINC_SMILES_HRR_256_F64_5G1NG4_CONFIG)
    ZINC_SMILES_HRR_1024_F64_5G1NG4 = ("ZINC_SMILES_HRR_1024_F64_5G1NG4", ZINC_SMILES_HRR_1024_F64_5G1NG4_CONFIG)
    ZINC_SMILES_HRR_2048_F64_5G1NG4 = ("ZINC_SMILES_HRR_2048_F64_5G1NG4", ZINC_SMILES_HRR_2048_F64_5G1NG4_CONFIG)
    QM9_SMILES_HRR_256_F64_G1NG3 = ("QM9_SMILES_HRR_256_F64_G1NG3", QM9_SMILES_HRR_256_CONFIG_F64_G1NG3_CONFIG)
    QM9_SMILES_HRR_1600_F64_G1NG3 = ("QM9_SMILES_HRR_1600_F64_G1NG3", QM9_SMILES_HRR_1600_CONFIG_F64_G1NG3_CONFIG)

    # No longer considered for the final experiments and can be ignored
    ZINC_SMILES = ("ZINC_SMILES", ZINC_SMILES_CONFIG)
    ZINC_SMILES_HRR_7744 = ("ZINC_SMILES_HRR_7744", ZINC_SMILES_HRR_7744_CONFIG)
    ZINC_SMILES_HRR_7744_F64 = ("ZINC_SMILES_HRR_7744_F64", ZINC_SMILES_HRR_7744_CONFIG_F64)
    ZINC_SMILES_HRR_5120_F64_G1G3 = ("ZINC_SMILES_HRR_5120_F64_G1G3", ZINC_SMILES_HRR_5120_G1G4_CONFIG)
    ZINC_SMILES_HRR_6144_F64_G1G3 = ("ZINC_SMILES_HRR_6144_F64_G1G3", ZINC_SMILES_HRR_6144_G1G4_CONFIG)
    QM9_SMILES = ("QM9_SMILES", QM9_SMILES_CONFIG)
    QM9_SMILES_HRR_1600 = ("QM9_SMILES_HRR_1600", QM9_SMILES_HRR_1600_CONFIG)
    QM9_SMILES_HRR_1600_F64 = ("QM9_SMILES_HRR_1600_F64", QM9_SMILES_HRR_1600_CONFIG_F64)
    QM9_SMILES_HRR_1600_F64_G1G3 = ("QM9_SMILES_HRR_1600_F64_G1G3", QM9_SMILES_HRR_1600_CONFIG_F64_G1G3_CONFIG)

    def __new__(cls, value: str, default_cfg: DSHDCConfig):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.default_cfg = default_cfg
        return obj


@dataclass
class FallbackDecoderSettings:
    """Settings for the greedy fallback decoder."""

    initial_limit: int = 2048
    limit: int = 2048
    beam_size: int = 2048
    pruning_method: str = "cos_sim"
    use_size_aware_pruning: bool = True
    use_one_initial_population: bool = False
    use_g3_instead_of_h3: bool = False
    use_modified_graph_embedding: bool = False
    random_sample_ratio: float = 0.0
    validate_ring_structure: bool | None = None  # Inherits from main if None
    top_k: int | None = None  # Inherits from main if None

    @property
    def pruning_fn(self) -> str:
        """Alias for pruning_method for internal use."""
        return self.pruning_method

    @property
    def graph_embedding_attr(self) -> str:
        """Returns the graph embedding attribute name."""
        return "g3" if self.use_g3_instead_of_h3 else "graph_embedding"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class DecoderSettings:
    """Main decoder settings for graph decoding."""

    iteration_budget: int = 3  # QM9 default
    max_graphs_per_iter: int = 1024
    _top_k: int = field(default=10, init=False, repr=False)
    sim_eps: float = 0.0001
    early_stopping: bool = True
    prefer_smaller_corrective_edits: bool = False
    only_correction_level_zero: bool = True
    use_correction: bool = True
    _validate_ring_structure: bool = field(default=False, init=False, repr=False)
    fallback_decoder_settings: FallbackDecoderSettings = field(default_factory=FallbackDecoderSettings)

    # Z3 decoder specific
    max_solutions: int = 1000

    def __post_init__(self):
        """Initialize inherited values in fallback settings."""
        # Set defaults for private fields if not already set
        if not hasattr(self, "_top_k"):
            self._top_k = 10
        if not hasattr(self, "_validate_ring_structure"):
            self._validate_ring_structure = False

        # Inherit into fallback settings
        updates = {}
        if self.fallback_decoder_settings.top_k is None:
            updates["top_k"] = self._top_k
        if self.fallback_decoder_settings.validate_ring_structure is None:
            updates["validate_ring_structure"] = self._validate_ring_structure

        if updates:
            self.fallback_decoder_settings = replace(self.fallback_decoder_settings, **updates)

    @property
    def top_k(self) -> int:
        """Top-k value for decoding."""
        return self._top_k

    @top_k.setter
    def top_k(self, value: int) -> None:
        """Set top_k and propagate to fallback settings."""
        self._top_k = value
        self.fallback_decoder_settings = replace(self.fallback_decoder_settings, top_k=value)

    @property
    def validate_ring_structure(self) -> bool:
        """Whether to validate ring structure during decoding."""
        return self._validate_ring_structure

    @validate_ring_structure.setter
    def validate_ring_structure(self, value: bool) -> None:
        """Set validate_ring_structure and propagate to fallback settings."""
        self._validate_ring_structure = value
        self.fallback_decoder_settings = replace(self.fallback_decoder_settings, validate_ring_structure=value)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def get_default_for(cls, base_dataset: BaseDataset) -> "DecoderSettings":
        """Returns default decoder settings for QM9 dataset."""
        if base_dataset == "qm9":
            return cls(
                iteration_budget=3,
                max_graphs_per_iter=1024,
                fallback_decoder_settings=FallbackDecoderSettings(limit=2048, beam_size=2048),
            )
        # zinc default
        return cls(
            iteration_budget=100,
            max_graphs_per_iter=512,
            fallback_decoder_settings=FallbackDecoderSettings(
                limit=256,
                beam_size=96,  # 48,64,96
            ),
        )
