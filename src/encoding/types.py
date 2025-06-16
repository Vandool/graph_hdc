import enum

from torchhd import FHRRTensor, HRRTensor, MAPTensor, VSATensor


class VSAModel(enum.Enum):
    # Multiply-Add-Permute (MAP)  # noqa: ERA001
    # - Representation: Bipolar {±1}^D
    # - Binding: Element-wise multiplication (self-inverse, commutative)
    # - Bundling: Element-wise sum; can retain real sum or apply sign function
    # - Characteristics: Analog version of BSC; supports counting; reversible if dimension is large enough
    MAP = ("MAP", MAPTensor)

    # Holographic Reduced Representation (HRR)
    # - Representation: Real continuous ℝ^D (often normalized)
    # - Binding: Circular convolution (commutative); inverse via circular correlation
    # - Bundling: Element-wise sum; can normalize or keep raw sums
    # - Characteristics: High capacity and robustness; reversible with small error; suitable for undirected graphs
    HRR = ("HRR", HRRTensor)

    # Fourier Holographic Reduced Representation (FHRR)
    # - Representation: Complex phasor ℂ^D (unit-length)
    # - Binding: Element-wise complex multiplication (phase addition); exact inverse via conjugate
    # - Bundling: Element-wise sum of complex vectors; often normalize each component’s magnitude
    # - Characteristics: Exact inverses; handles binding chains cleanly; suitable for reversible encoding
    FHRR = ("FHRR", FHRRTensor)

    def __new__(cls, value: str, t_class: VSATensor) -> "VSAModel":
        obj = object.__new__(cls)
        obj._value_ = value
        obj._vsa_type_ = t_class
        return obj

    @classmethod
    def is_supported(cls, vsa_type: VSATensor) -> bool:
        return any(vsa.tensor_class == vsa_type for vsa in cls)

    @property
    def tensor_class(self) -> VSATensor:
        return self._vsa_type_
