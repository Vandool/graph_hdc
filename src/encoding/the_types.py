import enum
from dataclasses import dataclass

from torchhd import FHRRTensor, HRRTensor, MAPTensor, VSATensor


class VSAModel(enum.Enum):
    # Multiply-Add-Permute (MAP)
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


@dataclass(frozen=True)
class Feat:
    """
    Hashable node feature (discrete indices).

    :param atom_type: Index of atom type (e.g., Br,C,Cl,F,I,N,O,P,S mapped to ints).
    :param degree_idx: Degree minus one, i.e. 0->deg 1, 4->deg 5.
    :param formal_charge_idx: Encoded as 0,1,2 for charges [0,1,-1] respectively.
    :param explicit_hs: Total explicit hydrogens (0..3).
    :param is_in_ring: Whether the atom is part of a ring (optional).
    """

    atom_type: int
    degree_idx: int
    formal_charge_idx: int
    explicit_hs: int
    is_in_ring: bool | None = None

    @property
    def target_degree(self) -> int:
        """Final/desired node degree (degree index + 1)."""
        return self.degree_idx + 1

    def to_tuple(self) -> tuple:
        """Return (atom_type, degree_idx, formal_charge_idx, explicit_hs, is_in_ring)."""
        res = [self.atom_type, self.degree_idx, self.formal_charge_idx, self.explicit_hs]
        if self.is_in_ring is not None:
            res.append(int(self.is_in_ring))
        return tuple(res)

    @staticmethod
    def from_tuple(t: tuple) -> "Feat":
        """
        Construct a Feat from a tuple of length 4 or 5.

        :param t: Tuple (atom_type, degree_idx, formal_charge_idx, explicit_hs[, is_in_ring]).
        :returns: Feat instance.
        :raises ValueError: If tuple length is not 4 or 5.
        """
        if len(t) == 4:
            a, d, c, h = t
            return Feat(int(a), int(d), int(c), int(h))

        a, d, c, h, r = t
        return Feat(int(a), int(d), int(c), int(h), bool(r) if r is not None else None)
