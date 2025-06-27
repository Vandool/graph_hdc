from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torchhd

from src.utils.utils import TupleIndexer


class AbstractFeatureEncoder(ABC):
    def __init__(
        self,
        dim: int,
        vsa: str,
        device: torch.device = torch.device("cpu"),
        seed: int | None = None,
        num_categories: int | None = None,
        idx_offset: int = 0,
    ):
        self.dim = dim
        self.vsa = vsa
        self.device = device
        self.num_categories = num_categories
        self.idx_offset = idx_offset
        if seed is not None:
            torch.manual_seed(seed)
        self.codebook = self.generate_codebook()

    @abstractmethod
    def generate_codebook(self) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def normalize(value: Any) -> Any:
        return value

    @abstractmethod
    def encode(self, value: Any) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, hv: torch.Tensor) -> Any:
        pass

    def get_codebook(self) -> torch.Tensor:
        return self.codebook

    @abstractmethod
    def decode_index(self, idx: int) -> torch.Tensor:
        pass


class CategoricalOneHotEncoder(AbstractFeatureEncoder):
    def __init__(
        self,
        dim: int,
        num_categories: int,
        vsa: str = "MAP",
        device: torch.device = torch.device("cpu"),
        seed: int | None = None,
        idx_offset: int = 0,
    ):
        super().__init__(
            dim=dim, vsa=vsa, device=device, seed=seed, num_categories=num_categories, idx_offset=idx_offset
        )

    def generate_codebook(self) -> torch.Tensor:
        return torchhd.random(self.num_categories, self.dim, vsa=self.vsa, device=self.device)

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        index = torch.argmax(data).item()
        return self.codebook[index]

    def decode(self, hv: torch.Tensor) -> torch.Tensor:
        similarities = torchhd.cosine_similarity(hv, self.codebook)
        index = torch.argmax(similarities)
        one_hot = torch.zeros(self.num_categories, device=self.device)
        one_hot[index] = 1
        return one_hot

    def decode_index(self, index: int) -> torch.Tensor:
        """
        Given a category index ∈ [0 .. num_categories-1], return the corresponding one-hot tensor.
        """
        if not (0 <= index < self.num_categories):
            err_msg = f"Index {index} out of range [0, {self.num_categories})"
            raise IndexError(err_msg)
        one_hot = torch.zeros(self.num_categories, device=self.device)
        one_hot[index] = 1
        return one_hot


class CategoricalIntegerEncoder(AbstractFeatureEncoder):
    def __init__(
        self,
        dim: int,
        num_categories: int,
        vsa: str = "MAP",
        device: torch.device = torch.device("cpu"),
        seed: int | None = None,
        idx_offset: int = 0,
    ):
        """
        :param dim: Dimensionality of the VSA hyper-vectors
        :param num_categories: Number of distinct integer categories
        :param vsa: VSA model to use for hypervector ops
        :param device: Torch device
        :param seed: Optional RNG seed
        """
        super().__init__(
            dim=dim, vsa=vsa, device=device, seed=seed, num_categories=num_categories, idx_offset=idx_offset
        )

    def generate_codebook(self) -> torch.Tensor:
        return torchhd.random(self.num_categories, self.dim, vsa=self.vsa, device=self.device)

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """
        Encode integer category indices into hyper-vector.

        :param data:
        :param data: Tensor of shape [..., N, 1]
            - ...: optional batch dimensions
            - N: number of things we have feature for
            - last dim: the feature value of the thing (each an integer index in [0, C))

        :returns: Tensor of shape [..., N, D]
            - D: embedding dimension of hyper-vector
        """
        # Advanced indexing: each integer index pulls one row from codebook
        # make sure the last dim is just a singleton index
        if data.shape[-1] != 1:
            msg = f"Expected last dim=1, got {data.shape[-1]}"
            raise ValueError(msg)
        # values contain the indexes of the codebook in float → [..., N]
        idx = data.squeeze(-1).long() - self.idx_offset  # [..., N]
        # directly index the codebook
        return self.codebook[idx]  # [..., N, D]

    def decode(self, hv: torch.Tensor) -> torch.LongTensor:
        """
        Decode hyper-vector back to integer category indices.

        :param hv: Tensor of shape [..., N, D]
            - ...: optional batch dimensions
            - N: number of nodes
            - D: embedding dimension
        :returns: LongTensor of shape [..., N, 1]
            - last dim=1: integer indices in [0, C)
        """
        if hv.shape[-1] != self.dim:
            msg = f"Expected last dim={self.dim}, got {hv.shape[-1]}"
            raise ValueError(msg)
        # Compute cosine similarities: [..., N, C]
        sims = torchhd.cosine_similarity(hv, self.codebook)
        # Best matching category per node
        idx = sims.argmax(dim=-1)  # [..., N
        idx += self.idx_offset
        return idx.unsqueeze(-1).float()  # [..., N, 1]

    def decode_index(self, label: int) -> torch.LongTensor:
        """
        Given a known integer category `label` ∈ [idx_offset .. idx_offset+num_categories-1],
        return a tensor of shape [1, 1] containing exactly that label—i.e. the “input” you
        would have supplied to `encode(...)`.

        Example:
            enc = CategoricalIntegerEncoder(dim=256, num_categories=10, idx_offset=0)
            single_input = enc.decode_index(3)  # → tensor([[3]])
            hv = enc.encode(single_input)       # same as if you had encoded `tensor([[3]])`
        """
        if not (self.idx_offset <= label < self.idx_offset + self.num_categories):
            msg = f"Label {label} out of range [{self.idx_offset} .. {self.idx_offset + self.num_categories - 1}]"
            raise IndexError(msg)
        # Create a Tensor of shape [1, 1] containing `label`.
        return torch.tensor([[label]], dtype=torch.long, device=self.device)


class CombinatoricIntegerEncoder(AbstractFeatureEncoder):
    def __init__(
        self,
        dim: int,
        num_categories: int,
        indexer: TupleIndexer = TupleIndexer([28, 6]),
        vsa: str = "MAP",
        device: torch.device = torch.device("cpu"),
        seed: int | None = None,
        idx_offset: int = 0,
    ):
        """
        :param dim: Dimensionality of the VSA hyper-vectors
        :param num_categories: Number of distinct integer categories
        :param vsa: VSA model to use for hypervector ops
        :param device: Torch device
        :param seed: Optional RNG seed
        """
        super().__init__(
            dim=dim, vsa=vsa, device=device, seed=seed, num_categories=num_categories, idx_offset=idx_offset
        )
        self.indexer = indexer

    def generate_codebook(self) -> torch.Tensor:
        return torchhd.random(self.num_categories, self.dim, vsa=self.vsa, device=self.device)

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """
        Encode integer category indices into hyper-vector.

        :param data:
        :param data: Tensor of shape [..., N, 1]
            - ...: optional batch dimensions
            - N: number of things we have feature for
            - last dim: the feature value of the thing (each an integer index in [0, C))

        :returns: Tensor of shape [..., N, D]
            - D: embedding dimension of hyper-vector
        """
        # Advanced indexing: each integer index pulls one row from codebook
        # make sure the last dim is just a singleton index
        if data.shape[-1] == 1:
            msg = f"Expected last dim>1, got {data.shape[-1]}"
            raise ValueError(msg)
        # values contain the indexes of the codebook in float → [..., N]
        tup = data.squeeze(-1).long() - self.idx_offset  # [..., N]
        tup = list(map(tuple, tup.tolist()))
        idxs = self.indexer.get_idxs(tup)
        idxs_tens = torch.tensor(idxs, dtype=torch.long, device=self.device)
        # directly index the codebook
        return self.codebook[idxs_tens]  # [..., N, D]

    def decode(self, hv: torch.Tensor) -> torch.LongTensor:
        """
        Decode hyper-vector back to integer category indices.

        :param hv: Tensor of shape [..., N, D]
            - ...: optional batch dimensions
            - N: number of nodes
            - D: embedding dimension
        :returns: LongTensor of shape [..., N, 1]
            - last dim=1: integer indices in [0, C)
        """
        if hv.shape[-1] != self.dim:
            msg = f"Expected last dim={self.dim}, got {hv.shape[-1]}"
            raise ValueError(msg)
        # Compute cosine similarities: [..., N, C]
        sims = torchhd.cosine_similarity(hv, self.codebook)
        # Best matching category per node
        idx = sims.argmax(dim=-1)  # [..., N
        idx += self.idx_offset
        idx = [int(i.item()) for i in idx.squeeze()]
        res = torch.tensor(self.indexer.get_tuples(idx))
        return res.float()  # [..., N, 1]

    def decode_index(self, label: int) -> torch.LongTensor:
        """
        Given a known integer category `label` ∈ [idx_offset .. idx_offset+num_categories-1],
        return a tensor of shape [1, 1] containing exactly that label—i.e. the “input” you
        would have supplied to `encode(...)`.

        Example:
            enc = CategoricalIntegerEncoder(dim=256, num_categories=10, idx_offset=0)
            single_input = enc.decode_index(3)  # → tensor([[3]])
            hv = enc.encode(single_input)       # same as if you had encoded `tensor([[3]])`
        """
        return torch.tensor([self.indexer.get_tuple(label)], dtype=torch.long, device=self.device)


class TrueFalseEncoder(AbstractFeatureEncoder):
    def __init__(
        self,
        dim: int,
        num_categories: int,  # noqa: ARG002
        vsa: str = "MAP",
        device: torch.device = torch.device("cpu"),
        seed: int | None = None,
    ):
        super().__init__(dim, vsa, device, seed, num_categories=1)

    def generate_codebook(self) -> torch.Tensor:
        true = torchhd.random(1, self.dim, vsa=self.vsa, device=self.device)
        false = true.inverse()
        return torch.stack([false, true])

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        idx = torch.argmax(data).item()
        return self.codebook[idx]

    def decode(self, hv: torch.Tensor) -> torch.Tensor:
        similarities = torchhd.cosine_similarity(hv, self.codebook)
        threshold = 0.5  # Adjust threshold based on practical requirements
        return (similarities > threshold).float()

    def decode_index(self, label: int) -> torch.Tensor:
        """
        Given a known boolean label (0 or 1), return exactly the one hot
        input that `encode(...)` would consume.

        Example:
            enc = TrueFalseEncoder(dim=256, vsa=VSAModel.MAP)
            one_hot_false = enc.decode_index(0)  # → tensor([1, 0])
            hv_false    = enc.encode(one_hot_false)

            one_hot_true  = enc.decode_index(1)  # → tensor([0, 1])
            hv_true       = enc.encode(one_hot_true)
        """
        if label not in (0, 1):
            msg = f"TrueFalseEncoder expects label 0 or 1, got {label}"
            raise IndexError(msg)

        # Build a one‐hot of length 2 on the correct device
        one_hot = torch.zeros(2, device=self.device)
        one_hot[label] = 1.0
        return one_hot


class CategoricalLevelEncoder(AbstractFeatureEncoder):
    def __init__(
        self,
        dim: int,
        num_categories: int,
        vsa: str = "MAP",
        device: torch.device = torch.device("cpu"),
        seed: int | None = None,
    ):
        """
        :param dim: Dimensionality of the VSA hyper-vectors
        :param num_categories: Number of distinct integer categories
        :param vsa: VSA model to use for hypervector ops
        :param device: Torch device
        :param seed: Optional RNG seed
        """
        super().__init__(dim=dim, vsa=vsa, device=device, seed=seed, num_categories=num_categories)

    def generate_codebook(self) -> torch.Tensor:
        return torchhd.level(self.num_categories, self.dim, vsa=self.vsa, device=self.device)

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """
        Encode integer category indices into hyper-vector.

        :param data:
        :param data: Tensor of shape [..., N, 1]
            - ...: optional batch dimensions
            - N: number of things we have feature for
            - last dim: the feature value of the thing (each an integer index in [0, C))

        :returns: Tensor of shape [..., N, D]
            - D: embedding dimension of hyper-vector
        """
        # Advanced indexing: each integer index pulls one row from codebook
        # make sure the last dim is just a singleton index
        if data.shape[-1] != 1:
            msg = f"Expected last dim=1, got {data.shape[-1]}"
            raise ValueError(msg)
        # values contain the indexes of the codebook in float → [..., N]
        idx = data.squeeze(-1).long()  # [..., N]
        # directly index the codebook
        return self.codebook[idx]  # [..., N, D]

    def decode(self, hv: torch.Tensor) -> torch.LongTensor:
        """
        Decode hyper-vector back to integer category indices.

        :param hv: Tensor of shape [..., N, D]
            - ...: optional batch dimensions
            - N: number of nodes
            - D: embedding dimension
        :returns: LongTensor of shape [..., N, 1]
            - last dim=1: integer indices in [0, C)
        """
        if hv.shape[-1] != self.dim:
            msg = f"Expected last dim={self.dim}, got {hv.shape[-1]}"
            raise ValueError(msg)
        # Compute cosine similarities: [..., N, C]
        sims = torchhd.cosine_similarity(hv, self.codebook)
        # Best matching category per node
        idx = sims.argmax(dim=-1)  # [..., N]
        return idx.unsqueeze(-1).float()  # [..., N, 1]

    def decode_index(self, label: int) -> torch.LongTensor:
        """
        Given a known integer category `label` ∈ [0..num_categories−1],
        return a tensor of shape [1, 1] containing that label—i.e. exactly what
        `encode(...)` would have taken as input for a singleton.

        Example:
            enc = CategoricalLevelEncoder(dim=256, num_categories=5)
            # Suppose we know “category = 3”:
            single_input = enc.decode_index(3)    # → tensor([[3]])
            hv = enc.encode(single_input)         # → hypervector for category 3
        """
        if not (0 <= label < self.num_categories):
            msg = f"Label {label} out of range [0..{self.num_categories - 1}]"
            raise IndexError(msg)

        # Return shape [1, 1] so that encode(...) sees last dim=1
        return torch.tensor([[label]], dtype=torch.long, device=self.device)
