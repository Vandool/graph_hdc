import itertools
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Union

import torch
import torchhd
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import scatter
from torchhd import VSATensor
from torchhd.tensors.hrr import HRRTensor
from torchhd.tensors.map import MAPTensor

from src.encoding.types import VSAModel

# ========= Paths =========
PATH = Path(__file__).parent.parent.absolute()
ARTIFACTS_PATH = PATH / "artifacts"
ASSETS_PATH = ARTIFACTS_PATH / "assets"
DATASET_TEST_PATH = ARTIFACTS_PATH / "datasets"



# ========= Torchhd Utils =========

# Type for any hypervector operation (bind or bundle)
ReductionOP = Literal["bind", "bundle"]


def scatter_hd(
    src: Tensor,
    index: Tensor,
    *,
    op: ReductionOP,
    dim_size: int | None = None,
) -> Tensor:
    """
    Scatter-reduce a batch of hypervectors along dim=0 using
    either torchhd.bind or torchhd.bundle, with minimal overhead
    for MAP, BSC and HRR models.

    Args:
        src (Tensor): hypervector batch of shape [N, D, ...] where
                      N is the “items” dimension to scatter over.
        index (LongTensor): shape [N], bucket indices in [0..dim_size).
        op (Callable): either torchhd.bind or torchhd.bundle.
        dim_size (int, optional): number of output buckets.
                                   If None, uses index.max()+1.

    Returns:
        Tensor: scattered & reduced hypervectors of shape
                [dim_size, D, ...], same dtype/device as src.
    """
    # infer output size
    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    # dispatch on type and op
    reduce = ""
    if isinstance(src, MAPTensor):
        # MAP bind == elementwise multiply → scatter-mul
        # MAP bundle == elementwise sum → use pyg scatter-sum
        reduce = "sum" if op == "bundle" else "mul"
    elif isinstance(src, HRRTensor) and op == "bundle":
        # HRR bundle == elementwise sum
        reduce = "sum"
        # HRR bind (circular conv) not supported by pyg
    if reduce:
        # When the dim_size is bigger than the addressed indexes, the scatter stacks zero vectors to reach the desired
        # dimensions, this is not correct in the hyperdimensional algebra. There we need identity vectors, such vectors
        # that when bound with a random-hypervector X, the result is X. Therefore we need to add them manually
        idx_dim = int(index.max().item()) + 1
        result = scatter(src, index, dim=0, dim_size=idx_dim, reduce=reduce)

        if (num_identity_vectors := dim_size - idx_dim) == 0:
            return result

        # TODO: Improve this
        vsa = VSAModel.HRR
        if VSAModel.MAP.value in repr(type(src)):
            vsa = VSAModel.MAP
        # elif VSAModel.BSC.value in repr(type(src)):
        #     vsa = VSAModel.BSC
        identities = torchhd.identity(
            num_vectors=num_identity_vectors, dimensions=src.shape[-1], vsa=vsa.value, device=src.device
        )
        return torch.cat([result, identities])

    # Generic fallback: group rows manually in Python (will be slower)
    # Currently no support for dim other than0
    buckets = [[] for _ in range(dim_size)]
    for i, b in enumerate(index.tolist()):
        buckets[b].append(src[i])

    # initialize output slots
    op_hd = torchhd.multibind if op == "bind" else torchhd.multibundle
    out = []
    for bucket in buckets:
        if not bucket:
            # empty bucket → identity for bind, zero for bundle
            identity = type(src).identity(1, src.shape[-1], device=src.device).squeeze(0)
            out.append(identity)
        else:
            # reduce the list by repeatedly applying op
            reduced = op_hd(torch.stack(bucket, dim=0))
            out.append(reduced)
    return torch.stack(out, dim=0)


def cartesian_bind_tensor(tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Given a list of K sets of hypervectors, each of shape [Ni, D] or [Ni],
    produce the full cross product of size (N_prod = N1 * … * NK) and bind
    each K-tuple into one [D] hypervector, returning [N_prod, D].

    Special cases:
      - K == 0 → ValueError
      - K == 1 → returns the single set (unsqueezed to 2D if needed, no self-binding)
    """
    # Clean-up the list
    tensors = [t for t in tensors if t is not None]
    if not tensors:
        err_msg = "Need at least one set"
        raise ValueError(err_msg)

    # ---- SHORT‐CIRCUIT SINGLE SET ----
    if len(tensors) == 1:
        t = tensors[0]
        # if it’s 1-D, treat it as [N,] → [N,1]; else assume [N,D]
        # We don't self-bind
        if t.dim() == 1:
            return t.unsqueeze(-1)  # [N,1]
        return t  # [N,D]

    # ---- GENERAL K > 1 CASE ----
    # 1) build the index grid for the cartesian product of sizes
    sizes = [t.shape[0] for t in tensors]
    idx_grids = torch.cartesian_prod(*[torch.arange(n, device=tensors[0].device) for n in sizes])  # [N_prod, K]

    # 2) gather each “column” of indices from its set
    hv_list = []
    for k, t in enumerate(tensors):
        idxs = idx_grids[:, k]

        # 1-D → gather to [N_prod], then unsqueeze to [N_prod,1]
        # 2-D → gather rows → [N_prod, D]
        hv = t[idxs] if t.dim() != 1 else t[idxs].unsqueeze(-1)  # [N_prod]
        hv_list.append(hv)

    # 3) stack along a new “slot” axis → [N_prod, K, D]
    stacked = torch.stack(hv_list, dim=1)

    # 4) bind all K hypervectors per row into one: output [N_prod, D]
    return torchhd.multibind(stacked)


def cartesian_bind_tensor_2(domains: Sequence[Tensor]) -> Tensor:
    """
    Given a list of K hypervector sets, each of shape [Nk, D],
    return a [Ntotal, D] tensor where Ntotal = N1 * N2 * ... * Nk,
    and each row is the bind of one K‐tuple drawn from each set.

    (No special 1-D logic—we always expect shape [Nk, D].)
    """
    if not domains:
        msg = "`domains` must contain at least one [Nk, D] tensor"
        raise ValueError(msg)

    # Verify consistent feature‐dimension and device
    D = domains[0].size(1)
    device = domains[0].device
    for H in domains:
        if H.ndim != 2 or H.size(1) != D:
            msg = f"All domains must be shape [Nk, D], got {tuple(H.shape)}"
            raise ValueError(msg)
        if H.device != device:
            msg = "All domains must be on the same device"
            raise ValueError(msg)

    # 1) Build Cartesian product of indices → shape [Ntotal, K]
    idx_grid = torch.cartesian_prod(*(torch.arange(H.size(0), device=device) for H in domains))

    # 2) Gather each domain’s hypervectors → list of [Ntotal, D]
    gathered: list[Tensor] = [domains[k][idx_grid[:, k]] for k in range(len(domains))]

    # 3) Stack into [Ntotal, K, D] and multibind → [Ntotal, D]
    stacked = torch.stack(gathered, dim=1)  # → shape [Ntotal, K, D]
    return torchhd.multibind(stacked)


def unbind(composite: VSATensor, factor: VSATensor) -> VSATensor:
    """
    Recover `other` from composite = bind(H, other), given H.

    This works for any VSA model supported by torchhd:

      • MAP   (±1 vectors):       bind = elementwise * ,   inverse = identity ⇒ bind(composite, H) = other
      • HRR   (real vectors):     bind = circular-conv,    inverse = circular-corr ⇒ bind(composite, H.inverse()) = other
      • FHRR  (complex phasors):  bind = complex-mul,      inverse = conj          ⇒ bind(composite, H.inverse()) = other
      • VTB   (block conv):       bind = VTB-bind,         inverse = VTB-inverse   ⇒ bind(composite, H.inverse()) = other

    Args:
        composite:  the bound hypervector       (bind(H, other))
        factor:     the known hypervector H

    Returns:
        the recovered hypervector `other`
    """
    # sanity checks
    assert type(composite) is type(factor), "both must be same VSATensor subclass"
    # factor.inverse() will return the proper inverse for each model
    return torchhd.bind(composite, factor.inverse())

# ========= Utils =========
class TupleIndexer:
    def __init__(self, sizes: Sequence[int]) -> None:
        # clean-up sizes: 0's should be kicked out
        sizes = [s for s in sizes if s]
        self.sizes = sizes
        # idx to tuple mapping can be a list
        self.idx_to_tuple: list[tuple[int, ...]] = list(itertools.product(*(range(N) for N in sizes))) if sizes else []
        self.tuple_to_idx: dict[tuple[int, ...], int] = (
            {t: idx for idx, t in enumerate(self.idx_to_tuple)} if sizes else {}
        )

    def get_tuple(self, idx: int) -> tuple[int, ...]:
        return self.idx_to_tuple[idx]

    def get_tuples(self, idxs: [int]) -> list[tuple[int, ...]]:
        return [self.idx_to_tuple[idx] for idx in idxs]

    def get_idx(self, tup: Union[tuple[int, ...], int]) -> int:
        if isinstance(tup, int):
            return self.tuple_to_idx.get((tup,))
        return self.tuple_to_idx.get(tup)

    def get_idxs(self, tuples: list[Union[tuple[int, ...], int]]) -> list[int]:
        return [self.get_idx(tup) for tup in tuples]

    def size(self) -> int:
        return len(self.idx_to_tuple)

    def get_sizes(self) -> Sequence[int]:
        return self.sizes


def get_deep_size(obj, seen_ids=None):
    """
    Recursively compute an approximate deep size of `obj` (in bytes),
    walking into nested mappings, sequences, and PyTorch tensors.
    Avoids double counting by tracking object IDs in `seen_ids`.
    """
    if seen_ids is None:
        seen_ids = set()

    obj_id = id(obj)
    if obj_id in seen_ids:
        return 0
    seen_ids.add(obj_id)

    size = 0

    # 1) If it's a PyTorch tensor, count its data buffer + wrapper overhead:
    if isinstance(obj, torch.Tensor):
        # data buffer:
        size += obj.numel() * obj.element_size()
        # wrapper overhead:
        size += sys.getsizeof(obj)
        return size

    # 2) Otherwise, start with the shallow size:
    size += sys.getsizeof(obj)

    # 3) If it's a mapping (dict, defaultdict, etc.), recurse on keys + values
    if isinstance(obj, Mapping):
        for key, val in obj.items():
            size += get_deep_size(key, seen_ids)
            size += get_deep_size(val, seen_ids)

    # 4) If it's a sequence (list, tuple, set, etc.) but not a str/bytes/bytearray:
    elif isinstance(obj, (Sequence, set, frozenset)) and not isinstance(obj, (str, bytes, bytearray)):
        for item in obj:
            size += get_deep_size(item, seen_ids)

    return size

def flatten_counter(c: Counter) -> list:
    return [k for k, v in c.items() for _ in range(v)]

class DataTransformer:
    @staticmethod
    def to_tuple_list(edge_index: Tensor) -> list[tuple[int, ...]]:
        return list(map(tuple, edge_index.T.tolist()))

    @staticmethod
    def get_edge_existence_counter(batch: int, data: Data, indexer: TupleIndexer) -> list[tuple[int, ...]]:
        """
        Returns a Counter of existing edges for a single graph in the batch,
        mapping (src_idx, dst_idx) to count.  It converts global node indices
        into local indices 0..(N-1) before encoding.
        """
        # 1) Mask to edges belonging to this graph
        edge_mask = (
                            data.batch[data.edge_index[0]] == batch
                    ) & (
                            data.batch[data.edge_index[1]] == batch
                    )
        # 2) Extract the global edge index pairs
        truth_edges_global = [
            tuple(pair)
            for pair in data.edge_index[:, edge_mask].t().tolist()
        ]

        # 3) Build mapping from global node index -> local (0..N-1)
        nodes_global = data.batch.eq(batch).nonzero(as_tuple=False).flatten().tolist()
        nodes_global.sort()
        global_to_local = {g: i for i, g in enumerate(nodes_global)}

        # 4) Gather local node feature tuples in the same order
        x_feats = data.x[data.batch == batch]
        # If features are vectors, convert each row to a tuple
        x_list = [tuple(row.tolist()) for row in x_feats] if x_feats.dim() > 1 else x_feats.squeeze(-1).tolist()

        # 5) Re‐encode each edge using the indexer on local node tuples
        truth_edge_idxs: list[tuple[int, int]] = []
        for g_u, g_v in truth_edges_global:
            u_loc = global_to_local[g_u]
            v_loc = global_to_local[g_v]
            idx_u = indexer.get_idx(x_list[u_loc])
            idx_v = indexer.get_idx(x_list[v_loc])
            truth_edge_idxs.append((idx_u, idx_v))

        # 6) Count and return
        return Counter(truth_edge_idxs)


    @staticmethod
    def get_edge_counter(data: Data, batch) -> list[tuple[int, ...]]:
        """
        Returns a Counter of existing edges for a single graph in the batch,
        mapping (src_idx, dst_idx) to count.  It converts global node indices
        into local indices 0..(N-1) before encoding.
        """
        # 1) Mask to edges belonging to this graph
        edge_mask = (
                            data.batch[data.edge_index[0]] == batch
                    ) & (
                            data.batch[data.edge_index[1]] == batch
                    )
        # 2) Extract the global edge index pairs
        truth_edges_global = [
            tuple(pair)
            for pair in data.edge_index[:, edge_mask].t().tolist()
        ]

        # 3) Build mapping from global node index -> local (0..N-1)
        nodes_global = data.batch.eq(batch).nonzero(as_tuple=False).flatten().tolist()
        nodes_global.sort()
        global_to_local = {g: i for i, g in enumerate(nodes_global)}

        # 4) Gather local node feature tuples in the same order
        x_feats = data.x[data.batch == batch].int()
        # 5) Re‐encode each edge using the indexer on local node tuples
        x_list = [tuple(row.tolist()) for row in x_feats] if x_feats.dim() > 1 else x_feats.squeeze(-1).tolist()
        truth_edge_idxs: list[tuple[int, int]] = []
        for g_u, g_v in truth_edges_global:
            u_loc = global_to_local[g_u]
            v_loc = global_to_local[g_v]
            truth_edge_idxs.append((x_list[u_loc], x_list[v_loc]))

        # 6) Count and return
        return Counter(truth_edge_idxs)






    @staticmethod
    def get_x_from_batch(batch: int, data: Data) -> Tensor:
        x = data.x.squeeze(-1).int()
        return x[data.batch == batch].tolist()

    @staticmethod
    def get_node_counter_from_batch(batch: int, data: Data) -> Counter[tuple[int, ...]]:
        node_list = DataTransformer.get_x_from_batch(batch, data)
        if isinstance(node_list[0], list):
            ## When we have multiple features like in ZINC_D
            return Counter([tuple(n) for n in node_list])
        return Counter([(n,) for n in node_list])

