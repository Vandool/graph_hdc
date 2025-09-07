import argparse
import itertools
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from random import random
from typing import Any, Literal, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchhd
from rdkit import Chem
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import scatter
from torchhd import VSATensor
from torchhd.tensors.hrr import HRRTensor
from torchhd.tensors.map import MAPTensor

from src.encoding.the_types import Feat, VSAModel

# ========= Paths =========
ROOT = Path(__file__)
while ROOT.stem != "graph_hdc":
    ROOT = ROOT.parent
ROOT = ROOT.absolute()

GLOBAL_DATASET_PATH = Path(os.getenv("GLOBAL_DATASETS", ROOT / "_datasets"))
GLOBAL_MODEL_PATH = Path(os.getenv("GLOBAL_MODELS", ROOT / "_models"))
GLOBAL_DATA_PATH = Path(os.getenv("GLOBAL_DATA", ROOT / ""))

TEST_ARTIFACTS_PATH = ROOT / "tests_new/artifacts"
TEST_ASSETS_PATH = ROOT / "tests_new/assets"

FORMAL_CHARGE_IDX_TO_VAL: dict[int, int] = {0: 0, 1: +1, 2: -1}


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


# ========= Utils =========
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jsonable(x: Any):
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, torch.device):
        return str(x)
    if torch.is_tensor(x):
        return x.item() if x.numel() == 1 else x.detach().cpu().tolist()
    if isinstance(x, Enum):
        return x.value
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    return str(x)


def plot_vector_distributions(vectors: np.ndarray, kind: str = "hist", bins: int = 32):
    """
    Plot the per-dimension distribution of a set of vectors.

    Parameters:
    - vectors (np.ndarray): shape (num_vectors, dim)
    - kind (str): one of {"hist", "box", "violin"}
    """
    num_vectors, dim = vectors.shape
    print(f"Shape: {num_vectors} vectors of dimension {dim}")

    if kind == "hist":
        cols = 8
        rows = int(np.ceil(dim / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), sharex=True, sharey=True)
        axes = axes.flatten()
        for i in range(dim):
            axes[i].hist(vectors[:, i], bins=bins, color="skyblue", edgecolor="black")
            axes[i].set_title(f"Dim {i}", fontsize=8)
            axes[i].tick_params(axis="both", labelsize=6)
        for j in range(dim, len(axes)):
            axes[j].axis("off")
        fig.suptitle("Histogram per Dimension", fontsize=14)
        plt.tight_layout()
        plt.show()

    elif kind == "box":
        plt.figure(figsize=(max(10, dim // 2), 5))
        plt.boxplot(vectors.T, showfliers=False)
        plt.xlabel("Dimension")
        plt.ylabel("Value")
        plt.title("Boxplot of Dimensions")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif kind == "violin":
        df = pd.DataFrame(vectors, columns=[f"dim_{i}" for i in range(dim)])
        df_melt = df.melt(var_name="dimension", value_name="value")
        plt.figure(figsize=(max(10, dim // 2), 5))
        sns.violinplot(data=df_melt, x="dimension", y="value", inner="quartile", cut=0)
        plt.xticks(rotation=90)
        plt.title("Violin Plot of Dimensions")
        plt.tight_layout()
        plt.show()

    else:
        raise ValueError("Invalid kind. Choose from {'hist', 'box', 'violin'}.")


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
    # Ensure device
    index = index.to(src.device, dtype=torch.long, non_blocking=True)

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


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pick_device_str() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DataTransformer:
    @staticmethod
    def to_tuple_list(edge_index: Tensor) -> list[tuple[int, ...]]:
        return list(map(tuple, edge_index.T.tolist()))

    @staticmethod
    def get_edge_existence_counter(batch: int, data: Data, indexer: TupleIndexer) -> Counter:
        """
        Returns a Counter of existing edges for a single graph in the batch,
        mapping (src_idx, dst_idx) to count.  It converts global node indices
        into local indices 0..(N-1) before encoding.
        """
        # 1) Mask to edges belonging to this graph
        edge_mask = (data.batch[data.edge_index[0]] == batch) & (data.batch[data.edge_index[1]] == batch)
        # 2) Extract the global edge index pairs
        truth_edges_global = [tuple(pair) for pair in data.edge_index[:, edge_mask].t().tolist()]

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
    def get_edge_counter(data: Data, batch) -> Counter[tuple[int, ...]]:
        """
        Returns a Counter of existing edges for a single graph in the batch,
        mapping (src_idx, dst_idx) to count.  It converts global node indices
        into local indices 0..(N-1) before encoding.
        """
        # 1) Mask to edges belonging to this graph
        edge_mask = (data.batch[data.edge_index[0]] == batch) & (data.batch[data.edge_index[1]] == batch)
        # 2) Extract the global edge index pairs
        truth_edges_global = [tuple(pair) for pair in data.edge_index[:, edge_mask].t().tolist()]

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

    @staticmethod
    def nx_to_pyg(G: nx.Graph) -> "Data":
        """
        Convert an undirected NetworkX graph with ``feat`` attributes to a
        :class:`torch_geometric.data.Data`.

        Node features are stacked as a dense matrix ``x`` of dtype ``long`` with
        columns ``[atom_type, degree_idx, formal_charge_idx, explicit_hs]``.

        Undirected edges are converted to a directed ``edge_index`` with both
        directions.

        :param G: NetworkX graph where each node has a ``feat: Feat`` attribute.
        :returns: PyG ``Data`` object with fields ``x`` and ``edge_index``.
        :raises RuntimeError: If PyTorch/PyG are not available.
        """
        if torch is None or Data is None:
            raise RuntimeError("torch / torch_geometric are required for nx_to_pyg")

        # Node ordering: use sorted ids for determinism
        nodes = sorted(G.nodes)
        idx_of: dict[int, int] = {n: i for i, n in enumerate(nodes)}

        # Features
        feats: list[list[int]] = [list(G.nodes[n]["feat"].to_tuple()) for n in nodes]
        x = torch.tensor(feats, dtype=torch.long)

        # Edges: add both directions
        src, dst = [], []
        for u, v in G.edges():
            iu, iv = idx_of[u], idx_of[v]
            src.extend([iu, iv])
            dst.extend([iv, iu])
        if src:
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index)

    @staticmethod
    def pyg_to_nx(
        data: "Data",
        *,
        strict_undirected: bool = True,
        allow_self_loops: bool = False,
    ) -> nx.Graph:
        """
        Convert a PyG ``Data`` (undirected, bidirectional edges) to a mutable NetworkX graph.

        Assumptions
        ----------
        - ``data.x`` has shape ``[N, 4]`` with integer-encoded features:
          ``[atom_type, degree_idx, formal_charge_idx, explicit_hs]``.
        - ``data.edge_index`` is bidirectional (both (u,v) and (v,u) present) for undirected graphs.
        - Features are **frozen** and represent the final target degrees.

        Node attributes
        ---------------
        - ``feat``: ``Feat`` instance (constructed from the 4-tuple).
        - ``target_degree``: ``feat.target_degree`` (== ``degree_idx + 1``).

        Parameters
        ----------
        data : Data
            PyG data object.
        strict_undirected : bool, optional
            If ``True``, assert that ``edge_index`` is symmetric.
        allow_self_loops : bool, optional
            If ``False``, drop self-loops.

        Returns
        -------
        nx.Graph
            Mutable undirected graph with node attributes.

        Raises
        ------
        ValueError
            If feature dimensionality is not ``[N, 4]`` or edges are not symmetric when required.
        """
        if data.x is None:
            raise ValueError("data.x is None (expected [N,4] features).")
        if data.edge_index is None:
            raise ValueError("data.edge_index is None.")

        x = data.x
        if x.dim() != 2 or x.size(1) != 4:
            raise ValueError(f"Expected data.x shape [N,4], got {tuple(x.size())}.")

        # Ensure integer features
        if not torch.is_floating_point(x):
            x_int = x.to(torch.long)
        else:
            x_int = x.to(torch.long)  # safe cast, your encoding is discrete

        N = x_int.size(0)
        ei = data.edge_index
        if ei.dim() != 2 or ei.size(0) != 2:
            raise ValueError("edge_index must be [2, E].")
        src = ei[0].to(torch.long)
        dst = ei[1].to(torch.long)

        # Build undirected edge set (dedup, optional self-loop handling)
        pairs = set()
        for u, v in zip(src.tolist(), dst.tolist(), strict=False):
            if u == v and not allow_self_loops:
                continue
            a, b = (u, v) if u < v else (v, u)
            pairs.add((a, b))

        if strict_undirected:
            # Check that for every (u,v) there is a (v,u) in the original directed list
            dir_pairs = set(zip(src.tolist(), dst.tolist(), strict=False))
            for u, v in list(pairs):
                if (u, v) not in dir_pairs or (v, u) not in dir_pairs:
                    raise ValueError(f"edge_index is not symmetric for undirected edge ({u},{v}).")

        # Construct NX graph
        G = nx.Graph()
        G.add_nodes_from(range(N))

        # Attach features and target degrees
        for n in range(N):
            t = tuple(int(z) for z in x_int[n].tolist())  # (atom_type, degree_idx, formal_charge_idx, explicit_hs)
            f = Feat.from_tuple(t)  # requires your Feat dataclass
            G.nodes[n]["feat"] = f
            G.nodes[n]["target_degree"] = f.target_degree

        # Add edges
        G.add_edges_from(pairs)
        return G

    @staticmethod
    def nx_to_mol(
        G: nx.Graph,
        *,
        atom_symbols=None,
        infer_bonds: bool = True,
        set_no_implicit: bool = True,
        set_explicit_hs: bool = True,
        set_atom_map_nums: bool = False,
        sanitize: bool = True,
        kekulize: bool = True,
        validate_heavy_degree: bool = False,
    ) -> tuple["Chem.Mol", dict[int, int]]:
        """
        Build an RDKit molecule from an **undirected** NetworkX graph with frozen features.

        Each node must have ``feat`` as a 4-tuple: ``(atom_type_idx, degree_idx, charge_idx, explicit_hs)``,
        or an object with ``to_tuple()`` returning that tuple.

        If ``infer_bonds=True``, a greedy valence-based routine assigns DOUBLE/TRIPLE bonds to reduce
        per-atom valence deficits (based on element & formal charge) before constructing the RDKit bonds.

        :param G: Undirected graph; node['feat'] is frozen feature tuple.
        :param atom_symbols: Mapping from atom_type_idx to element symbol.
        :param infer_bonds: If True, infer bond orders from valence targets; otherwise use SINGLE bonds.
        :param set_no_implicit: Set ``NoImplicit=True`` so RDKit won’t alter explicit H counts.
        :param set_explicit_hs: Apply ``NumExplicitHs`` from features.
        :param set_atom_map_nums: Annotate RDKit atoms with original NX ids (debug).
        :param sanitize: Run ``Chem.SanitizeMol`` (may fail if bonding is chemically impossible).
        :param kekulize: Try ``Chem.Kekulize`` after sanitize.
        :param validate_heavy_degree: Assert NX heavy degree equals ``degree_idx+1`` for all nodes.
        :returns: (mol, nx_to_rd_index_map)

        .. note::
           This inference is **heuristic** and not guaranteed to match real chemistry.
           If sanitize fails, the function falls back to SINGLE bonds (unless you set ``sanitize=False``).
        """

        if atom_symbols is None:
            atom_symbols = ["Br", "C", "Cl", "F", "I", "N", "O", "P", "S"]

        def _as_tuple(feat_obj) -> tuple[int, int, int, int]:
            if hasattr(feat_obj, "to_tuple"):
                t = feat_obj.to_tuple()
            else:
                t = tuple(int(x) for x in feat_obj)
            if len(t) != 4:
                raise ValueError(f"feat must be a 4-tuple, got {t}")
            return t

        # --- unpack features & basic guards ---
        feats: dict[int, tuple[int, int, int, int]] = {}
        symbols: dict[int, str] = {}
        charges: dict[int, int] = {}
        expHs: dict[int, int] = {}
        target_deg: dict[int, int] = {}

        for n in G.nodes:
            t = _as_tuple(G.nodes[n]["feat"])
            at_idx, deg_idx, ch_idx, hs = t
            if not (0 <= at_idx < len(atom_symbols)):
                raise ValueError(f"atom_type_idx out of range on node {n}: {at_idx}")
            sym = atom_symbols[at_idx]
            ch = FORMAL_CHARGE_IDX_TO_VAL.get(int(ch_idx), int(ch_idx))
            feats[n] = (at_idx, deg_idx, ch_idx, hs)
            symbols[n] = sym
            charges[n] = int(ch)
            expHs[n] = int(hs)
            target_deg[n] = int(deg_idx) + 1  # heavy neighbors count

        if validate_heavy_degree:
            for n in G.nodes:
                if G.degree[n] != target_deg[n]:
                    raise AssertionError(
                        f"Heavy degree mismatch at node {n}: NX={G.degree[n]} vs target={target_deg[n]}"
                    )

        # --- optional bond-order inference on the NX graph ---
        # Bond orders: dict of undirected edge -> order (1,2,3)
        order: dict[tuple[int, int], int] = {}
        for u, v in G.edges:
            a, b = (u, v) if u < v else (v, u)
            order[(a, b)] = 1  # start with single

        if infer_bonds and G.number_of_edges() > 0:
            # Element valence "menus" (very rough, neutral-first; charged variants added where common)
            # Chosen valence is the **smallest** >= current usage to avoid over-bonding.
            valence_menu = {
                "C": {0: (4,), -1: (3, 4), +1: (3,)},
                "N": {0: (3, 5), -1: (2, 3), +1: (4,)},
                "O": {0: (2,), -1: (1, 2), +1: (3,)},
                "P": {0: (3, 5), +1: (4, 5)},
                "S": {0: (2, 4, 6), +1: (3, 5), -1: (1, 2)},
                "F": {0: (1,)},
                "Cl": {0: (1,)},
                "Br": {0: (1,)},
                "I": {0: (1,)},
            }

            # ring preference: small cycles first
            try:
                cycles = nx.cycle_basis(G)
            except Exception:
                cycles = []
            ring_edges = set()
            for cyc in cycles:
                for i in range(len(cyc)):
                    a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                    a, b = (a, b) if a < b else (b, a)
                    ring_edges.add((a, b))

            def current_usage(n: int) -> int:
                """Valence usage = explicit Hs + sum(bond orders to heavy neighbors)."""
                s = expHs[n]
                for nb in G.neighbors(n):
                    a, b = (n, nb) if n < nb else (nb, n)
                    s += order[(a, b)]
                return s

            def target_valence(n: int) -> int:
                sym = symbols[n]
                ch = charges[n]
                menu = valence_menu.get(sym, {0: (target_deg[n] + expHs[n],)})
                opts = menu.get(ch, menu.get(0, (target_deg[n] + expHs[n],)))
                used = current_usage(n)
                for v in sorted(opts):
                    if v >= used:
                        return v
                return max(opts)

            # Greedy raise orders until deficits vanish or stuck
            def deficit(n: int) -> int:
                return max(0, target_valence(n) - current_usage(n))

            # atoms priority: hetero first, then carbons, larger deficit first
            hetero = {"N", "O", "S", "P"}

            def atom_priority(n: int) -> tuple[int, int]:
                return (1 if symbols[n] in hetero else 0, deficit(n))

            # bond escalation priority: prefer ring edges and hetero involvement
            def bond_priority(u: int, v: int) -> tuple[int, int, int]:
                a, b = (u, v) if u < v else (v, u)
                in_ring = 1 if (a, b) in ring_edges else 0
                hetero_count = int(symbols[u] in hetero) + int(symbols[v] in hetero)
                return (in_ring, hetero_count, order[(a, b)])

            changed = True
            iters = 0
            MAX_ITERS = 4 * G.number_of_edges()
            while changed and iters < MAX_ITERS:
                iters += 1
                changed = False

                # collect candidate atoms with positive deficit
                cand_atoms = [n for n in G.nodes if deficit(n) > 0]
                if not cand_atoms:
                    break

                # sort atoms by priority (hetero, deficit)
                cand_atoms.sort(key=lambda n: atom_priority(n), reverse=True)

                for n in cand_atoms:
                    if deficit(n) <= 0:
                        continue
                    # choose neighbor with deficit and smallest current order, with bond priority
                    nbs = [m for m in G.neighbors(n) if deficit(m) > 0 and order[(n, m) if n < m else (m, n)] < 3]
                    if not nbs:
                        continue
                    # sort neighbors/bonds by ring & hetero preference, then lowest existing order
                    nbs.sort(key=lambda m: bond_priority(n, m), reverse=True)
                    m = nbs[0]
                    a, b = (n, m) if n < m else (m, n)
                    # bump bond order by 1
                    order[(a, b)] += 1
                    changed = True

            # Optional final clip: avoid absurd orders on halogens
            for (a, b), bo in list(order.items()):
                if symbols[a] in ("F", "Cl", "Br", "I") or symbols[b] in ("F", "Cl", "Br", "I"):
                    order[(a, b)] = min(order[(a, b)], 1)

        # --- build RDKit RWMol ---
        rw = Chem.RWMol()
        nx_to_rd: dict[int, int] = {}
        for n in sorted(G.nodes):
            at_idx, deg_idx, ch_idx, hs = feats[n]
            sym = symbols[n]
            ch = charges[n]
            atom = Chem.Atom(sym)
            atom.SetFormalCharge(int(ch))
            if set_explicit_hs:
                atom.SetNumExplicitHs(int(hs))
            if set_no_implicit:
                atom.SetNoImplicit(True)
            rd_idx = rw.AddAtom(atom)
            nx_to_rd[n] = rd_idx

        # add bonds with (possibly inferred) orders
        for u, v in G.edges:
            a, b = (u, v) if u < v else (v, u)
            bo = order.get((a, b), 1)
            bt = Chem.BondType.SINGLE if bo == 1 else Chem.BondType.DOUBLE if bo == 2 else Chem.BondType.TRIPLE
            rw.AddBond(nx_to_rd[u], nx_to_rd[v], bt)

        mol = rw.GetMol()

        if set_atom_map_nums:
            for n, rd_idx in nx_to_rd.items():
                mol.GetAtomWithIdx(rd_idx).SetAtomMapNum(int(n))

        if sanitize:
            try:
                Chem.SanitizeMol(mol)
                if kekulize:
                    try:
                        Chem.Kekulize(mol, clearAromaticFlags=True)
                    except Exception:
                        pass
            except Exception:
                # fallback: rebuild with single bonds if sanitize fails
                rw2 = Chem.RWMol()
                idx_map = {}
                for n in sorted(G.nodes):
                    at_idx, deg_idx, ch_idx, hs = feats[n]
                    sym = symbols[n]
                    ch = charges[n]
                    a = Chem.Atom(sym)
                    a.SetFormalCharge(int(ch))
                    if set_explicit_hs:
                        a.SetNumExplicitHs(int(expHs[n]))
                    if set_no_implicit:
                        a.SetNoImplicit(True)
                    idx_map[n] = rw2.AddAtom(a)
                for u, v in G.edges:
                    rw2.AddBond(idx_map[u], idx_map[v], Chem.BondType.SINGLE)
                mol = rw2.GetMol()
                if set_atom_map_nums:
                    for n, rd_idx in idx_map.items():
                        mol.GetAtomWithIdx(rd_idx).SetAtomMapNum(int(n))
                if sanitize:
                    # try a gentle sanitize (may still fail if graph is impossible chemically)
                    try:
                        Chem.SanitizeMol(mol)
                    except Exception:
                        pass

        return mol, nx_to_rd


if __name__ == "__main__":
    print(ROOT)
