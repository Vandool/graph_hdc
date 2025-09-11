"""
We want a pairwise classifier f(G_1, G_2) to {0,1} that answers “is G_1 an induced subgraph of G_2” under
feature-aware matching, with node features frozen to those of the final graph.

•	Positives (P): connected, induced subgraphs G[S] of the full molecule G for sizes k=2,dots,K. Node features are
    taken from G (as you already do), edges are exactly those among S present in G.
•	Negatives (N): same node feature convention, but edge pattern (or node set) must differ so that G_1 is not an
    induced subgraph of G.

Positive Sampling:


"""

import collections
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import networkx as nx
import torch
from torch import Tensor
from torch_geometric.data import Data, Dataset
from torch_geometric.data import InMemoryDataset as _IMD
from tqdm.auto import tqdm

from src.datasets.zinc_smiles_generation import ZincSmiles
from src.utils.utils import GLOBAL_DATASET_PATH


# ────────────────────────────── utilities (graph I/O) ─────────────────────────
def _as_int_tuple_row(x_row: Sequence[float]) -> tuple[int, ...]:
    """Cast discrete-coded float row -> integer tuple label: (atom_id, degree_idx, charge_idx, num_Hs)."""
    return tuple(int(v) for v in x_row)


def pyg_to_nx(data: Data) -> nx.Graph:
    """PyG Data -> undirected NX graph with node attribute `label` = tuple[int,...]."""
    G = nx.Graph()
    x = data.x.cpu().numpy()
    for i in range(x.shape[0]):
        G.add_node(i, label=_as_int_tuple_row(x[i]))
    ei = data.edge_index.cpu().numpy()
    for u, v in zip(ei[0], ei[1], strict=False):
        if u < v:
            G.add_edge(int(u), int(v))
    return G


def induced_edge_index_from_node_set(edge_index: Tensor, nodes: Sequence[int]) -> Tensor:
    """Undirected both-direction edge_index for induced subgraph on `nodes`."""
    nodes = list(nodes)
    pos = {n: i for i, n in enumerate(nodes)}
    src, dst = edge_index.tolist()
    undirected = set()
    for u, v in zip(src, dst, strict=False):
        if u in pos and v in pos and u < v:
            undirected.add((pos[u], pos[v]))
    if not undirected:
        return torch.empty((2, 0), dtype=torch.long)
    s, d = [], []
    for u, v in undirected:
        s += [u, v]
        d += [v, u]
    return torch.tensor([s, d], dtype=torch.long)


def make_subgraph_data(full: Data, node_indices: Sequence[int], *, edge_index_override: Tensor | None = None) -> Data:
    """Create subgraph Data with features copied from the parent (no recompute)."""
    node_indices = list(node_indices)
    x_sub = full.x[node_indices].clone()
    ei_sub = (
        edge_index_override
        if edge_index_override is not None
        else induced_edge_index_from_node_set(full.edge_index, node_indices)
    )
    out = Data(x=x_sub, edge_index=ei_sub)
    out.parent_size = torch.tensor([full.num_nodes], dtype=torch.long)
    return out


def nx_to_edge_index_on_ordered_nodes(H: nx.Graph, ordered_nodes: Sequence[int]) -> Tensor:
    """NX Graph -> edge_index on local indices defined by `ordered_nodes`."""
    pos = {n: i for i, n in enumerate(ordered_nodes)}
    undirected = {(min(pos[u], pos[v]), max(pos[u], pos[v])) for (u, v) in H.edges()}
    if not undirected:
        return torch.empty((2, 0), dtype=torch.long)
    s, d = [], []
    for u, v in undirected:
        s += [u, v]
        d += [v, u]
    return torch.tensor([s, d], dtype=torch.long)


def low_degree_anchor_order(G: nx.Graph) -> list[int]:
    """Nodes sorted by (target_degree_from_label, actual_degree) ascending."""
    return sorted(
        G.nodes(),
        key=lambda v: (target_degree_from_label(G.nodes[v]["label"]), G.degree[v]),
    )


# ───────── helpers to guarantee negatives by construction ─────────


def node_label(G: nx.Graph, v: int) -> tuple[int, ...]:
    return G.nodes[v]["label"]


def parent_label_edge_set(G: nx.Graph) -> set[tuple[tuple[int, ...], tuple[int, ...]]]:
    S: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for u, v in G.edges():
        la, lb = node_label(G, u), node_label(G, v)
        S.add((la, lb) if la <= lb else (lb, la))
    return S


def unordered_label_pair(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (a, b) if a <= b else (b, a)


def wrong_add_candidates_anywhere(
    G_parent: nx.Graph,
    S: Sequence[int],
    parent_label_pairs: set[tuple[tuple[int, ...], tuple[int, ...]]],
) -> list[tuple[int, int]]:
    """All non-edges (u,v) in G_parent[S] such that residual[u]>0, residual[v]>0,
    and unordered label pair (label(u),label(v)) DOES NOT occur as an edge anywhere in parent."""
    H = G_parent.subgraph(S)
    res = residual_capacity_vector(G_parent, S)
    # all unordered pairs in S that are non-edges
    S_list = list(S)
    non_edges = [
        (S_list[i], S_list[j])
        for i in range(len(S_list))
        for j in range(i + 1, len(S_list))
        if not H.has_edge(S_list[i], S_list[j])
    ]
    bad = []
    for u, v in non_edges:
        if res[u] <= 0 or res[v] <= 0:
            continue
        lu = node_label(G_parent, u)
        lv = node_label(G_parent, v)
        if unordered_label_pair(lu, lv) not in parent_label_pairs:
            bad.append((u, v))
    return bad


# ───────────────────────── residual-degree utilities ──────────────────────────
def target_degree_from_label(label: tuple[int, ...]) -> int:
    """degree_idx is stored as 0..4 for degrees 1..5 → return target degree."""
    return int(label[1]) + 1


def residual_capacity_vector(G: nx.Graph, S: Sequence[int]) -> dict[int, int]:
    """
    residual(v) = target_degree_in_final(v) - current_degree_in_subgraph(v), for v in S.
    Target from node label; current from G.subgraph(S).
    """
    H = G.subgraph(S)
    res = {}
    for v in S:
        tgt = target_degree_from_label(G.nodes[v]["label"])
        cur = H.degree[v]
        res[v] = tgt - cur
    return res


def forbidden_pairs_k2(
    G: nx.Graph,
    parent_pairs: set[tuple[tuple[int, ...], tuple[int, ...]]],
) -> list[tuple[int, int]]:
    """
    All unordered node pairs (u,v), u<v, such that:
      - (u,v) is NOT an edge in G, and
      - the unordered label pair (label(u), label(v)) NEVER occurs as an edge anywhere in G, and
      - both nodes can support at least one incident edge in a 2-node start (target degree >= 1).
    """
    nodes = list(G.nodes())
    bad = []
    for i, u in enumerate(nodes):
        lu = node_label(G, u)
        if target_degree_from_label(lu) < 1:
            continue
        for v in nodes[i + 1 :]:
            if G.has_edge(u, v):
                continue
            lv = node_label(G, v)
            if target_degree_from_label(lv) < 1:
                continue
            if unordered_label_pair(lu, lv) not in parent_pairs:
                bad.append((u, v))
    return bad


# ────────────────────────────── positive samplers ─────────────────────────────


def sample_connected_kset_with_anchor(
    G: nx.Graph,
    k: int,
    anchor: int,
    *,
    rng: random.Random,
    max_expands: int = 4,
) -> list[int] | None:
    """
    Sample a connected k-node subset that **contains** a given anchor.

    The procedure is a randomized, anchored BFS-style growth:
    it starts from ``anchor``, repeatedly adds unvisited neighbors to the
    partial set, and—if growth stalls—tops up by adding any neighbor of
    the current set. The returned node set (if any) is connected and
    includes ``anchor``.

    :param G: Undirected NetworkX graph.
    :param k: Target subgraph size (``k >= 1``).
    :param anchor: Node id that must be included in the sample.
    :param rng: Python ``random.Random`` instance for reproducibility.
    :param max_expands: Number of independent restart attempts.
    :returns: List of node ids of length ``k`` if successful, else ``None``.

    Notes
    -----
    - If the anchor's connected component has fewer than ``k`` nodes, no
      sample exists and the function will return ``None``.
    - The “top-up” stage only adds neighbors of the current set, so
      connectivity is preserved by construction; the final
      ``nx.is_connected`` check is a safety guard.
    - Runtime per attempt is roughly ``O(k * avg_degree)`` on sparse graphs.
    """
    # Quick input guards: impossible k, empty graph, or bad anchor.
    if k <= 0 or k > G.number_of_nodes() or anchor not in G:
        return None

    for _ in range(max_expands):
        # Initialize the growing set S, a visited set, and a frontier set.
        # - S tracks the ordered nodes chosen so far (will be returned).
        # - visited prevents re-adding the same node.
        # - frontier contains boundary nodes from which we try to expand.
        S = [anchor]
        visited = {anchor}
        frontier = {anchor}

        # Stage 1: BFS-like randomized expansion along existing edges until |S| == k or frontier empties.
        while len(S) < k and frontier:
            # Randomly choose a boundary node to expand.
            u = rng.choice(list(frontier))

            # Consider only unvisited neighbors to avoid duplicates.
            nbrs = [w for w in G.neighbors(u) if w not in visited]

            if not nbrs:
                # Can't expand from u anymore → remove u from the frontier and try another boundary node.
                frontier.remove(u)
                continue

            # Pick a random neighbor and add it to the sample; it becomes part of the frontier as well.
            w = rng.choice(nbrs)
            S.append(w)
            visited.add(w)
            frontier.add(w)

        # Stage 2: If we still haven't reached size k, try to "top up" the set
        # by adding *any* neighbor of the current S (still preserves connectivity).
        if len(S) < k:
            # Collect neighbors of S excluding nodes already in S.
            candidates = {w for u in S for w in G.neighbors(u)} - set(S)
            while len(S) < k and candidates:
                w = rng.choice(list(candidates))
                S.append(w)
                candidates.remove(w)

        # Final safety checks: correct size, connectivity, and anchor containment.
        if len(S) == k and nx.is_connected(G.subgraph(S)) and anchor in S:
            return S

    # All attempts failed: either unlucky sampling or anchor's component < k.
    return None


# ───────────────────────────── verification (optional) ────────────────────────
def is_induced_subgraph_feature_aware(G_small: nx.Graph, G_big: nx.Graph) -> bool:
    """NetworkX VF2: is `G_small` an induced, label-preserving subgraph of `G_big`?"""
    nm = lambda a, b: a["label"] == b["label"]
    GM = nx.algorithms.isomorphism.GraphMatcher(G_big, G_small, node_match=nm)
    return GM.subgraph_is_isomorphic()


# ─────────────────────────────── pair container ───────────────────────────────
class PairData(Data):
    """
    Container for a (G1, G2, y) pair:
      - x1, edge_index1: candidate subgraph
      - x2, edge_index2: parent graph
      - y, k, neg_type, parent_idx: labels/metadata

      neg_type codes:
      • 0 = positive
      • 1 = local wrong-add (forbidden label-pair inside S; residual respected; for k=2 this is the forbidden-pair construction)
    """

    def __inc__(self, key, value, *args, **kwargs):
        # Ensure correct edge index shifting during batching
        if key == "edge_index1":
            return self.x1.size(0)
        if key == "edge_index2":
            return self.x2.size(0)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if "edge_index" in key:
            return 1
        return 0


# ───────── minimal PairConfig (no bloated switches) ─────────


class PairConfig:
    seed = 42
    k_min = 2
    k_max = None  # ← go up to N
    k_cap = 12  # main bucket

    # main bucket (k ≤ k_cap)
    pos_per_k_main = 2
    neg_per_pos_main = 2

    # tail bucket (k > k_cap)
    pos_per_k_tail = 1
    neg_per_pos_tail = 1

    # enable hard negatives with small caps (verified VF2)
    neg_wrong_edge_allowed = True
    neg_missing_edge_verified = True
    neg_rewire = True
    neg_anchor_aware = True
    neg_cross_parent = True

    cap_wrong_edge_allowed = 1
    cap_missing_edge = 1
    cap_rewire = 1
    cap_anchor_aware = 1
    cap_cross_parent = 1

    tail_anchor_fraction = 0.5


from enum import IntEnum

class PairType(IntEnum):
    r"""
    Enumeration of pair sample types.

    Members
    -------
    POSITIVE : 0
        Connected induced subgraph (label-frozen) — y=1.
    FORBIDDEN_ADD : 1
        Add a non-edge (u,v) in S whose **label pair never occurs** as an edge anywhere in the parent.
    WRONG_EDGE_ALLOWED : 2
        Add a non-edge (u,v) in S whose label pair **does occur somewhere** in the parent (contextually wrong).
    CROSS_PARENT : 3
        Use a positive G[S] from parent *i* against a different parent *j* that **does not** contain it.
    MISSING_EDGE : 4
        Remove an existing edge from G[S]; keep connected; verified **not** induced.
    REWIRE : 5
        Degree-preserving rewire of two disjoint edges inside S; keep connected; verified **not** induced.
    ANCHOR_AWARE : 6
        Add one real node w∉S and connect to feasible anchors in S; verified **not** induced.

    Notes
    -----
    - This matches the emitted ``neg_type`` values in ``_generate_pairs_stream``:
      1,2,3,4,5,6 as above; positives use 0.
    - Training policy: keep **Type 4** (MISSING_EDGE) and **Type 5** (REWIRE) each around **3–5%**.
    """
    POSITIVE = 0
    FORBIDDEN_ADD = 1
    WRONG_EDGE_ALLOWED = 2
    CROSS_PARENT = 3
    MISSING_EDGE = 4
    REWIRE = 5
    ANCHOR_AWARE = 6


# Optional: convenient reverse lookup for logging/summaries.
NEG_TYPE_NAME = {t.value: t.name for t in PairType}

class ZincPairsV2(Dataset):
    """
    ZincPairsV2 dataset:
    === Aggregated summary (train+valid+test) ===
    [ALL] size=45749267 | positives=8266188 (18.1%) | negatives=37483079 (81.9%)
    [ALL] neg_type histogram: 1:3398852 (9.1%), 2:7131081 (19.0%), 3:8266188 (22.1%), 4:4009247 (10.7%), 5:7019712 (18.7%), 6:7657999 (20.4%)

    In training the type 4 and Type 5 should be kept each around 3-5 %

    ZincPairsV2Dev dataset:
    === Aggregated summary (train+valid+test) ===
    [ALL] size=46781 | positives=8438 (18.0%) | negatives=38343 (82.0%)
    [ALL] neg_type histogram: 1:3365 (8.8%), 2:7300 (19.0%), 3:8437 (22.0%), 4:4222 (11.0%), 5:7174 (18.7%), 6:7845 (20.5%)
    """

    def __init__(
        self,
        base_dataset,
        split="train",
        root=GLOBAL_DATASET_PATH / "ZincPairsV2",
        cfg=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        shard_size=25_000,
        cache_shards=2,
        *,
        force_reprocess=False,
        dev: bool = False,
    ):
        # --- set everything process() will need BEFORE super().__init__ ---
        self.base = base_dataset
        self.split = split
        self.cfg = cfg or PairConfig()
        self.shard_size = shard_size
        self.cache_shards = cache_shards
        self._rng = random.Random(self.cfg.seed)

        if dev:
            root = GLOBAL_DATASET_PATH / "ZincPairsV2DEV"
        # idx_path must exist for process() to use
        self.idx_path = Path(root) / "processed" / f"index_{split}.pt"

        super().__init__(root, transform, pre_transform, pre_filter)

        # build if missing or forced
        if force_reprocess or not self.idx_path.exists():
            self.process()

        meta = torch.load(self.idx_path, map_location="cpu", weights_only=False)
        self._index = meta["index"]  # list[(shard_id, local_idx)]
        self._shards = meta["shards"]  # list[str] relative to processed_dir

        # LRU cache for shard tensors
        self._cache = collections.OrderedDict()

    @property
    def raw_file_names(self):
        return [f"depends_on_{type(self.base).__name__}.marker"]

    @property
    def processed_file_names(self):
        # PyG requires something here; we point to the index file we write.
        return [f"index_{self.split}.pt"]

    def download(self):
        Path(self.raw_dir).mkdir(parents=True, exist_ok=True)
        (Path(self.raw_dir) / self.raw_file_names[0]).write_text(f"split={self.split}; base_len={len(self.base)}\n")

    def len(self):
        return len(self._index)

    def get(self, idx: int):
        shard_id, local_idx = self._index[idx]
        data, slices = self._get_shard(shard_id)

        out = PairData()  # keep batching semantics (__inc__/__cat_dim__)
        ref = PairData()  # to query concat dims

        for key in data.keys():  # NOTE: .keys()
            val = data[key]
            s = slices[key]
            start = int(s[local_idx].item())
            end = int(s[local_idx + 1].item())

            if torch.is_tensor(val):
                cat_dim = ref.__cat_dim__(key, val)
                # Narrow along the correct axis (edge_index* is dim=1, others dim=0)
                out[key] = val.narrow(cat_dim, start, end - start)
            else:
                out[key] = val  # non-tensor payloads (rare)

        # Optional: quiet PyG "num_nodes" inference warning (use G1’s node count)
        if hasattr(out, "x1") and torch.is_tensor(out.x1):
            out.num_nodes = int(out.x1.size(0))

        return out

    def _get_shard(self, shard_id: int):
        # LRU cache with capacity = self.cache_shards
        if shard_id in self._cache:
            self._cache.move_to_end(shard_id)
            return self._cache[shard_id]
        shard_path = Path(self.processed_dir) / self._shards[shard_id]
        data, slices = torch.load(shard_path, map_location="cpu", weights_only=False)
        self._cache[shard_id] = (data, slices)
        while len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return data, slices

    def process(self):
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        shard_files, index = [], []
        buffer, schema_keys = [], None
        shard_id = 0

        def flush():
            nonlocal shard_id, buffer
            if not buffer:
                return
            # IMPORTANT: call collate as a static utility (no self)
            data, slices = _IMD.collate(buffer)
            fname = f"pairs_{self.split}_shard_{shard_id:05d}.pt"
            torch.save((data, slices), Path(self.processed_dir) / fname)
            index.extend((shard_id, i) for i in range(len(buffer)))
            shard_files.append(fname)
            buffer = []
            shard_id += 1

        for pair in self._generate_pairs_stream():
            # enforce uniform schema
            if schema_keys is None:
                schema_keys = set(pair.keys())
            elif set(pair.keys()) != schema_keys:
                raise RuntimeError(f"Non-uniform keys: got {set(pair.keys())} vs {schema_keys}")

            if self.pre_filter is None or self.pre_filter(pair):
                if self.pre_transform:
                    pair = self.pre_transform(pair)
                buffer.append(pair)
                if len(buffer) >= self.shard_size:
                    flush()

        flush()
        torch.save({"shards": shard_files, "index": index}, self.idx_path)

    def _generate_pairs_stream(self):
        r"""
        Build and serialize pair samples for **feature-aware induced-subgraph** containment.

        Mixed-k schedule
        ----------------
        For each parent graph `G` and for each `k` in `[k_min, min(N, k_max or N)]`:
        - For `k <= k_cap`, use *main* budgets: `pos_per_k_main`, `neg_per_pos_main`.
        - For `k >  k_cap`, use *tail* budgets: `pos_per_k_tail`, `neg_per_pos_tail`.
          The tail gives you coverage at large k without exploding size.

        Samples
        -------
        1) Positives: connected induced subgraphs G[S] with **frozen** node labels.
        2) Negatives:
           • Guaranteed-by-construction:
             Type1) Wrong-add with a **globally forbidden** label pair inside S.
                Add an edge (u,v) inside S where the label pair never occurs as an edge anywhere in the parent.
                Residual-respecting.
           • VF2-verified hard negatives (always correct):
             Type2) Wrong-edge (allowed pair): add a non-edge (u,v) in G[S] where the **label pair is allowed**
                somewhere in G.
             Type3) Cross-parent: use positive G[S] against a *different* parent that does not contain it.
                Ensures the oracle actually uses full_g_h and doesn’t devolve into a graph-only classifier. It’s a
                helpful regularizer for generalization (same motif may or may not be present depending on target).
                Should be kept capped in training
             Type4) Missing-edge: remove an existing edge from G[S].
                any candidate missing a required edge between already-selected nodes must be rejected by an
                induced-subgraph oracle. This will enforce the oracle to reject subgraphs, that are missing an edge to
                become a ring structure.
             Type5) Degree-preserving rewire: swap endpoints of two disjoint edges within S.
                This might help the decoder, to correct the previously wrnogly made decisions!
                The decoder can grow into a topology with the same degree sequence yet wrong adjacency due to a wrong
                early attachment. This family creates strong “looks plausible by degrees, but wrong by structure”
                examples. Good as hard negatives. Should be kept capped in training
             Type6) Anchor-aware (decoder-like): add **one real node** w∉S (its label from G), connect to feasible
                anchors in S. This is the closest simulation of what the decoder actually does: add one real node,
                attach to anchors by residuals, and see if the (k+1) candidate remains induced in the target. This
                directly teaches the boundary between “good next step” and “bad next step.” Core.

        All VF2 checks use `is_induced_subgraph_feature_aware`, ensuring labels are correct even without VF2 at inference.

        Notes
        -----
        - Positives per (parent,k) are deduplicated by S.
        - `cfg` may define:
            k_min, k_max, k_cap,
            pos_per_k_main, neg_per_pos_main,
            pos_per_k_tail, neg_per_pos_tail,
            neg_* flags (enable/disable families),
            cap_* per-family caps (shared by main/tail unless *_tail provided),
            tail_anchor_fraction for positive sampling.
        """
        rng, cfg = self._rng, self.cfg

        # -------------------------- local helpers (kept inside) --------------------------

        def _label(v: int, G: nx.Graph) -> tuple[int, ...]:
            return G.nodes[v]["label"]

        def _target_deg(label: tuple[int, ...]) -> int:
            # degree_idx (0..4) encodes degrees 1..5
            return int(label[1]) + 1

        def _allowed_label_pairs(G: nx.Graph) -> set[tuple[tuple[int, ...], tuple[int, ...]]]:
            """All unordered label pairs that appear as edges **anywhere** in G."""
            S: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
            for u, v in G.edges():
                a, b = _label(u, G), _label(v, G)
                S.add((a, b) if a <= b else (b, a))
            return S

        def _residuals_on_S(G: nx.Graph, S: list[int]) -> dict[int, int]:
            """Residual(v) = target_degree(v) - degree_in_induced(G,S)(v)."""
            H = G.subgraph(S)
            res = {}
            for v in S:
                res[v] = _target_deg(_label(v, G)) - H.degree[v]
            return res

        def _distinct_k_connected_with_anchor(G: nx.Graph, k: int, anchor: int) -> list[int] | None:
            """Thin wrapper around your anchored sampler."""
            return sample_connected_kset_with_anchor(G, k, anchor=anchor, rng=rng)

        # ----- Hard negative A: wrong-edge (allowed pair), return (S, edge_index_override) -----
        def _gen_wrong_edge_allowed(G: nx.Graph, S: list[int], max_neg: int) -> list[tuple[list[int], torch.Tensor]]:
            H = G.subgraph(S).copy()
            allowed = _allowed_label_pairs(G)
            out: list[tuple[list[int], torch.Tensor]] = []
            L = len(S)
            for i in range(L):
                for j in range(i + 1, L):
                    u, v = S[i], S[j]
                    if H.has_edge(u, v):
                        continue  # must be a non-edge in S
                    pair = (_label(u, G), _label(v, G))
                    pair = pair if pair[0] <= pair[1] else (pair[1], pair[0])
                    if pair not in allowed:
                        continue  # this is the "forbidden" family handled elsewhere
                    H2 = H.copy()
                    H2.add_edge(u, v)
                    if nx.is_connected(H2) and not is_induced_subgraph_feature_aware(H2, G):
                        ei = nx_to_edge_index_on_ordered_nodes(H2, S)
                        out.append((S, ei))
                        if len(out) >= max_neg:
                            return out
            return out

        # ----- Hard negative B: missing-edge (verified), return (S, edge_index_override) -----
        def _gen_missing_edge_verified(G: nx.Graph, S: list[int], max_neg: int) -> list[tuple[list[int], torch.Tensor]]:
            H = G.subgraph(S).copy()
            out: list[tuple[list[int], torch.Tensor]] = []
            for u, v in list(H.edges()):
                H2 = H.copy()
                H2.remove_edge(u, v)
                if nx.is_connected(H2) and not is_induced_subgraph_feature_aware(H2, G):
                    ei = nx_to_edge_index_on_ordered_nodes(H2, S)
                    out.append((S, ei))
                    if len(out) >= max_neg:
                        break
            return out

        # ----- Hard negative C: degree-preserving rewire (verified), return (S, edge_index_override) -----
        def _gen_rewire_degree_preserving(
            G: nx.Graph, S: list[int], max_neg: int
        ) -> list[tuple[list[int], torch.Tensor]]:
            H = G.subgraph(S).copy()
            out: list[tuple[list[int], torch.Tensor]] = []
            E = list(H.edges())
            for i in range(len(E)):
                a, b = E[i]
                for j in range(i + 1, len(E)):
                    c, d = E[j]
                    if len({a, b, c, d}) < 4:
                        continue  # need disjoint edges
                    for (x, y), (p, q) in [((a, c), (b, d)), ((a, d), (b, c))]:
                        if H.has_edge(*sorted((x, y))) or H.has_edge(*sorted((p, q))):
                            continue
                        H2 = H.copy()
                        H2.remove_edge(a, b)
                        H2.remove_edge(c, d)
                        H2.add_edge(*sorted((x, y)))
                        H2.add_edge(*sorted((p, q)))
                        if nx.is_connected(H2) and not is_induced_subgraph_feature_aware(H2, G):
                            ei = nx_to_edge_index_on_ordered_nodes(H2, S)
                            out.append((S, ei))
                            if len(out) >= max_neg:
                                return out
            return out

        # ----- Hard negative D: cross-parent indices (verified by VF2) -----
        def _gen_cross_parent_indices(
            G_pos: nx.Graph, parents: list[nx.Graph], self_idx: int, max_neg: int
        ) -> list[int]:
            out = []
            for j, G_other in enumerate(parents):
                if j == self_idx:
                    continue
                if not is_induced_subgraph_feature_aware(G_pos, G_other):
                    out.append(j)
                    if len(out) >= max_neg:
                        break
            return out

        # ----- Hard negative E: anchor-aware (decoder-like), return (S_plus, edge_index_override) -----
        def _gen_anchor_aware(G: nx.Graph, S: list[int], max_neg: int) -> list[tuple[list[int], torch.Tensor]]:
            """
            Add ONE real node w ∉ S (keep its label), connect to a feasible subset of anchors in S.
            Keep only if connected and NOT induced anywhere in G. Returns exact edge_index override.
            """
            H = G.subgraph(S).copy()
            res = _residuals_on_S(G, S)
            anchors_now = [v for v in S if res[v] > 0]
            if not anchors_now:
                return []

            pool = [w for w in G.nodes() if w not in S]
            if not pool:
                return []

            out: list[tuple[list[int], torch.Tensor]] = []
            tries = 0
            while len(out) < max_neg and tries < 50 * max_neg:
                tries += 1
                w = rng.choice(pool)
                lab_w = _label(w, G)
                deg_w = _target_deg(lab_w)

                d_new = min(max(1, deg_w), len(anchors_now))
                cand_anchors = [a for a in anchors_now if res[a] > 0]
                if len(cand_anchors) < d_new:
                    continue
                chosen = rng.sample(cand_anchors, d_new)

                H2 = H.copy()
                H2.add_node(w, label=lab_w)
                for a in chosen:
                    H2.add_edge(w, a)

                if nx.is_connected(H2) and not is_induced_subgraph_feature_aware(H2, G):
                    S_plus = S + [w]
                    ei = nx_to_edge_index_on_ordered_nodes(H2, S_plus)
                    out.append((S_plus, ei))
            return out

        # -------------------------- precompute NX views --------------------------
        nx_views = [pyg_to_nx(self.base[i]) for i in range(len(self.base))]

        # -------------------------- config (main vs tail budgets) --------------------------
        k_min = getattr(cfg, "k_min", 2)
        k_max = getattr(cfg, "k_max", None)  # None → use N
        k_cap = getattr(cfg, "k_cap", 10)

        pos_per_k_main = getattr(cfg, "pos_per_k_main", getattr(cfg, "pos_per_k", 2))
        neg_per_pos_main = getattr(cfg, "neg_per_pos_main", getattr(cfg, "neg_per_pos", 2))
        pos_per_k_tail = getattr(cfg, "pos_per_k_tail", 1)
        neg_per_pos_tail = getattr(cfg, "neg_per_pos_tail", 1)

        # enable/disable hard negatives
        use_wrong_edge_allowed = getattr(cfg, "neg_wrong_edge_allowed", True)
        use_missing_edge_verified = getattr(cfg, "neg_missing_edge_verified", True)
        use_rewire = getattr(cfg, "neg_rewire", True)
        use_cross_parent = getattr(cfg, "neg_cross_parent", True)
        use_anchor_aware = getattr(cfg, "neg_anchor_aware", True)

        # per-family caps (shared for main/tail unless *_tail provided)
        cap_wrong_edge_allowed = getattr(cfg, "cap_wrong_edge_allowed", 1)
        cap_missing_edge = getattr(cfg, "cap_missing_edge", 1)
        cap_rewire = getattr(cfg, "cap_rewire", 1)
        cap_cross_parent = getattr(cfg, "cap_cross_parent", 1)
        cap_anchor_aware = getattr(cfg, "cap_anchor_aware", 1)

        cap_wrong_edge_allowed_t = getattr(cfg, "cap_wrong_edge_allowed_tail", cap_wrong_edge_allowed)
        cap_missing_edge_t = getattr(cfg, "cap_missing_edge_tail", cap_missing_edge)
        cap_rewire_t = getattr(cfg, "cap_rewire_tail", cap_rewire)
        cap_cross_parent_t = getattr(cfg, "cap_cross_parent_tail", cap_cross_parent)
        cap_anchor_aware_t = getattr(cfg, "cap_anchor_aware_tail", cap_anchor_aware)

        tail_anchor_fraction = getattr(cfg, "tail_anchor_fraction", 0.5)

        # -------------------------- iterate parents --------------------------
        parent_iter = tqdm(
            range(len(self.base)),
            desc=f"{self.split}: parents",
            unit="mol",
            dynamic_ncols=True,
        )

        for parent_idx in parent_iter:
            full: Data = self.base[parent_idx]
            G = nx_views[parent_idx]
            parent_pairs = _allowed_label_pairs(G)

            N = G.number_of_nodes()
            if k_min > N:
                continue

            k_hi = N if (k_max is None) else min(k_max, N)

            k_iter = tqdm(
                range(k_min, k_hi + 1),
                desc=f"mol#{parent_idx} k",
                unit="k",
                leave=False,
                dynamic_ncols=True,
            )

            for k in k_iter:
                # -------- pick budgets for this k (main vs tail) --------
                is_tail = k > k_cap
                pos_budget = pos_per_k_tail if is_tail else pos_per_k_main
                neg_per_pos = neg_per_pos_tail if is_tail else neg_per_pos_main

                capA = cap_wrong_edge_allowed_t if is_tail else cap_wrong_edge_allowed
                capB = cap_missing_edge_t if is_tail else cap_missing_edge
                capC = cap_rewire_t if is_tail else cap_rewire
                capE = cap_anchor_aware_t if is_tail else cap_anchor_aware
                capD = cap_cross_parent_t if is_tail else cap_cross_parent

                # ---- collect positives for this (parent,k) ----
                pos_sets: list[list[int]] = []
                pos_seen: set[tuple[int, ...]] = set()

                anchors_order = low_degree_anchor_order(G)
                n_anchor = max(1, int(round(pos_budget * tail_anchor_fraction)))
                n_anchor = min(n_anchor, len(anchors_order))

                # anchored slice
                tries = 0
                i = 0
                max_tries = 30 * max(1, pos_budget)
                while len(pos_sets) < n_anchor and tries < max_tries:
                    tries += 1
                    anchor = anchors_order[i % max(1, n_anchor)]
                    i += 1
                    S = _distinct_k_connected_with_anchor(G, k, anchor=anchor)
                    if S is None:
                        continue
                    key = tuple(sorted(S))
                    if key in pos_seen:
                        continue
                    pos_sets.append(S)
                    pos_seen.add(key)

                # top up to pos_budget
                tries = 0
                max_tries = 30 * (pos_budget - len(pos_sets) + 1)
                j = 0
                while len(pos_sets) < pos_budget and tries < max_tries:
                    tries += 1
                    anchor = anchors_order[j % max(1, len(anchors_order))]
                    j += 1
                    S = _distinct_k_connected_with_anchor(G, k, anchor=anchor)
                    if S is None:
                        continue
                    key = tuple(sorted(S))
                    if key in pos_seen:
                        continue
                    pos_sets.append(S)
                    pos_seen.add(key)

                try:
                    k_iter.set_postfix_str(f"pos={len(pos_sets)}/{pos_budget}{' tail' if is_tail else ''}")
                except Exception:
                    pass

                # ---- emit positives + negatives ----
                for S in pos_sets:
                    # Positive
                    g1_pos = make_subgraph_data(full, S)
                    yield PairData(
                        x1=g1_pos.x,
                        edge_index1=g1_pos.edge_index,
                        x2=full.x,
                        edge_index2=full.edge_index,
                        y=torch.tensor([1]),
                        k=torch.tensor([k]),
                        neg_type=torch.tensor([PairType.POSITIVE]),  # positive
                        parent_idx=torch.tensor([parent_idx]),
                    )

                    # ----- Guaranteed negatives: wrong-add (globally forbidden label pair) -----
                    bad_pairs_forbidden = wrong_add_candidates_anywhere(G, S, parent_pairs)
                    if bad_pairs_forbidden and neg_per_pos > 0:
                        m = min(len(bad_pairs_forbidden), neg_per_pos)
                        for u, v in rng.sample(bad_pairs_forbidden, m):
                            H = G.subgraph(S).copy()
                            H.add_edge(u, v)
                            ei_neg = nx_to_edge_index_on_ordered_nodes(H, S)
                            g1_neg = make_subgraph_data(full, S, edge_index_override=ei_neg)
                            yield PairData(
                                x1=g1_neg.x,
                                edge_index1=g1_neg.edge_index,
                                x2=full.x,
                                edge_index2=full.edge_index,
                                y=torch.tensor([0]),
                                k=torch.tensor([k]),
                                neg_type=torch.tensor([PairType.FORBIDDEN_ADD]),  # forbidden-pair wrong-add
                                parent_idx=torch.tensor([parent_idx]),
                            )

                    # ----- Hard A: wrong-edge (allowed pair), verified -----
                    if use_wrong_edge_allowed and k >= 2 and capA > 0:
                        for Sx, ei in _gen_wrong_edge_allowed(G, S, max_neg=capA):
                            g1_neg = make_subgraph_data(full, Sx, edge_index_override=ei)
                            yield PairData(
                                x1=g1_neg.x,
                                edge_index1=g1_neg.edge_index,
                                x2=full.x,
                                edge_index2=full.edge_index,
                                y=torch.tensor([0]),
                                k=torch.tensor([k]),
                                neg_type=torch.tensor([PairType.WRONG_EDGE_ALLOWED]),  # allowed-pair wrong-edge
                                parent_idx=torch.tensor([parent_idx]),
                            )

                    # ----- Hard B: missing-edge (verified) -----
                    if use_missing_edge_verified and k >= 2 and capB > 0:
                        for Sx, ei in _gen_missing_edge_verified(G, S, max_neg=capB):
                            g1_neg = make_subgraph_data(full, Sx, edge_index_override=ei)
                            yield PairData(
                                x1=g1_neg.x,
                                edge_index1=g1_neg.edge_index,
                                x2=full.x,
                                edge_index2=full.edge_index,
                                y=torch.tensor([0]),
                                k=torch.tensor([k]),
                                neg_type=torch.tensor([PairType.MISSING_EDGE]),  # missing-edge
                                parent_idx=torch.tensor([parent_idx]),
                            )

                    # ----- Hard C: degree-preserving rewire (verified) -----
                    if use_rewire and k >= 4 and capC > 0:
                        for Sx, ei in _gen_rewire_degree_preserving(G, S, max_neg=capC):
                            g1_neg = make_subgraph_data(full, Sx, edge_index_override=ei)
                            yield PairData(
                                x1=g1_neg.x,
                                edge_index1=g1_neg.edge_index,
                                x2=full.x,
                                edge_index2=full.edge_index,
                                y=torch.tensor([0]),
                                k=torch.tensor([k]),
                                neg_type=torch.tensor([PairType.REWIRE]),  # rewire
                                parent_idx=torch.tensor([parent_idx]),
                            )

                    # ----- Hard E: anchor-aware (decoder-like), verified -----
                    if use_anchor_aware and k >= 2 and capE > 0:
                        for Sx, ei in _gen_anchor_aware(G, S, max_neg=capE):
                            g1_neg = make_subgraph_data(full, Sx, edge_index_override=ei)
                            yield PairData(
                                x1=g1_neg.x,
                                edge_index1=g1_neg.edge_index,
                                x2=full.x,
                                edge_index2=full.edge_index,
                                y=torch.tensor([0]),
                                k=torch.tensor([len(Sx)]),  # note: this is k+1
                                neg_type=torch.tensor([PairType.ANCHOR_AWARE]),  # anchor-aware
                                parent_idx=torch.tensor([parent_idx]),
                            )

                # ----- Extra: k=2 forbidden-pair negatives tied to decoder start -----
                if k == 2 and pos_sets:
                    k2_neg_budget = len(pos_sets) * neg_per_pos
                    bad_pairs2 = forbidden_pairs_k2(G, parent_pairs)
                    if bad_pairs2 and k2_neg_budget > 0:
                        take = min(len(bad_pairs2), k2_neg_budget)
                        for u, v in rng.sample(bad_pairs2, take):
                            S2 = [u, v]
                            H2 = nx.Graph()
                            H2.add_nodes_from(S2)
                            H2.add_edge(u, v)
                            ei2 = nx_to_edge_index_on_ordered_nodes(H2, S2)
                            g1_neg = make_subgraph_data(full, S2, edge_index_override=ei2)
                            yield PairData(
                                x1=g1_neg.x,
                                edge_index1=g1_neg.edge_index,
                                x2=full.x,
                                edge_index2=full.edge_index,
                                y=torch.tensor([0]),
                                k=torch.tensor([2]),
                                neg_type=torch.tensor([PairType.FORBIDDEN_ADD]),  # forbidden-pair wrong-add
                                parent_idx=torch.tensor([parent_idx]),
                            )

                # ----- Hard D: cross-parent negatives for each POSITIVE we just emitted -----
                if use_cross_parent and pos_sets and capD > 0:
                    for S in pos_sets:
                        G_pos = G.subgraph(S).copy()
                        other_idxs = _gen_cross_parent_indices(G_pos, nx_views, parent_idx, max_neg=capD)
                        if not other_idxs:
                            continue
                        g1_pos = make_subgraph_data(full, S)
                        for j in other_idxs:
                            other = self.base[j]
                            yield PairData(
                                x1=g1_pos.x,
                                edge_index1=g1_pos.edge_index,
                                x2=other.x,
                                edge_index2=other.edge_index,
                                y=torch.tensor([0]),
                                k=torch.tensor([k]),
                                neg_type=torch.tensor([PairType.CROSS_PARENT]),  # cross-parent
                                parent_idx=torch.tensor([j]),
                            )


if __name__ == "__main__":
    # Tiny base split (1 molecule) just for sanity
    base = ZincSmiles(split="test")[:10]

    pairs = ZincPairsV2(
        base_dataset=base,
        split="test",
        cfg=PairConfig(),  # use your current config
    )

    print(f"base graphs: {len(base)}")
    print(f"pair samples: {len(pairs)}")

    from torch_geometric.loader import DataLoader

    loader = DataLoader(pairs, batch_size=1, shuffle=False)

    n_pos = 0
    n_neg = 0

    for i, batch in enumerate(tqdm(loader, total=len(pairs), desc="VF2 sanity")):
        # batch_size=1 ⇒ this is effectively one PairData
        y = int(batch.y.item())
        neg_code = int(batch.neg_type.item())

        # Rebuild PyG graphs for VF2 (candidate = G1, parent = G2)
        g1 = Data(x=batch.x1.cpu(), edge_index=batch.edge_index1.cpu())
        g2 = Data(x=batch.x2.cpu(), edge_index=batch.edge_index2.cpu())

        # Label-aware induced subgraph check (VF2)
        is_sub = is_induced_subgraph_feature_aware(
            pyg_to_nx(g1),  # small / candidate
            pyg_to_nx(g2),  # big / parent
        )

        if y == 1:
            assert is_sub, f"[idx={i}] Positive mislabeled: not found as induced subgraph."
            n_pos += 1
        elif y == 0:
            assert not is_sub, f"[idx={i}] Negative mislabeled: embeds in parent (neg_type={neg_code})."
            n_neg += 1
        else:
            raise AssertionError(f"[idx={i}] Unexpected label y={y} (expected 0/1).")

    print(f"Sanity OK ✓  positives: {n_pos}, negatives: {n_neg}")
