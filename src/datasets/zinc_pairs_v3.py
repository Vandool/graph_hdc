import collections
import random
from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations_with_replacement
from pathlib import Path

import networkx as nx
import torch
from matplotlib import pyplot as plt
from torch_geometric.data import Data, Dataset
from torch_geometric.data import InMemoryDataset as _IMD
from tqdm.auto import tqdm

from src.encoding.decoder import is_induced_subgraph_by_features, residual_degree, wl_hash
from src.encoding.the_types import Feat
from src.utils.utils import GLOBAL_DATASET_PATH, DataTransformer
from src.utils.visualisations import draw_nx_with_atom_colorings

# Plotting
plt.show(block=True)


# ────────────────────────────── Plotting (Debugging) ─────────────────────────
def draw(
    g: nx.Graph, label: str, full_g: nx.Graph | None = None, highlight: list[tuple[int, int]] | None = None
) -> None:
    draw_nx_with_atom_colorings(
        H=g,
        dataset="ZincSmiles",
        label=label,
        overlay_full_graph=full_g,
        overlay_draw_nodes=True,
        overlay_labels=True,
        overlay_node_size=1400,
        overlay_alpha=0.40,
        # keep overlay behind subgraph labels
        # (only needed if you added the overlay-label z knob)
        # overlay_label_z=1.8,
        highlight_edges=highlight,
        highlight_color="#ef4444",
        highlight_width=5.0,
        figsize=(12, 9),
    )
    plt.tight_layout()
    plt.show(block=False)


# ────────────────────────────── utilities (graph I/O) ─────────────────────────
def make_subgraph_data(full: Data, S: Sequence[int]) -> Data:
    r"""
    Build a PyG ``Data`` that contains only nodes in ``S`` and edges between them.

    Parameters
    ----------
    full : Data
        Parent PyG graph (undirected represented as bidirectional directed edges).
    S : Sequence[int]
        Node indices (from ``full``) to keep. Order defines the new node order.

    Returns
    -------
    Data
        Subgraph with ``x`` sliced to ``S`` and ``edge_index`` filtered/remapped.

    Notes
    -----
    * Keeps *only* edges whose endpoints are both in ``S``.
    * If no such edges exist, ``edge_index`` is an empty tensor of shape ``[2, 0]``.
    """
    S = list(map(int, S))
    idx_map: dict[int, int] = {orig: i for i, orig in enumerate(S)}
    x_sub = full.x[S] if full.x is not None else None

    if full.edge_index is None or full.edge_index.numel() == 0:
        ei_sub = torch.empty((2, 0), dtype=torch.long)
        return Data(x=x_sub, edge_index=ei_sub)

    src = full.edge_index[0].tolist()
    dst = full.edge_index[1].tolist()
    S_set = set(S)

    e_src, e_dst = [], []
    for u, v in zip(src, dst, strict=False):
        if u in S_set and v in S_set:
            e_src.append(idx_map[u])
            e_dst.append(idx_map[v])

    if not e_src:
        ei_sub = torch.empty((2, 0), dtype=torch.long)
    else:
        ei_sub = torch.tensor([e_src, e_dst], dtype=torch.long)

    return Data(x=x_sub, edge_index=ei_sub)


def _feat_tuple(G: nx.Graph, n: int) -> tuple[int, int, int, int]:
    r"""
    Return the canonical 4-tuple feature for node ``n``.

    Assumes each node carries ``feat: Feat``.
    """
    return G.nodes[n]["feat"].to_tuple()


def _norm_pair(a: tuple, b: tuple) -> tuple[tuple, tuple]:
    r"""
    Return a canonical unordered pair of feature tuples: ``(min(a,b), max(a,b))``.
    """
    return (a, b) if a <= b else (b, a)


def distinct_edge_feature_pairs(G: nx.Graph) -> dict[tuple[tuple, tuple], tuple[int, int]]:
    r"""
    Map each *distinct* unordered **feature-pair** present as an edge in ``G``
    to one representative node pair ``(u, v)``.

    Distinctness is by node **features**, not node IDs.
    """
    rep: dict[tuple[tuple, tuple], tuple[int, int]] = {}
    for u, v in G.edges():
        fu, fv = _feat_tuple(G, u), _feat_tuple(G, v)
        key = _norm_pair(fu, fv)
        if key not in rep:
            rep[key] = (u, v)
    return rep


def distinct_nonedge_feature_pairs(G: nx.Graph) -> dict[tuple[tuple, tuple], tuple[int, int]]:
    r"""
    Map each unordered **feature-pair** that occurs in the node set but **never**
    as an edge in ``G`` to one representative **non-edge** node pair.

    Returns
    -------
    dict
        Keys are unordered feature-pair tuples; values are node pairs ``(u, v)``
        with those features such that ``not G.has_edge(u, v)``.
    """
    # Nodes by feature
    nodes_by_feat: dict[tuple, list[int]] = defaultdict(list)
    for n in G.nodes:
        nodes_by_feat[_feat_tuple(G, n)].append(n)

    feat_list = sorted(nodes_by_feat.keys())
    edge_pairs = set(distinct_edge_feature_pairs(G).keys())

    out: dict[tuple[tuple, tuple], tuple[int, int]] = {}

    for fa, fb in combinations_with_replacement(feat_list, 2):
        key = _norm_pair(fa, fb)
        if key in edge_pairs:
            continue

        A, B = nodes_by_feat[fa], nodes_by_feat[fb]
        if fa == fb:
            if len(A) < 2:
                continue
            found = None
            for i in range(len(A)):
                for j in range(i + 1, len(A)):
                    u, v = A[i], A[j]
                    if not G.has_edge(u, v):
                        found = (u, v)
                        break
                if found:
                    break
        else:
            found = None
            for u in A:
                for v in B:
                    if not G.has_edge(u, v):
                        found = (u, v)
                        break
                if found:
                    break

        if found is not None:
            out[key] = found

    return out


# ────────────────────────────── positive samplers ─────────────────────────────
def _sample_connected_set_bfs(G: nx.Graph, k: int, *, tries: int = 64, rng) -> list[int] | None:
    r"""
    Sample a connected node set ``S`` of size ``k`` using random-BFS roots.

    Returns
    -------
    list[int] | None
        Node ids if successful; otherwise ``None`` after ``tries`` attempts.
    """
    nodes = tuple(G.nodes)
    if not nodes:
        return None
    for _ in range(tries):
        s = rng.choice(nodes)
        order = list(nx.bfs_tree(G, s).nodes())
        if len(order) >= k:
            return order[:k]
    return None


def _pick_positive_sets(G: nx.Graph, *, k: int, budget: int, min_edges: int = 2, rng) -> list[list[int]]:
    r"""
    Pick up to ``budget`` **connected** induced subgraphs of size ``k``,
    deduplicated by WL hash (feat-aware). Enforces ``#E >= min_edges``.
    """
    out: list[list[int]] = []
    seen: set[str] = set()
    # generous attempt cap to avoid rare degenerate cases
    max_attempts = max(64, 20 * budget)
    attempts = 0
    while len(out) < budget and attempts < max_attempts:
        attempts += 1
        S = _sample_connected_set_bfs(G, k, rng=rng)
        if not S:
            continue
        H = G.subgraph(S).copy()
        if H.number_of_edges() < min_edges:
            continue
        key = wl_hash(H)  # uses node 'feat' labels
        if key in seen:
            continue
        seen.add(key)
        out.append(S)
    return out


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


class ZincPairConfig:
    # reproducibility
    seed = 42

    # k schedule
    k_max = None
    k_cap = 12

    # budgets (k > 2)
    pos_per_k_main = 4
    neg_per_pos_main = 5  # ↑ was 2
    pos_per_k_tail = 2 # Not relvant for QM9
    neg_per_pos_tail = 3 # Not relevant for QM9

    # enable families
    neg_forbidden_add = True
    neg_wrong_edge_allowed = True
    neg_missing_edge_verified = True
    neg_rewire = True
    neg_anchor_aware = True
    neg_cross_parent = True

    # per-family caps (per positive)
    cap_forbidden_add = 1  # new
    cap_wrong_edge_allowed = 2  # ↑ was 1
    cap_missing_edge = 1
    cap_rewire = 1
    cap_anchor_aware = 2  # ↑ was 1
    cap_cross_parent = 1  # keep 1 but gate by rate below

    # tail caps (defaults to main if not set)
    # Not relevant for QM9
    cap_forbidden_add_tail = cap_forbidden_add
    cap_wrong_edge_allowed_tail = cap_wrong_edge_allowed
    cap_missing_edge_tail = cap_missing_edge
    cap_rewire_tail = cap_rewire
    cap_anchor_aware_tail = cap_anchor_aware
    cap_cross_parent_tail = cap_cross_parent

    # cross-parent throttling
    cross_parent_rate = 0.06  # ← only ~7% of positives get a cross-parent neg


from enum import IntEnum


class PairType(IntEnum):
    r"""
    Enumeration of pair sample types.

    Members
    -------
    POSITIVE_EDGE : -1
        All the distinct edges in the parent graph. (Graph with two nodes and one edge) k == 2
    POSITIVE : 0
        Connected induced subgraph (label-frozen) — y=1. k > 2
    FORBIDDEN_ADD : 1
        Add a non-edge (u,v) in S whose **label pair never occurs** as an edge anywhere in the parent.
        The new added node is selected from the leftovers
    WRONG_EDGE_ALLOWED : 2
        Add a non-edge (u,v) in S whose label pair **does occur somewhere** in the parent (contextually wrong).
        The new added node is selected from the leftovers.
    CROSS_PARENT : 3
        Use a positive G[S] from parent *i* against a different parent *j* that **does not** contain it.
    MISSING_EDGE : 4
        Repeatedly Remove an existing edge from G[S]; keep connected; verified **not** induced.
        On each repeat add the negative sample to the negative pool. Until no more edges can be removed.
    REWIRE : 5
        Degree-preserving rewire of two disjoint edges inside S; keep connected; verified **not** induced.
    ANCHOR_AWARE : 6
        Add one real node w∉S and connect to possible anchors in S; verified **not** induced.
    NEGATIVE_EDGE : 7
        Edges that does not appear in the parent graph, where nodes appear. (Graph with two nodes and no edge), k == 2

    """

    POSITIVE_EDGE = -1
    POSITIVE = 0
    FORBIDDEN_ADD = 1
    WRONG_EDGE_ALLOWED = 2
    CROSS_PARENT = 3
    MISSING_EDGE = 4
    REWIRE = 5
    ANCHOR_AWARE = 6
    NEGATIVE_EDGE = 7


# Optional: convenient reverse lookup for logging/summaries.
NEG_TYPE_NAME = {t.value: t.name for t in PairType}


class ZincPairsV3(Dataset):
    def __init__(
        self,
        base_dataset,
        split="train",
        root=GLOBAL_DATASET_PATH / "ZincPairsV3",
        cfg=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        shard_size=25_000,
        cache_shards=2,
        *,
        force_reprocess=False,
        dev: bool = False,
        debug: bool = False,
    ):
        # --- set everything process() will need BEFORE super().__init__ ---
        self.base = base_dataset
        self.split = split
        self.cfg = cfg or ZincPairConfig()
        self.shard_size = shard_size
        self.cache_shards = cache_shards
        self._rng = random.Random(self.cfg.seed)
        self.debug = debug

        if dev:
            root = GLOBAL_DATASET_PATH / "ZincPairsV3DEV"
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

        for pair in self._generate_pairs_stream(debug=self.debug):
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

    def _generate_pairs_stream(self, debug: bool = False):
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
        1) Positives
            Type -1) Positive edge: all distinct edges in the parent graph. (Graph with two nodes and one edge)
            Type 0) The rest: connected induced subgraphs G[S] with **frozen** node labels.
                    (Graph with nodes > 2 and edge > 1)
        2) Negatives:
           • Guaranteed-by-construction:
             Type 1) Wrong-add with a **globally forbidden** label pair inside S.
                Add an edge (u,v) inside S where the label pair never occurs as an edge anywhere in the parent.
                Residual-respecting.
           • VF2-verified hard negatives (always correct):
             Type 2) Wrong-edge (allowed pair): add a non-edge (u,v) in G[S] where the **label pair is allowed**
                somewhere in G.
             Type 3) Cross-parent: use positive G[S] against a *different* parent that does not contain it.
                Ensures the oracle actually uses full_g_h and doesn’t devolve into a graph-only classifier. It’s a
                helpful regularizer for generalization (same motif may or may not be present depending on target).
                Should be kept capped in training
             Type 4) Missing-edge: remove an existing edge from G[S].
                any candidate missing a required edge between already-selected nodes must be rejected by an
                induced-subgraph oracle. This will enforce the oracle to reject subgraphs, that are missing an edge to
                become a ring structure.
             Type 5) Degree-preserving rewire: swap endpoints of two disjoint edges within S.
                This might help the decoder, to correct the previously wrnogly made decisions!
                The decoder can grow into a topology with the same degree sequence yet wrong adjacency due to a wrong
                early attachment. This family creates strong “looks plausible by degrees, but wrong by structure”
                examples. Good as hard negatives. Should be kept capped in training
             Type 6) Anchor-aware (decoder-like): add **one real node** w∉S (its label from G), connect to feasible
                anchors in S. This is the closest simulation of what the decoder actually does: add one real node,
                attach to anchors by residuals, and see if the (k+1) candidate remains induced in the target. This
                directly teaches the boundary between “good next step” and “bad next step.” Core.
             Type 7) Illegal-edge: edges that does not appear in the parent graph, where nodes appear.

        All VF2 checks use `is_induced_subgraph_feature_aware`, ensuring labels are correct even without VF2 at inference.
        """
        rng, cfg = self._rng, self.cfg

        # -------------------------- precompute NX views --------------------------
        nx_views = [DataTransformer.pyg_to_nx(self.base[i]) for i in range(len(self.base))]

        # -------------------------- config (main vs tail budgets) --------------------------
        k_min = getattr(cfg, "k_min", 2)
        k_max = getattr(cfg, "k_max", None)  # None → use N
        k_cap = getattr(cfg, "k_cap", 10)

        pos_per_k_main = getattr(cfg, "pos_per_k_main", getattr(cfg, "pos_per_k", 2))
        neg_per_pos_main = getattr(cfg, "neg_per_pos_main", getattr(cfg, "neg_per_pos", 2))
        pos_per_k_tail = getattr(cfg, "pos_per_k_tail", 1)
        neg_per_pos_tail = getattr(cfg, "neg_per_pos_tail", 1)

        cross_parent_rate = getattr(cfg, "cross_parent_rate", 0.2)
        # enable/disable hard negatives
        use_wrong_edge_allowed = getattr(cfg, "neg_wrong_edge_allowed", True)
        use_missing_edge_verified = getattr(cfg, "neg_missing_edge_verified", True)
        use_rewire = getattr(cfg, "neg_rewire", True)
        use_cross_parent = getattr(cfg, "neg_cross_parent", True)
        use_anchor_aware = getattr(cfg, "neg_anchor_aware", True)
        use_forbidden_add = getattr(cfg, "neg_forbidden_add", True)
        cap_forbidden_add = getattr(cfg, "cap_forbidden_add", 1)
        cap_forbidden_add_t = getattr(cfg, "cap_forbidden_add_tail", cap_forbidden_add)

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
            # Distinct feature-pairs that occur as edges in parent G
            edge_feat_keys = set(distinct_edge_feature_pairs(G).keys())
            if debug:
                draw_nx_with_atom_colorings(H=G, label=f"ORIGINAL Parent {parent_idx}")

            N = G.number_of_nodes()

            k_hi = N if (k_max is None) else min(k_max, N)

            k_iter = tqdm(
                range(k_min, k_hi + 1),
                desc=f"mol#{parent_idx} k",
                unit="k",
                leave=False,
                dynamic_ncols=True,
            )

            for k in k_iter:
                seen_set: set[tuple[tuple, tuple]] = set()

                if k == 2:
                    # ---- positives: distinct edges by feature-pair ----
                    pos_map = distinct_edge_feature_pairs(G)  # { (featA,featB) : (u,v) }
                    pos_keys_sorted = sorted(pos_map.keys())  # deterministic order
                    num_pos = 0

                    for key in pos_keys_sorted:
                        if key in seen_set:
                            continue
                        u, v = pos_map[key]
                        g1_pos = make_subgraph_data(full, [u, v])

                        if self.debug:
                            draw(DataTransformer.pyg_to_nx(g1_pos), label="POSITIVE_EDGE", full_g=G, highlight=[(0, 1)])

                        yield PairData(
                            x1=g1_pos.x,
                            edge_index1=g1_pos.edge_index,
                            x2=full.x,
                            edge_index2=full.edge_index,
                            y=torch.tensor([1], dtype=torch.long),
                            k=torch.tensor([2], dtype=torch.long),
                            neg_type=torch.tensor([PairType.POSITIVE_EDGE], dtype=torch.long),
                            parent_idx=torch.tensor([parent_idx], dtype=torch.long),
                        )
                        seen_set.add(key)
                        num_pos += 1

                    # ---- negatives: distinct non-edges by feature-pair (match count) ----
                    neg_map = distinct_nonedge_feature_pairs(G)  # { (featA,featB) : (u,v) }
                    neg_keys_sorted = sorted(k for k in neg_map.keys() if k not in seen_set)

                    num_neg_needed = num_pos
                    neg_emitted = 0
                    for key in neg_keys_sorted:
                        if neg_emitted >= num_neg_needed:
                            break
                        u, v = neg_map[key]
                        # Sanity: must be non-edge in parent
                        if G.has_edge(u, v):
                            continue
                        # Build a 2-node candidate with the SAME node features but FORCE an edge between them.
                        H_neg = nx.Graph()
                        fu = G.nodes[u]["feat"]
                        fv = G.nodes[v]["feat"]
                        H_neg.add_node(0, feat=fu, target_degree=fu.target_degree)
                        H_neg.add_node(1, feat=fv, target_degree=fv.target_degree)
                        H_neg.add_edge(0, 1)  # illegal: this feature-pair never appears as an edge in the parent

                        # Safety: should NOT embed if pair truly forbidden
                        if is_induced_subgraph_by_features(H_neg, G, require_connected=True):
                            continue

                        g1_neg = DataTransformer.nx_to_pyg(H_neg)

                        if self.debug:
                            draw_nx_with_atom_colorings(
                                DataTransformer.pyg_to_nx(g1_neg), label="NEGATIVE_EDGE", overlay_full_graph=G
                            )

                        yield PairData(
                            x1=g1_neg.x,
                            edge_index1=g1_neg.edge_index,  # expected shape [2, 0]
                            x2=full.x,
                            edge_index2=full.edge_index,
                            y=torch.tensor([0], dtype=torch.long),
                            k=torch.tensor([2], dtype=torch.long),
                            neg_type=torch.tensor([PairType.NEGATIVE_EDGE], dtype=torch.long),
                            parent_idx=torch.tensor([parent_idx], dtype=torch.long),
                        )
                        seen_set.add(key)
                        neg_emitted += 1

                    # done with k==2 for this parent
                    continue

                # -------------------- k > 2 positives -------------------------
                # budgets per k
                if k <= k_cap:
                    pos_budget = pos_per_k_main
                    neg_per_pos = neg_per_pos_main
                    capA, capB, capC, capD, capE, capF = (
                        cap_wrong_edge_allowed,
                        cap_missing_edge,
                        cap_rewire,
                        cap_cross_parent,
                        cap_anchor_aware,
                        cap_forbidden_add,
                    )
                else:
                    pos_budget = pos_per_k_tail
                    neg_per_pos = neg_per_pos_tail
                    capA, capB, capC, capD, capE, capF = (
                        cap_wrong_edge_allowed_t,
                        cap_missing_edge_t,
                        cap_rewire_t,
                        cap_cross_parent_t,
                        cap_anchor_aware_t,
                        cap_forbidden_add_t,
                    )

                # choose connected induced subgraphs; require ≥2 edges for k>2
                pos_sets: list[list[int]] = _pick_positive_sets(G, k=k, budget=pos_budget, min_edges=2, rng=rng)

                # emit positives
                for S in pos_sets:
                    H_pos = G.subgraph(S).copy()
                    g1_pos = DataTransformer.nx_to_pyg(H_pos)

                    if self.debug:
                        draw_nx_with_atom_colorings(H=H_pos, label=f"POSITIVE k={k}", overlay_full_graph=G)

                    yield PairData(
                        x1=g1_pos.x,
                        edge_index1=g1_pos.edge_index,
                        x2=full.x,
                        edge_index2=full.edge_index,
                        y=torch.tensor([1], dtype=torch.long),
                        k=torch.tensor([k], dtype=torch.long),
                        neg_type=torch.tensor([PairType.POSITIVE], dtype=torch.long),
                        parent_idx=torch.tensor([parent_idx], dtype=torch.long),
                    )

                    # ---- Negatives per positive (A..E), WL-dedupe within this S ----
                    neg_seen: set[str] = set()
                    neg_emitted = 0

                    # ---------- Type-1 FORBIDDEN_ADD: add a non-edge in S whose feature-pair NEVER occurs as an edge anywhere in parent ----------
                    if use_forbidden_add and k >= 2 and capF > 0 and neg_emitted < neg_per_pos:
                        # Non-edges within S
                        non_edges = [
                            (u, v) for u in H_pos.nodes for v in H_pos.nodes if u < v and not H_pos.has_edge(u, v)
                        ]
                        # Keep only those whose feature-pair is globally forbidden in this parent (not present as ANY edge)
                        cand = []
                        for u, v in non_edges:
                            fu = G.nodes[u]["feat"].to_tuple()
                            fv = G.nodes[v]["feat"].to_tuple()
                            key = (fu, fv) if fu <= fv else (fv, fu)
                            if key not in edge_feat_keys:
                                cand.append((u, v))
                        rng.shuffle(cand)

                        taken = 0
                        for u, v in cand:
                            if taken >= min(capF, neg_per_pos - neg_emitted):
                                break
                            H_neg = H_pos.copy()
                            # Decoder respects the residuals
                            if (
                                residual_degree(H_neg, u) <= 0 or residual_degree(H_neg, v) <= 0
                            ) and rng.random() < 0.5:
                                continue
                            H_neg.add_edge(u, v)  # guaranteed to break inducedness against G[S]
                            # Safety: skip if somehow still induced (shouldn't happen)
                            if is_induced_subgraph_by_features(H_neg, G, require_connected=True):
                                continue
                            hkey = wl_hash(H_neg)
                            if hkey in neg_seen:
                                continue
                            neg_seen.add(hkey)

                            g1_neg = DataTransformer.nx_to_pyg(H_neg)
                            if self.debug:
                                draw_nx_with_atom_colorings(
                                    H=H_neg,
                                    label=f"NEG FORBIDDEN_ADD k={k}",
                                    overlay_full_graph=G,
                                    highlight_edges=[(u, v)],
                                )

                            yield PairData(
                                x1=g1_neg.x,
                                edge_index1=g1_neg.edge_index,
                                x2=full.x,
                                edge_index2=full.edge_index,
                                y=torch.tensor([0], dtype=torch.long),
                                k=torch.tensor([k], dtype=torch.long),
                                neg_type=torch.tensor([PairType.FORBIDDEN_ADD], dtype=torch.long),
                                parent_idx=torch.tensor([parent_idx], dtype=torch.long),
                            )
                            taken += 1
                            neg_emitted += 1

                    # ---------- A) WRONG_EDGE_ALLOWED (add non-edge in S where feat-pair is allowed in parent) ----------
                    if use_wrong_edge_allowed and k >= 2 and capA > 0 and neg_emitted < neg_per_pos:
                        non_edges = [
                            (u, v) for u in H_pos.nodes for v in H_pos.nodes if u < v and not H_pos.has_edge(u, v)
                        ]
                        # prioritize feature-pairs that are allowed somewhere in parent
                        cand = []
                        for u, v in non_edges:
                            fu = G.nodes[u]["feat"].to_tuple()
                            fv = G.nodes[v]["feat"].to_tuple()
                            key = (fu, fv) if fu <= fv else (fv, fu)
                            if key in edge_feat_keys:
                                cand.append((u, v))
                        rng.shuffle(cand)

                        taken = 0
                        for u, v in cand:
                            if taken >= min(capA, neg_per_pos - neg_emitted):
                                break
                            H_neg = H_pos.copy()
                            # Decoder respects the residuals
                            if (
                                residual_degree(H_neg, u) <= 0 or residual_degree(H_neg, v) <= 0
                            ) and rng.random() < 0.5:
                                continue
                            H_neg.add_edge(u, v)
                            # Adding an edge keeps connectivity; verify non-induced:
                            if is_induced_subgraph_by_features(H_neg, G, require_connected=True):
                                continue
                            hkey = wl_hash(H_neg)
                            if hkey in neg_seen:
                                continue
                            neg_seen.add(hkey)

                            g1_neg = DataTransformer.nx_to_pyg(H_neg)
                            if self.debug:
                                draw_nx_with_atom_colorings(
                                    H=H_neg,
                                    label=f"NEG A: wrong-edge-allowed k={k}",
                                    overlay_full_graph=G,
                                    highlight_edges=[(u, v)],
                                )

                            yield PairData(
                                x1=g1_neg.x,
                                edge_index1=g1_neg.edge_index,
                                x2=full.x,
                                edge_index2=full.edge_index,
                                y=torch.tensor([0], dtype=torch.long),
                                k=torch.tensor([k], dtype=torch.long),
                                neg_type=torch.tensor([PairType.WRONG_EDGE_ALLOWED], dtype=torch.long),
                                parent_idx=torch.tensor([parent_idx], dtype=torch.long),
                            )
                            taken += 1
                            neg_emitted += 1

                    # ---------- B) MISSING_EDGE (remove existing edge; keep connected) ----------
                    if use_missing_edge_verified and k >= 2 and capB > 0 and neg_emitted < neg_per_pos:
                        edges = list(H_pos.edges())
                        rng.shuffle(edges)
                        taken = 0
                        for u, v in edges:
                            if taken >= min(capB, neg_per_pos - neg_emitted):
                                break
                            H_neg = H_pos.copy()
                            H_neg.remove_edge(u, v)
                            if not nx.is_connected(H_neg):
                                continue
                            # Removing a real edge should break inducedness:
                            if is_induced_subgraph_by_features(H_neg, G, require_connected=True):
                                continue
                            hkey = wl_hash(H_neg)
                            if hkey in neg_seen:
                                continue
                            neg_seen.add(hkey)

                            g1_neg = DataTransformer.nx_to_pyg(H_neg)
                            if self.debug:
                                draw_nx_with_atom_colorings(
                                    H=H_neg, label=f"NEG B: missing-edge k={k}", overlay_full_graph=G
                                )

                            yield PairData(
                                x1=g1_neg.x,
                                edge_index1=g1_neg.edge_index,
                                x2=full.x,
                                edge_index2=full.edge_index,
                                y=torch.tensor([0], dtype=torch.long),
                                k=torch.tensor([k], dtype=torch.long),
                                neg_type=torch.tensor([PairType.MISSING_EDGE], dtype=torch.long),
                                parent_idx=torch.tensor([parent_idx], dtype=torch.long),
                            )
                            taken += 1
                            neg_emitted += 1

                    # ---------- C) REWIRE: degree-preserving swap of two disjoint edges inside S ----------
                    if use_rewire and k >= 4 and capC > 0 and neg_emitted < neg_per_pos:
                        edges = list(H_pos.edges())
                        if len(edges) >= 2:
                            rng.shuffle(edges)
                            # Try pairs of disjoint edges
                            taken = 0
                            for i in range(len(edges)):
                                if taken >= min(capC, neg_per_pos - neg_emitted):
                                    break
                                a, b = edges[i]
                                for j in range(i + 1, len(edges)):
                                    c, d = edges[j]
                                    # Four distinct endpoints?
                                    if len({a, b, c, d}) < 4:
                                        continue

                                    # Two possible rewires: (a,c)+(b,d) or (a,d)+(b,c)
                                    candidates = [((a, c), (b, d)), ((a, d), (b, c))]
                                    rng.shuffle(candidates)

                                    success = False
                                    for (e1u, e1v), (e2u, e2v) in candidates:
                                        if e1u == e1v or e2u == e2v:
                                            continue
                                        # avoid adding existing edges
                                        if H_pos.has_edge(e1u, e1v) or H_pos.has_edge(e2u, e2v):
                                            continue
                                        H_neg = H_pos.copy()
                                        H_neg.remove_edge(a, b)
                                        H_neg.remove_edge(c, d)
                                        H_neg.add_edge(e1u, e1v)
                                        H_neg.add_edge(e2u, e2v)
                                        if not nx.is_connected(H_neg):
                                            continue
                                        # Rewired structure must not be an induced subgraph
                                        if is_induced_subgraph_by_features(H_neg, G, require_connected=True):
                                            continue
                                        hkey = wl_hash(H_neg)
                                        if hkey in neg_seen:
                                            continue
                                        neg_seen.add(hkey)

                                        g1_neg = DataTransformer.nx_to_pyg(H_neg)
                                        if self.debug:
                                            draw_nx_with_atom_colorings(
                                                H=H_neg,
                                                label=f"NEG C: rewire k={k}",
                                                overlay_full_graph=G,
                                                highlight_edges=[(e1u, e1v), (e2u, e2v)],
                                            )

                                        yield PairData(
                                            x1=g1_neg.x,
                                            edge_index1=g1_neg.edge_index,
                                            x2=full.x,
                                            edge_index2=full.edge_index,
                                            y=torch.tensor([0], dtype=torch.long),
                                            k=torch.tensor([k], dtype=torch.long),
                                            neg_type=torch.tensor([PairType.REWIRE], dtype=torch.long),
                                            parent_idx=torch.tensor([parent_idx], dtype=torch.long),
                                        )
                                        taken += 1
                                        neg_emitted += 1
                                        success = True
                                        break  # move to next edge-pair
                                    if success and taken >= min(capC, neg_per_pos - neg_emitted):
                                        break

                    # ---------- E) ANCHOR_AWARE: add one real node w∉S and connect to feasible anchors in S ----------
                    if use_anchor_aware and capE > 0 and neg_emitted < neg_per_pos:
                        outside = [n for n in G.nodes if n not in H_pos.nodes]
                        if outside:
                            rng.shuffle(outside)

                            # current anchors in S (residual > 0)
                            S_anchors = [n for n in H_pos.nodes if residual_degree(H_pos, n) > 0]
                            if S_anchors:
                                taken = 0
                                for w in outside:
                                    if taken >= min(capE, neg_per_pos - neg_emitted):
                                        break
                                    fw: Feat = G.nodes[w]["feat"]
                                    w_target = fw.target_degree

                                    # Feasible anchors limited by their residual and w's degree
                                    feas = [a for a in S_anchors if residual_degree(H_pos, a) > 0]
                                    if not feas:
                                        break
                                    rng.shuffle(feas)
                                    max_attach = min(w_target, len(feas))
                                    if max_attach <= 0:
                                        continue

                                    if k <= k_cap:
                                        attach_cnt = rng.randint(1, max_attach)
                                    else:
                                        # tail uses a fixed fraction (>=1)
                                        attach_cnt = max(1, int(max_attach * tail_anchor_fraction))

                                    use_anchors = feas[:attach_cnt]

                                    H_neg = H_pos.copy()
                                    # add real node w with its attributes
                                    H_neg.add_node(w, feat=fw, target_degree=fw.target_degree)
                                    for a in use_anchors:
                                        H_neg.add_edge(w, a)

                                    # Must be connected (it is if attach_cnt>=1) and NOT induced
                                    if is_induced_subgraph_by_features(H_neg, G, require_connected=True):
                                        continue
                                    hkey = wl_hash(H_neg)
                                    if hkey in neg_seen:
                                        continue
                                    neg_seen.add(hkey)

                                    g1_neg = DataTransformer.nx_to_pyg(H_neg)
                                    if self.debug:
                                        draw_nx_with_atom_colorings(
                                            H=H_neg,
                                            label=f"NEG E: anchor-aware k={k}, w={w}",
                                            overlay_full_graph=G,
                                            highlight_edges=[(w, a) for a in use_anchors],
                                        )

                                    yield PairData(
                                        x1=g1_neg.x,
                                        edge_index1=g1_neg.edge_index,
                                        x2=full.x,
                                        edge_index2=full.edge_index,
                                        y=torch.tensor([0], dtype=torch.long),
                                        k=torch.tensor([k], dtype=torch.long),
                                        neg_type=torch.tensor([PairType.ANCHOR_AWARE], dtype=torch.long),
                                        parent_idx=torch.tensor([parent_idx], dtype=torch.long),
                                    )
                                    taken += 1
                                    neg_emitted += 1

                    # ----- D) CROSS_PARENT: for each POSITIVE, pick other parents that do NOT contain it -----
                    if use_cross_parent and capD > 0 and pos_sets:
                        other_parents = [j for j in range(len(nx_views)) if j != parent_idx]
                        rng.shuffle(other_parents)

                        for S in pos_sets:
                            # throttle: only some positives produce a cross-parent negative
                            if rng.random() > cross_parent_rate:
                                continue

                            H_pos = G.subgraph(S).copy()
                            g1_pos = DataTransformer.nx_to_pyg(H_pos)
                            taken = 0
                            for j in other_parents:
                                if taken >= capD:
                                    break
                                Gj = nx_views[j]
                                if is_induced_subgraph_by_features(H_pos, Gj, require_connected=True):
                                    continue
                                full_j: Data = self.base[j]

                                if self.debug:
                                    draw_nx_with_atom_colorings(
                                        H=H_pos, label=f"NEG D: cross-parent vs parent#{j}", overlay_full_graph=Gj
                                    )

                                yield PairData(
                                    x1=g1_pos.x,
                                    edge_index1=g1_pos.edge_index,
                                    x2=full_j.x,
                                    edge_index2=full_j.edge_index,
                                    y=torch.tensor([0], dtype=torch.long),
                                    k=torch.tensor([k], dtype=torch.long),
                                    neg_type=torch.tensor([PairType.CROSS_PARENT], dtype=torch.long),
                                    parent_idx=torch.tensor([j], dtype=torch.long),
                                )
                                taken += 1
