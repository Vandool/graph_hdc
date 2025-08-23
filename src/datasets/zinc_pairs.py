"""
We want a pairwise classifier f(G_1, G_2) to {0,1} that answers “is G_1 an induced subgraph of G_2” under
feature-aware matching, with node features frozen to those of the final graph.

•	Positives (P): connected, induced subgraphs G[S] of the full molecule G for sizes k=2,dots,K. Node features are
    taken from G (as you already do), edges are exactly those among S present in G.
•	Negatives (N): same node feature convention, but edge pattern (or node set) must differ so that G_1 is not an
    induced subgraph of G.

Positive Sampling:


"""
import random
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, Optional
from typing import Sequence

import networkx as nx
import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset
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
    for u, v in zip(ei[0], ei[1]):
        if u < v:
            G.add_edge(int(u), int(v))
    return G


def induced_edge_index_from_node_set(edge_index: Tensor, nodes: Sequence[int]) -> Tensor:
    """Undirected both-direction edge_index for induced subgraph on `nodes`."""
    nodes = list(nodes)
    pos = {n: i for i, n in enumerate(nodes)}
    src, dst = edge_index.tolist()
    undirected = set()
    for u, v in zip(src, dst):
        if u in pos and v in pos and u < v:
            undirected.add((pos[u], pos[v]))
    if not undirected:
        return torch.empty((2, 0), dtype=torch.long)
    s, d = [], []
    for u, v in undirected:
        s += [u, v]
        d += [v, u]
    return torch.tensor([s, d], dtype=torch.long)


def make_subgraph_data(full: Data, node_indices: Sequence[int], *,
                       edge_index_override: Optional[Tensor] = None) -> Data:
    """Create subgraph Data with features copied from the parent (no recompute)."""
    node_indices = list(node_indices)
    x_sub = full.x[node_indices].clone()
    ei_sub = edge_index_override if edge_index_override is not None \
        else induced_edge_index_from_node_set(full.edge_index, node_indices)
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


def pick_wrong_add_at_anchor(
        G_parent: nx.Graph,
        S: Sequence[int],
        anchor: int,
        *,
        rng: random.Random,
        require_residual: bool = True,
        parent_pairs: Optional[set[tuple[tuple[int, ...], tuple[int, ...]]]] = None,
) -> Optional[nx.Graph]:
    if anchor not in S:
        return None
    H = G_parent.subgraph(S).copy()
    S = list(S)
    res = residual_capacity_vector(G_parent, S) if require_residual else {v: 1 for v in S}
    parent_pairs = parent_pairs or parent_label_edge_set(G_parent)
    la = node_label(G_parent, anchor)

    cands: list[int] = []
    for v in S:
        if v == anchor or H.has_edge(anchor, v):
            continue
        if res.get(anchor, 0) <= 0 or res.get(v, 0) <= 0:
            continue
        lv = node_label(G_parent, v)
        if unordered_label_pair(la, lv) not in parent_pairs:
            cands.append(v)

    if not cands:
        return None
    v = rng.choice(cands)
    H.add_edge(anchor, v)
    return H


def permute_features_with_forbidden_edge(
        full: Data,
        S: Sequence[int],
        G_parent: nx.Graph,
        *,
        rng: random.Random,
        max_tries: int = 64,
        parent_pairs: Optional[set[tuple[tuple[int, ...], tuple[int, ...]]]] = None,
) -> Optional[Data]:
    S = list(S)
    ei_sub = induced_edge_index_from_node_set(full.edge_index, S)
    if ei_sub.size(1) == 0:
        return None  # no edges -> cannot create a forbidden label-pair by permutation
    parent_pairs = parent_pairs or parent_label_edge_set(G_parent)

    tries = 0
    while tries < max_tries:
        tries += 1
        perm = S[:]  # positions in full.x we’ll take features from
        rng.shuffle(perm)
        x_perm = full.x[perm].clone()

        # labels applied to local node i are the labels of original node perm[i]
        applied_labels = [node_label(G_parent, S_perm) for S_perm in perm]

        src, dst = ei_sub.tolist()
        ok = False
        for u, v in set((min(a, b), max(a, b)) for a, b in zip(src, dst)):
            lu = applied_labels[u]
            lv = applied_labels[v]
            if unordered_label_pair(lu, lv) not in parent_pairs:
                ok = True
                break

        if ok:
            out = Data(x=x_perm, edge_index=ei_sub)
            out.parent_size = torch.tensor([full.num_nodes], dtype=torch.long)
            return out

    return None


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


# ────────────────────────────── positive samplers ─────────────────────────────
def sample_connected_bfs(G: nx.Graph, k: int, *, rng: random.Random) -> Optional[list[int]]:
    if k <= 0 or k > G.number_of_nodes():
        return None
    if G.number_of_edges() > 0 and k >= 2:
        u, v = rng.choice(list(G.edges()))
        S = [u, v]
    else:
        u = rng.choice(list(G.nodes()))
        S = [u]
    frontier, visited = set(S), set(S)
    while len(S) < k and frontier:
        u = rng.choice(list(frontier))
        nbrs = [w for w in G.neighbors(u) if w not in visited]
        if not nbrs:
            frontier.remove(u);
            continue
        w = rng.choice(nbrs)
        S.append(w);
        visited.add(w);
        frontier.add(w)
    if len(S) < k:
        candidates = {w for u in S for w in G.neighbors(u)} - set(S)
        while len(S) < k and candidates:
            S.append(rng.choice(list(candidates)))
            candidates = {w for u in S for w in G.neighbors(u)} - set(S)
        if len(S) < k:
            return None
    return S if nx.is_connected(G.subgraph(S)) else None


def sample_uniform_connected_kset(G: nx.Graph, k: int, *, rng: random.Random, max_tries: int = 50) -> Optional[
    list[int]]:
    nodes = list(G.nodes())
    if k > len(nodes) or k <= 0:
        return None
    for _ in range(max_tries):
        S = rng.sample(nodes, k)
        if nx.is_connected(G.subgraph(S)):
            return S
    return None


def enumerate_connected_ksets_small(G: nx.Graph, k: int, *, max_count: int | None = None) -> Iterable[list[int]]:
    count, nodes = 0, list(G.nodes())
    for S in combinations(nodes, k):
        if nx.is_connected(G.subgraph(S)):
            yield list(S)
            count += 1
            if max_count is not None and count >= max_count:
                return


def sample_connected_kset_with_anchor(
        G: nx.Graph,
        k: int,
        anchor: int,
        *,
        rng: random.Random,
        max_expands: int = 4,
) -> Optional[list[int]]:
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
      Negative type code (or 0 for positive):
      • 0 = positive
      • 2 = feature-permutation negative (at least one edge’s unordered label-pair is forbidden in parent)
      • 3 = anchored add-by-construction (added (anchor,v) within S whose unordered label-pair never occurs anywhere in
            parent; residual respected)
      • 4 = full-graph add (added a non-edge in the full graph). All negatives are guaranteed non-embeddings by
            construction.

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
    seed: int = 42
    k_min: int = 2
    k_max: Optional[int] = None  # None ⇒ go up to N
    pos_per_k: int = 6
    neg_per_pos: int = 2
    use_enumeration_up_to_k: int = 5
    enum_cap_per_k: int = 64
    include_full_graph: bool = True
    full_graph_neg_per_parent: int = 6
    tail_anchor_fraction: float = 0.5


class ZincPairs(InMemoryDataset):
    def __init__(self,
                 base_dataset,
                 root: Path | str = GLOBAL_DATASET_PATH / "ZincPairs",
                 split: str = "train",
                 cfg: PairConfig | None = None,
                 transform: Callable | None = None,
                 pre_transform: Callable | None = None,
                 pre_filter: Callable | None = None,
                 force_reload: bool = False):
        self.base = base_dataset
        self.split = split
        self.cfg = cfg or PairConfig()
        self._rng = random.Random(self.cfg.seed)

        # EITHER pass force_reload=True OR point root to a unique folder
        # when your base dataset changes (recommended in tests).
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        # Load the processed tensors
        with open(self.processed_paths[0], "rb") as f:
            self.data, self.slices = torch.load(f, map_location="cpu", weights_only=False)

    @property
    def raw_file_names(self) -> list[str]:
        return [f"depends_on_{type(self.base).__name__}.marker"]

    @property
    def processed_file_names(self) -> list[str]:
        return [f"pairs_{self.split}.pt"]

    def download(self):
        # Just ensure the raw dir exists & create a marker file (optional)
        Path(self.raw_dir).mkdir(parents=True, exist_ok=True)
        Path(self.raw_paths[0]).write_text(f"split={self.split}; base_len={len(self.base)}\n")

    def process(self):
        r"""
        Build and serialize pair samples for **feature-aware induced-subgraph** containment.

        Workflow
        --------
        For each parent graph in the base dataset:

        1. For each target subgraph order ``k`` in ``[cfg.k_min, N]``:
           a. Generate a set of **positive**, connected node subsets ``S`` using:
              - Anchored sampling (must include a chosen anchor),
              - Enumeration for small ``k`` (capped by ``enum_cap_per_k``),
              - Stochastic top-up (BFS or uniform-connected).
           b. For each positive subset ``S``:
              - Materialize a positive pair (induced subgraph on ``S`` vs. full parent).
              - Materialize ``neg_per_pos`` **guaranteed negatives**:
                * **Anchored add-by-construction**: add exactly one edge at the anchor to a node such that
                  the resulting edge’s unordered **label pair does not occur anywhere** in the parent;
                  also requires residual capacity on both ends.
                * **Feature-permutation fallback**: permute node features inside ``S`` until at least one
                  existing edge has a **forbidden** (parent-absent) label pair.
        2. Add **full-graph coverage** (``k = N``):
           - One exact positive (full vs. full),
           - Several add-only negatives by toggling one **non-edge** to an edge.

        Feature semantics
        -----------------
        Node features in any candidate subgraph are **copied from the parent graph** (never recomputed),
        enforcing feature-aware induced-subgraph matching.

        Side effects
        ------------
        Writes the collated tensors to ``processed/pairs_<split>.pt``.

        Notes
        -----
        - ``pre_filter`` and ``pre_transform`` are applied to each :class:`PairData`.
        - Positive subsets per ``(parent, k)`` are **deduplicated** to improve diversity.
        - Set ``cfg.k_max = None`` to include all sizes up to each parent’s ``N``; otherwise, an explicit upper bound is enforced.
        """
        rng, cfg = self._rng, self.cfg
        data_list: list[PairData] = []

        # Precompute NX views to avoid repeated conversions during verification
        nx_views = [pyg_to_nx(self.base[i]) for i in range(len(self.base))]

        # Iterate parent molecules; this is the grouping unit for splits
        parent_iter = tqdm(
            range(len(self.base)),
            desc=f"{self.split}: parents",
            unit="mol",
            dynamic_ncols=True,
        )
        # Iterate parent molecules; this is the grouping unit for splits
        for parent_idx in parent_iter:
            full: Data = self.base[parent_idx]
            G = nx_views[parent_idx]  # NetworkX view with node label tuples
            parent_pairs = parent_label_edge_set(G)
            N = G.number_of_nodes()
            if N < cfg.k_min:  # Nothing to generate for tiny graphs
                continue

            # Upper bound for k: either all the way to N or the configured min(N, k_max)
            k_hi = N if (getattr(cfg, "k_max", None) is None) else min(cfg.k_max, N)

            # Progress bar over k for this parent (nested; does not clutter the console)
            k_iter = tqdm(
                range(cfg.k_min, k_hi + 1),
                desc=f"mol#{parent_idx} k",
                unit="k",
                leave=False,
                dynamic_ncols=True,
            )

            # Subgraph orders k that mirror decoder growth (k >= 2 recommended)
            for k in k_iter:
                pos_sets: list[list[int]] = []
                pos_seen: set[tuple[int, ...]] = set()  # dedup key = sorted tuple of node ids

                # ---- Anchored positives: force inclusion of an anchor node ----
                n_anchor = int(round(cfg.pos_per_k * cfg.tail_anchor_fraction))
                for _ in range(n_anchor):
                    anchor = rng.choice(list(G.nodes()))
                    S = sample_connected_kset_with_anchor(G, k, anchor=anchor, rng=rng)
                    if S is None:
                        continue
                    key = tuple(sorted(S))
                    if key in pos_seen:
                        continue
                    pos_sets.append(S)
                    pos_seen.add(key)

                # ---- Enumeration for small k (coverage, capped to avoid blow-up) ----
                if k <= cfg.use_enumeration_up_to_k and len(pos_sets) < cfg.pos_per_k:
                    for S in enumerate_connected_ksets_small(G, k, max_count=cfg.enum_cap_per_k):
                        key = tuple(sorted(S))
                        if key in pos_seen:
                            continue
                        pos_sets.append(S)
                        pos_seen.add(key)
                        if len(pos_sets) >= cfg.pos_per_k:
                            break

                # ---- Stochastic top-up to reach the target count per k ----
                # Guard against duplicate-heavy sampling: cap attempts.
                needed = cfg.pos_per_k - len(pos_sets)
                if needed > 0:
                    max_topup_tries = 50 * needed  # tuneable; 50–200x usually plenty
                    tries = 0
                    while len(pos_sets) < cfg.pos_per_k and tries < max_topup_tries:
                        tries += 1
                        S = (sample_connected_bfs(G, k, rng=rng)
                             or sample_uniform_connected_kset(G, k, rng=rng))
                        if S is None:
                            break
                        key = tuple(sorted(S))
                        if key in pos_seen:
                            continue
                        pos_sets.append(S)
                        pos_seen.add(key)

                    # (optional) show fill ratio for this k
                    try:
                        k_iter.set_postfix_str(f"pos={len(pos_sets)}/{cfg.pos_per_k}, tries={tries}")
                    except Exception:
                        pass

                # ---- Materialize positives and their negatives ----
                for S in pos_sets:
                    # Positive
                    g1_pos = make_subgraph_data(full, S)
                    pair_pos = PairData(
                        x1=g1_pos.x, edge_index1=g1_pos.edge_index,
                        x2=full.x, edge_index2=full.edge_index,
                        y=torch.tensor([1]), k=torch.tensor([k]),
                        neg_type=torch.tensor([0]),
                        parent_idx=torch.tensor([parent_idx]),
                    )
                    if self.pre_filter is None or self.pre_filter(pair_pos):
                        if self.pre_transform:
                            pair_pos = self.pre_transform(pair_pos)
                        data_list.append(pair_pos)

                    # -------------------- NEGATIVES (guaranteed-by-construction) --------------------
                    for _ in range(cfg.neg_per_pos):
                        emitted = False

                        # (A) Anchored add-by-construction: add an edge (anchor, v) whose label-pair
                        #     never appears as any edge in the parent; both endpoints must have residual>0.
                        #     This cannot embed in the parent → guaranteed negative.
                        # Pick anchor among nodes with residual>0 if possible.
                        res = residual_capacity_vector(G, S)
                        candidates = [v for v in S if res[v] > 0]
                        anchor = rng.choice(candidates) if candidates else rng.choice(list(S))

                        H = pick_wrong_add_at_anchor(G, S, anchor=anchor, rng=rng, require_residual=True,
                                                     parent_pairs=parent_pairs)
                        if H is not None:
                            ei_neg = nx_to_edge_index_on_ordered_nodes(H, S)
                            g1_neg = make_subgraph_data(full, S, edge_index_override=ei_neg)
                            pair_neg = PairData(
                                x1=g1_neg.x, edge_index1=g1_neg.edge_index,
                                x2=full.x, edge_index2=full.edge_index,
                                y=torch.tensor([0]), k=torch.tensor([k]),
                                neg_type=torch.tensor([3]),  # 3 = anchored add-by-construction
                                parent_idx=torch.tensor([parent_idx]),
                            )
                            if self.pre_filter is None or self.pre_filter(pair_neg):
                                if self.pre_transform:
                                    pair_neg = self.pre_transform(pair_neg)
                                data_list.append(pair_neg)
                                emitted = True

                        if emitted:
                            continue

                        # (B) Fallback: feature-permutation-by-construction.
                        #     Permute features on S until at least one edge label-pair is forbidden
                        #     relative to the parent. Guaranteed negative if produced.
                        g1_perm = permute_features_with_forbidden_edge(full, S, G_parent=G, rng=rng, max_tries=64,
                                                                       parent_pairs=parent_pairs)
                        if g1_perm is not None:
                            pair_neg = PairData(
                                x1=g1_perm.x, edge_index1=g1_perm.edge_index,
                                x2=full.x, edge_index2=full.edge_index,
                                y=torch.tensor([0]), k=torch.tensor([k]),
                                neg_type=torch.tensor([2]),  # 2 = feature-permutation by construction
                                parent_idx=torch.tensor([parent_idx]),
                            )
                            if self.pre_filter is None or self.pre_filter(pair_neg):
                                if self.pre_transform:
                                    pair_neg = self.pre_transform(pair_neg)
                                data_list.append(pair_neg)
                                emitted = True

                        # If neither (A) nor (B) worked, silently skip this negative.
            # ───────── full-graph (k=N) coverage: keep positive, add-only negatives ─────────
            if cfg.include_full_graph and N >= 2:
                # Positive: full vs full
                pair_full_pos = PairData(
                    x1=full.x, edge_index1=full.edge_index,
                    x2=full.x, edge_index2=full.edge_index,
                    y=torch.tensor([1]), k=torch.tensor([N]),
                    neg_type=torch.tensor([0]),
                    parent_idx=torch.tensor([parent_idx]),
                )
                if self.pre_filter is None or self.pre_filter(pair_full_pos):
                    if self.pre_transform:
                        pair_full_pos = self.pre_transform(pair_full_pos)
                    data_list.append(pair_full_pos)

                # Add-only negatives (choose a non-edge; prefer touching a random anchor)
                nodes_list = list(G.nodes())
                undirected = {(min(u, v), max(u, v)) for (u, v) in G.edges()}
                all_pairs = {(min(u, v), max(u, v))
                             for i, u in enumerate(nodes_list)
                             for v in nodes_list[i + 1:]}
                non_edges = list(all_pairs - undirected)

                for _ in range(cfg.full_graph_neg_per_parent):
                    if not non_edges:
                        break
                    anchor = self._rng.choice(nodes_list)
                    touching_non = [(u, v) for (u, v) in non_edges if anchor in (u, v)]
                    add_uv = self._rng.choice(touching_non) if touching_non else self._rng.choice(non_edges)

                    H = G.copy()
                    u, v = add_uv
                    if not H.has_edge(u, v):
                        H.add_edge(u, v)

                    ei_neg = nx_to_edge_index_on_ordered_nodes(H, nodes_list)
                    g1_neg = make_subgraph_data(full, nodes_list, edge_index_override=ei_neg)
                    pair_full_neg = PairData(
                        x1=g1_neg.x, edge_index1=g1_neg.edge_index,
                        x2=full.x, edge_index2=full.edge_index,
                        y=torch.tensor([0]),
                        k=torch.tensor([N]),
                        neg_type=torch.tensor([4]),  # 4 = full-graph add
                        parent_idx=torch.tensor([parent_idx]),
                    )
                    if self.pre_filter is None or self.pre_filter(pair_full_neg):
                        if self.pre_transform:
                            pair_full_neg = self.pre_transform(pair_full_neg)
                        data_list.append(pair_full_neg)

        if data_list:
            keys0 = set(data_list[0].keys())
            for i, d in enumerate(data_list):
                if set(d.keys()) != keys0:
                    missing = keys0 - set(d.keys())
                    extra = set(d.keys()) - keys0
                    raise RuntimeError(f"Non-uniform attributes at item {i}: missing={missing}, extra={extra}")

        # Collate to in-memory tensors and persist
        data, slices = self.collate(data_list)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    # Tiny base split (1 molecule) just for sanity
    base = ZincSmiles(split="test")[:80]

    pairs = ZincPairs(
        base_dataset=base,
        split="test",
        cfg=PairConfig(),   # use your current config
        force_reload=True
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
            pyg_to_nx(g1),   # small / candidate
            pyg_to_nx(g2),   # big / parent
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
