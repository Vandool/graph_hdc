"""
We want a pairwise classifier f(G_1, G_2) to {0,1} that answers “is G_1 an induced subgraph of G_2” under
feature-aware matching, with node features frozen to those of the final graph.

•	Positives (P): connected, induced subgraphs G[S] of the full molecule G for sizes k=2,dots,K. Node features are
    taken from G (as you already do), edges are exactly those among S present in G.
•	Negatives (N): same node feature convention, but edge pattern (or node set) must differ so that G_1 is not an
    induced subgraph of G.

Positive Sampling:


"""
import math
import random
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, Optional
from typing import Sequence
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset

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


# ────────────────────────────── negative samplers ─────────────────────────────
def edge_flip_negative_within_S(G: nx.Graph, S: Sequence[int], *, rng: random.Random) -> Optional[nx.Graph]:
    """Generic single-edge toggle inside S (no residual constraints)."""
    H = G.subgraph(S).copy()
    existing = list(H.edges())
    all_pairs = [(u, v) for i, u in enumerate(S) for v in S[i + 1:]]
    non_edges = [(u, v) for (u, v) in all_pairs if not H.has_edge(u, v)]
    choices: list[tuple[str, tuple[int, int]]] = []
    if existing:
        choices.append(("remove", rng.choice(existing)))
    if non_edges:
        choices.append(("add", rng.choice(non_edges)))
    if not choices:
        return None
    op, (u, v) = rng.choice(choices)
    if op == "remove":
        H.remove_edge(u, v)
    else:
        H.add_edge(u, v)
    return H


def edge_flip_negative_adjacent_to_anchor_residual_aware(
        G: nx.Graph, S: Sequence[int], anchor: int, *, rng: random.Random
) -> Optional[nx.Graph]:
    """
    Toggle one edge incident to `anchor`, but:
      - for ADD: require residual(u)>0 and residual(v)>0 (plausible decoder action),
      - for REMOVE: allow removing any existing (models “missing required edge”).
    """
    if anchor not in S:
        return None
    H = G.subgraph(S).copy()
    res = residual_capacity_vector(G, S)

    # candidates to remove: existing (anchor, v)
    rem = [(min(anchor, v), max(anchor, v)) for v in H.neighbors(anchor)]

    # candidates to add: non-edges (anchor, v) with residual>0 on both ends
    add = []
    for v in S:
        if v == anchor:
            continue
        u, w = (min(anchor, v), max(anchor, v))
        if not H.has_edge(u, w) and res[anchor] > 0 and res[v] > 0:
            add.append((u, w))

    choices: list[tuple[str, tuple[int, int]]] = []
    if add:
        choices.append(("add", rng.choice(add)))
    if rem:
        choices.append(("remove", rng.choice(rem)))
    if not choices:
        return None

    op, (u, v) = rng.choice(choices)
    if op == "remove":
        H.remove_edge(u, v)
    else:
        H.add_edge(u, v)
    return H


def feature_permutation_negative_S(full: Data, S: Sequence[int], *, rng: random.Random) -> Data:
    """Permute node features among S while keeping induced edges (feature-mismatch negative)."""
    S = list(S)
    ei_sub = induced_edge_index_from_node_set(full.edge_index, S)
    perm = S[:]
    rng.shuffle(perm)
    x_perm = full.x[perm].clone()
    out = Data(x=x_perm, edge_index=ei_sub)
    out.parent_size = torch.tensor([full.num_nodes], dtype=torch.long)
    # Keep both maps to avoid ambiguity later
    out.idx_map_edges = torch.tensor(S, dtype=torch.long)
    out.idx_map_feats = torch.tensor(perm, dtype=torch.long)
    return out


# ───────────────────────────── verification (optional) ────────────────────────
def is_induced_subgraph_feature_aware(G_small: nx.Graph, G_big: nx.Graph) -> bool:
    """NetworkX VF2: is `G_small` an induced, label-preserving subgraph of `G_big`?"""
    nm = lambda a, b: a["label"] == b["label"]
    GM = nx.algorithms.isomorphism.GraphMatcher(G_big, G_small, node_match=nm)
    return GM.subgraph_is_isomorphic()

# helpers (place near your VF2 utilities)
def has_label_duplicates(G: nx.Graph, S: Sequence[int]) -> bool:
    labels = [G.nodes[v]["label"] for v in S]
    # count duplicates
    return len(labels) != len(set(labels))

def is_present_anywhere_pyg(g_small: Data, G_big: nx.Graph) -> bool:
    return is_induced_subgraph_feature_aware(pyg_to_nx(g_small), G_big)

# ─────────────────────────────── pair container ───────────────────────────────
class PairData(Data):
    """
    Container for a (G1, G2, y) pair:
      - x1, edge_index1: candidate subgraph
      - x2, edge_index2: parent graph
      - y, k, neg_type, parent_idx: labels/metadata
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


# ─────────────────────────────── dataset builder ──────────────────────────────
class PairConfig:
    seed: int = 42
    k_min: int = 2
    k_max: Optional[int] = None  # None ⇒ go up to N for each parent
    pos_per_k: int = 6
    neg_per_pos: int = 2
    use_enumeration_up_to_k: int = 5
    enum_cap_per_k: int = 64
    verify_cross_graph_negatives: bool = True
    cross_graph_neg_prob: float = 0.15  # ← how often to try cross-graph
    cross_graph_max_attempts: int = 5  # ← resample budget if match found
    tail_anchor_fraction: float = 0.5
    include_full_graph: bool = True
    full_graph_neg_per_parent: int = 6
    anchor_neg_prob: float = 0.8
    verify_local_negatives: bool = True
    verify_local_negatives_ambiguous_only: bool = True   # only when labels duplicate
    local_neg_max_attempts: int = 6


class ZincPairs(InMemoryDataset):
    def __init__(self,
                 base_dataset,
                 root: Path | str = GLOBAL_DATASET_PATH / "ZincPairs",
                 split: str = "train",
                 cfg: PairConfig | None = None,
                 transform: Callable | None = None,
                 pre_transform: Callable | None = None,
                 pre_filter: Callable | None = None):
        """
        Notes
        -----
        - We rely on PyG to call `download()`/`process()` inside `super().__init__`.
        - As a safety net, we *also* ensure the processed file exists afterwards,
          and if not, we create the raw marker and call `process()` manually.
        """
        self.base = base_dataset
        self.split = split
        self.cfg = cfg or PairConfig()
        self._rng = random.Random(self.cfg.seed)

        # Let PyG check freshness and (if needed) call download()/process()
        super().__init__(root, transform, pre_transform, pre_filter)

        # Safety net: if for any reason the processed file still doesn't exist,
        # create the raw marker and process now. This prevents FileNotFoundError.
        pp = Path(self.processed_paths[0])
        if not pp.exists():
            Path(self.raw_dir).mkdir(parents=True, exist_ok=True)
            Path(self.raw_paths[0]).write_text("")  # raw marker to satisfy PyG expectations
            self.process()  # build pairs and write pairs_<split>.pt

        # Now load the in-memory representation
        with open(self.processed_paths[0], "rb") as f:
            self.data, self.slices = torch.load(f, map_location="cpu", weights_only=False)

    @property
    def raw_file_names(self) -> list[str]:
        # A sentinel that declares our dependency on the base dataset
        return [f"depends_on_{type(self.base).__name__}.marker"]

    @property
    def processed_file_names(self) -> list[str]:
        return [f"pairs_{self.split}.pt"]

    def download(self):
        # Ensure the raw marker exists so PyG will be happy and call `process()`
        Path(self.raw_dir).mkdir(parents=True, exist_ok=True)
        Path(self.raw_paths[0]).write_text("")

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
              - Materialize ``neg_per_pos`` negatives, in order of preference:
                * **Cross-graph negatives** (optional): sample a connected k-subgraph from a *different*
                  parent; if ``cfg.verify_cross_graph_negatives`` is True, reject it if it matches
                  as an induced, feature-aware subgraph of the current parent (VF2 check).
                * **Anchor-adjacent, residual-aware** single-edge toggles (adds only if both endpoints
                  have non-zero residual degree; removals allowed),
                * Generic single-edge toggle,
                * **Feature-permutation** within ``S``.
        2. Add **full-graph coverage** (``k = N``):
           - One exact positive (full vs. full),
           - Several completion negatives by toggling exactly one edge
             (prefer edges incident to a random anchor).

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
        - Set ``cfg.k_max = None`` to include all sizes up to each parent’s ``N``; otherwise,
          an explicit upper bound is enforced.
        - Cross-graph negatives are controlled by:
          * ``cross_graph_neg_prob``: probability to *attempt* a cross-graph negative,
          * ``cross_graph_max_attempts``: resampling budget,
          * ``verify_cross_graph_negatives``: enable VF2 label-aware rejection of accidental positives.

        Raises
        ------
        Exceptions from I/O (e.g., ``torch.save``) or user-provided ``pre_filter`` / ``pre_transform``.
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
                    # Positive: induced subgraph with features copied from the full graph
                    g1_pos = make_subgraph_data(full, S)
                    pair_pos = PairData(
                        x1=g1_pos.x, edge_index1=g1_pos.edge_index,  # candidate subgraph
                        x2=full.x,  edge_index2=full.edge_index,     # parent graph
                        y=torch.tensor([1]),                        # label = positive
                        k=torch.tensor([k]),
                        neg_type=torch.tensor([0]),                 # 0 marks positive
                        parent_idx=torch.tensor([parent_idx]),
                    )

                    # Apply optional hooks and keep
                    if self.pre_filter is None or self.pre_filter(pair_pos):
                        if self.pre_transform:
                            pair_pos = self.pre_transform(pair_pos)
                        data_list.append(pair_pos)

                    # Generate negatives for this positive
                    for _ in range(cfg.neg_per_pos):
                        emitted = False

                        # --------- (1) Try cross-graph negative with small probability ---------
                        # To prevent the classifier from overfitting to “one-edge flips only.
                        cross_ok = getattr(cfg, "cross_graph_neg_prob", 0.0) > 0.0 and len(self.base) > 1
                        if cross_ok and rng.random() < cfg.cross_graph_neg_prob:
                            attempts = 0
                            max_attempts = getattr(cfg, "cross_graph_max_attempts", 5)
                            while attempts < max_attempts:
                                attempts += 1
                                other_idx = rng.randrange(len(self.base))
                                if other_idx == parent_idx:
                                    continue  # don't use the same parent

                                other_full: Data = self.base[other_idx]
                                G_other = nx_views[other_idx]

                                # Sample a connected k-set in the other parent
                                S_other = (sample_connected_bfs(G_other, k, rng=rng)
                                           or sample_uniform_connected_kset(G_other, k, rng=rng))
                                if S_other is None:
                                    continue

                                g1_cross = make_subgraph_data(other_full, S_other)

                                # Optional VF2 check: reject if the cross-graph subgraph actually appears in current parent
                                if getattr(cfg, "verify_cross_graph_negatives", False):
                                    G_small = pyg_to_nx(g1_cross)
                                    if is_induced_subgraph_feature_aware(G_small, G):
                                        # Found a real match → not a valid negative; try another
                                        continue

                                pair_neg = PairData(
                                    x1=g1_cross.x, edge_index1=g1_cross.edge_index,  # G1 from *other* parent
                                    x2=full.x,     edge_index2=full.edge_index,      # G2 = current parent
                                    y=torch.tensor([0]),
                                    k=torch.tensor([k]),
                                    neg_type=torch.tensor([6]),                      # 6 = cross-graph negative
                                    parent_idx=torch.tensor([parent_idx]),           # group by target parent
                                )
                                # (optional) record source graph id
                                pair_neg.src_parent_idx1 = torch.tensor([other_idx])

                                if self.pre_filter is None or self.pre_filter(pair_neg):
                                    if self.pre_transform:
                                        pair_neg = self.pre_transform(pair_neg)
                                    data_list.append(pair_neg)
                                    emitted = True
                                    break  # done with this negative

                        if emitted:
                            continue  # next negative for this positive

                        # --------- (2) Anchored residual-aware local negative (preferred) ---------
                        use_anchor = (rng.random() < cfg.anchor_neg_prob)
                        if use_anchor:
                            res = residual_capacity_vector(G, S)      # residuals on the *current* k-set
                            candidates = [v for v in S if res[v] > 0]
                            anchor = rng.choice(candidates) if candidates else rng.choice(S)

                            H = edge_flip_negative_adjacent_to_anchor_residual_aware(G, S, anchor=anchor, rng=rng)
                            if H is not None:
                                ei_neg = nx_to_edge_index_on_ordered_nodes(H, S)
                                g1_neg = make_subgraph_data(full, S, edge_index_override=ei_neg)
                                neg_code = 3  # anchored residual-aware flip
                            else:
                                # Fallbacks: generic flip, then feature permutation
                                H = edge_flip_negative_within_S(G, S, rng=rng)
                                if H is None:
                                    g1_neg = feature_permutation_negative_S(full, S, rng=rng)
                                    neg_code = 2
                                else:
                                    ei_neg = nx_to_edge_index_on_ordered_nodes(H, S)
                                    g1_neg = make_subgraph_data(full, S, edge_index_override=ei_neg)
                                    neg_code = 1
                        else:
                            # --------- (3) Generic local negative path ---------
                            H = edge_flip_negative_within_S(G, S, rng=rng)
                            if H is None:
                                g1_neg = feature_permutation_negative_S(full, S, rng=rng)
                                neg_code = 2
                            else:
                                ei_neg = nx_to_edge_index_on_ordered_nodes(H, S)
                                g1_neg = make_subgraph_data(full, S, edge_index_override=ei_neg)
                                neg_code = 1

                        # --- FIX: verify that the local negative does NOT occur anywhere in G (feature-aware, induced)
                        if getattr(cfg, "verify_local_negatives", True):
                            ok = not is_induced_subgraph_feature_aware(pyg_to_nx(g1_neg), G)
                            tries = 0
                            max_local = getattr(cfg, "local_neg_max_attempts", 6)
                            while (not ok) and tries < max_local:
                                tries += 1
                                if use_anchor:
                                    # try anchored again; fallback to generic then permutation
                                    H2 = edge_flip_negative_adjacent_to_anchor_residual_aware(G, S, anchor=anchor, rng=rng)
                                    if H2 is None:
                                        H2 = edge_flip_negative_within_S(G, S, rng=rng)
                                        if H2 is None:
                                            g1_neg = feature_permutation_negative_S(full, S, rng=rng)
                                            neg_code = 2
                                        else:
                                            ei2 = nx_to_edge_index_on_ordered_nodes(H2, S)
                                            g1_neg = make_subgraph_data(full, S, edge_index_override=ei2)
                                            neg_code = 1
                                    else:
                                        ei2 = nx_to_edge_index_on_ordered_nodes(H2, S)
                                        g1_neg = make_subgraph_data(full, S, edge_index_override=ei2)
                                        neg_code = 3
                                else:
                                    # generic path: generic flip → permutation
                                    H2 = edge_flip_negative_within_S(G, S, rng=rng)
                                    if H2 is None:
                                        g1_neg = feature_permutation_negative_S(full, S, rng=rng)
                                        neg_code = 2
                                    else:
                                        ei2 = nx_to_edge_index_on_ordered_nodes(H2, S)
                                        g1_neg = make_subgraph_data(full, S, edge_index_override=ei2)
                                        neg_code = 1
                                ok = not is_induced_subgraph_feature_aware(pyg_to_nx(g1_neg), G)

                            if not ok:
                                # could not realize a strict negative → skip this one
                                continue
                        # --- END FIX

                        pair_neg = PairData(
                            x1=g1_neg.x, edge_index1=g1_neg.edge_index,
                            x2=full.x,   edge_index2=full.edge_index,
                            y=torch.tensor([0]),
                            k=torch.tensor([k]),
                            neg_type=torch.tensor([neg_code]),          # 1/2/3 = local variants
                            parent_idx=torch.tensor([parent_idx]),
                        )

                        if self.pre_filter is None or self.pre_filter(pair_neg):
                            if self.pre_transform:
                                pair_neg = self.pre_transform(pair_neg)
                            data_list.append(pair_neg)

            # ---- Full-graph (k=N) coverage: exactness and completion errors ----
            if cfg.include_full_graph and N >= 2:
                # Positive: full graph vs itself (teaches strict completion)
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

                # Candidate edge toggles (prefer those incident to a random anchor)
                undirected = {(min(u, v), max(u, v)) for (u, v) in G.edges()}
                all_pairs = {
                    (min(u, v), max(u, v))
                    for i, u in enumerate(G.nodes())
                    for v in list(G.nodes())[i + 1:]
                }
                non_edges = list(all_pairs - undirected)  # additions
                true_edges = list(undirected)             # deletions

                for _ in range(cfg.full_graph_neg_per_parent):
                    anchor = rng.choice(list(G.nodes()))
                    touching_true = [(u, v) for (u, v) in true_edges if anchor in (u, v)]
                    touching_non  = [(u, v) for (u, v) in non_edges if anchor in (u, v)]

                    choices: list[tuple[str, tuple[int, int]]] = []
                    if touching_non:
                        choices.append(("add", rng.choice(touching_non)))
                    if touching_true:
                        choices.append(("remove", rng.choice(touching_true)))
                    if not choices:
                        if non_edges:  choices.append(("add", rng.choice(non_edges)))
                        if true_edges: choices.append(("remove", rng.choice(true_edges)))
                    if not choices:
                        continue  # degenerate case already guarded

                    op, (u, v) = rng.choice(choices)
                    H = G.copy()
                    if op == "remove" and H.has_edge(u, v):
                        H.remove_edge(u, v)
                    elif op == "add" and not H.has_edge(u, v):
                        H.add_edge(u, v)

                    ei_neg = nx_to_edge_index_on_ordered_nodes(H, list(G.nodes()))
                    g1_neg = make_subgraph_data(full, list(G.nodes()), edge_index_override=ei_neg)
                    pair_full_neg = PairData(
                        x1=g1_neg.x, edge_index1=g1_neg.edge_index,
                        x2=full.x,   edge_index2=full.edge_index,
                        y=torch.tensor([0]),
                        k=torch.tensor([N]),
                        neg_type=torch.tensor([4 if op == "add" else 5]),  # 4:add, 5:remove
                        parent_idx=torch.tensor([parent_idx]),
                    )
                    if self.pre_filter is None or self.pre_filter(pair_full_neg):
                        if self.pre_transform:
                            pair_full_neg = self.pre_transform(pair_full_neg)
                        data_list.append(pair_full_neg)

        if data_list:
            keys0 = set(data_list[0].keys)
            for i, d in enumerate(data_list):
                if set(d.keys) != keys0:
                    missing = keys0 - set(d.keys)
                    extra = set(d.keys) - keys0
                    raise RuntimeError(f"Non-uniform attributes at item {i}: missing={missing}, extra={extra}")

        # Collate to in-memory tensors and persist
        data, slices = self.collate(data_list)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    base = ZincSmiles(split="test")[:1]
    pairs = ZincPairs(
        base_dataset=base,
        split="test",
        cfg=PairConfig()
    )

    print(len(base))
    print(len(pairs))