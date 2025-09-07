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
    seed: int = 42
    k_min: int = 2
    k_max: int | None = None  # None ⇒ go up to N
    pos_per_k: int = 8
    neg_per_pos: int = 8
    tail_anchor_fraction: float = 0.5


class ZincPairs(Dataset):
    def __init__(
        self,
        base_dataset,
        split="train",
        root=GLOBAL_DATASET_PATH / "ZincPairs",
        cfg=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        shard_size=100_000,
        cache_shards=1,
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
            root = GLOBAL_DATASET_PATH / "ZincPairsDEV"

        # idx_path must exist for process() to use
        self.idx_path = Path(root) / "processed" / f"index_{split}.pt"

        super().__init__(root, transform, pre_transform, pre_filter)

        # build if missing or forced
        if force_reprocess or not self.idx_path.exists():
            self.process()

        meta = torch.load(self.idx_path, map_location="cpu")
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

        Workflow
        --------
        For each parent graph G in the base dataset and for each k in [cfg.k_min, min(N, cfg.k_max or N)]:

        1) Positives:
           - Sample connected node sets S of size k via anchored sampling.
             Anchors are chosen in low-degree order (by target degree from labels, then actual degree).
           - Materialize the positive pair: (G[S], G), induced edges; node features copied from G.

        2) Negatives (guaranteed-by-construction):
           - For each positive S, list all non-edges (u,v) in G[S] such that:
               * residual(u) > 0 and residual(v) > 0; and
               * the unordered label pair (label(u), label(v)) never occurs as an edge anywhere in G.
             Sample up to cfg.neg_per_pos of these and add a single (u,v) to form G'[S].
           - Each such candidate is a non-embedding by construction under feature-aware induced-subgraph matching.

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
            if cfg.k_min > N:  # Nothing to generate for tiny graphs
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
                # ---- Anchored positives: force inclusion of a low-degree anchor ----
                anchors_order = low_degree_anchor_order(G)
                n_anchor = max(1, int(round(cfg.pos_per_k * cfg.tail_anchor_fraction)))
                n_anchor = min(n_anchor, len(anchors_order))

                tries = 0
                i = 0
                max_tries = 30 * cfg.pos_per_k  # bounded
                while len(pos_sets) < n_anchor and tries < max_tries:
                    tries += 1
                    anchor = anchors_order[i % n_anchor]  # only the lowest-degree slice
                    i += 1
                    S = sample_connected_kset_with_anchor(G, k, anchor=anchor, rng=rng)
                    if S is None:
                        continue
                    key = tuple(sorted(S))
                    if key in pos_seen:
                        continue
                    pos_sets.append(S)
                    pos_seen.add(key)

                # ---- Top-up to reach the target count per k (still anchor-based) ----
                tries = 0
                max_tries = 30 * (cfg.pos_per_k - len(pos_sets) + 1)
                j = 0
                while len(pos_sets) < cfg.pos_per_k and tries < max_tries:
                    tries += 1
                    # cycle over the whole low-degree order now
                    anchor = anchors_order[j % max(1, len(anchors_order))]
                    j += 1
                    S = sample_connected_kset_with_anchor(G, k, anchor=anchor, rng=rng)
                    if S is None:
                        continue
                    key = tuple(sorted(S))
                    if key in pos_seen:
                        continue
                    pos_sets.append(S)
                    pos_seen.add(key)

                # (optional) show fill ratio
                try:
                    k_iter.set_postfix_str(f"pos={len(pos_sets)}/{cfg.pos_per_k}")
                except Exception:
                    pass

                # ---- Materialize positives and their negatives ----
                for S in pos_sets:
                    # Positive
                    g1_pos = make_subgraph_data(full, S)
                    pair_pos = PairData(
                        x1=g1_pos.x,
                        edge_index1=g1_pos.edge_index,
                        x2=full.x,
                        edge_index2=full.edge_index,
                        y=torch.tensor([1]),
                        k=torch.tensor([k]),
                        neg_type=torch.tensor([0]),
                        parent_idx=torch.tensor([parent_idx]),
                    )
                    yield pair_pos

                    # -------------------- NEGATIVES (guaranteed-by-construction) --------------------
                    bad_pairs = wrong_add_candidates_anywhere(G, S, parent_pairs)

                    if bad_pairs:
                        m = min(len(bad_pairs), cfg.neg_per_pos)
                        for u, v in rng.sample(bad_pairs, m):
                            H = G.subgraph(S).copy()
                            H.add_edge(u, v)
                            ei_neg = nx_to_edge_index_on_ordered_nodes(H, S)
                            g1_neg = make_subgraph_data(full, S, edge_index_override=ei_neg)
                            pair_neg = PairData(
                                x1=g1_neg.x,
                                edge_index1=g1_neg.edge_index,
                                x2=full.x,
                                edge_index2=full.edge_index,
                                y=torch.tensor([0]),
                                k=torch.tensor([k]),
                                neg_type=torch.tensor([1]),  # local wrong-add by construction
                                parent_idx=torch.tensor([parent_idx]),
                            )
                            yield pair_neg
                        # If there are no bad_pairs for this S, we emit fewer (possibly zero) negatives.

                # --- Extra: k=2 negatives (decoder start) ---
                if k == 2:
                    # Total budget tied to positives we just emitted for k=2:
                    k2_neg_budget = len(pos_sets) * cfg.neg_per_pos
                    bad_pairs2 = forbidden_pairs_k2(G, parent_pairs)
                    if bad_pairs2 and k2_neg_budget > 0:
                        # Sample without replacement to avoid duplicates
                        take = min(len(bad_pairs2), k2_neg_budget)
                        for u, v in rng.sample(bad_pairs2, take):
                            S2 = [u, v]
                            H2 = nx.Graph()
                            H2.add_nodes_from(S2)
                            H2.add_edge(u, v)  # the forbidden edge
                            ei2 = nx_to_edge_index_on_ordered_nodes(H2, S2)

                            g1_neg = make_subgraph_data(full, S2, edge_index_override=ei2)
                            pair_neg = PairData(
                                x1=g1_neg.x,
                                edge_index1=g1_neg.edge_index,
                                x2=full.x,
                                edge_index2=full.edge_index,
                                y=torch.tensor([0]),
                                k=torch.tensor([2]),
                                neg_type=torch.tensor([1]),  # same code: local wrong-add (k=2 construction)
                                parent_idx=torch.tensor([parent_idx]),
                            )
                            yield pair_neg


if __name__ == "__main__":
    # Tiny base split (1 molecule) just for sanity
    base = ZincSmiles(split="test")[:10]

    pairs = ZincPairs(
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
