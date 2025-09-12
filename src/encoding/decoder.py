import itertools
from collections import Counter
from collections.abc import Sequence
from itertools import chain, combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from networkx.algorithms.isomorphism import (
    GraphMatcher,
    categorical_edge_match,
    categorical_node_match,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
)

from src.encoding.oracles import Oracle
from src.encoding.the_types import Feat
from src.utils.visualisations import draw_nx_with_atom_colorings


def is_induced_subgraph_by_features(
    g1: nx.Graph,
    g2: nx.Graph,
    *,
    node_keys: list[str] | None = None,
    edge_keys: Sequence[str] = (),
    require_connected: bool = True,
) -> bool:
    """
    Return True iff G1 is isomorphic to a node-induced subgraph of G2 while
    preserving specified node (and optional edge) attributes.

    Parameters
    ----------
    g1, g2 : nx.Graph
        Pattern graph (g1) and target graph (g2).
    node_keys : str | Sequence[str], optional
        Node attribute key(s) that define a node's identity (labels).
        Nodes are matched by equality of these attributes; node IDs are ignored.
        Default: "feature".
    edge_keys : Sequence[str], optional
        Edge attribute key(s) that must also match categorically. Default: ().
    require_connected : bool, optional
        If True, fail fast when g1 is non-empty and not connected.

    Returns
    -------
    bool
        True if an induced subgraph isomorphism exists; False otherwise.

    Notes
    -----
    - Uses NetworkX VF2 with semantic checks via `categorical_node_match`
      and `categorical_edge_match`. Subgraph here means *node-induced*.
    - Performs a quick multiset pre-check on node features to prune impossible cases.

    """
    if require_connected and g1.number_of_nodes() and not nx.is_connected(g1):
        return False

    if node_keys is None:
        node_keys = ["feat"]

    # Fast prune: g1's feature multiset must be a subset of g2's

    def feat_tuple(G: nx.Graph, n) -> tuple:
        data = G.nodes[n]
        return tuple(data.get(k) for k in node_keys)

    # Quick check: g1's feature multiset must be a subset of g2's
    c1 = Counter(feat_tuple(g1, n) for n in g1.nodes)
    c2 = Counter(feat_tuple(g2, n) for n in g2.nodes)
    for k, need in c1.items():
        if c2.get(k, 0) < need:
            return False

    # Perform the full graph isomorphism check
    nm = categorical_node_match(
        node_keys if len(node_keys) > 1 else node_keys[0], [None] * len(node_keys) if len(node_keys) > 1 else None
    )

    em = categorical_edge_match(list(edge_keys), [None] * len(edge_keys)) if edge_keys else None

    GM = GraphMatcher(g2, g1, node_match=nm, edge_match=em)
    return GM.subgraph_is_isomorphic()


def perfect_oracle(gs: list[nx.Graph], full_g_nx: nx.Graph) -> list[bool]:
    return [is_induced_subgraph_by_features(g1=g, g2=full_g_nx, node_keys=["feat"]) for g in gs]


def show_confusion_matrix(ys: list[bool], ps: list[bool]) -> None:
    disp = ConfusionMatrixDisplay.from_predictions(
        ys,
        ps,
        labels=[False, True],
        normalize="true",  # row-normalize => 0..1
        cmap="Blues",
        values_format=".0%",  # show percents
    )
    disp.im_.set_clim(0, 1)  # fix color scale (no auto-rescale)
    plt.tight_layout()
    plt.show()


def feature_counter_from_graph(G: nx.Graph) -> Counter[tuple[int, int, int, int]]:
    """
    Count node features in a graph.

    :returns: Counter keyed by ``feat.to_tuple()``.
    """
    c = Counter()
    for n in G.nodes:
        c[G.nodes[n]["feat"].to_tuple()] += 1
    return c


def leftover_features(full: Counter[tuple[int, int, int, int]], G: nx.Graph) -> Counter:
    """
    Remaining features to be placed given the current graph.

    :param full: Full target multiset of node features.
    :param G: Current partial graph.
    :returns: Non-negative counter of leftover feature tuples.
    """
    left = full.copy()
    left.subtract(feature_counter_from_graph(G))
    # drop non-positive
    for k in list(left):
        if left[k] <= 0:
            del left[k]
    return left


def current_degree(G: nx.Graph, node: int) -> int:
    """
    Current degree of ``node`` (undirected).

    :param G: Graph.
    :param node: Node id.
    :returns: Degree in the *current* partial graph.
    """
    return G.degree[node]


def residual_degree(G: nx.Graph, node: int) -> int:
    """
    Residual degree capacity = ``target_degree - current_degree``.

    :param G: Graph.
    :param node: Node id.
    :returns: Remaining stubs (>= 0).
    """
    return int(G.nodes[node]["target_degree"]) - current_degree(G, node)


def residuals(G: nx.Graph) -> dict[int, int]:
    """
    Residual degrees for all nodes.

    :param G: Graph.
    :returns: Mapping ``node -> residual_degree``.
    """
    return {n: residual_degree(G, n) for n in G.nodes}


def anchors(G: nx.Graph) -> list[int]:
    """
    Nodes that can still accept edges (residual > 0).

    :param G: Graph.
    :returns: list of node ids.
    """
    return [n for n in G.nodes if residual_degree(G, n) > 0]


def add_edge_if_possible(G: nx.Graph, u: int, v: int, *, strict: bool = True) -> bool:
    """
    Add an undirected edge if constraints allow it.

    Constraints:
    - ``u != v``
    - Edge must not already exist.
    - Both endpoints must have residual > 0.
    - If ``strict``: never exceed target degrees.

    :param G: Graph (modified in place).
    :param u: Endpoint node.
    :param v: Endpoint node.
    :param strict: Enforce residual checks.
    :returns: ``True`` if edge was added, else ``False``.
    """
    if u == v or G.has_edge(u, v):
        return False
    if strict and (residual_degree(G, u) <= 0 or residual_degree(G, v) <= 0):
        return False
    G.add_edge(u, v)
    if strict and (residual_degree(G, u) < 0 or residual_degree(G, v) < 0):
        # rollback if we over-shot due to concurrent edits
        G.remove_edge(u, v)
        return False
    return True


def total_edges_count(feat_ctr: Counter[tuple[int, int, int, int]]) -> int:
    """
    Compute the total number of edges implied by a multiset of features.

    The degree in the encoding is ``degree_idx = degree - 1``. The sum of all
    target degrees divided by 2 yields the number of undirected edges.

    :param feat_ctr: Counter mapping feature tuples to multiplicities.
    :returns: Total number of edges in the final graph.
    """
    return sum(((deg_idx + 1) * v) for (_, deg_idx, _, _), v in feat_ctr.items()) // 2


def add_node_with_feat(G: nx.Graph, feat: Feat, node_id: int | None = None) -> int:
    """
    Add a node with frozen features.

    :param G: Target graph (modified in place).
    :param feat: Frozen node features.
    :param node_id: Optional explicit node id. If ``None``, uses ``max_id+1`` or ``0``.
    :returns: The node id used.
    """
    if node_id is None:
        node_id = 0 if not G.nodes else (max(G.nodes) + 1)
    G.add_node(node_id, feat=feat, target_degree=feat.target_degree)
    return node_id


def add_node_and_connect(G: nx.Graph, feat: Feat, connect_to: Sequence[int], total_nodes: int) -> int | None:
    """
    Add a node and try to connect it to a set of anchors (greedy, respects residuals).

    :param G: Graph (modified in place).
    :param feat: Features of the new node.
    :param connect_to: Candidate anchor nodes to attempt connections.
    :returns: New node id, or ``None`` if no valid placement (edge constraints violated).
    """
    nid = add_node_with_feat(G, feat)
    return connect_all_if_possible(G, nid, connect_to, total_nodes)


def connect_all_if_possible(G: nx.Graph, nid: int, connect_to: Sequence[int], total_nodes: int) -> int | None:
    """
    Add a node and try to connect it to a set of anchors (greedy, respects residuals).

    :param G: Graph (modified in place).
    :param feat: Features of the new node.
    :param connect_to: Candidate anchor nodes to attempt connections.
    :returns: New node id, or ``None`` if no valid placement (edge constraints violated).
    """
    ok = True
    for a in connect_to:
        if residual_degree(G, nid) <= 0:
            break
        if residual_degree(G, a) <= 0:
            continue
        if not add_edge_if_possible(G, nid, a, strict=True):
            ok = False
            break
    # Don't finish the graph if we still have leftover nodes
    if not ok or (len(anchors(G)) <= 0 and G.number_of_nodes() != total_nodes):
        G.remove_node(nid)
        return None
    return nid


def powerset(iterable):
    """
    Return the power set of the input iterable.

    Example
    -------
    >>> list(powerset([1, 2, 3]))
    [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def wl_hash(G: nx.Graph, *, iters: int = 3) -> str:
    """WL hash that respects `feat`."""
    H = G.copy()
    for n in H.nodes:
        f = H.nodes[n]["feat"]
        H.nodes[n]["__wl_label__"] = ",".join(map(str, f.to_tuple()))
    return nx.weisfeiler_lehman_graph_hash(H, node_attr="__wl_label__", iterations=iters)

def hash(G: nx.Graph) -> tuple[str, int, int]:
    return wl_hash(G), G.number_of_nodes(), G.number_of_edges()

def order_leftovers_by_degree_distinct(ctr: Counter) -> list[tuple[int, int, int, int]]:
    """Unique feature tuples, sorted by final degree (asc), then lexicographically."""
    uniq = list(ctr.keys())
    uniq.sort(key=lambda t: (t[1] + 1, t))
    return uniq

def greedy_oracle_decoder(
    node_multiset: Counter,
    full_g_h: torch.Tensor | None = None,  # [D] final graph hyper vector
    oracle: Oracle | None = None,
    *,
    beam_size: int = 32,
    expand_on_n_anchors: int | None = None,
    draw: bool = False,
    oracle_threshold: float = 0.5,
    strict: bool = True,
    full_g_nx: nx.Graph = None,
    report_cnf_matrix: bool = False,
    use_perfect_oracle: bool = False,
    use_pair_feasibility: bool = False,
) -> list[nx.Graph]:
    full_ctr: Counter = node_multiset.copy()
    total_edges = total_edges_count(full_ctr)
    total_nodes = sum(full_ctr.values())
    print(f"Decoding a graph with {total_nodes} nodes and {total_edges} edges.")
    ys = []
    ps = []



    def _call_oracle(Gs: list[nx.Graph]) -> list[bool]:
        if use_perfect_oracle:
            return perfect_oracle(gs=Gs, full_g_nx=full_g_nx)
        probs = oracle.is_induced_graph(small_gs=Gs, final_h=full_g_h)
        mask = probs.reshape(-1).gt(oracle_threshold)
        return mask.detach().cpu().tolist()

    def _is_valid_final_graph(G: nx.Graph) -> bool:
        node_condition = G.number_of_nodes() == total_nodes
        edge_condition = G.number_of_edges() == total_edges
        leftover_condition = leftover_features(full_ctr, G).total() == 0
        residual_condition = sum(residuals(G).values()) == 0
        return node_condition and edge_condition and leftover_condition and residual_condition

    def _apply_strict_filter(population: list[nx.Graph]) -> list[nx.Graph]:
        return [g for g in population if _is_valid_final_graph(g)]

    # Cache: from a 2-node test we can learn if an edge between two feature types is plausible.
    # Key is an ordered pair of feature tuples (t_small, t_big) with t_small <= t_big.
    pair_ok: dict[tuple[tuple[int, int, int, int], tuple[int, int, int, int]], bool] = {}

    # Global dedup across the whole search (WL hash based)
    global_seen: set = set()

    # ---------------------------
    # 1) Initial population (2-node graphs)
    # ---------------------------

    feat_types = order_leftovers_by_degree_distinct(full_ctr)

    # Trivial case 1
    if total_nodes == 1 and len(feat_types) == 1:
        G = nx.Graph()
        add_node_with_feat(G, Feat.from_tuple(feat_types[0]))
        return [G]

    # Trivial case 2
    if total_nodes == 2:
        G = nx.Graph()
        nodes = list(full_ctr.elements())
        n1 = add_node_with_feat(G, Feat.from_tuple(nodes[0]))
        n2 = add_node_with_feat(G, Feat.from_tuple(nodes[1]))
        if add_edge_if_possible(G, n1, n2) and _is_valid_final_graph(G):
            return [G]
        return []

    # build all distinct ordered pairs (i <= j) where at least one has residual > 0 (deg_idx > 0)
    first_pairs: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    for i in range(len(feat_types)):
        for j in range(i, len(feat_types)):
            u = feat_types[i]
            v = feat_types[j]
            # skip the impossible 0-0 pair (both final degrees == 0)
            if u[1] == 0 and v[1] == 0 and total_nodes > 2:
                continue
            # require multiplicity when u == v
            if u == v and full_ctr[u] < 2 < total_nodes:
                continue
            first_pairs.append((u, v))

    # materialize 2-node candidates
    first_pop: list[nx.Graph] = []
    for u_t, v_t in first_pairs:
        G = nx.Graph()
        uid = add_node_with_feat(G, Feat.from_tuple(u_t))
        if add_node_and_connect(G, Feat.from_tuple(v_t), connect_to=[uid], total_nodes=total_nodes) is None:
            continue

        key = hash(G)
        if key in global_seen:
            continue
        global_seen.add(key)
        first_pop.append(G)
    # print(f"First population size: {len(first_pop)}")

    # oracle filter for 2-node graphs; also learn pair feasibility
    healthy_pop: list[nx.Graph] = []
    oracle_results: list[bool] = _call_oracle(first_pop)
    if full_g_nx is not None:
        perfect_oracle_results: list[bool] = perfect_oracle(gs=first_pop, full_g_nx=full_g_nx)
        assert len(oracle_results) == len(perfect_oracle_results) == len(first_pop)
        if draw:
            for g, (actual, pred) in zip(
                first_pop, zip(perfect_oracle_results, oracle_results, strict=False), strict=False
            ):
                draw_nx_with_atom_colorings(
                    g, label=f"actual: {actual}, pred: {pred}", overlay_full_graph=full_g_nx, overlay_draw_nodes=True
                )
                plt.show()
        y_true = np.asarray(perfect_oracle_results, dtype=bool)
        ys.extend(y_true.tolist())
        y_pred = np.asarray(oracle_results, dtype=bool)
        ps.extend(y_pred.tolist())
        print(f"[population@iter0] Acc {(accuracy_score(y_true, y_pred)) * 100:.2f} at size {len(first_pop)}")

    for i, G in enumerate(first_pop):
        # learn pair feasibility from G (it has 1 or 0 edges)
        if G.number_of_nodes() == 2:
            nodes = list(G.nodes)
            t0 = G.nodes[nodes[0]]["feat"].to_tuple()
            t1 = G.nodes[nodes[1]]["feat"].to_tuple()
            k = tuple(sorted((t0, t1)))
            pair_ok[k] = bool(G.number_of_edges() == 1 and oracle_results[i])
        if oracle_results[i]:
            healthy_pop.append(G)
    # Don't limit the first population
    # print(f"Healthy population size: {len(healthy_pop)}")

    if not healthy_pop:
        return []

    # ---------------------------
    # 2) Iterative expansion
    # ---------------------------

    population = healthy_pop
    # loop until we place all nodes; guard with max_iters to avoid infinite loops
    max_iters = total_nodes + 5
    curr_nodes = 2

    while curr_nodes <= max_iters:
        # Expand each growing candidate by adding ONE new node connected to ONE anchor,
        # and attempting to connect the new node to all the anchors.
        children: list[nx.Graph] = []
        local_seen: set = set()  # per-iteration dedup to keep branching under control

        for G in population:
            leftovers_ctr = leftover_features(full_ctr, G)
            if not leftovers_ctr:
                continue

            # choose types in ascending final degree (distinct types)
            leftover_types = order_leftovers_by_degree_distinct(leftovers_ctr)
            ancrs = anchors(G)
            if not ancrs:
                # cannot expand this candidate
                continue

            for a, lo_t in list(itertools.product(ancrs[:expand_on_n_anchors], leftover_types)):
                a_t = G.nodes[a]["feat"].to_tuple()
                # Early prune using 2-node oracle knowledge (if seen and false)
                k = tuple(sorted((a_t, lo_t)))
                if use_pair_feasibility and k in pair_ok and pair_ok[k] is False:
                    continue

                # First child is adding the leftover to the graph and connecting it to the anchor
                C = G.copy()
                nid = add_node_and_connect(C, Feat.from_tuple(lo_t), connect_to=[a], total_nodes=total_nodes)
                if nid is None or C.number_of_edges() > total_edges:
                    continue

                key = hash(C)
                if key not in global_seen and key not in local_seen:
                    global_seen.add(key)
                    local_seen.add(key)
                    children.append(C)

                # Try and find all the edges that the node has with the rest of the anchors (i.e. closing rings)
                ancrs_rest = [a_ for a_ in ancrs if a_ != a]
                for subset in powerset(ancrs_rest):
                    if len(subset) == 0:
                        continue

                    H = C.copy()
                    nid = connect_all_if_possible(H, nid, connect_to=list(subset), total_nodes=total_nodes)
                    if nid is None or H.number_of_edges() > total_edges:
                        continue

                    key = hash(H)
                    if key not in global_seen and key not in local_seen:
                        global_seen.add(key)
                        local_seen.add(key)
                        children.append(H)

        if not children:
            if full_g_nx is not None and report_cnf_matrix:
                show_confusion_matrix(ys=ys, ps=ps)
            # No more expansions possible
            if strict:
                return _apply_strict_filter(population)
            return population

        # Oracle filter children
        accepted: list[nx.Graph] = []
        oracle_results: list[bool] = _call_oracle(children)
        if full_g_nx is not None:
            perfect_oracle_results: list[bool] = perfect_oracle(gs=children, full_g_nx=full_g_nx)
            assert len(oracle_results) == len(perfect_oracle_results) == len(children)
            if draw:
                for g, (actual, pred) in zip(
                    children, zip(perfect_oracle_results, oracle_results, strict=False), strict=False
                ):
                    draw_nx_with_atom_colorings(
                        g,
                        label=f"actual: {actual}, pred: {pred}",
                        overlay_full_graph=full_g_nx,
                        overlay_draw_nodes=True,
                    )
                    plt.show()
            y_true = np.asarray(perfect_oracle_results, dtype=bool)
            ys.extend(y_true.tolist())
            y_pred = np.asarray(oracle_results, dtype=bool)
            ps.extend(y_pred.tolist())
            print(
                f"[population@iter{curr_nodes}] Acc: {(accuracy_score(y_true, y_pred)) * 100:.2f}% at size {len(children)}"
            )
        for i, H in enumerate(children):
            if oracle_results[i]:
                accepted.append(H)
                if len(accepted) >= beam_size:
                    break

        if not accepted:
            if full_g_nx is not None and report_cnf_matrix:
                show_confusion_matrix(ys=ys, ps=ps)
            # Oracle rejected all -> stop with the current population
            if strict:
                return _apply_strict_filter(population)
            return population

        # Next generation
        population = accepted
        curr_nodes += 1

    # Safeguard exit
    if full_g_nx is not None and report_cnf_matrix:
        show_confusion_matrix(ys=ys, ps=ps)
    if strict:
        return _apply_strict_filter(population)
    return population


def greedy_oracle_decoder_faster(
    node_multiset: Counter,
    full_g_h: torch.Tensor,  # [D] final graph hyper vector
    oracle: Oracle,
    *,
    beam_size: int = 32,
    expand_on_n_anchors: int | None = None,
    oracle_threshold: float = 0.5,
    strict: bool = True,
    use_pair_feasibility: bool = False,
) -> list[nx.Graph]:
    full_ctr: Counter = node_multiset.copy()
    total_edges = total_edges_count(full_ctr)
    total_nodes = sum(full_ctr.values())
    print(f"Decoding a graph with {total_nodes} nodes and {total_edges} edges.")

    # -- local helpers --
    def _wl_hash(G: nx.Graph, *, iters: int = 3) -> str:
        """WL hash that respects `feat`."""
        H = G.copy()
        for n in H.nodes:
            f = H.nodes[n]["feat"]
            H.nodes[n]["__wl_label__"] = ",".join(map(str, f.to_tuple()))
        return nx.weisfeiler_lehman_graph_hash(H, node_attr="__wl_label__", iterations=iters)

    def _hash(G: nx.Graph) -> tuple[str, int, int]:
        return _wl_hash(G), G.number_of_nodes(), G.number_of_edges()

    def _order_leftovers_by_degree_distinct(ctr: Counter) -> list[tuple[int, int, int, int]]:
        """Unique feature tuples, sorted by final degree (asc), then lexicographically."""
        uniq = list(ctr.keys())
        uniq.sort(key=lambda t: (t[1] + 1, t))
        return uniq

    def _call_oracle(Gs: list[nx.Graph]) -> tuple[list[int], list[int]]:
        probs = oracle.is_induced_graph(small_gs=Gs, final_h=full_g_h).flatten()  # torch.sigmoid(logits)
        sorted_indices = torch.argsort(probs, descending=True).tolist()
        return probs.tolist(), sorted_indices

    def _is_valid_final_graph(G: nx.Graph) -> bool:
        node_condition = G.number_of_nodes() == total_nodes
        edge_condition = G.number_of_edges() == total_edges
        leftover_condition = leftover_features(full_ctr, G).total() == 0
        residual_condition = sum(residuals(G).values()) == 0
        return node_condition and edge_condition and leftover_condition and residual_condition

    def _apply_strict_filter(population: list[nx.Graph]) -> list[nx.Graph]:
        return [g for g in population if _is_valid_final_graph(g)]

    # Cache: from a 2-node test we can learn if an edge between two feature types is plausible.
    # Key is an ordered pair of feature tuples (t_small, t_big) with t_small <= t_big.
    pair_ok: dict[tuple[tuple[int, int, int, int], tuple[int, int, int, int]], bool] = {}

    # Global dedup across the whole search (WL hash based)
    global_seen: set = set()

    # ---------------------------
    # 1) Initial population (2-node graphs)
    # ---------------------------

    feat_types = _order_leftovers_by_degree_distinct(full_ctr)

    # Trivial case 1
    if total_nodes == 1 and len(feat_types) == 1:
        G = nx.Graph()
        add_node_with_feat(G, Feat.from_tuple(feat_types[0]))
        return [G]

    # Trivial case 2
    if total_nodes == 2:
        G = nx.Graph()
        nodes = list(full_ctr.elements())
        n1 = add_node_with_feat(G, Feat.from_tuple(nodes[0]))
        n2 = add_node_with_feat(G, Feat.from_tuple(nodes[1]))
        if add_edge_if_possible(G, n1, n2) and _is_valid_final_graph(G):
            return [G]
        return []

    # build all distinct ordered pairs (i <= j) where at least one has residual > 0 (deg_idx > 0)
    first_pairs: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    for i in range(len(feat_types)):
        for j in range(i, len(feat_types)):
            u = feat_types[i]
            v = feat_types[j]
            # skip the impossible 0-0 pair (both final degrees == 0)
            if u[1] == 0 and v[1] == 0 and total_nodes > 2:
                continue
            # require multiplicity when u == v
            if u == v and full_ctr[u] < 2 < total_nodes:
                continue
            first_pairs.append((u, v))

    # materialize 2-node candidates
    first_pop: list[nx.Graph] = []
    for u_t, v_t in first_pairs:
        G = nx.Graph()
        uid = add_node_with_feat(G, Feat.from_tuple(u_t))
        if add_node_and_connect(G, Feat.from_tuple(v_t), connect_to=[uid], total_nodes=total_nodes) is None:
            continue

        key = _hash(G)
        if key in global_seen:
            continue
        global_seen.add(key)
        first_pop.append(G)
    # print(f"First population size: {len(first_pop)}")

    # oracle filter for 2-node graphs; also learn pair feasibility
    healthy_pop: list[nx.Graph] = []
    oracle_results, _ = _call_oracle(first_pop)
    for i, G in enumerate(first_pop):
        # learn pair feasibility from G (it has 1 or 0 edges)
        if G.number_of_nodes() == 2:
            nodes = list(G.nodes)
            t0 = G.nodes[nodes[0]]["feat"].to_tuple()
            t1 = G.nodes[nodes[1]]["feat"].to_tuple()
            k = tuple(sorted((t0, t1)))
            pair_ok[k] = bool(G.number_of_edges() == 1 and oracle_results[i] >= oracle_threshold)
        if oracle_results[i] >= oracle_threshold:
            healthy_pop.append(G)
    # Don't limit the first population
    # print(f"Healthy population size: {len(healthy_pop)}")

    if not healthy_pop:
        return []

    # ---------------------------
    # 2) Iterative expansion
    # ---------------------------

    population = healthy_pop
    # loop until we place all nodes; guard with max_iters to avoid infinite loops
    max_iters = total_nodes + 5
    curr_nodes = 2

    while curr_nodes <= max_iters:
        # Expand each growing candidate by adding ONE new node connected to ONE anchor,
        # and attempting to connect the new node to all the anchors.
        children: list[nx.Graph] = []
        local_seen: set = set()  # per-iteration dedup to keep branching under control

        for G in population:
            leftovers_ctr = leftover_features(full_ctr, G)
            if not leftovers_ctr:
                continue

            # choose types in ascending final degree (distinct types)
            leftover_types = _order_leftovers_by_degree_distinct(leftovers_ctr)
            ancrs = anchors(G)
            if not ancrs:
                # cannot expand this candidate
                continue

            for a, lo_t in list(itertools.product(ancrs[:expand_on_n_anchors], leftover_types)):
                a_t = G.nodes[a]["feat"].to_tuple()
                # Early prune using 2-node oracle knowledge (if seen and false)
                k = tuple(sorted((a_t, lo_t)))
                if use_pair_feasibility and k in pair_ok and pair_ok[k] is False:
                    continue

                # First child is adding the leftover to the graph and connecting it to the anchor
                C = G.copy()
                nid = add_node_and_connect(C, Feat.from_tuple(lo_t), connect_to=[a], total_nodes=total_nodes)
                if nid is None or C.number_of_edges() > total_edges:
                    continue

                key = _hash(C)
                if key not in global_seen and key not in local_seen:
                    global_seen.add(key)
                    local_seen.add(key)
                    children.append(C)

                # Try and find all the edges that the node has with the rest of the anchors (i.e. closing rings)
                ancrs_rest = [a_ for a_ in ancrs if a_ != a]
                for subset in powerset(ancrs_rest):
                    if len(subset) == 0:
                        continue

                    H = C.copy()
                    nid = connect_all_if_possible(H, nid, connect_to=list(subset), total_nodes=total_nodes)
                    if nid is None or H.number_of_edges() > total_edges:
                        continue

                    key = _hash(H)
                    if key not in global_seen and key not in local_seen:
                        global_seen.add(key)
                        local_seen.add(key)
                        children.append(H)

        if not children:
            # No more expansions possible
            if strict:
                return _apply_strict_filter(population)
            return population

        # Oracle filter children
        accepted: list[nx.Graph] = []
        oracle_results, sorted_idx = _call_oracle(children)
        for idx in sorted_idx[:beam_size]:
            if oracle_results[idx] >= oracle_threshold:
                accepted.append(children[idx])

        if not accepted:
            # Oracle rejected all -> stop with the current population
            if strict:
                return _apply_strict_filter(population)
            return population

        # Next generation
        population = accepted
        curr_nodes += 1

    # Safeguard exit
    if strict:
        return _apply_strict_filter(population)
    return population
