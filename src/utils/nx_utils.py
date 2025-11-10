from collections import Counter
from collections.abc import Sequence
from itertools import chain, combinations

import networkx as nx
from networkx.algorithms.isomorphism import (
    GraphMatcher,
    categorical_edge_match,
    categorical_node_match,
)

from src.encoding.the_types import Feat


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


def  residual_degree(G: nx.Graph, node: int) -> int:
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


def _wl_hash(G: nx.Graph, *, iters: int = 3) -> str:
    """WL hash that respects `feat`."""
    H = G.copy()
    for n in H.nodes:
        f = H.nodes[n]["feat"]
        H.nodes[n]["__wl_label__"] = ",".join(map(str, f.to_tuple()))
    return nx.weisfeiler_lehman_graph_hash(H, node_attr="__wl_label__", iterations=iters)


def _hash(G: nx.Graph) -> tuple[str, int, int]:
    return _wl_hash(G), G.number_of_nodes(), G.number_of_edges()


def order_leftovers_by_degree_distinct(ctr: Counter) -> list[tuple[int, int, int, int]]:
    """Unique feature tuples, sorted by final degree (asc), then lexicographically."""
    uniq = list(ctr.keys())
    uniq.sort(key=lambda t: (t[1] + 1, t))
    return uniq
