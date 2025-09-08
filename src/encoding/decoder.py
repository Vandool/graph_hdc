from collections import Counter
from collections.abc import Sequence
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import torch

from src.encoding.oracles import Oracle
from src.encoding.the_types import Feat
from src.utils import visualisations


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


def add_node_and_connect(G: nx.Graph, feat: Feat, connect_to: Sequence[int]) -> int | None:
    """
    Add a node and try to connect it to a set of anchors (greedy, respects residuals).

    :param G: Graph (modified in place).
    :param feat: Features of the new node.
    :param connect_to: Candidate anchor nodes to attempt connections.
    :returns: New node id, or ``None`` if no valid placement (edge constraints violated).
    """
    nid = add_node_with_feat(G, feat)
    ok = True
    for a in connect_to:
        if residual_degree(G, nid) <= 0:
            break
        if residual_degree(G, a) <= 0:
            continue
        if not add_edge_if_possible(G, nid, a, strict=True):
            ok = False
            break
    if not ok:
        G.remove_node(nid)
        return None
    return nid


def greedy_oracle_decoder(
    node_multiset: Counter,
    full_g_h: torch.Tensor,  # [D] final graph hyper vector
    oracle: Oracle,
    *,
    beam_size: int = 32,
    draw: bool = False,
    oracle_threshold: float = 0.5,
    strict: bool = True,
) -> list[nx.Graph]:
    """Greedy/beam search decoder outline using an oracle.

    :param full_g_h: final graph hypervector
    :param oracle_threshold: threshold to mark a prob positive
    :param draw: if draw the sub graphs
    :param node_multiset: Multiset of target node features (final graph nodes).
    :param oracle: Callable that scores/filters a candidate. Prefer passing a closure
                   that captures the truth graph, so it can be called as `oracle(G)`.
                   If your oracle has signature `(G, truth)`, this function will try
                   `oracle(G, full_graph_nx)` where `full_graph_nx` must exist globally.
    :param beam_size: Max population size to keep each iteration.
    :return: list of candidate graphs (ideally size 1 at the end).
    """
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

    def _dedup_key(G: nx.Graph) -> tuple[str, int, int]:
        return _wl_hash(G), G.number_of_nodes(), G.number_of_edges()

    def _order_leftovers_by_degree_distinct(ctr: Counter) -> list[tuple[int, int, int, int]]:
        """Unique feature tuples, sorted by final degree (asc), then lexicographically."""
        uniq = list(ctr.keys())
        uniq.sort(key=lambda t: (t[1] + 1, t))
        return uniq

    def _call_oracle(Gs: list[nx.Graph]) -> list[bool]:
        probs = oracle.is_induced_graph(small_gs=Gs, final_h=full_g_h)
        # print(probs)
        return (probs > oracle_threshold).tolist()

    def _apply_strict_filter(population: list[nx.Graph]) -> list[nx.Graph]:
        return list(
            filter(lambda x: (x.number_of_nodes() == total_nodes and x.number_of_edges() == total_edges), population)
        )

    # Cache: from a 2-node test we can learn if an edge between two feature types is plausible.
    # Key is an ordered pair of feature tuples (t_small, t_big) with t_small <= t_big.
    pair_ok: dict[tuple[tuple[int, int, int, int], tuple[int, int, int, int]], bool] = {}

    # Global dedup across the whole search (WL hash based)
    global_seen: set = set()

    # ---------------------------
    # 1) Initial population (2-node graphs)
    # ---------------------------

    feat_types = _order_leftovers_by_degree_distinct(full_ctr)

    # build all distinct ordered pairs (i <= j) where at least one has residual > 0 (deg_idx > 0)
    first_pairs: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    for i in range(len(feat_types)):
        for j in range(i, len(feat_types)):
            u = feat_types[i]
            v = feat_types[j]
            # skip impossible 0-0 pair (both final degree == 0)
            if u[1] == 0 and v[1] == 0:
                continue
            # require multiplicity when u == v
            if u == v and full_ctr[u] < 2:
                continue
            first_pairs.append((u, v))

    # materialize 2-node candidates
    first_pop: list[nx.Graph] = []
    for u_t, v_t in first_pairs:
        G = nx.Graph()
        uid = add_node_with_feat(G, Feat.from_tuple(u_t))
        _ = add_node_and_connect(G, Feat.from_tuple(v_t), connect_to=[uid])  # may attach or fail
        # if attachment failed (shouldn't, unless deg==0 on v_t), ensure at least nodes exist
        if G.number_of_nodes() < 2:
            vid = add_node_with_feat(G, Feat.from_tuple(v_t))
            # try to connect if feasible
            add_edge_if_possible(G, uid, vid, strict=True)

        key = _dedup_key(G)
        if key in global_seen:
            continue
        if len(first_pop) >= beam_size:
            continue
        global_seen.add(key)
        first_pop.append(G)
    # print(f"First population size: {len(first_pop)}")

    # oracle filter for 2-node graphs; also learn pair feasibility
    healthy_pop: list[nx.Graph] = []
    oracle_results: list[bool] = _call_oracle(first_pop)
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
    iters = 0

    while iters < max_iters:
        iters += 1
        # print(f"Iteration: {iters} | max pop: {len(population)}")
        if draw:
            print("Drawings...")
            for G in population:
                _ = visualisations.draw_nx_with_atom_colorings(G)
                plt.show()

        # Check if any candidate already has all nodes placed
        done = []
        still_growing = []
        for G in population:
            left = leftover_features(full_ctr, G)
            if not left:
                done.append(G)
            else:
                still_growing.append(G)

        # If we have any 'done' graphs, try to finish edges if needed (optional) and return them
        if done:
            # If desired, you can attempt to add remaining edges among anchors here.
            # For now, return those that don't exceed the total edge budget.
            finals = [g for g in done if g.number_of_edges() <= total_edges]

            # Prefer fully saturated (all residuals zero)
            def _all_saturated(g: nx.Graph) -> bool:
                return all((g.nodes[n]["target_degree"] - g.degree[n]) == 0 for n in g.nodes)

            finals.sort(key=lambda g: (_all_saturated(g), g.number_of_edges()), reverse=True)
            return finals[:beam_size]

        # Expand each growing candidate by adding ONE new node connected to ONE anchor,
        # then optionally one extra edge from the new node to another anchor.
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

            # Enumerate attachments
            for a in ancrs:
                a_t = G.nodes[a]["feat"].to_tuple()
                for lo_t in leftover_types:
                    # Early prune using 2-node oracle knowledge (if seen and false)
                    k = tuple(sorted((a_t, lo_t)))
                    if k in pair_ok and pair_ok[k] is False:
                        continue

                    H = G.copy()
                    nid = add_node_and_connect(H, Feat.from_tuple(lo_t), connect_to=[a])
                    if nid is None:
                        continue

                    # Do not exceed target total edges
                    if H.number_of_edges() > total_edges:
                        continue

                    key = _dedup_key(H)
                    if key not in global_seen and key not in local_seen:
                        global_seen.add(key)
                        local_seen.add(key)
                        children.append(H)

                    # Optionally try adding ONE more edge from new node to another anchor
                    # (keeps branching modest; you can widen this later if needed)
                    H_anchors = anchors(H)
                    for b in H_anchors:
                        if b == nid:
                            continue
                        H2 = H.copy()
                        if add_edge_if_possible(H2, b, nid, strict=True):
                            if H2.number_of_edges() <= total_edges:
                                key2 = _dedup_key(H2)
                                if key2 not in global_seen and key2 not in local_seen:
                                    global_seen.add(key2)
                                    local_seen.add(key2)
                                    children.append(H2)

        if not children:
            # No more expansions possible
            if strict:
                return _apply_strict_filter(population)
            return population

        # Oracle filter children
        accepted: list[nx.Graph] = []
        oracle_results: list[bool] = _call_oracle(children)
        for i, H in enumerate(children):
            if oracle_results[i]:
                accepted.append(H)
                if len(accepted) >= beam_size:
                    break

        if not accepted:
            # Oracle rejected all; stop with current population
            if strict:
                return _apply_strict_filter(population)
            return population

        # Next generation
        population = accepted

    # Safeguard exit
    if strict:
        return _apply_strict_filter(population)
    return population
