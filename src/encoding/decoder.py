import copy
import itertools
import logging
from collections import Counter, deque

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torchhd
from sklearn.metrics import (
    accuracy_score,
)
from torch_geometric.data import Batch
from tqdm import tqdm

from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import QM9_SMILES_HRR_1600_CONFIG_F64, ZINC_SMILES_HRR_7744_CONFIG
from src.encoding.graph_encoders import load_or_create_hypernet
from src.encoding.oracles import Oracle, SimpleVoterOracle
from src.encoding.the_types import Feat
from src.utils.nx_utils import (
    _hash,
    add_edge_if_possible,
    add_node_and_connect,
    add_node_with_feat,
    anchors,
    connect_all_if_possible,
    leftover_features,
    order_leftovers_by_degree_distinct,
    perfect_oracle,
    powerset,
    residual_degree,
    residuals,
    total_edges_count,
)
from src.utils.utils import DataTransformer, TupleIndexer, show_confusion_matrix
from src.utils.visualisations import draw_nx_with_atom_colorings


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


# configure your logger once in your app:
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def greedy_oracle_decoder_faster(
    node_multiset: Counter,
    full_g_h: torch.Tensor,  # [D] final graph hyper vector
    oracle: "Oracle",
    *,
    beam_size: int = 32,
    expand_on_n_anchors: int | None = None,
    oracle_threshold: float = 0.5,
    strict: bool = False,
    use_pair_feasibility: bool = False,
    skip_n_nodes: int = 0,
    trace_back_settings: dict[str, int] | None = None,
) -> tuple[list[nx.Graph], list[bool]]:
    full_ctr: Counter = node_multiset.copy()
    total_edges = total_edges_count(full_ctr)
    total_nodes = sum(full_ctr.values())

    ## Guard the decoder
    if (skip_n_nodes and total_nodes > skip_n_nodes) or total_edges > skip_n_nodes:
        return [nx.Graph()], [False]

    # By default no trace back
    trace_back_setting = trace_back_settings or {
        "beam_size_multiplier": 1,
        "trace_back_attempts": 0,
        "agitated_rounds": 0,  # how many rounds after applying trace back keep the beam size larger
    }

    def _order_leftovers_by_degree_distinct(ctr: Counter) -> list[tuple[int, int, int, int]]:
        """Unique feature tuples, sorted by final degree (asc), then lexicographically."""
        uniq = list(ctr.keys())
        uniq.sort(key=lambda t: (t[1] + 1, t))
        return uniq

    def _call_oracle(Gs: list[nx.Graph]) -> tuple[list[float], list[int]]:
        probs = oracle.is_induced_graph(small_gs=Gs, final_h=full_g_h).flatten()  # torch.sigmoid(logits)
        sorted_indices = torch.argsort(probs, descending=True).tolist()
        return probs.tolist(), sorted_indices

    def _is_valid_final_graph(G: nx.Graph) -> bool:
        node_condition = G.number_of_nodes() == total_nodes
        edge_condition = G.number_of_edges() == total_edges
        leftover_condition = leftover_features(full_ctr, G).total() == 0
        # Even the dataset itself has some radical atoms, we should not enforce this.
        residual_condition = sum(residuals(G).values()) == 0 or True
        return node_condition and edge_condition and leftover_condition and residual_condition

    def _generate_response(population: list[nx.Graph]) -> tuple[list[nx.Graph], list[bool]]:
        final_flags = [_is_valid_final_graph(g) for g in population]
        if strict:
            if sum(final_flags) == 0:
                return [nx.Graph()], [False]
            return [g for g, final in zip(population, final_flags, strict=True) if final], [True] * sum(final_flags)
        return population, final_flags

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
        return [G], [True]

    # Trivial case 2
    if total_nodes == 2:
        G = nx.Graph()
        nodes = list(full_ctr.elements())
        n1 = add_node_with_feat(G, Feat.from_tuple(nodes[0]))
        n2 = add_node_with_feat(G, Feat.from_tuple(nodes[1]))
        ok = add_edge_if_possible(G, n1, n2)
        if ok and _is_valid_final_graph(G):
            return [G], [True]
        return [nx.Graph()], [False]

    # build all distinct ordered pairs (i <= j) where at least one has residual > 0 (deg_idx > 0)
    first_pairs: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    for i in range(len(feat_types)):
        for j in range(i, len(feat_types)):
            u = feat_types[i]
            v = feat_types[j]
            if u[1] == 0 and v[1] == 0 and total_nodes > 2:
                continue
            if u == v and full_ctr[u] < 2 < total_nodes:
                continue
            first_pairs.append((u, v))

    # materialize 2-node candidates
    first_pop: list[nx.Graph] = []
    for k, (u_t, v_t) in enumerate(first_pairs):
        G = nx.Graph()
        uid = add_node_with_feat(G, Feat.from_tuple(u_t))
        ok = add_node_and_connect(G, Feat.from_tuple(v_t), connect_to=[uid], total_nodes=total_nodes) is not None
        if not ok:
            continue
        key = _hash(G)
        if key in global_seen:
            continue
        global_seen.add(key)
        first_pop.append(G)

    # oracle filter for 2-node graphs; also learn pair feasibility
    healthy_pop: list[nx.Graph] = []
    oracle_results, _ = _call_oracle(first_pop)
    for i, G in enumerate(first_pop):
        if G.number_of_nodes() == 2:
            nodes = list(G.nodes)
            tup0 = G.nodes[nodes[0]]["feat"].to_tuple()
            tup1 = G.nodes[nodes[1]]["feat"].to_tuple()
            kk = tuple(sorted((tup0, tup1)))
            pair_ok[kk] = bool(G.number_of_edges() == 1 and oracle_results[i] >= oracle_threshold)
        if oracle_results[i] >= oracle_threshold:
            healthy_pop.append(G)

    if not healthy_pop:
        return [nx.Graph()], [False]

    # ---------------------------
    # 2) Iterative expansion
    # ---------------------------

    # Track back settings
    beam_size_growth_factor = trace_back_setting["beam_size_multiplier"]
    trace_back_attempts_left_max = trace_back_setting["trace_back_attempts"]
    initial_agitated_rounds = trace_back_setting["agitated_rounds"]
    agitated_rounds = 0
    initial_beam_size = beam_size

    keep_history = trace_back_attempts_left_max > 0
    history: list[tuple[int, list[nx.Graph]]] = []
    # Worst case history size, so we cap it
    history_cap = (2**trace_back_attempts_left_max) * beam_size + beam_size
    trace_back_attempts_left = trace_back_attempts_left_max
    is_back_tracing = False

    population = healthy_pop
    # loop until we place all nodes; guard with max_iters to avoid infinite loops
    max_iters = total_nodes
    curr_nodes = 2

    last_none_empty_pop = None
    with tqdm(total=max_iters - 2, desc="Iterations", unit="node") as pbar:
        while curr_nodes < max_iters:
            children: list[nx.Graph] = []
            local_seen: set = set()  # per-iteration dedup to keep branching under control

            for gi, G in enumerate(population):
                leftovers_ctr = leftover_features(full_ctr, G)
                if not leftovers_ctr:
                    continue

                leftover_types = _order_leftovers_by_degree_distinct(leftovers_ctr)
                ancrs = anchors(G)
                if not ancrs:
                    continue

                # how many anchors we’ll actually use (slice semantics preserved)
                inner_iter = 0
                for a, lo_t in list(itertools.product(ancrs[:expand_on_n_anchors], leftover_types)):
                    inner_iter += 1
                    a_t = G.nodes[a]["feat"].to_tuple()
                    kpair = tuple(sorted((a_t, lo_t)))
                    if use_pair_feasibility and kpair in pair_ok and pair_ok[kpair] is False:
                        continue

                    C = G.copy()
                    nid = add_node_and_connect(C, Feat.from_tuple(lo_t), connect_to=[a], total_nodes=total_nodes)
                    if nid is None:
                        continue
                    if C.number_of_edges() > total_edges:
                        continue

                    keyC = _hash(C)
                    if keyC in global_seen or keyC in local_seen:
                        continue
                    global_seen.add(keyC)
                    local_seen.add(keyC)
                    children.append(C)

                    ancrs_rest = [a_ for a_ in ancrs if a_ != a]

                    sub_counter = 0
                    for subset in powerset(ancrs_rest):
                        if len(subset) == 0:
                            continue
                        sub_counter += 1

                        H = C.copy()
                        new_nid = connect_all_if_possible(H, nid, connect_to=list(subset), total_nodes=total_nodes)
                        if new_nid is None:
                            continue
                        if H.number_of_edges() > total_edges:
                            continue

                        keyH = _hash(H)
                        if keyH in global_seen or keyH in local_seen:
                            continue
                        global_seen.add(keyH)
                        local_seen.add(keyH)
                        children.append(H)

            if not children:
                return _generate_response(population)

            oracle_results, sorted_idx = _call_oracle(children)
            accepted: list[nx.Graph] = [children[idx] for idx in sorted_idx if oracle_results[idx] >= oracle_threshold]

            if not accepted:
                if trace_back_attempts_left:
                    is_back_tracing = True
                    beam_size = initial_beam_size
                    agitated_rounds = initial_agitated_rounds
                    trace_back_attempts_left -= 1

                    if len(history) <= 1:
                        # not enough history -> abort gracefully
                        return _generate_response(population)

                    # history[-1] is the most recent completed iteration
                    # Find the latest history entry that has more than beam size population members,
                    tb_iter_idx, tb_population_full = None, None
                    # Skip the last one, because it's the current iteration!
                    for i, p in reversed(history[:-1]):
                        if len(p) >= beam_size:
                            tb_iter_idx, tb_population_full = i, p
                            break

                    if tb_population_full is None:
                        return _generate_response(population)

                    # optional slicing window on that older population
                    last_none_empty_pop = population
                    slice_start = beam_size
                    slice_end = slice_start + beam_size_growth_factor * beam_size
                    population = tb_population_full[slice_start:slice_end]

                    # move the logical counter back to that iteration index
                    curr_nodes = tb_iter_idx

                    # tqdm: show traceback phase + move bar backward
                    pbar.n = curr_nodes
                    pbar.set_postfix_str("TRACEBACK")
                    pbar.refresh()
                    continue

                # No lives left (default path)
                population = population if len(population) > 1 else last_none_empty_pop
                return _generate_response(population)

            if is_back_tracing:
                is_back_tracing = False
                print(f"BACK TRACING SUCCEEDED -> New healthy population: {len(accepted)}")

            # store snapshot of the *accepted* population for this completed iteration
            if keep_history:
                history.append((curr_nodes, copy.deepcopy(accepted[:history_cap])))

            if agitated_rounds > 0:
                beam_size = initial_beam_size * beam_size_growth_factor
                pbar.set_postfix_str("AGITATED FORWARD")
                agitated_rounds -= 1
            else:
                pbar.set_postfix_str("FORWARD")

            population = accepted[:beam_size]
            curr_nodes += 1
            pbar.refresh()
            pbar.update(1)

    return _generate_response(population)


def greedy_oracle_decoder_voter_oracle(
    node_multiset: Counter,
    full_g_h: torch.Tensor,  # [D] final graph hyper vector
    oracle: SimpleVoterOracle,
    *,
    beam_size: int = 32,
    expand_on_n_anchors: int | None = None,
    strict: bool = False,
    use_pair_feasibility: bool = False,
    skip_n_nodes: int = 0,
    trace_back_setting: dict[str, int] | None = None,
) -> tuple[list[nx.Graph], bool]:
    full_ctr: Counter = node_multiset.copy()
    total_edges = total_edges_count(full_ctr)
    total_nodes = sum(full_ctr.values())

    # By default no trace back
    trace_back_setting = trace_back_setting or {
        "beam_size_multiplier": 1,
        "trace_back_attempts": 0,
        "trace_back_to_last_nth_iter": 0,
    }

    ## Guard the decoder
    if (skip_n_nodes and total_nodes > skip_n_nodes) or total_edges > skip_n_nodes:
        return [nx.Graph()], False

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

    def _call_oracle(Gs: list[nx.Graph]) -> list[bool]:
        probs = oracle.is_induced_graph(small_gs=Gs, final_h=full_g_h).flatten()
        return probs.tolist()

    def _is_valid_final_graph(G: nx.Graph) -> bool:
        node_condition = G.number_of_nodes() == total_nodes
        edge_condition = G.number_of_edges() == total_edges
        leftover_condition = leftover_features(full_ctr, G).total() == 0
        # Even the dataset itself has some radical atoms, we should not enforce this.
        residual_condition = sum(residuals(G).values()) == 0 or True
        return node_condition and edge_condition and leftover_condition and residual_condition

    def _apply_strict_filter(population: list[nx.Graph]) -> tuple[list[nx.Graph], bool]:
        final_graphs = [g for g in population if _is_valid_final_graph(g)]
        are_final = len(final_graphs) > 0
        if not are_final:
            final_graphs.append(nx.Graph())
        return final_graphs, are_final

    def _generate_response(population: list[nx.Graph]) -> tuple[list[nx.Graph], bool]:
        final_graphs, are_final = _apply_strict_filter(population=population)
        if strict:
            return final_graphs, are_final
        return population, are_final

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
        return [G], True

    # Trivial case 2
    if total_nodes == 2:
        G = nx.Graph()
        nodes = list(full_ctr.elements())
        n1 = add_node_with_feat(G, Feat.from_tuple(nodes[0]))
        n2 = add_node_with_feat(G, Feat.from_tuple(nodes[1]))
        ok = add_edge_if_possible(G, n1, n2)
        if ok and _is_valid_final_graph(G):
            return [G], True
        return [nx.Graph()], False

    # build all distinct ordered pairs (i <= j) where at least one has residual > 0 (deg_idx > 0)
    first_pairs: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    for i in range(len(feat_types)):
        for j in range(i, len(feat_types)):
            u = feat_types[i]
            v = feat_types[j]
            if u[1] == 0 and v[1] == 0 and total_nodes > 2:
                continue
            if u == v and full_ctr[u] < 2 < total_nodes:
                continue
            first_pairs.append((u, v))

    # materialize 2-node candidates
    first_pop: list[nx.Graph] = []
    for k, (u_t, v_t) in enumerate(first_pairs):
        G = nx.Graph()
        uid = add_node_with_feat(G, Feat.from_tuple(u_t))
        ok = add_node_and_connect(G, Feat.from_tuple(v_t), connect_to=[uid], total_nodes=total_nodes) is not None
        if not ok:
            continue
        key = _hash(G)
        if key in global_seen:
            continue
        global_seen.add(key)
        first_pop.append(G)

    # oracle filter for 2-node graphs; also learn pair feasibility
    healthy_pop: list[nx.Graph] = []
    oracle_results = _call_oracle(first_pop)
    for i, G in enumerate(first_pop):
        if G.number_of_nodes() == 2:
            nodes = list(G.nodes)
            tup0 = G.nodes[nodes[0]]["feat"].to_tuple()
            tup1 = G.nodes[nodes[1]]["feat"].to_tuple()
            kk = tuple(sorted((tup0, tup1)))
            pair_ok[kk] = bool(G.number_of_edges() == 1 and oracle_results[i])
        if oracle_results[i]:
            healthy_pop.append(G)

    if not healthy_pop:
        return [nx.Graph()], False

    # ---------------------------
    # 2) Iterative expansion
    # ---------------------------

    # Track back settings
    beam_size_growth_factor = trace_back_setting["beam_size_multiplier"]
    trace_back_to_last_nth_iter = trace_back_setting["trace_back_to_last_nth_iter"]
    trace_back_attempts_left_max = trace_back_setting["trace_back_attempts"]

    history = deque(maxlen=total_nodes)
    trace_back_attempts_left = trace_back_attempts_left_max
    is_back_tracing = False

    population = healthy_pop
    # loop until we place all nodes; guard with max_iters to avoid infinite loops
    max_iters = total_nodes
    curr_nodes = 2

    with tqdm(total=max_iters - 2, desc="Iterations", unit="node") as pbar:
        while curr_nodes < max_iters:
            children: list[nx.Graph] = []
            local_seen: set = set()  # per-iteration dedup to keep branching under control

            for gi, G in enumerate(population):
                leftovers_ctr = leftover_features(full_ctr, G)
                if not leftovers_ctr:
                    continue

                leftover_types = _order_leftovers_by_degree_distinct(leftovers_ctr)
                ancrs = anchors(G)
                if not ancrs:
                    continue

                # how many anchors we’ll actually use (slice semantics preserved)
                inner_iter = 0
                for a, lo_t in list(itertools.product(ancrs[:expand_on_n_anchors], leftover_types)):
                    inner_iter += 1
                    a_t = G.nodes[a]["feat"].to_tuple()
                    kpair = tuple(sorted((a_t, lo_t)))
                    if use_pair_feasibility and kpair in pair_ok and pair_ok[kpair] is False:
                        continue

                    C = G.copy()
                    nid = add_node_and_connect(C, Feat.from_tuple(lo_t), connect_to=[a], total_nodes=total_nodes)
                    if nid is None:
                        continue
                    if C.number_of_edges() > total_edges:
                        continue

                    keyC = _hash(C)
                    if keyC in global_seen or keyC in local_seen:
                        continue
                    global_seen.add(keyC)
                    local_seen.add(keyC)
                    children.append(C)

                    ancrs_rest = [a_ for a_ in ancrs if a_ != a]

                    sub_counter = 0
                    for subset in powerset(ancrs_rest):
                        if len(subset) == 0:
                            continue
                        sub_counter += 1

                        H = C.copy()
                        new_nid = connect_all_if_possible(H, nid, connect_to=list(subset), total_nodes=total_nodes)
                        if new_nid is None:
                            continue
                        if H.number_of_edges() > total_edges:
                            continue

                        keyH = _hash(H)
                        if keyH in global_seen or keyH in local_seen:
                            continue
                        global_seen.add(keyH)
                        local_seen.add(keyH)
                        children.append(H)

            if not children:
                return _generate_response(population)

            oracle_results = _call_oracle(children)
            accepted: list[nx.Graph] = [c for c, o in zip(children, oracle_results, strict=True) if o][:beam_size]

            if not accepted:
                return _generate_response(population)

            population = accepted
            curr_nodes += 1
            pbar.update(1)

    return _generate_response(population)


def new_decoder(nodes_multiset: Counter, edge_terms, encoder, graph_terms):
    ## First get the node multiset
    num_edges = sum([(e_idx + 1) * n for (_, e_idx, _, _), n in nodes_multiset.items()])

    n_count = nodes_multiset.total()
    e_count = num_edges
    print(n_count)
    print(e_count)

    node_idxs = encoder.nodes_indexer.get_idxs(nodes_multiset.keys())
    idx_tensor = torch.tensor(node_idxs, dtype=torch.long, device=torch.device("cpu"))
    cb = encoder.nodes_codebook[idx_tensor]
    print(cb.shape)

    residuals = {k: (k[1] + 1) * n * 2 for k, n in nodes_multiset.items()}  # how much each node can spend on edges
    # edges_left = [(node_tuples[a], node_tuples[b]) for a, b in edge_tuples]
    result = Counter()
    print(f"ITERATIONS: {e_count}")
    result_edge_index = []
    black_list = set()
    edge_terms_clone = edge_terms.clone()
    edges = []
    all_edges = list(itertools.product(nodes_multiset.keys(), nodes_multiset.keys()))
    for a, b in all_edges:
        # if a == b: continue
        idx_a = encoder.nodes_indexer.get_idx(a)
        idx_b = encoder.nodes_indexer.get_idx(b)
        hd_a = encoder.nodes_codebook[idx_a]
        hd_b = encoder.nodes_codebook[idx_b]

        # bind
        edges.append(hd_a.bind(hd_b))
    t = torch.stack(edges).as_subclass(torchhd.HRRTensor)
    for i in range(e_count // 2):
        sims = torchhd.cos(edge_terms, t)
        max = sims.max().item()
        eps = 1e-9
        # max_idxs = (sims >= sims.max() - eps).nonzero(as_tuple=True)[0]
        idx_max = torch.argmax(sims).item()
        a_found, b_found = all_edges[idx_max]
        # for idx in max_idxs:
        #     if idx in black_list:
        #         continue
        #     a_found, b_found = all_edges[idx]
        #     if residuals[a_found] <= 0:
        #         continue
        #     if residuals[b_found] <= 0:
        #         continue
        #     if (a_found, b_found) not in edges_left:
        #         print("ALARM")
        #     break

        if not a_found or not b_found:
            print("Failed to decode")
            break
        # edges_left.remove((a_found, b_found))
        # edges_left.remove((b_found, a_found))
        hd_a_found = encoder.nodes_codebook[encoder.nodes_indexer.get_idx(a_found)]
        hd_b_found = encoder.nodes_codebook[encoder.nodes_indexer.get_idx(b_found)]
        edge_terms -= hd_a_found.bind(hd_b_found)
        edge_terms -= hd_b_found.bind(hd_a_found)
        result[(a_found, b_found)] += 1
        result[(b_found, a_found)] += 1

        residuals[a_found] -= 2
        residuals[b_found] -= 2

        result_edge_index.append((a_found, b_found))
        result_edge_index.append((b_found, a_found))
        print(f"Edge {i} done")
        print(f"Sim max: {sims.max().item()}")

    # actual_edges = [(node_tuples[a], node_tuples[b]) for a, b in edge_tuples]
    # if result_edge_index != actual_edges:
    #     print("ALARM")
    # extras = []
    # for tup in result_edge_index:
    #     if tup not in actual_edges:
    #         extras.append(tup)
    #
    # not_found = []
    # for tup in actual_edges:
    #     if tup not in result_edge_index:
    #         not_found.append(tup)

    # print(result.total())
    # print(result)
    # print(result_edge_index)
    # print("ENCODED")
    # e_c = Counter(result_edge_index)
    # print(sorted(e_c.items(), key=lambda x: x[1], reverse=True))
    # print("ACTUAL")
    # # a_c = Counter([(node_tuples[a], node_tuples[b]) for a, b in edge_tuples])
    # print(sorted(a_c.items(), key=lambda x: x[1], reverse=True))

    ## We have the multiset of nodes and the multiset of edges
    first_pop: list[tuple[nx.Graph, list[tuple]]] = []
    global_seen: set = set()
    for k, (u_t, v_t) in enumerate(result_edge_index):
        G = nx.Graph()
        uid = add_node_with_feat(G, Feat.from_tuple(u_t))
        ok = add_node_and_connect(G, Feat.from_tuple(v_t), connect_to=[uid], total_nodes=n_count) is not None
        if not ok:
            continue
        key = _hash(G)
        if key in global_seen:
            continue
        global_seen.add(key)
        remaining_edges = result_edge_index.copy()
        remaining_edges.remove((u_t, v_t))
        remaining_edges.remove((v_t, u_t))
        first_pop.append((G, remaining_edges))
    print(len(first_pop))

    # Start with a child with on satisfied node
    selected = [(G, l) for G, l in first_pop if len(anchors(G)) == 2]
    population = selected if len(selected) >= 1 else first_pop
    for i in tqdm(range(2, n_count)):
        children: list[tuple[nx.Graph, list[tuple]]] = []
        local_seen: set = set()  # per-iteration dedup to keep branching under control

        for gi, (G, edges_left) in enumerate(population):
            leftovers_ctr = leftover_features(nodes_multiset, G)
            if not leftovers_ctr:
                continue

            leftover_types = order_leftovers_by_degree_distinct(leftovers_ctr)
            ancrs = anchors(G)
            if not ancrs:
                continue
            lowest_degree_ancrs = sorted(ancrs, key=lambda n: residual_degree(G, n))[:1]

            # how many anchors we’ll actually use (slice semantics preserved)
            for a, lo_t in list(itertools.product(lowest_degree_ancrs, leftover_types)):
                a_t = G.nodes[a]["feat"].to_tuple()
                if (a_t, lo_t) not in edges_left:
                    continue

                C = G.copy()
                nid = add_node_and_connect(C, Feat.from_tuple(lo_t), connect_to=[a], total_nodes=n_count)
                if nid is None:
                    continue
                if C.number_of_edges() > e_count:
                    continue

                keyC = _hash(C)
                if keyC in global_seen or keyC in local_seen:
                    continue
                # # ## --------------------------
                # batch = Batch.from_data_list([DataTransformer.nx_to_pyg(C)])
                # enc_out = encoder.forward(batch)
                # g_terms = enc_out["graph_embedding"]
                # sims_c = torchhd.cos(graph_terms, g_terms).tolist()[0]
                # print(f"SIM: {sims_c[0]}")
                # draw_nx_with_atom_colorings(C, dataset="ZincSmiles", label=sims_c[0])
                # plt.show()
                # # ## --------------------------
                remaining_edges = edges_left.copy()
                remaining_edges.remove((a_t, lo_t))
                remaining_edges.remove((lo_t, a_t))
                global_seen.add(keyC)
                local_seen.add(keyC)
                children.append((C, remaining_edges))

                ancrs_rest = [a_ for a_ in ancrs if a_ != a]

                for subset in powerset(ancrs_rest):
                    if len(subset) == 0:
                        continue

                    # Skip if subsets edges are not in the edge list
                    all_new_connection = []
                    nid_t = C.nodes[nid]["feat"].to_tuple()
                    subset_ts = [C.nodes[s]["feat"].to_tuple() for s in subset]
                    should_continue = False
                    for st in subset_ts:
                        ts = (nid_t, st)
                        if ts not in remaining_edges:
                            should_continue = True
                            break
                        all_new_connection.append(ts)

                    if should_continue:
                        continue

                    all_new_counter = Counter(all_new_connection)
                    # if both ends of an edge is the same tuple, it should be considered twice
                    for k, v in all_new_counter.items():
                        if k[0] == k[1]:
                            all_new_counter[k] = 2 * v
                    left_over_edges_counter = Counter(remaining_edges)
                    for k, v in all_new_counter.items():
                        if left_over_edges_counter[k] < v:
                            should_continue = True
                            break

                    if should_continue:
                        continue

                    H = C.copy()
                    new_nid = connect_all_if_possible(H, nid, connect_to=list(subset), total_nodes=n_count)
                    if new_nid is None:
                        continue
                    if H.number_of_edges() > e_count:
                        continue

                    keyH = _hash(H)
                    if keyH in global_seen or keyH in local_seen:
                        continue
                    remaining_edges_ = remaining_edges.copy()
                    for a_t, b_t in all_new_connection:
                        try:
                            remaining_edges_.remove((a_t, b_t))
                            remaining_edges_.remove((b_t, a_t))
                        except Exception as e:
                            print(e)
                            continue

                    # ## ---------------------------------
                    # batch = Batch.from_data_list([DataTransformer.nx_to_pyg(H)])
                    # enc_out = encoder.forward(batch)
                    # g_terms = enc_out["graph_embedding"]
                    # sims_c = torchhd.cos(graph_terms, g_terms).tolist()[0]
                    # print(f"SIM: {sims_c[0]}")
                    # draw_nx_with_atom_colorings(H, dataset="ZincSmiles", label=sims_c[0])
                    # plt.show()
                    # ## ---------------------------------

                    global_seen.add(keyH)
                    local_seen.add(keyH)
                    children.append((H, remaining_edges_))

        ## Collect the children with highest number of edges
        # edge_max = max([G.number_of_edges() for G, _ in children])
        # children = [(G, l) for G, l in children if G.number_of_edges() >= edge_max]
        # if (i % 3) == 0:
        if not children:
            graphs, edges_left = zip(*population, strict=True)
            are_final = [len(i) == 0 for i in edges_left]
            return graphs, are_final

        def ring_bonus(cycles: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
            """
            Apply a *very small* additive boost if the graph has any cycle.
            cycles: tensor[int] with number of simple cycles per graph.
            alpha: small positive constant controlling how much rings matter.
            """
            return alpha * torch.log1p(cycles)

        # --- pipeline ---
        limit = 128
        keep = limit // 2

        # Encode and compute similarity
        batch = Batch.from_data_list([DataTransformer.nx_to_pyg(c) for c, _ in children])
        enc_out = encoder.forward(batch)
        g_terms = enc_out["graph_embedding"]
        sims = torchhd.cos(graph_terms, g_terms)[0]

        # Sort by similarity first
        sim_order = torch.argsort(sims, descending=True)
        children = [children[i.item()] for i in sim_order]

        # Optional ring-boost if too many
        if len(children) > limit:
            top_children = children[:limit]
            cyc_counts = torch.tensor([len(nx.cycle_basis(c)) for c, _ in top_children])

            # very light boost (alpha ≈ 0.05)
            boosted = sims[sim_order[:limit]] + ring_bonus(cyc_counts, alpha=0.00)

            # Re-rank by boosted score
            final_order_local = torch.argsort(boosted, descending=True)
            children = [top_children[i.item()] for i in final_order_local[:keep]]

        population = children

    graphs, edges_left = zip(*population, strict=True)
    are_final = [len(i) == 0 for i in edges_left]
    return graphs, are_final


if __name__ == "__main__":
    base_dataset = "zinc"
    ds = ZincSmiles(split="train") if base_dataset == "zinc" else QM9Smiles(split="train")

    data = ds[0]

    nx_g = DataTransformer.pyg_to_nx(data)
    # draw_nx_with_atom_colorings(nx_g, dataset=base_dataset)
    # plt.show()
    mol, _ = DataTransformer.nx_to_mol_v2(nx_g, dataset=base_dataset)

    print(mol.GetNumAtoms())
    print(mol.GetNumBonds())
    print(nx_g.number_of_nodes())
    print(nx_g.number_of_edges())
    print(data.x)
    print(data.edge_index)

    # device = pick_device()
    device = torch.device("cpu")
    node_tuples = [tuple(i) for i in data.x.tolist()]
    edge_tuples = [tuple(e) for e in data.edge_index.t().cpu().tolist()]

    ds_config = ZINC_SMILES_HRR_7744_CONFIG if base_dataset == "zinc" else QM9_SMILES_HRR_1600_CONFIG_F64
    encoder = load_or_create_hypernet(cfg=ds_config).to(device, dtype=torch.float64)
    encoder.nodes_codebook = encoder.nodes_codebook.to(torch.float64).as_subclass(torchhd.HRRTensor)
    node_cb = encoder.nodes_codebook
    node_cb = node_cb.to(torch.float64).as_subclass(torchhd.HRRTensor)
    print(node_cb.shape)
    node_idxer: TupleIndexer = encoder.nodes_indexer

    # Manually compute the edge terms
    edge_terms_manul = None
    edges = []
    for a, b in edge_tuples:
        idx_a = node_idxer.get_idx(node_tuples[a])
        idx_b = node_idxer.get_idx(node_tuples[b])
        hd_a = node_cb[idx_a]
        hd_b = node_cb[idx_b]

        # bind
        edges.append(hd_a.bind(hd_b))

    t = torch.stack(edges)
    print(t.shape)
    edge_terms_manul = torchhd.multibundle(t)
    print(edge_terms_manul.shape)

    edget_terms_m_copy = edge_terms_manul.clone()
    # Now just reverse to see if you get 0
    for a, b in edge_tuples:
        idx_a = node_idxer.get_idx(node_tuples[a])
        idx_b = node_idxer.get_idx(node_tuples[b])
        hd_a = node_cb[idx_a]
        hd_b = node_cb[idx_b]

        # bind
        edget_terms_m_copy -= hd_a.bind(hd_b)

    zero_hd = torch.zeros_like(edget_terms_m_copy)
    sum_elements = edget_terms_m_copy.abs().sum().item()
    print(sum_elements)
    # prints 2.470420440658927e-05 <-- why not zero?
    # with dtype float64 -> 4.542902902140598e-14 almost zero

    # Compare the manually created one with the hypernet

    batch = Batch.from_data_list([data])
    forward = encoder.forward(batch)
    edge_terms = forward["edge_terms"]
    graph_terms = forward["graph_embedding"]

    eps = 1e-9
    ok_mask = (edge_terms - edge_terms_manul).abs() <= eps  # bool tensor
    all_ok = ok_mask.all()  # << GOOD
    print(all_ok.item())

    draw_mol(mol=mol, save_path="candidate-input.png", fmt="png")
    ## Now let's fucking decode the edges
    node_counter = DataTransformer.get_node_counter_from_batch(0, batch)
    candidates, edges_left = new_decoder(
        nodes_multiset=node_counter,
        edge_terms=edge_terms,
        encoder=encoder,
        graph_terms=graph_terms,
    )
    data_list = [DataTransformer.nx_to_pyg(c) for c in candidates]

    batch = Batch.from_data_list(data_list)
    enc_out = encoder.forward(batch)
    g_terms = enc_out["graph_embedding"]  # [B, D]

    q = graph_terms.to(g_terms.device, g_terms.dtype)  # [D]
    sims_ = torchhd.cos(q, g_terms).tolist()[0]
    print(sims_)
    print(max(sims_))

    best_idx = sims_.index(max(sims_))
    best = candidates[best_idx]
    print(f"Edges Left Best: {edges_left[best_idx]}")

    draw_mol(mol=mol, save_path="candidate-input.png", fmt="png")
    for i, c in enumerate(candidates):
        best_mol, _ = DataTransformer.nx_to_mol_v2(c, dataset=base_dataset)
        draw_mol(mol=best_mol, save_path=f"candidate-{i}-best-{best_idx}.png", fmt="png")
