import random

import networkx as nx

# ============================================================================
# Graph Isomorphism Search via Random Matchings
# ============================================================================


def deduplicate_edges(edges_multiset: list[tuple]) -> list[tuple]:
    """
    Remove bidirectional duplicates from an edge multiset.

    For each pair (a, b) and (b, a), keeps only one canonical representative
    (the one with smaller first element).

    Parameters
    ----------
    edges_multiset : list[tuple]
        List of directed edges that may contain both (a,b) and (b,a)

    Returns
    -------
    list[tuple]
        Deduplicated list with one edge per undirected pair

    Example
    -------
    >>> deduplicate_edges([(1, 2), (2, 1), (1, 3), (3, 1)])
    [(1,2), (1,3)]
    """
    temp = []
    for edge_vec in edges_multiset:
        if edge_vec[0] > edge_vec[1]:
            temp.append((edge_vec[1], edge_vec[0]))
        else:
            temp.append((edge_vec[0], edge_vec[1]))
    temp.sort()

    # Keep only every second entry (removes duplicates)
    deduplicated = [temp[i] for i in range(0, len(temp), 2)]
    return deduplicated


def compute_sampling_structure(nodes_multiset: list[tuple], edges_multiset: list[tuple]) -> tuple[dict, dict]:
    """
    Construct a bipartite matching structure for efficient random graph sampling.

    This creates a data structure representing multiple complete bipartite graphs (K_n,n)
    where each component corresponds to a node type. The structure enables efficient
    sampling of random valid matchings.

    Parameters
    ----------
    nodes_multiset : list[tuple]
        List of node feature tuples (may contain duplicates)
    edges_multiset : list[tuple]
        List of edge tuples (pairs of node features)

    Returns
    -------
    matching_components : dict
        Dictionary mapping node types to their nodes and edges lists.
        Keys are node feature tuples, values are dicts with 'nodes' and 'edges' lists.
    id_to_type : dict
        Mapping from string IDs (e.g., 'n0', 'e1') to feature tuples

    Notes
    -----
    - Node IDs are formatted as 'n{i}' where i is the node index
    - Edge IDs are formatted as 'e{k}' where k is the edge index
    - Each node appears in the 'nodes' list according to its degree (degree+1 times)
    """
    nodes_multiset = sorted(nodes_multiset)
    edges_multiset = sorted(edges_multiset)

    deduplicated_edges = deduplicate_edges(edges_multiset)

    matching_components: dict = {}
    id_to_type: dict = {}

    for node_vec in nodes_multiset:
        matching_components.setdefault(node_vec, {"nodes": [], "edges": []})

    for i, node_vec in enumerate(nodes_multiset):
        node_degree = node_vec[1] + 1  # degree_idx + 1 gives target degree
        id_to_type[f"n{i}"] = node_vec

        for _ in range(node_degree):
            matching_components.setdefault(node_vec, {"nodes": []})["nodes"].append(f"n{i}")

    for k, edge_vec in enumerate(deduplicated_edges):
        edge_beginning = tuple(edge_vec[0])
        edge_ending = tuple(edge_vec[1])
        id_to_type[f"e{k}"] = (edge_beginning, edge_ending)

        matching_components.setdefault(edge_beginning, {"edges": []})["edges"].append(f"e{k}")
        matching_components.setdefault(edge_ending, {"edges": []})["edges"].append(f"e{k}")

    return matching_components, id_to_type


def draw_random_matching(sampling_structure: dict) -> list[tuple[str, str]]:
    """
    Draw a random matching from the sampling structure.

    Randomly assigns edges to nodes within each component by shuffling edges
    and pairing them with nodes.

    Parameters
    ----------
    sampling_structure : dict
        Dictionary mapping node types to their nodes and edges lists

    Returns
    -------
    list[tuple[str, str]]
        Sorted list of (edge_id, node_id) pairs representing the matching
    """
    matching = []

    for component in sampling_structure.values():
        nodes = component["nodes"]
        edges = component["edges"]
        permuted_edges = edges[:]
        random.shuffle(permuted_edges)
        for node, edge in zip(nodes, permuted_edges, strict=False):
            matching.append((edge, node))

    return sorted(matching)


def compute_graph_from_matching(matching: list[tuple[str, str]], id_to_type: dict) -> nx.Graph:
    """
    Construct a NetworkX graph from a matching.

    Parameters
    ----------
    matching : list[tuple[str, str]]
        Sorted list of (edge_id, node_id) pairs. Consecutive entries with
        the same edge_id form an edge in the graph.
    id_to_type : dict
        Mapping from string IDs to feature tuples

    Returns
    -------
    nx.Graph
        NetworkX graph with node attribute 'type' containing feature tuples
    """
    G = nx.Graph()

    # Process matching in pairs (each edge connects two nodes)
    for i in range(0, len(matching), 2):
        edge_id_1, node_id_1 = matching[i]
        edge_id_2, node_id_2 = matching[i + 1]

        G.add_edge(node_id_1, node_id_2)
        G.nodes[node_id_1]["type"] = id_to_type[node_id_1]
        G.nodes[node_id_2]["type"] = id_to_type[node_id_2]

    return G


def draw_random_graph_from_sampling_structure(matching_components: dict, id_to_type: dict) -> nx.Graph:
    """
    Sample a random molecular graph by drawing a random matching.

    This is the main sampling function that combines random matching generation
    and graph construction.

    Parameters
    ----------
    matching_components : dict
        Dictionary mapping node types to their nodes and edges lists
    id_to_type : dict
        Mapping from string IDs to feature tuples

    Returns
    -------
    nx.Graph
        Randomly sampled graph (may not be valid - check with graph_is_valid)

    Notes
    -----
    The sampling may be biased towards certain graph structures.
    """
    random_matching = draw_random_matching(matching_components)
    G = compute_graph_from_matching(random_matching, id_to_type)
    return G


def graph_is_valid(G: nx.Graph) -> bool:
    """
    Check if a graph is valid (connected and has no self-loops).

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph to validate

    Returns
    -------
    bool
        True if graph is connected and has no self-loops, False otherwise
    """
    return nx.is_connected(G) and nx.number_of_selfloops(G) == 0


def try_find_isomorphic_graph(
    matching_components: dict,
    id_to_type: dict,
    *,
    max_samples: int = 200000,
) -> list[nx.Graph]:
    """
    Generate valid molecular graphs by sampling random matchings.

    Repeatedly samples random matchings from the bipartite structure until
    reaching the maximum number of attempts. Collects all valid (connected,
    non-self-looping) graphs found during the search.

    Parameters
    ----------
    matching_components : dict
        Dictionary mapping node types to their nodes and edges lists,
        constructed by compute_sampling_structure
    id_to_type : dict
        Mapping from string IDs to feature tuples
    max_samples : int, optional
        Maximum number of random matchings to try, by default 200000
    report_interval : int, optional
        Print progress every N attempts, by default 1000

    Returns
    -------
    list[nx.Graph]
        List of valid graphs found during sampling. May be empty if no
        valid graphs were found.

    Notes
    -----
    - This function does not guarantee finding any specific target graph
    - The search stops after max_samples attempts regardless of success
    - All returned graphs have node attribute 'type' containing feature tuples
    """
    count = 0
    graphs = []

    max_attempts = 10 * max_samples
    attempts = 0
    while True:
        G = draw_random_graph_from_sampling_structure(matching_components, id_to_type)
        attempts += 1
        if attempts > max_attempts:
            print("stop")
        if attempts >= max_attempts and len(graphs) > 0:
            break

        if not graph_is_valid(G):
            continue

        graphs.append(G)
        count += 1

        if count >= max_samples:
            break

    return graphs
