import argparse
import math
import os
import random
import time
from collections import Counter, OrderedDict
from pprint import pprint

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torchhd
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.datasets.utils import get_split
from src.encoding.configs_and_constants import (
    DSHDCConfig,
    FeatureConfig,
    Features,
    IndexRange,
    SupportedDataset,
)
from src.encoding.decoder import new_decoder  # noqa: F401  # noqa: F401
from src.encoding.feature_encoders import CombinatoricIntegerEncoder
from src.encoding.graph_encoders import HyperNet, load_or_create_hypernet
from src.encoding.the_types import VSAModel
from src.utils.utils import GLOBAL_ARTEFACTS_PATH, DataTransformer, pick_device, pick_device_str

PLOT = False

HV_DIMS = {
    "qm9": [
        # 128,
        256,
        512,
        1024,
        1280,
        1536,
        1600,
    ],
    "zinc": [
        # 128,
        # 256,
        512,
        1024,
        1280,
        1536,
        1600,
        2048,
    ],
}


DTYPE = torch.float32
torch.set_default_dtype(DTYPE)


def eval_retrieval(ds: SupportedDataset, n_samples: int = 1):
    for ds in [
        SupportedDataset.ZINC_SMILES_HRR_5120_F64_G1G3,
        # SupportedDataset.QM9_SMILES_HRR_1600_F64_G1NG3,
    ]:
        ds_config = ds.default_cfg
        base_dataset = ds_config.base_dataset
        for hv_dim in HV_DIMS[base_dataset]:
            device = pick_device()
            print(f"Running on {device}")
            ds_config.hv_dim = hv_dim
            ds_config.device = device

            if ds.default_cfg.base_dataset == "zinc":
                ds_config = DSHDCConfig(
                    seed=42,
                    name=f"ZincSmilesFeat5HRR{hv_dim}G1NG3",
                    base_dataset="zinc",
                    vsa=VSAModel.HRR,
                    hv_dim=hv_dim,
                    device=pick_device_str(),
                    node_feature_configs=OrderedDict(
                        [
                            (
                                Features.ATOM_TYPE,
                                FeatureConfig(
                                    count=math.prod([9, 6, 3, 4, 2]),
                                    encoder_cls=CombinatoricIntegerEncoder,
                                    index_range=IndexRange((0, 5)),
                                    bins=[9, 6, 3, 4, 2],
                                ),
                            ),
                        ]
                    ),
                    normalize=True,
                    hypernet_depth=4,
                    dtype="float64",
                )

            ds_config.name = f"DELETE_{ds_config.name}_HRR{hv_dim}_"
            hypernet: HyperNet = (
                load_or_create_hypernet(cfg=ds_config, use_edge_codebook=False).to(device=device, dtype=DTYPE).eval()
            )
            hypernet.decoding_limit_for = base_dataset

            dataset = get_split(split="train", ds_config=ds.default_cfg, use_no_suffix=True)

            nodes_set = set(map(tuple, dataset.x.long().tolist()))
            hypernet.limit_nodes_codebook(limit_node_set=nodes_set)

            dataset = dataset[:n_samples]
            dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

            hits = []
            ts = []
            results = []
            best_sims = []
            acc = []
            for data in tqdm(dataloader):
                # if PLOT:
                nx_g = DataTransformer.pyg_to_nx(data)
                # draw_nx_with_atom_colorings(
                #     nx_g, dataset="QM9Smiles" if dataset == "qm9" else "ZincSmiles", label="Original"
                # )
                # plt.show()

                # Real Data
                node_tuples = [tuple(i) for i in data.x.int().tolist()]
                edge_tuples = [(node_tuples[e[0]], node_tuples[e[1]]) for e in data.edge_index.t().int().cpu().tolist()]
                real_edge_counter = Counter(edge_tuples)

                # Decoded
                edge_terms = hypernet.forward(data)["edge_terms"][0]
                decoded_edges = hypernet.decode_order_one_no_node_terms(edge_terms.clone())
                decoded_edge_counter = Counter(decoded_edges)

                # Graph Decoding
                matching_components, id_to_type = compute_sampling_structure(node_tuples, edge_tuples)
                t_d = time.perf_counter()
                budegt = 1 if ds.default_cfg.base_dataset == "qm9" else 10

                best_sims_per_budget = []  # Store top 3 sims for each budget iteration
                all_similarities = []  # Track all top sims to find overall best

                # Ground Truth
                the_graph = hypernet.forward(Batch.from_data_list([data]))["graph_embedding"][0]
                for i in range(budegt):
                    decoded_graphs = try_find_isomorphic_graph(
                        matching_components=matching_components, id_to_type=id_to_type, max_samples=1024
                    )
                    # custom nx -> data
                    pyg_graphs = [nx_to_pyg_jp(g, id_to_type) for g in decoded_graphs]
                    batch = Batch.from_data_list(pyg_graphs)
                    g_hdc = hypernet.forward(batch)["graph_embedding"]

                    sims = torchhd.cos(the_graph, g_hdc)

                    # Get top 3 similarities
                    top_3_sims, top_3_indices = torch.topk(sims, k=3)
                    top_3_sims = top_3_sims.cpu().numpy()

                    best_sims_per_budget.append(top_3_sims)
                    all_similarities.extend(top_3_sims)

                    # Print info for this iteration
                    print(f"Budget {i} - Top 3 Similarities: {top_3_sims}")

                # Get overall best similarity across all budgets
                overall_best_sim = max(all_similarities)
                print(f"Overall Best Sim: {overall_best_sim:.4f}")

                # Optional: Store results if needed for your original tracking
                best_sims = [max(sims_list) for sims_list in best_sims_per_budget]
                acc = [abs(sim - 1.0) < 0.0001 for sim in best_sims]

                is_hit = decoded_edge_counter == real_edge_counter
                hits.append(is_hit)

            ts = np.array(ts)
            results.append(
                {
                    "n_samples": n_samples,
                    "dataset": ds_config.base_dataset,
                    "vsa": ds_config.vsa.value,
                    "hv_dim": ds_config.hv_dim,
                    "device": str(device),
                    "time_per_sample": ts.mean(),
                    "edge_accuracy": sum(hits) / len(hits),
                    "accuracy": sum(acc) / len(acc),
                    "avg_sim": round(sum(best_sims) / len(best_sims), 4),
                    "dtype": "float32" if torch.float32 == DTYPE else "float64",
                }
            )
            pprint(results[-1])

            # --- save metrics to disk ---
            asset_dir = GLOBAL_ARTEFACTS_PATH / "graph_retrieval"
            asset_dir.mkdir(parents=True, exist_ok=True)

            parquet_path = asset_dir / "graph_retrieval.parquet"
            csv_path = asset_dir / "graph_retrieval.csv"

            metrics_df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()

            new_row = pd.DataFrame(results)
            metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

            # write back out
            metrics_df.to_parquet(parquet_path, index=False)
            metrics_df.to_csv(csv_path, index=False)


def nx_to_pyg_jp(G: nx.Graph, id_to_type) -> Data:
    """
    Convert an undirected NetworkX graph with ``feat`` attributes to a
    :class:`torch_geometric.data.Data`.

    Node features are stacked as a dense matrix ``x`` of dtype ``long`` with
    columns ``[atom_type, degree_idx, formal_charge_idx, explicit_hs]``.

    Undirected edges are converted to a directed ``edge_index`` with both
    directions.

    :param G: NetworkX graph where each node has a ``feat: Feat`` attribute.
    :returns: PyG ``Data`` object with fields ``x`` and ``edge_index``.
    :raises RuntimeError: If PyTorch/PyG are not available.
    """
    if torch is None or Data is None:
        raise RuntimeError("torch / torch_geometric are required for nx_to_pyg")

    # Node ordering: use sorted ids for determinism
    nodes = sorted(G.nodes)
    idx_of: dict[int, int] = {n: i for i, n in enumerate(nodes)}

    # Features
    feats: list[list[int]] = [list(G.nodes[n]["type"]) for n in nodes]
    x = torch.tensor(feats, dtype=torch.long)

    # Edges: add both directions
    src, dst = [], []
    for u, v in G.edges():
        iu, iv = idx_of[u], idx_of[v]
        src.extend([iu, iv])
        dst.extend([iv, iu])
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


def compute_solution_graph_from_example(ordered_nodes, associated_edge_idxs):
    """
    Computes the labeled expected solution graph from the input json
    """

    G = nx.Graph()
    for i, node_vec in enumerate(ordered_nodes):
        node_id = f"n{i}"
        G.add_node(node_id, type=tuple(node_vec))
    for a, b in associated_edge_idxs:
        node_id_1 = f"n{a}"
        node_id_2 = f"n{b}"
        G.add_edge(node_id_1, node_id_2)

    return G


def draw_random_matching(sampling_structure):
    """
    Enumerating all matchings is not efficient, as multiple matchings map to the same (isomorphic) candidate graph.
    Instead, we can draw random matchings from the matching components.
    In each component, we randomly permute the edges and assign them to the nodes.
    We can construct a candidate graph from the resulting matching.
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


def compute_graph_from_matching(matching, id_to_type):
    G = nx.Graph()

    # matching is sorted list of (edge_id, node_id) pairs
    # two consecutive entries always have the same edge

    for i in range(0, len(matching), 2):
        edge_id_1, node_id_1 = matching[i]
        edge_id_2, node_id_2 = matching[i + 1]

        G.add_edge(node_id_1, node_id_2)
        G.nodes[node_id_1]["type"] = id_to_type[node_id_1]
        G.nodes[node_id_2]["type"] = id_to_type[node_id_2]

    return G


def draw_random_graph_from_sampling_structure(matching_components, id_to_type):
    """
    This is the main function of this file.

    It allows to draw a random molecule (graph) by drawing a random matching and then constructs the corresponding graph.
    Graphs are probably drawn biased.
    """
    random_matching = draw_random_matching(matching_components)
    G = compute_graph_from_matching(random_matching, id_to_type)
    return G


def graph_is_valid(G):
    """
    We dont't care about molecules that are disconnected or have selfloops.
    This functions discards those graphs.
    """
    return nx.is_connected(G) and nx.number_of_selfloops(G) == 0


def try_find_isomorphic_graph(matching_components, id_to_type, max_samples=200000, report_interval=1000):
    """
    Tries to find a graph isomorphic to solution_graph by drawing random matchings.
    """
    count = 0

    graphs = []
    while True:
        G = draw_random_graph_from_sampling_structure(matching_components, id_to_type)

        if not graph_is_valid(G):
            continue
        graphs.append(G)
        #
        # if are_isomorphic(solution_graph, G):
        #     return G

        count += 1
        if count % report_interval == 0:
            print(f"Tried {count} random matchings...")
        if count >= max_samples:
            print(f"Reached {max_samples} attempts without finding isomorphism. Giving up.", flush=True)
            return graphs


def deduplicate_edges(edges_multiset):
    """
    Helper function to remove bidirectional edges.
    If, we have edge (a,b) and (b,a), only keep one of them.
    Result will still be a multiset
    """
    temp = []
    for edge_vec in edges_multiset:
        if edge_vec[0] > edge_vec[1]:
            temp.append((edge_vec[1], edge_vec[0]))
        else:
            temp.append((edge_vec[0], edge_vec[1]))
    temp.sort()
    # remove every second entry
    deduplicated = []
    for i in range(len(temp)):
        if i % 2 == 0:
            deduplicated.append(temp[i])
    return deduplicated


def compute_sampling_structure(nodes_multiset, edges_multiset):
    """
    Constructs a datastructure where we can efficiently sample random matchings from.

    It is essentialy a bipartite graph with a very simple structure (multiple K_nn) so we can represent it like this:
    - Each K_nn represents a node type (like a c atom).
    - Each K_nn has a node and a edge part.
    - We identify each nodes/edge with a unique label and also store the mapping to the type information
    """
    nodes_multiset = sorted(nodes_multiset)
    edges_multiset = sorted(edges_multiset)

    deduplicated_edges = deduplicate_edges(edges_multiset)

    matching_components = {}
    id_to_type = {}

    for node_vec in nodes_multiset:
        matching_components.setdefault(node_vec, {"nodes": [], "edges": []})

    for i, node_vec in enumerate(nodes_multiset):
        node_degree = node_vec[1] + 1  # modified
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluation retrieval of full graph from encoded graph")
    p.add_argument(
        "--dataset",
        type=str,
        default=SupportedDataset.QM9_SMILES_HRR_1600_F64_G1NG3.value,
        choices=[ds.value for ds in SupportedDataset],
    )
    p.add_argument("--n_samples", type=int, default=100)
    args = p.parse_args()
    ds = SupportedDataset(args.dataset)

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    pprint(args)
    eval_retrieval(n_samples=args.n_samples, ds=ds)
