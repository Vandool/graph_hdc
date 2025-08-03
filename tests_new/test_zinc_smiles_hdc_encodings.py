import time
from collections import Counter, OrderedDict
from datetime import datetime
from math import prod

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
import torchhd
from pytorch_lightning import seed_everything
from torch_geometric.loader import DataLoader
from torchhd import structures

from src import evaluation_metrics
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import DatasetConfig, FeatureConfig, Features, IndexRange
from src.encoding.feature_encoders import CombinatoricIntegerEncoder
from src.encoding.graph_encoders import HyperNet
from src.encoding.the_types import VSAModel
from src.utils import utils
from src.utils.utils import TEST_ARTIFACTS_PATH, DataTransformer


@pytest.mark.parametrize(
    "vsa",
    [
        VSAModel.HRR,
        VSAModel.MAP
    ],
)
@pytest.mark.parametrize(
    "hv_dim",
    [
        # 32 * 32,
        # 40 * 40,
        # 48 * 48,
        # 56 * 56,
        64 * 64,
        72 * 72,
        80 * 80,
        88 * 88,
        96 * 96,
        104 * 104,
        112 * 112,
        120 * 120,
        128 * 128,
    ],
)
@pytest.mark.parametrize(
    "normalise_graph_embedding",
    [True, False],
)
@pytest.mark.parametrize(
    "depth",
    [2, 3],
)
def test_node_terms_edge_terms_decoding(hv_dim, vsa, depth, normalise_graph_embedding):
    seed = 42
    utils.set_seed(seed)
    seed_everything(seed)

    zinc_feature_bins = [9, 6, 3, 4]
    dataset_config = DatasetConfig(
        seed=seed,
        name="ZINC_SMILES",
        vsa=vsa,
        hv_dim=hv_dim,
        node_feature_configs=OrderedDict(
            [
                (
                    Features.ATOM_TYPE,
                    FeatureConfig(
                        # Atom types size: 9
                        # Atom types: ['Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S']
                        # Degrees size: 5
                        # Degrees: {1, 2, 3, 4, 5}
                        # Formal Charges size: 3
                        # Formal Charges: {0, 1, -1}
                        # Explicit Hs size: 4
                        # Explicit Hs: {0, 1, 2, 3}
                        count=prod(
                            zinc_feature_bins
                        ),  # 9 Atom Types, 6 Unique Node Degrees: [0.0 (for ease of indexing), 1.0, 2.0, 3.0, 4.0, 5.0]
                        encoder_cls=CombinatoricIntegerEncoder,
                        index_range=IndexRange((0, 4)),
                        bins=zinc_feature_bins,
                    ),
                ),
            ]
        ),
    )

    hypernet = HyperNet(config=dataset_config, depth=depth, use_edge_codebook=False)
    assert not hypernet.use_edge_features()
    assert not hypernet.use_graph_features()

    batch_size = 1028
    zinc_smiles = ZincSmiles(split="train")
    dataloader = DataLoader(dataset=zinc_smiles, batch_size=batch_size, shuffle=False)
    batch = next(iter(dataloader))

    # Encode the whole graph in one HV
    encoded_data = hypernet.forward(batch)
    node_term = encoded_data["node_terms"]
    edge_term = encoded_data["edge_terms"]
    graph_term = encoded_data["graph_embedding"]

    if normalise_graph_embedding:
        graph_term = graph_term.normalize()

    s0_hyper_graph = graph_term
    s2_hyper_graph = node_term.bundle(graph_term)
    s3_hyper_graph = node_term.bundle(edge_term).bundle(graph_term)

    ### ----- H2
    ## Create a HashTable
    v_n = torchhd.random(1, dimensions=hv_dim, vsa=vsa.value)
    v_g = torchhd.random(1, dimensions=hv_dim, vsa=vsa.value)
    node_term_key_value = v_n.bind(node_term)
    graph_term_key_value = v_g.bind(graph_term)

    stacked = torch.stack([node_term_key_value, graph_term_key_value], dim=0).transpose(0, 1)
    g_hv = torchhd.multiset(stacked)

    # Extract the node_terms from Graph Hyper Vector
    node_term_extract_2_levels = torchhd.bind(v_n.inverse(), g_hv)
    graph_term_extract_2_levels = torchhd.bind(v_g.inverse(), g_hv)

    print("Similarity metrics after decoding")
    H2_node_term_sim = F.cosine_similarity(node_term_extract_2_levels, node_term, dim=1).mean().item()
    H2_graph_term_sim = F.cosine_similarity(graph_term_extract_2_levels, graph_term, dim=1).mean().item()

    ### ----- H3
    ## Create a HashTable
    hach_table_3_levels = structures.HashTable(dim_or_input=hv_dim, vsa=vsa.value)
    h3_var = torchhd.random(3, hv_dim, vsa=vsa.value)
    hach_table_3_levels.add(key=h3_var[0], value=node_term)
    hach_table_3_levels.add(key=h3_var[1], value=edge_term)
    hach_table_3_levels.add(key=h3_var[2], value=graph_term)

    # Extract the node_terms from Graph Hyper Vector
    node_term_extract_3_levels = hach_table_3_levels.get(h3_var[0])
    edge_term_extract_3_levels = hach_table_3_levels.get(h3_var[1])
    graph_term_extract_3_levels = hach_table_3_levels.get(h3_var[2])

    H3_node_term_sim = F.cosine_similarity(node_term_extract_3_levels, node_term, dim=1).mean().item()
    H3_edge_term_sim = F.cosine_similarity(edge_term_extract_3_levels, edge_term, dim=1).mean().item()
    H3_graph_term_sim = F.cosine_similarity(graph_term_extract_3_levels, graph_term, dim=1).mean().item()

    # Build ground‐truth Counters per graph
    ground_truth_counters = {}
    for g in range(batch_size):
        ground_truth_counters[g] = DataTransformer.get_node_counter_from_batch(batch=g, data=batch)

    ## ---- Order 0 - H0
    nodes_decoded_counter = hypernet.decode_order_zero_counter(node_term)

    # Compute accuracy per graph:
    H0_order_zero_f1s = []
    H0_order_zero_precisions = []
    for g in range(batch_size):
        decoded_ctr = nodes_decoded_counter.get(g, Counter())
        truth_ctr = ground_truth_counters[g]

        p, _, f1 = evaluation_metrics.calculate_p_a_f1(pred=decoded_ctr, true=truth_ctr)
        H0_order_zero_f1s.append(f1)
        H0_order_zero_precisions.append(p)
    H0_F1_avg_node_terms = sum(H0_order_zero_f1s) / batch_size
    H0_P_avg_node_terms = sum(H0_order_zero_precisions) / batch_size

    ## ---- Order 0 - H2
    H2_nodes_decoded_counter = hypernet.decode_order_zero_counter(node_term_extract_2_levels)

    # Compute accuracy per graph:
    H2_order_zero_f1s = []
    H2_order_zero_precisions = []
    for g in range(batch_size):
        decoded_ctr = H2_nodes_decoded_counter.get(g, Counter())
        truth_ctr = ground_truth_counters[g]

        p, _, f1 = evaluation_metrics.calculate_p_a_f1(pred=decoded_ctr, true=truth_ctr)
        H2_order_zero_f1s.append(f1)
        H2_order_zero_precisions.append(p)
    H2_F1_avg_node_terms = sum(H2_order_zero_f1s) / batch_size
    H2_P_avg_node_terms = sum(H2_order_zero_precisions) / batch_size

    ## ---- Order 0 - H3
    H3_nodes_decoded_counter = hypernet.decode_order_zero_counter(node_term_extract_3_levels)

    # Compute accuracy per graph:
    H3_order_zero_f1s = []
    H3_order_zero_precisions = []
    for g in range(batch_size):
        decoded_ctr = H3_nodes_decoded_counter.get(g, Counter())
        truth_ctr = ground_truth_counters[g]

        p, _, f1 = evaluation_metrics.calculate_p_a_f1(pred=decoded_ctr, true=truth_ctr)
        H3_order_zero_f1s.append(f1)
        H3_order_zero_precisions.append(p)
    H3_F1_avg_node_terms = sum(H3_order_zero_f1s) / batch_size
    H3_P_avg_node_terms = sum(H3_order_zero_precisions) / batch_size

    ## ---- Order 0 - S0
    s0_nodes_decoded_counter = hypernet.decode_order_zero_counter(s0_hyper_graph)

    # Compute accuracy per graph:
    s0_order_zero_f1s = []
    s0_order_zero_precisions = []
    for g in range(batch_size):
        decoded_ctr = s0_nodes_decoded_counter.get(g, Counter())
        truth_ctr = ground_truth_counters[g]

        p, _, f1 = evaluation_metrics.calculate_p_a_f1(pred=decoded_ctr, true=truth_ctr)
        s0_order_zero_f1s.append(f1)
        s0_order_zero_precisions.append(p)
    S0_F1_avg_node_terms = sum(s0_order_zero_f1s) / batch_size
    S0_P_avg_node_terms = sum(s0_order_zero_precisions) / batch_size

    ## ---- Order 0 - S2
    s2_nodes_decoded_counter = hypernet.decode_order_zero_counter(s2_hyper_graph)

    # Compute accuracy per graph:
    s2_order_zero_f1s = []
    s2_order_zero_precisions = []
    for g in range(batch_size):
        decoded_ctr = s2_nodes_decoded_counter.get(g, Counter())
        truth_ctr = ground_truth_counters[g]

        p, _, f1 = evaluation_metrics.calculate_p_a_f1(pred=decoded_ctr, true=truth_ctr)
        s2_order_zero_f1s.append(f1)
        s2_order_zero_precisions.append(p)
    S2_F1_avg_node_terms = sum(s2_order_zero_f1s) / batch_size
    S2_P_avg_node_terms = sum(s2_order_zero_precisions) / batch_size

    ## ---- Order 0 - S3
    s3_nodes_decoded_counter = hypernet.decode_order_zero_counter(s3_hyper_graph)

    # Compute accuracy per graph:
    s3_order_zero_f1s = []
    s3_order_zero_precisions = []
    for g in range(batch_size):
        decoded_ctr = s3_nodes_decoded_counter.get(g, Counter())
        truth_ctr = ground_truth_counters[g]

        p, _, f1 = evaluation_metrics.calculate_p_a_f1(pred=decoded_ctr, true=truth_ctr)
        s3_order_zero_f1s.append(f1)
        s3_order_zero_precisions.append(p)
    S3_F1_avg_node_terms = sum(s3_order_zero_f1s) / batch_size
    S3_P_avg_node_terms = sum(s3_order_zero_precisions) / batch_size

    ### Save metrics
    run_metrics = {
        "dataset": "ZincSmiles (4 features - codebook size 540)",
        "date": datetime.now().isoformat(),
        "depth": depth,
        "normalize_graph_term": normalise_graph_embedding,
        "n_samples": batch_size,
        "vsa": vsa.value,
        "hv_dim": hv_dim,
        "seed": seed,
        "H0_P_avg_node_terms": H0_P_avg_node_terms,
        "H2_P_avg_node_terms": H2_P_avg_node_terms,
        "H3_P_avg_node_terms": H3_P_avg_node_terms,
        "S0_P_avg_node_terms": S0_P_avg_node_terms,
        "S2_P_avg_node_terms": S2_P_avg_node_terms,
        "S3_P_avg_node_terms": S3_P_avg_node_terms,
        "H0_F1_node_terms": H0_F1_avg_node_terms,
        "H2_F1_node_terms": H2_F1_avg_node_terms,
        "H3_F1_node_terms": H3_F1_avg_node_terms,
        "S0_F1_node_terms": S0_F1_avg_node_terms,
        "S2_F1_node_terms": S2_F1_avg_node_terms,
        "S3_F1_node_terms": S3_F1_avg_node_terms,
        "H2_avg_node_term_cos_sim": H2_node_term_sim,
        "H2_avg_graph_term_cos_sim": H2_graph_term_sim,
        "H3_avg_node_term_cos_sim": H3_node_term_sim,
        "H3_avg_edge_term_cos_sim": H3_edge_term_sim,
        "H3_avg_graph_term_cos_sim": H3_graph_term_sim,
    }

    # --- new code starts here ---
    # --- save metrics to disk ---
    asset_dir = TEST_ARTIFACTS_PATH / "nodes_and_edges" / "node_terms"
    asset_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = asset_dir / "res_normalizing.parquet"

    metrics_df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()

    # append current row
    new_row = pd.DataFrame([run_metrics])
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    # write back out
    metrics_df.to_parquet(parquet_path, index=False)
