import logging
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
import torch
import torchhd
from torch_geometric.data import Batch
from torch_geometric.datasets import ZINC

from src import evaluation_metrics
from src.datasets.utils import Compose, AddNodeDegree, AddNeighbourhoodEncodings
from src.encoding.configs_and_constants import FeatureConfig, Features, IndexRange, SupportedDataset
from src.encoding.feature_encoders import CombinatoricIntegerEncoder
from src.encoding.graph_encoders import HyperNet, load_or_create_hypernet
from src.encoding.the_types import VSAModel
from src.utils.utils import DataTransformer
from tests.utils import ARTIFACTS_PATH

logging.basicConfig()
logger = logging.getLogger(__name__)


def test_hypernet_encode_properties_work(batch_data):
    ds = SupportedDataset.ZINC

    hypernet = HyperNet(
        config=ds.default_cfg,
    )

    data = batch_data(ds_enum=ds, batch_size=10)

    encoded_data = hypernet.encode_properties(data)

    assert hasattr(encoded_data, "node_hv")
    assert hasattr(encoded_data, "edge_hv")
    assert hasattr(encoded_data, "graph_hv")

    assert encoded_data.node_hv.shape == (data.x.shape[0], ds.default_cfg.hv_dim)
    assert encoded_data.edge_hv.shape == (data.edge_attr.shape[0], ds.default_cfg.hv_dim)


@pytest.mark.parametrize("vsa", [VSAModel.MAP, VSAModel.HRR, VSAModel.FHRR])
def test_hypernet_forward_works(batch_data, vsa):
    ds = SupportedDataset.ZINC
    ds.default_cfg.vsa = vsa

    hypernet = HyperNet(
        config=ds.default_cfg,
        hidden_dim=ds.default_cfg.hv_dim,
    )

    batch_size = 10
    data = batch_data(ds_enum=ds, batch_size=batch_size)

    encoded_data = hypernet.forward(data)

    assert isinstance(encoded_data, dict)

    assert "graph_embedding" in encoded_data
    assert "node_hv_stack" in encoded_data

    assert encoded_data["graph_embedding"].shape == (batch_size, ds.default_cfg.hv_dim)

    num_nodes = data.x.shape[0]
    assert encoded_data["node_hv_stack"].shape == (num_nodes, hypernet.depth + 1, ds.default_cfg.hv_dim)

    assert vsa.value in repr(type(encoded_data["node_hv_stack"]))
    assert vsa.value in repr(type(encoded_data["graph_embedding"]))


@pytest.mark.parametrize("vsa", [VSAModel.MAP, VSAModel.HRR, VSAModel.FHRR])
def test_hypernet_forward_works_combinatorial_encoder(batch_data, vsa):
    ds = SupportedDataset.ZINC_NODE_DEGREE_COMB
    ds.default_cfg.vsa = vsa

    hypernet = HyperNet(
        config=ds.default_cfg,
        hidden_dim=ds.default_cfg.hv_dim,
    )

    batch_size = 2
    data = batch_data(ds_enum=ds, batch_size=batch_size)

    encoded_data = hypernet.forward(data)

    assert isinstance(encoded_data, dict)

    assert "graph_embedding" in encoded_data
    assert "node_hv_stack" in encoded_data

    assert encoded_data["graph_embedding"].shape == (batch_size, ds.default_cfg.hv_dim)

    num_nodes = data.x.shape[0]
    assert encoded_data["node_hv_stack"].shape == (num_nodes, hypernet.depth + 1, ds.default_cfg.hv_dim)

    assert vsa.value in repr(type(encoded_data["node_hv_stack"]))
    assert vsa.value in repr(type(encoded_data["graph_embedding"]))


@pytest.mark.parametrize(
    "separate_levels",
    [
        True,
        # False
    ],
)
@pytest.mark.parametrize(
    "normalize",
    [
        # True,
        False
    ],
)
@pytest.mark.parametrize(
    "depth",
    [
        # 1,
        # 2,
        # 3
    ],
)
@pytest.mark.parametrize(
    "dataset",
    [
        # SupportedDataset.ZINC,
        SupportedDataset.ZINC_NODE_DEGREE,
        SupportedDataset.ZINC_NODE_DEGREE_COMB,
        # SupportedDataset.ZINC_NODE_DEGREE_COMB_NHA,
    ],
)
@pytest.mark.parametrize("vsa", [VSAModel.HRR, VSAModel.MAP])
@pytest.mark.parametrize(
    "hv_dim",
    [
        32 * 32,
        48 * 48,
        # 96 * 96,
        # 80 * 80,
    ],
)
@pytest.mark.parametrize(
    "nha_depth",
    [1, 2, 3],
)
@pytest.mark.parametrize("use_explain_away", [True, False])
def test_hypernet_decode_order_one_is_good_enough_counter(
    batch_data, dataset, depth, vsa, hv_dim, use_explain_away, normalize, separate_levels, nha_depth
):
    import time  # add to the top if not already there

    ds = dataset
    ds.default_cfg.vsa = vsa
    # Disable edge and graph features
    ds.default_cfg.edge_feature_configs = {}
    ds.default_cfg.graph_feature_configs = {}
    ds.default_cfg.hv_dim = hv_dim

    hypernet = HyperNet(config=ds.default_cfg, depth=depth, use_explain_away=use_explain_away)
    assert not hypernet.use_edge_features()
    assert not hypernet.use_graph_features()

    batch_size = 256
    data = batch_data(ds_enum=ds, batch_size=batch_size)

    encoded_data = hypernet.forward(data, normalize=normalize, separate_levels=separate_levels)
    graph_embedding_0 = encoded_data["node_terms"] if separate_levels else encoded_data["graph_embedding"]

    start_time = time.perf_counter()
    ## ---- Order 0
    nodes_decoded_counter = hypernet.decode_order_zero_counter(graph_embedding_0)
    # Build ground‐truth Counters per graph
    ground_truth_counters = {}
    for g in range(batch_size):
        ground_truth_counters[g] = DataTransformer.get_node_counter_from_batch(batch=g, data=data)

    # Compute accuracy per graph:
    order_zero_f1s = []
    order_zero_precisions = []
    for g in range(batch_size):
        decoded_ctr = nodes_decoded_counter.get(g, Counter())
        truth_ctr = ground_truth_counters[g]

        p, _, f1 = evaluation_metrics.calculate_p_a_f1(pred=decoded_ctr, true=truth_ctr)
        order_zero_f1s.append(f1)
        order_zero_precisions.append(p)
    avg_F1_node = sum(order_zero_f1s) / batch_size
    avg_pr_node = sum(order_zero_precisions) / batch_size

    ## ---- Order 1
    graph_embedding_1 = encoded_data["edge_terms"] if separate_levels else encoded_data["graph_embedding"]
    if normalize and separate_levels:
        graph_embedding_1 = graph_embedding_1.normalize()
    unique_nodes_decoded = [sorted(nodes_decoded_counter[b].keys()) for b in range(batch_size)]
    order_one_fn = (
        hypernet.decode_order_one_counter_explain_away if use_explain_away else hypernet.decode_order_one_counter
    )

    edges_decoded_counter = order_one_fn(graph_embedding_1, unique_nodes_decoded)
    order_one_f1s = []
    order_one_precisions = []
    for b in range(batch_size):
        truth_counter = DataTransformer.get_edge_existence_counter(batch=b, data=data, indexer=hypernet.nodes_indexer)
        truth_counter = {k: 1 for k, _ in truth_counter.items()}
        pred_countre = edges_decoded_counter[b]
        p, _, f1 = evaluation_metrics.calculate_p_a_f1(pred=pred_countre, true=truth_counter)
        order_one_f1s.append(f1)
        order_one_precisions.append(p)

    avg_F1_edge = sum(order_one_f1s) / batch_size
    avg_pr_edge = sum(order_one_precisions) / batch_size

    run_time_order_one = time.perf_counter() - start_time

    logger.info(
        f"\n<-- [{ds=}] | [n_samples: {batch_size}] | [{vsa.name=}] | [{hv_dim=}] | [{depth=}] | [{normalize=}] | [{separate_levels=}] | [{use_explain_away=}] -->"
    )
    logger.info(f"\tAverage node (order 0) f1:  {avg_F1_node:.2f}")
    logger.info(f"\tAverage edge (order 1) f1:  {avg_F1_edge:.2f}\n")

    ### Save metrics
    run_metrics = {
        "dataset": ds.value,
        "n_samples": batch_size,
        "vsa": vsa.value,
        "hv_dim": hv_dim,
        "depth": depth,
        "nha_depth": nha_depth,
        "normalize": normalize,
        "separate_levels": separate_levels,
        "use_explain_away": use_explain_away,
        "runtime_order_one_s": run_time_order_one,
        "P_order_zero": avg_pr_node,
        "F1_order_zero": avg_F1_node,
        "P_order_one": avg_pr_edge,
        "F1_order_one": avg_F1_edge,
    }

    # --- new code starts here ---
    # --- save metrics to disk ---
    asset_dir = ARTIFACTS_PATH / "nodes_and_edges" / "run8_small_dim"
    asset_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = asset_dir / "res.parquet"
    csv_path = asset_dir / "res.csv"

    metrics_df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()

    # append current row
    new_row = pd.DataFrame([run_metrics])
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    # write back out
    metrics_df.to_parquet(parquet_path, index=False)
    metrics_df.to_csv(csv_path, index=False)


@pytest.mark.parametrize(
    "separate_levels",
    [
        True,
        # False
    ],
)
@pytest.mark.parametrize(
    "normalize",
    [
        # True,
        False
    ],
)
@pytest.mark.parametrize(
    "depth",
    [
        # 1,
        # 2,
        3,
    ],
)
@pytest.mark.parametrize(
    "dataset",
    [
        SupportedDataset.ZINC_NODE_DEGREE_COMB,
        # SupportedDataset.ZINC_NODE_DEGREE,
    ],
)
@pytest.mark.parametrize("vsa", [VSAModel.HRR, VSAModel.MAP])
@pytest.mark.parametrize(
    "hv_dim",
    [
        # 96 * 96,
        80 * 80,
    ],
)
@pytest.mark.parametrize(
    "use_explain_away",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "use_node_degrees",
    [
        True,
        # False
    ],
)
def test_hypernet_reconstruct_works(
    batch_data,
    dataset,
    depth,
    vsa,
    hv_dim,
    normalize,
    separate_levels,
    use_explain_away,
    use_node_degrees,
):
    ds = dataset
    ds.default_cfg.vsa = vsa
    # Disable edge and graph features
    ds.default_cfg.hv_dim = hv_dim
    ds.default_cfg.edge_feature_configs = {}
    ds.default_cfg.graph_feature_configs = {}
    ds.default_cfg.seed = 42

    start_time = time.perf_counter()
    hypernet = HyperNet(
        config=ds.default_cfg, hidden_dim=ds.default_cfg.hv_dim, depth=depth, use_explain_away=use_explain_away
    )
    assert not hypernet.use_edge_features()
    assert not hypernet.use_graph_features()

    batch_size = 1
    data = batch_data(ds_enum=ds, batch_size=batch_size)

    encoded_data = hypernet.forward(data, normalize=normalize, separate_levels=separate_levels)

    p_nodes = []
    f1_nodes = []
    p_edges = []
    f1_edges = []
    p_graphs = []
    f1_graphs = []
    f1_graphs_tests = []
    node_edit_distances = []
    edge_edit_distances = []

    data_list = data.to_data_list()
    for b in range(batch_size):
        graph_embedding = encoded_data["graph_embedding"][b]
        node_terms = encoded_data["node_terms"][b]
        edge_terms = encoded_data["edge_terms"][b]
        data_dec, node_counter_dec, edge_counter_dec = hypernet.reconstruct(
            graph_embedding, node_terms, edge_terms, use_node_degree=use_node_degrees, is_undirected=True
        )

        ## Nodes
        truth_node_counter = DataTransformer.get_node_counter_from_batch(batch=b, data=data)
        p_0, _, f1_node = evaluation_metrics.calculate_p_a_f1(pred=node_counter_dec[0], true=truth_node_counter)
        p_nodes.append(p_0)
        f1_nodes.append(f1_node)

        ## Edge Existence
        actual_edge_counter = DataTransformer.get_edge_existence_counter(
            batch=b, data=data, indexer=hypernet.nodes_indexer
        )
        edge_existence_counter = Counter({k: 1 for k, _ in actual_edge_counter.items()})
        p_1, _, f1_edge = evaluation_metrics.calculate_p_a_f1(pred=edge_counter_dec[0], true=edge_existence_counter)
        p_edges.append(p_1)
        f1_edges.append(f1_edge)

        ## Reconstruction
        actual_edge_counter = DataTransformer.get_edge_counter(batch=b, data=data)
        ## Calculate predicted edges
        x_list = data_dec.x.int().tolist()
        edges = list(zip(data_dec.edge_index[0], data_dec.edge_index[1], strict=False))
        edges_as_ints = [(a.item(), b.item()) for a, b in edges]
        pred_edges = []
        for u, v in edges_as_ints:
            pred_edges.append((tuple(x_list[u]), tuple(x_list[v])))
        pred_edge_counter = Counter(pred_edges)

        p_g, _, f1_graph = evaluation_metrics.calculate_p_a_f1(pred=pred_edge_counter, true=actual_edge_counter)

        print("-----")
        print(f"{pred_edge_counter=}")

        ## Testing injecting the opposite edge
        edges_test = list({e for u, v in edges_as_ints for e in [(v, u), (u, v)]})
        pred_edges = []
        for u, v in edges_test:
            pred_edges.append((tuple(x_list[u]), tuple(x_list[v])))
        pred_edge_counter_test = Counter(pred_edges)

        p_g_test, _, f1_graph_test = evaluation_metrics.calculate_p_a_f1(
            pred=pred_edge_counter_test, true=actual_edge_counter
        )
        print(f"{pred_edge_counter_test=}")
        print(f"{actual_edge_counter=}")
        print(f"{f1_graph=}")
        print(f"{f1_graph_test=}")
        print("-----")

        p_graphs.append(p_g)
        f1_graphs.append(f1_graph)
        f1_graphs_tests.append(f1_graph_test)

        delta_node, delta_edge, _ = evaluation_metrics.graph_edit_distance(data_pred=data_dec, data_true=data_list[b])
        node_edit_distances.append(delta_node)
        edge_edit_distances.append(delta_edge)

    rund_time = time.perf_counter() - start_time

    # === (optional) display in a notebook ===
    ### Save metrics
    run_metrics = {
        "dataset": ds.value,
        "n_samples": batch_size,
        "vsa": vsa.value,
        "hv_dim": hv_dim,
        "depth": depth,
        "normalize": normalize,
        "separate_levels": separate_levels,
        "use_explain_away": use_explain_away,
        "rund_time": rund_time,
        "avg_P_order_zero": sum(p_nodes) / batch_size,
        "avg_F1_order_zero": sum(f1_nodes) / batch_size,
        "avg_P_order_one": sum(p_edges) / batch_size,
        "avg_F1_order_one": sum(f1_edges) / batch_size,
        "avg_P_recons": sum(p_graphs) / batch_size,
        "avg_F1_recons": sum(f1_graphs) / batch_size,
        "avg_F1_recons_test": sum(f1_graphs_tests) / batch_size,
        "avg_node_edit_dist": sum(node_edit_distances) / batch_size,
        "avg_edge_edit_dist": sum(edge_edit_distances) / batch_size,
        "total_edges": data.edge_index.shape[1],
    }

    logger.info(run_metrics)
    # === save to CSV in your assets folder ===
    # --- at the end of the test, write out to your assets folder ---
    asset_dir = ARTIFACTS_PATH / "reconstruction" / "run3"
    asset_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = asset_dir / "res.parquet"
    csv_path = asset_dir / "res.csv"

    metrics_df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()

    # append current row
    new_row = pd.DataFrame([run_metrics])
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    # write back out
    metrics_df.to_parquet(parquet_path, index=False)
    metrics_df.to_csv(csv_path, index=False)


@pytest.mark.parametrize(
    "dataset",
    [
        # SupportedDataset.ZINC_NODE_DEGREE_COMB_NHA,
        SupportedDataset.ZINC_NODE_DEGREE_COMB,
    ],
)
@pytest.mark.parametrize(
    "vsa",
    [VSAModel.HRR, VSAModel.MAP],
)
@pytest.mark.parametrize(
    "nha_depth",
    [
        # 1, 2,
        3,
        # 4
    ],
)
@pytest.mark.parametrize(
    "nha_bins",
    [
        # 3, 4, 5, 6, 7, 8, 9,
        10
    ],
)
@pytest.mark.parametrize(
    "hv_dim",
    [
        32 * 32,
        48 * 48,
        64 * 64,
        # 80 * 80,
        # 96 * 96,
    ],
)
def test_hypernet_decode_order_one_is_good_enough_counter_nha(dataset, vsa, hv_dim, nha_depth, nha_bins):
    import time  # add to the top if not already there

    from pytorch_lightning import seed_everything

    seed_everything(42)

    global_model_dir = "/Users/arvandkaveh/Projects/kit/graph_hdc/_models"
    global_dataset_dir = Path("/Users/arvandkaveh/Projects/kit/graph_hdc/_datasets")

    ds = dataset
    ds.default_cfg.vsa = vsa
    # Disable edge and graph features
    ds.default_cfg.edge_feature_configs = {}
    ds.default_cfg.graph_feature_configs = {}
    ds.default_cfg.hv_dim = hv_dim
    ds.default_cfg.seed = 42
    ds.default_cfg.nha_bins = nha_bins
    ds.default_cfg.nha_depth = nha_depth
    # ds.default_cfg.node_feature_configs[Features.ATOM_TYPE] = FeatureConfig(
    #     # Added Neighbourhood awareness encodings (n distinct values)
    #     count=28
    #     * 6
    #     * nha_bins,  # 28 Atom Types, 6 Unique Node Degrees: [0.0 (for ease of indexing), 1.0, 2.0, 3.0, 4.0, 5.0]
    #     encoder_cls=CombinatoricIntegerEncoder,
    #     index_range=IndexRange((0, 3)),
    # )

    hypernet = load_or_create_hypernet(path=Path(global_model_dir), ds=ds, use_edge_codebook=False)

    assert not hypernet.use_edge_features()
    assert not hypernet.use_graph_features()

    batch_size = 256
    # Construct the composed transform
    pre_transform = Compose(
        [
            AddNodeDegree(),
            AddNeighbourhoodEncodings(depth=nha_depth, bins=nha_bins),
        ]
    )

    # Use it in your dataset
    zinc = ZINC(
        root=global_dataset_dir / f"zinc_nd_comb_nha_d{nha_depth}_b{nha_bins}", pre_transform=pre_transform, subset=True
    )
    data_list = [zinc[i] for i in range(batch_size)]
    data = Batch.from_data_list(data_list)

    encoded_data = hypernet.forward(data)

    start_time = time.perf_counter()
    ## ---- Order 0
    nodes_decoded_counter = hypernet.decode_order_zero_counter(encoded_data["node_terms"])
    # Build ground‐truth Counters per graph
    ground_truth_counters = {}
    for g in range(batch_size):
        ground_truth_counters[g] = DataTransformer.get_node_counter_from_batch(batch=g, data=data)

    # Compute accuracy per graph:
    order_zero_f1s = []
    order_zero_precisions = []
    for g in range(batch_size):
        decoded_ctr = nodes_decoded_counter.get(g, Counter())
        truth_ctr = ground_truth_counters[g]

        p, _, f1 = evaluation_metrics.calculate_p_a_f1(pred=decoded_ctr, true=truth_ctr)
        order_zero_f1s.append(f1)
        order_zero_precisions.append(p)
    avg_F1_node = sum(order_zero_f1s) / batch_size
    avg_pr_node = sum(order_zero_precisions) / batch_size

    # Count graphs with unique nodes
    unique_count = 0
    for g in zinc:
        t = g.x
        flattened = t.view(t.size(0), -1)
        unique_rows = torch.unique(flattened, dim=0)
        unique_count += int(unique_rows.size(0) == t.size(0))
    unique_proportion = unique_count / len(data_list)

    run_time_order_one = time.perf_counter() - start_time

    logger.info(f"\n<-- [{ds=}] | [n_samples: {batch_size}] | [{vsa.name=}] | [{hv_dim=}]] -->")
    logger.info(f"\tAverage node (order 0) f1:  {avg_F1_node:.2f}")

    ### Save metrics
    run_metrics = {
        "dataset": ds.value,
        "n_samples": batch_size,
        "vsa": vsa.value,
        "hv_dim": hv_dim,
        "nha_depth": nha_depth,
        "nha_bins": nha_bins,
        "unique_node_count": unique_count,
        "unique_proportion": unique_proportion,
        "runtime_order_one_s": run_time_order_one,
        "P_order_zero": avg_pr_node,
        "F1_order_zero": avg_F1_node,
    }

    # --- new code starts here ---
    # --- save metrics to disk ---
    asset_dir = ARTIFACTS_PATH / "nodes_and_edges" / "run8_nha_small_dimensions"
    asset_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = asset_dir / "res.parquet"
    csv_path = asset_dir / "res.csv"

    metrics_df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()

    # append current row
    new_row = pd.DataFrame([run_metrics])
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    # write back out
    metrics_df.to_parquet(parquet_path, index=False)
    metrics_df.to_csv(csv_path, index=False)


@pytest.mark.parametrize(
    "dataset",
    [
        SupportedDataset.ZINC_NODE_DEGREE_COMB_NHA,
    ],
)
@pytest.mark.parametrize(
    "vsa",
    [VSAModel.HRR, VSAModel.MAP],
)
@pytest.mark.parametrize(
    "nha_depth",
    [1, 2, 3, 4],
)
@pytest.mark.parametrize(
    "nha_bins",
    [
        # 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        # 16,
        # 17,
        # 18,
        # 19,
        20,
    ],
)
@pytest.mark.parametrize(
    "hv_dim",
    [80 * 80, 88 * 88, 96 * 96, 112 * 112],
)
def test_hypernet_decode_order_one_is_good_enough_counter_nha__one_hv_dict(dataset, vsa, hv_dim, nha_depth, nha_bins):
    import time  # add to the top if not already there

    import torch.nn.functional as F
    from pytorch_lightning import seed_everything

    seed_everything(42)

    global_model_dir = "/Users/arvandkaveh/Projects/kit/graph_hdc/_models"
    global_dataset_dir = Path("/Users/arvandkaveh/Projects/kit/graph_hdc/_datasets")

    ds = dataset
    ds.default_cfg.vsa = vsa
    # Disable edge and graph features
    ds.default_cfg.edge_feature_configs = {}
    ds.default_cfg.graph_feature_configs = {}
    ds.default_cfg.hv_dim = hv_dim
    ds.default_cfg.seed = 42
    ds.default_cfg.nha_bins = nha_bins
    ds.default_cfg.nha_depth = nha_depth
    # ds.default_cfg.node_feature_configs[Features.ATOM_TYPE] = FeatureConfig(
    #     # Added Neighbourhood awareness encodings (n distinct values)
    #     count=28
    #     * 6
    #     * nha_bins,  # 28 Atom Types, 6 Unique Node Degrees: [0.0 (for ease of indexing), 1.0, 2.0, 3.0, 4.0, 5.0]
    #     encoder_cls=CombinatoricIntegerEncoder,
    #     index_range=IndexRange((0, 3)),
    # )

    hypernet = load_or_create_hypernet(path=Path(global_model_dir), ds=ds, use_edge_codebook=False)

    assert not hypernet.use_edge_features()
    assert not hypernet.use_graph_features()

    batch_size = 512
    # Construct the composed transform
    pre_transform = Compose(
        [
            AddNodeDegree(),
            AddNeighbourhoodEncodings(depth=nha_depth, bins=nha_bins),
        ]
    )

    # Use it in your dataset
    zinc = ZINC(
        root=global_dataset_dir / f"zinc_nd_comb_nha_d{nha_depth}_b{nha_bins}", pre_transform=pre_transform, subset=True
    )
    data_list = [zinc[i] for i in range(batch_size)]
    data = Batch.from_data_list(data_list)

    # Encode the whole graph in one HV
    encoded_data = hypernet.forward(data)
    node_term = encoded_data["node_terms"]
    graph_term = encoded_data["graph_embedding"]
    v_n = torchhd.random(1, dimensions=hv_dim, vsa=vsa.value)
    v_g = torchhd.random(1, dimensions=hv_dim, vsa=vsa.value)
    node_term_key_value = v_n.bind(node_term)
    graph_term_key_value = v_g.bind(graph_term)

    stacked = torch.stack([node_term_key_value, graph_term_key_value], dim=0).transpose(0, 1)
    g_hv = torchhd.multiset(stacked)

    # Extract the node_terms from Graph Hyper Vector
    node_term_decoded = torchhd.bind(v_n.inverse(), g_hv)
    graph_term_decoded = torchhd.bind(v_g.inverse(), g_hv)

    print("Similarity metrics after decoding")
    node_term_sims = F.cosine_similarity(node_term_decoded, node_term, dim=1)
    graph_term_sims = F.cosine_similarity(graph_term_decoded, graph_term, dim=1)

    avg_node_term_sims = torch.mean(node_term_sims, dim=0).item()
    avg_graph_term_sims = torch.mean(graph_term_sims, dim=0).item()

    start_time = time.perf_counter()
    ## ---- Order 0
    nodes_decoded_counter = hypernet.decode_order_zero_counter(node_term_decoded)
    # Build ground‐truth Counters per graph
    ground_truth_counters = {}
    for g in range(batch_size):
        ground_truth_counters[g] = DataTransformer.get_node_counter_from_batch(batch=g, data=data)

    # Compute accuracy per graph:
    order_zero_f1s = []
    order_zero_precisions = []
    for g in range(batch_size):
        decoded_ctr = nodes_decoded_counter.get(g, Counter())
        truth_ctr = ground_truth_counters[g]

        p, _, f1 = evaluation_metrics.calculate_p_a_f1(pred=decoded_ctr, true=truth_ctr)
        order_zero_f1s.append(f1)
        order_zero_precisions.append(p)
    avg_F1_node = sum(order_zero_f1s) / batch_size
    avg_pr_node = sum(order_zero_precisions) / batch_size

    # Count graphs with unique nodes
    unique_count = 0
    for g in zinc:
        t = g.x
        flattened = t.view(t.size(0), -1)
        unique_rows = torch.unique(flattened, dim=0)
        unique_count += int(unique_rows.size(0) == t.size(0))
    unique_proportion = unique_count / len(data_list)

    run_time_order_one = time.perf_counter() - start_time

    logger.info(f"\n<-- [{ds=}] | [n_samples: {batch_size}] | [{vsa.name=}] | [{hv_dim=}]] -->")
    logger.info(f"\tAverage node (order 0) f1:  {avg_F1_node:.2f}")

    ### Save metrics
    run_metrics = {
        "dataset": ds.value,
        "n_samples": batch_size,
        "vsa": vsa.value,
        "hv_dim": hv_dim,
        "nha_depth": nha_depth,
        "nha_bins": nha_bins,
        "unique_node_count": unique_count,
        "unique_proportion": unique_proportion,
        "runtime_order_one_s": run_time_order_one,
        "avg_node_term_sim": avg_node_term_sims,
        "avg_graph_term_sim": avg_graph_term_sims,
        "P_order_zero": avg_pr_node,
        "F1_order_zero": avg_F1_node,
    }

    # --- new code starts here ---
    # --- save metrics to disk ---
    asset_dir = ARTIFACTS_PATH / "nodes_and_edges" / "run8_hash_table1"
    asset_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = asset_dir / "res.parquet"
    csv_path = asset_dir / "res.csv"

    metrics_df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()

    # append current row
    new_row = pd.DataFrame([run_metrics])
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    # write back out
    metrics_df.to_parquet(parquet_path, index=False)
    metrics_df.to_csv(csv_path, index=False)


@pytest.mark.parametrize(
    "dataset",
    [
        SupportedDataset.ZINC_NODE_DEGREE_COMB_NHA,
    ],
)
@pytest.mark.parametrize(
    "vsa",
    [VSAModel.HRR, VSAModel.MAP],
)
@pytest.mark.parametrize(
    "nha_depth",
    [
        # 1,
        # 2,
        3,
        # 4
    ],
)
@pytest.mark.parametrize(
    "nha_bins",
    [
        # 3,
        # 5,
        # 7,
        # 9,
        10,
        # 13,
        # 17,
        # 20,
    ],
)
@pytest.mark.parametrize(
    "hv_dim",
    [
        32 * 32,
        48 * 48,
        64 * 64,
        # 80 * 80, 88 * 88, 96 * 96, 112 * 112,
        # 120 * 120,
        # 128 * 128,
        # 136 * 136,
    ],
)
def test_hypernet_decode_order_one_is_good_enough_counter_nha__one_hv_dict__three_levels(
    dataset, vsa, hv_dim, nha_depth, nha_bins
):
    import time  # add to the top if not already there

    import torch.nn.functional as F
    from pytorch_lightning import seed_everything

    seed_everything(42)

    global_model_dir = "/Users/arvandkaveh/Projects/kit/graph_hdc/_models"
    global_dataset_dir = Path("/Users/arvandkaveh/Projects/kit/graph_hdc/_datasets")

    ds = dataset
    ds.default_cfg.vsa = vsa
    # Disable edge and graph features
    ds.default_cfg.edge_feature_configs = {}
    ds.default_cfg.graph_feature_configs = {}
    ds.default_cfg.hv_dim = hv_dim
    ds.default_cfg.seed = 42
    ds.default_cfg.nha_bins = nha_bins
    ds.default_cfg.nha_depth = nha_depth
    ds.default_cfg.node_feature_configs[Features.ATOM_TYPE] = FeatureConfig(
        # Added Neighbourhood awareness encodings (n distinct values)
        count=28
        * 6
        * nha_bins,  # 28 Atom Types, 6 Unique Node Degrees: [0.0 (for ease of indexing), 1.0, 2.0, 3.0, 4.0, 5.0]
        encoder_cls=CombinatoricIntegerEncoder,
        index_range=IndexRange((0, 3)),
    )

    hypernet = load_or_create_hypernet(path=Path(global_model_dir), ds=ds, use_edge_codebook=False)

    assert not hypernet.use_edge_features()
    assert not hypernet.use_graph_features()

    batch_size = 512
    # Construct the composed transform
    pre_transform = Compose(
        [
            AddNodeDegree(),
            AddNeighbourhoodEncodings(depth=nha_depth, bins=nha_bins),
        ]
    )

    # Use it in your dataset
    zinc = ZINC(
        root=global_dataset_dir / f"zinc_nd_comb_nha_d{nha_depth}_b{nha_bins}", pre_transform=pre_transform, subset=True
    )
    data_list = [zinc[i] for i in range(batch_size)]
    data = Batch.from_data_list(data_list)

    # Encode the whole graph in one HV
    encoded_data = hypernet.forward(data)
    node_term = encoded_data["node_terms"]
    edge_term = encoded_data["edge_terms"]
    graph_term = encoded_data["graph_embedding"]
    v_n = torchhd.random(1, dimensions=hv_dim, vsa=vsa.value)
    v_e = torchhd.random(1, dimensions=hv_dim, vsa=vsa.value)
    v_g = torchhd.random(1, dimensions=hv_dim, vsa=vsa.value)
    node_term_key_value = v_n.bind(node_term)
    edge_term_key_value = v_e.bind(edge_term)
    graph_term_key_value = v_g.bind(graph_term)

    stacked = torch.stack([node_term_key_value, edge_term_key_value, graph_term_key_value], dim=0).transpose(0, 1)
    g_hv = torchhd.multiset(stacked)

    # Extract the node_terms from Graph Hyper Vector
    node_term_extract = torchhd.bind(v_n.inverse(), g_hv)
    edge_term_extract = torchhd.bind(v_e.inverse(), g_hv)
    graph_term_extract = torchhd.bind(v_g.inverse(), g_hv)

    print("Similarity metrics after decoding")
    node_term_sims = F.cosine_similarity(node_term_extract, node_term, dim=1)
    edge_term_sims = F.cosine_similarity(edge_term_extract, edge_term, dim=1)
    graph_term_sims = F.cosine_similarity(graph_term_extract, graph_term, dim=1)

    avg_node_term_sims = torch.mean(node_term_sims, dim=0).item()
    avg_edge_term_sims = torch.mean(edge_term_sims, dim=0).item()
    avg_graph_term_sims = torch.mean(graph_term_sims, dim=0).item()

    start_time = time.perf_counter()
    ## ---- Order 0
    nodes_decoded_counter = hypernet.decode_order_zero_counter(node_term_extract)
    # Build ground‐truth Counters per graph
    ground_truth_counters = {}
    for g in range(batch_size):
        ground_truth_counters[g] = DataTransformer.get_node_counter_from_batch(batch=g, data=data)

    # Compute accuracy per graph:
    order_zero_f1s = []
    order_zero_precisions = []
    for g in range(batch_size):
        decoded_ctr = nodes_decoded_counter.get(g, Counter())
        truth_ctr = ground_truth_counters[g]

        p, _, f1 = evaluation_metrics.calculate_p_a_f1(pred=decoded_ctr, true=truth_ctr)
        order_zero_f1s.append(f1)
        order_zero_precisions.append(p)
    avg_F1_node = sum(order_zero_f1s) / batch_size
    avg_pr_node = sum(order_zero_precisions) / batch_size

    # Count graphs with unique nodes
    unique_count = 0
    for g in zinc:
        t = g.x
        flattened = t.view(t.size(0), -1)
        unique_rows = torch.unique(flattened, dim=0)
        unique_count += int(unique_rows.size(0) == t.size(0))
    unique_proportion = unique_count / len(data_list)

    run_time_order_one = time.perf_counter() - start_time

    logger.info(f"\n<-- [{ds=}] | [n_samples: {batch_size}] | [{vsa.name=}] | [{hv_dim=}]] -->")
    logger.info(f"\tAverage node (order 0) f1:  {avg_F1_node:.2f}")

    ### Save metrics
    run_metrics = {
        "dataset": ds.value,
        "n_samples": batch_size,
        "vsa": vsa.value,
        "hv_dim": hv_dim,
        "nha_depth": nha_depth,
        "nha_bins": nha_bins,
        "unique_node_count": unique_count,
        "unique_proportion": unique_proportion,
        "runtime_order_one_s": run_time_order_one,
        "avg_node_term_sim": avg_node_term_sims,
        "avg_edge_term_sim": avg_edge_term_sims,
        "avg_graph_term_sim": avg_graph_term_sims,
        "P_order_zero": avg_pr_node,
        "F1_order_zero": avg_F1_node,
    }

    # --- new code starts here ---
    # --- save metrics to disk ---
    asset_dir = ARTIFACTS_PATH / "nodes_and_edges" / "run9_hna_two_level"
    asset_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = asset_dir / "res.parquet"
    csv_path = asset_dir / "res.csv"

    metrics_df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()

    # append current row
    new_row = pd.DataFrame([run_metrics])
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    # write back out
    metrics_df.to_parquet(parquet_path, index=False)
    metrics_df.to_csv(csv_path, index=False)


@pytest.mark.parametrize(
    "dataset",
    [
        SupportedDataset.ZINC_NODE_DEGREE_COMB_NHA,
    ],
)
@pytest.mark.parametrize(
    "vsa",
    [VSAModel.HRR, VSAModel.MAP],
)
@pytest.mark.parametrize(
    "nha_depth",
    [
        1,
        2,
        3,
    ],
)
@pytest.mark.parametrize(
    "nha_bins",
    [
        # 3,
        # 5,
        # 7,
        # 9,
        10,
        # 13,
        # 17,
        # 20,
    ],
)
@pytest.mark.parametrize(
    "hv_dim",
    [80 * 80, 88 * 88, 96 * 96, 112 * 112, 120 * 120, 128 * 128, 136 * 136],
)
def test_hypernet_decode_order_one_is_good_enough_counter_nha__one_hv_dict__two_levels(
    dataset, vsa, hv_dim, nha_depth, nha_bins
):
    import time  # add to the top if not already there

    import torch.nn.functional as F
    from pytorch_lightning import seed_everything

    seed_everything(42)

    global_model_dir = "/Users/arvandkaveh/Projects/kit/graph_hdc/_models"
    global_dataset_dir = Path("/Users/arvandkaveh/Projects/kit/graph_hdc/_datasets")

    ds = dataset
    ds.default_cfg.vsa = vsa
    # Disable edge and graph features
    ds.default_cfg.edge_feature_configs = {}
    ds.default_cfg.graph_feature_configs = {}
    ds.default_cfg.hv_dim = hv_dim
    ds.default_cfg.seed = 42
    ds.default_cfg.nha_bins = nha_bins
    ds.default_cfg.nha_depth = nha_depth
    ds.default_cfg.node_feature_configs[Features.ATOM_TYPE] = FeatureConfig(
        # Added Neighbourhood awareness encodings (n distinct values)
        count=28
        * 6
        * nha_bins,  # 28 Atom Types, 6 Unique Node Degrees: [0.0 (for ease of indexing), 1.0, 2.0, 3.0, 4.0, 5.0]
        encoder_cls=CombinatoricIntegerEncoder,
        index_range=IndexRange((0, 3)),
    )

    hypernet = load_or_create_hypernet(path=Path(global_model_dir), ds=ds, use_edge_codebook=False)

    assert not hypernet.use_edge_features()
    assert not hypernet.use_graph_features()

    batch_size = 1024
    # Construct the composed transform
    pre_transform = Compose(
        [
            AddNodeDegree(),
            AddNeighbourhoodEncodings(depth=nha_depth, bins=nha_bins),
        ]
    )

    # Use it in your dataset
    zinc = ZINC(
        root=global_dataset_dir / f"zinc_nd_comb_nha_d{nha_depth}_b{nha_bins}", pre_transform=pre_transform, subset=True
    )
    data_list = [zinc[i] for i in range(batch_size)]
    data = Batch.from_data_list(data_list)

    from torchhd import structures

    # Encode the whole graph in one HV
    encoded_data = hypernet.forward(data)
    node_term = encoded_data["node_terms"]
    edge_term = encoded_data["edge_terms"]
    graph_term = encoded_data["graph_embedding"]

    ## Create a HashTable
    graph_hash_table = structures.HashTable(dim_or_input=hv_dim, vsa=vsa.value)
    var = torchhd.random(3, hv_dim, vsa=vsa.value)
    graph_hash_table.add(key=var[0], value=node_term)
    graph_hash_table.add(key=var[1], value=edge_term)
    graph_hash_table.add(key=var[2], value=graph_term)

    # Extract the node_terms from Graph Hyper Vector
    node_term_extract = graph_hash_table.get(var[0])
    edge_term_extract = graph_hash_table.get(var[1])
    graph_term_extract = graph_hash_table.get(var[2])

    print("Similarity metrics after decoding")
    node_term_sim = F.cosine_similarity(node_term_extract, node_term, dim=1).mean().item()
    edge_term_sim = F.cosine_similarity(edge_term_extract, edge_term, dim=1).mean().item()
    graph_term_sim = F.cosine_similarity(graph_term_extract, graph_term, dim=1).mean().item()

    start_time = time.perf_counter()
    ## ---- Order 0
    nodes_decoded_counter = hypernet.decode_order_zero_counter(node_term_extract)
    # Build ground‐truth Counters per graph
    ground_truth_counters = {}
    for g in range(batch_size):
        ground_truth_counters[g] = DataTransformer.get_node_counter_from_batch(batch=g, data=data)

    # Compute accuracy per graph:
    order_zero_f1s = []
    order_zero_precisions = []
    for g in range(batch_size):
        decoded_ctr = nodes_decoded_counter.get(g, Counter())
        truth_ctr = ground_truth_counters[g]

        p, _, f1 = evaluation_metrics.calculate_p_a_f1(pred=decoded_ctr, true=truth_ctr)
        order_zero_f1s.append(f1)
        order_zero_precisions.append(p)
    avg_F1_node = sum(order_zero_f1s) / batch_size
    avg_pr_node = sum(order_zero_precisions) / batch_size

    # Count graphs with unique nodes
    unique_count = 0
    for g in zinc:
        t = g.x
        flattened = t.view(t.size(0), -1)
        unique_rows = torch.unique(flattened, dim=0)
        unique_count += int(unique_rows.size(0) == t.size(0))
    unique_proportion = unique_count / len(data_list)

    run_time_order_one = time.perf_counter() - start_time

    logger.info(f"\tAverage node (order 0) f1:  {avg_F1_node:.2f}")
    logger.info(f"\n<-- [{ds=}] | [n_samples: {batch_size}] | [{vsa.name=}] | [{hv_dim=}]] -->")

    ### Save metrics
    run_metrics = {
        "dataset": ds.value,
        "n_samples": batch_size,
        "vsa": vsa.value,
        "hv_dim": hv_dim,
        "nha_depth": nha_depth,
        "nha_bins": nha_bins,
        "unique_node_count": unique_count,
        "unique_proportion": unique_proportion,
        "runtime_order_one_s": run_time_order_one,
        "avg_node_term_cos_sim": node_term_sim,
        "avg_edge_term_coos_sim": edge_term_sim,
        "avg_graph_term_cos_sim": graph_term_sim,
        "P_order_zero": avg_pr_node,
        "F1_order_zero": avg_F1_node,
    }

    # --- new code starts here ---
    # --- save metrics to disk ---
    asset_dir = ARTIFACTS_PATH / "nodes_and_edges" / "run8_hash_table_three_lvl"
    asset_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = asset_dir / "torchhd_hash_table.parquet"
    csv_path = asset_dir / "torchhd_hach_table.csv"

    metrics_df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()

    # append current row
    new_row = pd.DataFrame([run_metrics])
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    # write back out
    metrics_df.to_parquet(parquet_path, index=False)
    metrics_df.to_csv(csv_path, index=False)


@pytest.mark.parametrize(
    "dataset",
    [
        SupportedDataset.ZINC_NODE_DEGREE_COMB,
    ],
)
@pytest.mark.parametrize(
    "vsa",
    [VSAModel.HRR, VSAModel.MAP],
)
@pytest.mark.parametrize(
    "hv_dim",
    [48 * 48, 56 * 56, 64 * 64, 72 * 72, 80 * 80, 88 * 88, 96 * 96, 112 * 112],
)
def test_hypernet_decode_order_one_is_good_enough_counter_comb_two_level_hash_table(dataset, vsa, hv_dim, batch_data):
    import time  # add to the top if not already there

    import torch.nn.functional as F
    from pytorch_lightning import seed_everything

    seed = 42
    seed_everything(seed)

    ds = dataset
    ds.default_cfg.vsa = vsa
    # Disable edge and graph features
    ds.default_cfg.edge_feature_configs = {}
    ds.default_cfg.graph_feature_configs = {}
    ds.default_cfg.hv_dim = hv_dim

    hypernet = HyperNet(config=ds.default_cfg, use_edge_codebook=False)
    assert not hypernet.use_edge_features()
    assert not hypernet.use_graph_features()

    batch_size = 128
    data = batch_data(ds_enum=ds, batch_size=batch_size)

    from torchhd import structures

    # Encode the whole graph in one HV
    encoded_data = hypernet.forward(data)
    node_term = encoded_data["node_terms"]
    edge_term = encoded_data["edge_terms"]
    graph_term = encoded_data["graph_embedding"]

    ## Create a HashTable
    graph_hash_table = structures.HashTable(dim_or_input=hv_dim, vsa=vsa.value)
    var = torchhd.random(3, hv_dim, vsa=vsa.value)
    graph_hash_table.add(key=var[0], value=node_term)
    graph_hash_table.add(key=var[1], value=edge_term)
    graph_hash_table.add(key=var[2], value=graph_term)

    # Extract the node_terms from Graph Hyper Vector
    node_term_extract = graph_hash_table.get(var[0])
    edge_term_extract = graph_hash_table.get(var[1])
    graph_term_extract = graph_hash_table.get(var[2])

    print("Similarity metrics after decoding")
    node_term_sim = F.cosine_similarity(node_term_extract, node_term, dim=1).mean().item()
    # edge_term_sim = F.cosine_similarity(edge_term_extract, edge_term, dim=1).mean().item()
    graph_term_sim = F.cosine_similarity(graph_term_extract, graph_term, dim=1).mean().item()

    start_time = time.perf_counter()
    ## ---- Order 0
    nodes_decoded_counter = hypernet.decode_order_zero_counter(node_term_extract)
    # Build ground‐truth Counters per graph
    ground_truth_counters = {}
    for g in range(batch_size):
        ground_truth_counters[g] = DataTransformer.get_node_counter_from_batch(batch=g, data=data)

    # Compute accuracy per graph:
    order_zero_f1s = []
    order_zero_precisions = []
    for g in range(batch_size):
        decoded_ctr = nodes_decoded_counter.get(g, Counter())
        truth_ctr = ground_truth_counters[g]

        p, _, f1 = evaluation_metrics.calculate_p_a_f1(pred=decoded_ctr, true=truth_ctr)
        order_zero_f1s.append(f1)
        order_zero_precisions.append(p)
    avg_F1_node = sum(order_zero_f1s) / batch_size
    avg_pr_node = sum(order_zero_precisions) / batch_size

    run_time_order_one = time.perf_counter() - start_time

    logger.info(f"\n<-- [{ds=}] | [n_samples: {batch_size}] | [{vsa.name=}] | [{hv_dim=}]] -->")
    logger.info(f"\tAverage node (order 0) f1:  {avg_F1_node:.2f}")

    ### Save metrics
    run_metrics = {
        "dataset": ds.value,
        "seed": seed,
        "n_samples": batch_size,
        "vsa": vsa.value,
        "hv_dim": hv_dim,
        "runtime_order_one_s": run_time_order_one,
        "avg_node_term_sim": node_term_sim,
        "avg_graph_term_sim": graph_term_sim,
        "P_order_zero": avg_pr_node,
        "F1_order_zero": avg_F1_node,
    }

    # --- new code starts here ---
    # --- save metrics to disk ---
    asset_dir = ARTIFACTS_PATH / "nodes_and_edges" / "run_10_comb_two_level_hashtable"
    asset_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = asset_dir / "res_three_levels.parquet"
    csv_path = asset_dir / "res_three_levels.csv"

    metrics_df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()

    # append current row
    new_row = pd.DataFrame([run_metrics])
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    # write back out
    metrics_df.to_parquet(parquet_path, index=False)
    metrics_df.to_csv(csv_path, index=False)


@pytest.mark.parametrize(
    "dataset",
    [
        SupportedDataset.ZINC_NODE_DEGREE_COMB,
    ],
)
@pytest.mark.parametrize(
    "vsa",
    [VSAModel.HRR, VSAModel.MAP],
)
@pytest.mark.parametrize(
    "hv_dim",
    [64 * 64, 72 * 72, 80 * 80, 88 * 88, 96 * 96, 112 * 112, 120 * 120, 128 * 128, 136 * 136],
)
def test_hypernet_decode_order_one_is_good_enough_counter_comb_one_hv_two_levels(
    dataset,
    vsa,
    hv_dim,
):
    import time  # add to the top if not already there

    import torch.nn.functional as F
    from pytorch_lightning import seed_everything

    seed = 42
    seed_everything(seed)

    global_model_dir = "/Users/arvandkaveh/Projects/kit/graph_hdc/_models"
    global_dataset_dir = Path("/Users/arvandkaveh/Projects/kit/graph_hdc/_datasets")

    ds = dataset
    ds.default_cfg.vsa = vsa
    # Disable edge and graph features
    ds.default_cfg.edge_feature_configs = {}
    ds.default_cfg.graph_feature_configs = {}
    ds.default_cfg.hv_dim = hv_dim
    ds.default_cfg.seed = 42

    hypernet = load_or_create_hypernet(path=Path(global_model_dir), ds=ds, use_edge_codebook=False)

    assert not hypernet.use_edge_features()
    assert not hypernet.use_graph_features()

    batch_size = 2048
    # Use it in your dataset
    zinc = ZINC(root=global_dataset_dir / ds.value, pre_transform=AddNodeDegree(), subset=True)
    data_list = [zinc[i] for i in range(batch_size)]
    data = Batch.from_data_list(data_list)

    from torchhd import structures

    # Encode the whole graph in one HV
    encoded_data = hypernet.forward(data)
    node_term = encoded_data["node_terms"]
    edge_term = encoded_data["edge_terms"]
    graph_term = encoded_data["graph_embedding"]

    ## Create a HashTable
    graph_hash_table = structures.HashTable(dim_or_input=hv_dim, vsa=vsa.value)
    var = torchhd.random(3, hv_dim, vsa=vsa.value)
    graph_hash_table.add(key=var[0], value=node_term)
    graph_hash_table.add(key=var[1], value=edge_term)
    graph_hash_table.add(key=var[2], value=graph_term)

    # Extract the node_terms from Graph Hyper Vector
    node_term_extract = graph_hash_table.get(var[0])
    edge_term_extract = graph_hash_table.get(var[1])
    graph_term_extract = graph_hash_table.get(var[2])

    print("Similarity metrics after decoding")
    node_term_sim = F.cosine_similarity(node_term_extract, node_term, dim=1).mean().item()
    edge_term_sim = F.cosine_similarity(edge_term_extract, edge_term, dim=1).mean().item()
    graph_term_sim = F.cosine_similarity(graph_term_extract, graph_term, dim=1).mean().item()

    start_time = time.perf_counter()
    ## ---- Order 0
    nodes_decoded_counter = hypernet.decode_order_zero_counter(node_term_extract)
    # Build ground‐truth Counters per graph
    ground_truth_counters = {}
    for g in range(batch_size):
        ground_truth_counters[g] = DataTransformer.get_node_counter_from_batch(batch=g, data=data)

    # Compute accuracy per graph:
    order_zero_f1s = []
    order_zero_precisions = []
    for g in range(batch_size):
        decoded_ctr = nodes_decoded_counter.get(g, Counter())
        truth_ctr = ground_truth_counters[g]

        p, _, f1 = evaluation_metrics.calculate_p_a_f1(pred=decoded_ctr, true=truth_ctr)
        order_zero_f1s.append(f1)
        order_zero_precisions.append(p)
    avg_F1_node = sum(order_zero_f1s) / batch_size
    avg_pr_node = sum(order_zero_precisions) / batch_size

    # Count graphs with unique nodes
    unique_count = 0
    for g in zinc:
        t = g.x
        flattened = t.view(t.size(0), -1)
        unique_rows = torch.unique(flattened, dim=0)
        unique_count += int(unique_rows.size(0) == t.size(0))
    unique_proportion = unique_count / len(data_list)

    run_time_order_one = time.perf_counter() - start_time

    logger.info(f"\tAverage node (order 0) f1:  {avg_F1_node:.2f}")
    logger.info(f"\n<-- [{ds=}] | [n_samples: {batch_size}] | [{vsa.name=}] | [{hv_dim=}]] -->")

    ### Save metrics
    run_metrics = {
        "dataset": ds.value,
        "n_samples": batch_size,
        "vsa": vsa.value,
        "hv_dim": hv_dim,
        "seed": seed,
        "unique_node_count": unique_count,
        "unique_proportion": unique_proportion,
        "runtime_order_one_s": run_time_order_one,
        "avg_node_term_cos_sim": node_term_sim,
        "avg_edge_term_cos_sim": edge_term_sim,
        "avg_graph_term_cos_sim": graph_term_sim,
        "P_order_zero": avg_pr_node,
        "F1_order_zero": avg_F1_node,
    }

    # --- new code starts here ---
    # --- save metrics to disk ---
    asset_dir = ARTIFACTS_PATH / "nodes_and_edges" / "run_11_zinc_comb_one_hv_two_levels"
    asset_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = asset_dir / "torchhd_hash_table__3.parquet"
    csv_path = asset_dir / "torchhd_hach_table__3.csv"

    metrics_df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()

    # append current row
    new_row = pd.DataFrame([run_metrics])
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    # write back out
    metrics_df.to_parquet(parquet_path, index=False)
    metrics_df.to_csv(csv_path, index=False)
