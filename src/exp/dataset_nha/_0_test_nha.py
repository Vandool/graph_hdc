import time
from collections import Counter
from itertools import product
from pathlib import Path

import pandas as pd
from torch_geometric.data import Batch
from torch_geometric.datasets import ZINC

from src import evaluation_metrics
from src.datasets import Compose, AddNodeDegree, AddNeighbourhoodEncodings
from src.encoding.configs_and_constants import SupportedDataset, Features, FeatureConfig, IndexRange
from src.encoding.feature_encoders import CombinatoricIntegerEncoder
from src.encoding.graph_encoders import load_or_create_hypernet
from src.encoding.the_types import VSAModel
from src.utils.utils import DataTransformer


def run_all_combinations():
    global_model_dir = "/Users/arvandkaveh/Projects/kit/graph_hdc/_models"
    global_dataset_dir = "/Users/arvandkaveh/Projects/kit/graph_hdc/_datasets"

    asset_dir = Path("artifacts/nodes_and_edges/run7_nha")
    asset_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = asset_dir / "res.parquet"
    csv_path = asset_dir / "res.csv"

    metrics_df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()

    param_grid = product(
        [SupportedDataset.ZINC_NODE_DEGREE_COMB_NHA],
        [VSAModel.HRR, VSAModel.MAP],
        [80 * 80, 96 * 96],
        [True, False],
        [1, 2, 3, 4],
        [3],
    )

    for dataset, vsa, hv_dim, use_explain_away, nha_depth, nha_bins in param_grid:
        ds = dataset
        ds.default_cfg.vsa = vsa
        ds.default_cfg.edge_feature_configs = {}
        ds.default_cfg.graph_feature_configs = {}
        ds.default_cfg.hv_dim = hv_dim
        ds.default_cfg.nha_bins = nha_bins
        ds.default_cfg.nha_depth = nha_depth
        ds.default_cfg.node_feature_configs[Features.ATOM_TYPE] = FeatureConfig(
            count=28 * 6 * nha_bins,
            encoder_cls=CombinatoricIntegerEncoder,
            index_range=IndexRange((0, 3)),
        )

        hypernet = load_or_create_hypernet(path=Path(global_model_dir), ds=ds)

        assert not hypernet.use_edge_features()
        assert not hypernet.use_graph_features()

        batch_size = 16
        pre_transform = Compose([
            AddNodeDegree(),
            AddNeighbourhoodEncodings(depth=nha_depth, bins=nha_bins),
        ])
        dataset = ZINC(root=global_dataset_dir, pre_transform=pre_transform)
        data_list = [dataset[i] for i in range(batch_size)]
        data = Batch.from_data_list(data_list)
        encoded_data = hypernet.forward(data)

        start_time = time.perf_counter()

        # --- Order 0
        nodes_decoded_counter = hypernet.decode_order_zero_counter(encoded_data['node_terms'])
        ground_truth_counters = {
            g: DataTransformer.get_node_counter_from_batch(batch=g, data=data)
            for g in range(batch_size)
        }

        order_zero_f1s, order_zero_precisions = [], []
        for g in range(batch_size):
            decoded_ctr = nodes_decoded_counter.get(g, Counter())
            truth_ctr = ground_truth_counters[g]
            p, _, f1 = evaluation_metrics.calculate_p_a_f1(pred=decoded_ctr, true=truth_ctr)
            order_zero_f1s.append(f1)
            order_zero_precisions.append(p)

        avg_F1_node = sum(order_zero_f1s) / batch_size
        avg_pr_node = sum(order_zero_precisions) / batch_size

        # --- Order 1
        unique_nodes_decoded = [sorted(nodes_decoded_counter[b].keys()) for b in range(batch_size)]
        edges_decoded_counter = hypernet.decode_order_one_counter_explain_away_faster(
            encoded_data["edge_terms"], unique_nodes_decoded
        )

        order_one_f1s, order_one_precisions = [], []
        for b in range(batch_size):
            truth_counter = DataTransformer.get_edge_existence_counter(batch=b, data=data, indexer=hypernet.nodes_indexer)
            truth_counter = {k: 1 for k in truth_counter}
            pred_countre = edges_decoded_counter[b]
            p, _, f1 = evaluation_metrics.calculate_p_a_f1(pred=pred_countre, true=truth_counter)
            order_one_f1s.append(f1)
            order_one_precisions.append(p)

        avg_F1_edge = sum(order_one_f1s) / batch_size
        avg_pr_edge = sum(order_one_precisions) / batch_size
        run_time_order_one = time.perf_counter() - start_time

        print(
            f"\n<-- [{ds=}] | [n_samples: {batch_size}] | [{vsa.name=}] | [{hv_dim=}] | [{use_explain_away=}] -->"
        )
        print(f"\tAverage node (order 0) f1:  {avg_F1_node:.2f}")
        print(f"\tAverage edge (order 1) f1:  {avg_F1_edge:.2f}\n")

        run_metrics = {
            "dataset": ds.value,
            "n_samples": batch_size,
            "vsa": vsa.value,
            "hv_dim": hv_dim,
            "nha_depth": nha_depth,
            "nha_bins": nha_bins,
            "use_explain_away": use_explain_away,
            "runtime_order_one_s": run_time_order_one,
            "P_order_zero": avg_pr_node,
            "F1_order_zero": avg_F1_node,
            "P_order_one": avg_pr_edge,
            "F1_order_one": avg_F1_edge,
        }

        metrics_df = pd.concat([metrics_df, pd.DataFrame([run_metrics])], ignore_index=True)
        metrics_df.to_parquet(parquet_path, index=False)
        metrics_df.to_csv(csv_path, index=False)