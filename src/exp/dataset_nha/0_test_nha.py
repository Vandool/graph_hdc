import time
from collections import Counter
from itertools import product
from pathlib import Path

import pandas as pd
import torch
from pytorch_lightning import seed_everything
from torch_geometric.data import Batch
from torch_geometric.datasets import ZINC

from src import evaluation_metrics
from src.datasets import AddNeighbourhoodEncodings, AddNodeDegree, Compose
from src.encoding.configs_and_constants import FeatureConfig, Features, IndexRange, SupportedDataset
from src.encoding.feature_encoders import CombinatoricIntegerEncoder
from src.encoding.graph_encoders import load_or_create_hypernet
from src.encoding.the_types import VSAModel
from src.utils.utils import DataTransformer

seed_everything(42)  # For reproducibility


def run_all_combinations():
    PROJECT_DIR = Path("/home/ka/ka_iti/ka_zi9629/projects/graph_hdc")
    global_model_dir = "/home/ka/ka_iti/ka_zi9629/projects/graph_hdc/_models"
    global_dataset_dir = Path("/home/ka/ka_iti/ka_zi9629/projects/graph_hdc/_datasets")

    asset_dir = PROJECT_DIR / Path("src/exp/dataset_nha/results")
    asset_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = asset_dir / "res.parquet"
    csv_path = asset_dir / "res.csv"

    metrics_df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()

    param_grid = product(
        [SupportedDataset.ZINC_NODE_DEGREE_COMB_NHA],
        [VSAModel.HRR, VSAModel.MAP],
        [80 * 80, 96 * 96],
        [1, 2, 3, 4],  # nha_depth
        [3, 4, 5, 6, 7, 8, 9, 10],  # nha_bins
    )

    for dataset, vsa, hv_dim, nha_depth, nha_bins in param_grid:
        ds = dataset
        ds.default_cfg.vsa = vsa
        ds.default_cfg.edge_feature_configs = {}
        ds.default_cfg.graph_feature_configs = {}
        ds.default_cfg.hv_dim = hv_dim
        ds.default_cfg.seed = 42
        ds.default_cfg.nha_bins = nha_bins
        ds.default_cfg.nha_depth = nha_depth
        ds.default_cfg.node_feature_configs[Features.ATOM_TYPE] = FeatureConfig(
            count=28 * 6 * nha_bins,
            encoder_cls=CombinatoricIntegerEncoder,
            index_range=IndexRange((0, 3)),
        )

        hypernet = load_or_create_hypernet(path=Path(global_model_dir), ds=ds, use_edge_codebook=False)

        assert not hypernet.use_edge_features()
        assert not hypernet.use_graph_features()

        batch_size = 512
        pre_transform = Compose(
            [
                AddNodeDegree(),
                AddNeighbourhoodEncodings(depth=nha_depth, bins=nha_bins),
            ]
        )
        zinc = ZINC(root=global_dataset_dir / f"zinc_nd_comb_nha_d{nha_depth}_b{nha_bins}", pre_transform=pre_transform)
        data_list = [zinc[i] for i in range(batch_size)]
        data = Batch.from_data_list(data_list)
        encoded_data = hypernet.forward(data)

        start_time = time.perf_counter()

        # --- Order 0
        nodes_decoded_counter = hypernet.decode_order_zero_counter(encoded_data["node_terms"])
        ground_truth_counters = {
            g: DataTransformer.get_node_counter_from_batch(batch=g, data=data) for g in range(batch_size)
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

        run_time_order_zero = time.perf_counter() - start_time

        # Count graphs with unique nodes
        unique_count = 0
        for g in zinc:
            t = g.x
            flattened = t.view(t.size(0), -1)
            unique_rows = torch.unique(flattened, dim=0)
            unique_count += int(unique_rows.size(0) == t.size(0))
        unique_proportion = unique_count / len(data_list)

        print(f"\n<-- [{ds=}] | [n_samples: {batch_size}] | [{vsa.name=}] | [{hv_dim=}]-->")
        print(f"\tAverage node (order 0) f1:  {avg_F1_node:.2f}")

        run_metrics = {
            "dataset": ds.value,
            "n_samples": batch_size,
            "vsa": vsa.value,
            "hv_dim": hv_dim,
            "nha_depth": nha_depth,
            "nha_bins": nha_bins,
            "unique_node_count": unique_count,
            "unique_proportion": unique_proportion,
            "runtime_dec_node": run_time_order_zero,
            "P_order_zero": avg_pr_node,
            "F1_order_zero": avg_F1_node,
        }

        print(run_metrics)
        metrics_df = pd.concat([metrics_df, pd.DataFrame([run_metrics])], ignore_index=True)
        metrics_df.to_parquet(parquet_path, index=False)
        metrics_df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    run_all_combinations()
