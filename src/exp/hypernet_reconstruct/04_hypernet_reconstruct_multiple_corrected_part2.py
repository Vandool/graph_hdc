import itertools  # noqa: INP001
import time
from collections import Counter
from pathlib import Path
from pprint import pformat
from typing import Any

import pandas as pd
import torch
from pycomex.functional.experiment import Experiment
from pycomex.util import file_namespace, folder_path
from torch_geometric import loader
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import QM9, ZINC

from src import evaluation_metrics
from src.datasets import AddNodeDegree
from src.encoding.configs_and_constants import SupportedDataset
from src.encoding.graph_encoders import HyperNet
from src.encoding.the_types import VSAModel
from src.utils.utils import DataTransformer, set_seed

### === PARAMETERS
# :param PROJECT_DIR:
#       String a path of the project directory
PROJECT_DIR: str = "/home/ka/ka_iti/ka_zi9629/projects/graph_hdc"


# :param SEED:
#       int the seed!
SEED: int = 40


# :param DATASET:
#       SupportedDataset representation of a supported dataset
DATASET: SupportedDataset = SupportedDataset.ZINC_NODE_DEGREE_COMB

# :param DATA_BATCH_SIZE:
#       int batch size of the dataloader
DATA_BATCH_SIZE: int = 32

# :param VSA:
#       str representation of a supported vsa (hypervector) types
VSA: str = "MAP"

# :param HV_DIM:
#       int the dimension of the hypervectors
HV_DIM: int = 80 * 80  # 6400

# :param HYPERNET_DEPTH:
#       int the depth of the message passing when decoding the graph using hypernet encoder
HYPERNET_DEPTH: int = 3

# :param MAX_MAIN_LOOP
#       int the max number of times the main training loop should run.
MAX_MAIN_LOOP: int = 1


REC_NUM_ITERS: list[int] = [10, 25, 50]
REC_LEARNING_RATES: list[float] = [1.0, 0.1]
REC_ALPHA: list[float] = [0.0, 0.1, 0.5]
REC_LAMBDA_L1: list[float] = [0.0, 0.01, 0.001, 0.0001]
REC_LOW_HIGH_BATCH_SIZE: list[tuple[int, int, int]] = [
    (10, 10, 1),
    (-1, 1, 3),
    (-1, 1, 10),
    (-1, 1, 25),
    (0, 3, 3),
    (0, 3, 10),
    (0, 3, 25),
]

GRIDS = list(itertools.product(REC_NUM_ITERS, REC_LEARNING_RATES, REC_ALPHA, REC_LAMBDA_L1, REC_LOW_HIGH_BATCH_SIZE))

### === Experiment
experiment = Experiment(base_path=folder_path(__file__), namespace=file_namespace(__file__), glob=globals())


@experiment.hook("get_dataset", replace=False, default=True)
def get_dataset(e: Experiment, dataset: str | SupportedDataset, root: Path) -> Dataset:
    ds = dataset if isinstance(dataset, SupportedDataset) else SupportedDataset(dataset)
    e.log("Loading dataset ...")
    e.log(f"{dataset=}")
    if ds == SupportedDataset.ZINC:
        return ZINC(root=str(root))
    if ds in {SupportedDataset.ZINC_NODE_DEGREE, SupportedDataset.ZINC_NODE_DEGREE_COMB}:
        return ZINC(root=str(root), pre_transform=AddNodeDegree())
    return QM9(root=str(root))


@experiment.hook("get_device", replace=False, default=True)
def get_device(e: Experiment) -> Any:
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        e.log(f"CUDA is available. Detected {count} GPU device{'s' if count != 1 else ''}.")
        return torch.device("cuda")

    e.log("CUDA is not available.")
    return torch.device("cpu")


@experiment
def experiment(e: Experiment):
    e.log(pformat({k: (v, type(v)) for k, v in e.parameters.items()}, indent=2))
    device = e.apply_hook("get_device")
    e.log(f"{device=}")

    ## Apply configs
    vsa = VSAModel(e.VSA)
    ds = SupportedDataset(e.DATASET)
    ds.default_cfg.vsa = vsa
    ds.default_cfg.hv_dim = e.HV_DIM
    ds.default_cfg.device = device
    ds.default_cfg.seed = e.SEED
    # Disable edge and graph features
    ds.default_cfg.edge_feature_configs = {}
    ds.default_cfg.graph_feature_configs = {}

    ## Create Paths
    dataset_root = Path(e.PROJECT_DIR) / "datasets" / ds.value
    dataset_root.mkdir(parents=True, exist_ok=True)

    eval_root = Path(e.path) / "evaluations"
    eval_root.mkdir(parents=True, exist_ok=True)
    parquet_path = eval_root / "res.parquet"
    csv_path    = eval_root / "res.csv"

    e.log(f"setting global seed: {e.SEED=}")
    set_seed(e.SEED)

    batch_size = e.DATA_BATCH_SIZE

    # GRID SEARCH
    for exp_nr, (num_iters, lr, alpha, lambda_l1, (low, high, rec_batch_size)) in enumerate(GRIDS):
        ## We timed out, so we continue
        if vsa == VSAModel.HRR and e.HV_DIM == 6400 and exp_nr <= 47:
            continue
        if vsa == VSAModel.MAP and e.HV_DIM == 6400 and exp_nr <= 93:
            continue
        if vsa == VSAModel.HRR and e.HV_DIM == 9216 and exp_nr <= 33:
            continue
        if vsa == VSAModel.MAP and e.HV_DIM == 9216 and exp_nr <= 40:
            continue
        e.log(
            f"-------\nNew grid: {exp_nr=}, {num_iters=}, {lr=}, {alpha=}, {lambda_l1=}, {low=}, {high=}, {rec_batch_size=}"
        )

        # To store evals of the exp
        grid_metrics: list[dict] = []

        ## Get the Dataset
        e.log(f"{e.DATASET}")
        dataset = e.apply_hook("get_dataset", dataset=e.DATASET, root=dataset_root)
        e.log(f"Dataset has been loaded. {dataset=!r}")

        dataloader = loader.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

        ### Initialize Hypernet and evals
        hypernet = HyperNet(config=ds.default_cfg, hidden_dim=ds.default_cfg.hv_dim, depth=e.HYPERNET_DEPTH)

        ## Run the Experiment and
        for i, batch in enumerate(dataloader):
            if i >= e.MAX_MAIN_LOOP:
                break

            ## Encode
            batch: Data
            start_time_forward = time.perf_counter()
            encoded_data = hypernet.forward(data=batch)
            forward_time = time.perf_counter() - start_time_forward

            ## Get ready to evaluate
            p_nodes = []
            f1_nodes = []
            p_edges = []
            f1_edges = []
            p_graphs = []
            f1_graphs = []
            node_edit_distances = []
            edge_edit_distances = []

            data_list = batch.to_data_list()

            ## Time the whole reconstruction + eval
            start_time = time.perf_counter()
            for b in range(batch_size):
                graph_embedding = encoded_data["graph_embedding"][b]
                node_terms = encoded_data["node_terms"][b]
                edge_terms = encoded_data["edge_terms"][b]
                data_dec, node_counter_dec, edge_counter_dec = hypernet.reconstruct(
                    graph_hv=graph_embedding,
                    node_terms=node_terms,
                    edge_terms=edge_terms,
                    learning_rate=lr,
                    batch_size=rec_batch_size,
                    low=low,
                    high=high,
                    alpha=alpha,
                    lambda_l1=lambda_l1,
                    use_node_degree=bool(alpha),  # Alpha = 0, zeroes the node_degree loss
                )

                ## Nodes
                truth_node_counter: Counter = DataTransformer.get_node_counter_from_batch(batch=b, data=batch)
                p_0, _, f1_node = evaluation_metrics.calculate_p_a_f1(pred=node_counter_dec[0], true=truth_node_counter)
                p_nodes.append(p_0)
                f1_nodes.append(f1_node)

                ## Edge Existence
                actual_edge_counter: Counter = DataTransformer.get_edge_existence_counter(
                    batch=b, data=batch, indexer=hypernet.nodes_indexer
                )
                edge_existence_counter = Counter({k: 1 for k, _ in actual_edge_counter.items()})
                p_1, _, f1_edge = evaluation_metrics.calculate_p_a_f1(
                    pred=edge_counter_dec[0], true=edge_existence_counter
                )
                p_edges.append(p_1)
                f1_edges.append(f1_edge)

                ## Reconstruction
                actual_edge_counter = DataTransformer.get_edge_counter(batch=b, data=batch)
                ## Calculate predicted edges
                x_list = data_dec.x.int().tolist()
                edges = list(zip(data_dec.edge_index[0], data_dec.edge_index[1], strict=False))
                edges_as_ints = [(a.item(), b.item()) for a, b in edges]
                pred_edges = []
                for u, v in edges_as_ints:
                    pred_edges.append((tuple(x_list[u]), tuple(x_list[v])))
                pred_edge_counter = Counter(pred_edges)

                p_g, _, f1_graph = evaluation_metrics.calculate_p_a_f1(pred=pred_edge_counter, true=actual_edge_counter)

                ## Testing injecting the opposite edge
                edges_test = list({e for u, v in edges_as_ints for e in [(v, u), (u, v)]})
                pred_edges = []
                for u, v in edges_test:
                    pred_edges.append((tuple(x_list[u]), tuple(x_list[v])))
                pred_edge_counter_test = Counter(pred_edges)

                p_g_test, _, f1_graph_test = evaluation_metrics.calculate_p_a_f1(
                    pred=pred_edge_counter_test, true=actual_edge_counter
                )
                e.log(f"{pred_edge_counter_test=}")
                e.log(f"{actual_edge_counter=}")
                e.log(f"{f1_graph=}")

                p_graphs.append(p_g)
                f1_graphs.append(f1_graph)

                delta_node, delta_edge, _ = evaluation_metrics.graph_edit_distance(
                    data_pred=data_dec, data_true=data_list[b]
                )
                node_edit_distances.append(delta_node)
                edge_edit_distances.append(delta_edge)

            reconstruct_time = time.perf_counter() - start_time

            ### Save metrics
            run_metrics = {
                "exp_nr": exp_nr,
                "seed": hypernet.seed,
                "n_samples": batch_size,
                "vsa": hypernet.vsa.value,
                "hv_dim": hypernet.hv_dim,
                "depth": hypernet.depth,
                "normalize": True,
                "separate_levels": True,
                "rec_num_iterations": num_iters,
                "rec_lr": lr,
                "rec_batch_size": rec_batch_size,
                "rec_low": low,
                "rec_high": high,
                "rec_alpha": alpha,
                "rec_lambda": lambda_l1,
                "avg_reconstruct_time": reconstruct_time / batch_size,
                "avg_forward_time": forward_time / batch_size,
                "avg_P_order_zero": sum(p_nodes) / batch_size,
                "avg_F1_order_zero": sum(f1_nodes) / batch_size,
                "avg_P_order_one": sum(p_edges) / batch_size,
                "avg_F1_order_one": sum(f1_edges) / batch_size,
                "avg_P_recons": sum(p_graphs) / batch_size,
                "avg_F1_recons": sum(f1_graphs) / batch_size,
                "avg_node_edit_dist": sum(node_edit_distances) / batch_size,
                "avg_edge_edit_dist": sum(edge_edit_distances) / batch_size,
                "total_edges": batch.edge_index.shape[1],
            }

            grid_metrics.append(run_metrics)
            e.log(pformat(run_metrics, indent=2))

        # after this grid completes, append to disk:
        df_grid = pd.DataFrame(grid_metrics)
        if parquet_path.exists():
            df_all = pd.read_parquet(parquet_path)
            df_all = pd.concat([df_all, df_grid], ignore_index=True)
        else:
            df_all = df_grid
        df_all.to_parquet(parquet_path, index=False)

        # likewise for CSV if you want
        if csv_path.exists():
            df_all.to_csv(csv_path, index=False)
        else:
            df_grid.to_csv(csv_path, index=False)

        e.log(f"Saved grid {exp_nr} → {len(grid_metrics)} rows; "
              f"total so far {len(df_all)} rows")


experiment.run_if_main()
