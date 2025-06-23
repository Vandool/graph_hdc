import time
from collections import Counter
from pathlib import Path
from pprint import pformat
from typing import Any

import pandas as pd
from pycomex.functional.experiment import Experiment
from pycomex.util import file_namespace, folder_path
from torch_geometric import loader
from torch_geometric.data import Dataset
from torch_geometric.datasets import QM9, ZINC

from src import evaluation_metrics
from src.datasets import AddNodeDegree
from src.encoding.configs_and_constants import SupportedDataset
from src.encoding.graph_encoders import HyperNet
from src.encoding.types import VSAModel
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
#       VSAModel representation of a supported vsa (hypervector) types
VSA: VSAModel = VSAModel.HRR

# :param HV_DIM:
#       int the dimension of the hypervectors
HV_DIM: int = 80 * 80  # 6400

# :param HYPERNET_DEPTH:
#       int the depth of the message passing when decoding the graph using hypernet encoder
HYPERNET_DEPTH: int = 3

# :param MAX_MAIN_LOOP
#       int the max number of times the main training loop should run.
MAX_MAIN_LOOP: int = 1


# :param REC_NUM_ITER
#       int the number iterations the reconstruction algorithm.
REC_NUM_ITER: int = 10

# :param REC_LEARNING_RATE
#       float the learning parameter of the optimizer
REC_LEARNING_RATE: float = 1

# :param REC_BATCH_SIZE
#       int indicated how many copies of the datapoints be re-constructed to find the best one
REC_BATCH_SIZE: int = 1

# :param REC_LOW
#       int the lower bound of the randomly generated edge weight
REC_LOW: int = 10

# :param REC_HIGH
#       int the upper bound of the randomly generated edge weight
REC_HIGH: int = 10

# :param REC_ALPHA
#       int to what degree the node_degree loss should be considered
REC_ALPHA: int = 0

# :param REC_LAMBDA_L1
#       int to what degree the L1 loss should be considered
REC_LAMBDA_L1: int = 0


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
    try:
        import torch
    except ImportError:
        e.log("PyTorch is not installed in this environment.")
        raise

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        e.log(f"CUDA is available. Detected {count} GPU device{'s' if count != 1 else ''}.")
        return torch.device("cuda")

    e.log("CUDA is not available.")
    return torch.device("cpu")


@experiment
def experiment(e: Experiment):
    e.log(f"{e.path=}")
    e.log(pformat({k: (v, type(v)) for k, v in e.parameters.items()}, indent=2))
    device = e.apply_hook("get_device")
    e.log(f"{device=}")

    ## Apply configs
    ds = SupportedDataset(e.DATASET)
    ds.default_cfg.vsa = VSAModel(e.VSA)
    # Disable edge and graph features
    ds.default_cfg.edge_feature_configs = {}
    ds.default_cfg.graph_feature_configs = {}
    ds.default_cfg.hv_dim = e.HV_DIM
    batch_size = e.DATA_BATCH_SIZE
    e.log(f"{batch_size=}")
    e.log(f"{type(batch_size)=}")

    e.log(f"setting global seed: {e.SEED=}")
    set_seed(e.SEED)

    ## Create Paths
    dataset_root = Path(e.PROJECT_DIR) / "datasets" / ds.value
    dataset_root.mkdir(parents=True, exist_ok=True)

    eval_root = Path(e.path) / "evaluations"
    eval_root.mkdir(parents=True, exist_ok=True)

    ## Get the Dataset
    e.log(f"{e.DATASET}")
    dataset = e.apply_hook("get_dataset", dataset=e.DATASET, root=dataset_root)
    e.log(f"Dataset has been loaded. {dataset=!r}")

    dataloader = loader.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    ### Initialize Hypernet and evals
    hypernet = HyperNet(config=ds.default_cfg, hidden_dim=ds.default_cfg.hv_dim, depth=e.HYPERNET_DEPTH)
    all_metrics = []

    ## Run the Experiment and
    for batch in dataloader:
        encoded_data = hypernet.forward(data=batch)

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
                graph_embedding, node_terms, edge_terms, use_node_degree=bool(e.ALPHA)
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
            p_1, _, f1_edge = evaluation_metrics.calculate_p_a_f1(pred=edge_counter_dec[0], true=edge_existence_counter)
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
            "dataset": ds.value,
            "n_samples": batch_size,
            "vsa": hypernet.vsa.value,
            "hv_dim": hypernet.hv_dim,
            "depth": hypernet.depth,
            "normalize": True,
            "separate_levels": True,
            "avg_reconstruct_time": reconstruct_time / batch_size,
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

        all_metrics.append(run_metrics)
        e.log(pformat(run_metrics, indent=2))

    # Persist the evals
    metrics_df = pd.DataFrame(all_metrics)

    parquet_path = eval_root / "res.parquet"
    csv_path = eval_root / "res.csv"

    metrics_df.to_parquet(parquet_path, index=False)
    metrics_df.to_csv(csv_path, index=False)

    e.log(f"Saved {len(metrics_df)} rows to:\n  {parquet_path}\n  {csv_path}")
    e.log("DONE")


experiment.run_if_main()
