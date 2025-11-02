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
from src.encoding.decoder import (
    compute_sampling_structure,
    new_decoder,  # noqa: F401
    try_find_isomorphic_graph,
)
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
        # 512,
        # 1024,
        # 1280,
        # 1536,
        # 1600,
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
                                Features.NODE_FEATURES,
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
                    pyg_graphs = [DataTransformer.nx_to_pyg_with_type_attr(g) for g in decoded_graphs]
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
