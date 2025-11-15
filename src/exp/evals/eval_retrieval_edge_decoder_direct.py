import argparse
import datetime
import os
import time
from collections import Counter
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from networkx.algorithms import isomorphism
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.datasets.utils import get_split
from src.encoding.configs_and_constants import (
    SupportedDataset,
)
from src.encoding.graph_encoders import DecodingResult, HyperNet, load_or_create_hypernet
from src.utils.utils import DataTransformer, pick_device

PLOT = False

HV_DIMS = {
    "qm9": [256],
    "zinc": [256],
}

DECODER_SETTINGS = {
    "qm9": [
        {
            "initial_limit": 2048,
            "limit": 1024,
            "beam_size": 1024,
            "pruning_method": "cos_sim",
            "use_size_aware_pruning": True,
            "use_one_initial_population": False,
            "use_g3_instead_of_h3": False,
            "validate_ring_structure": False,  # qm9 does not have information about ring structure
            "use_modified_graph_embedding": True,
            "random_sample_ratio": 0.0,
        }
    ],
    "zinc": [
        {
            "initial_limit": 4096,
            "limit": 2048 + 1024,
            "beam_size": 96,
            "pruning_method": "cos_sim",
            "use_size_aware_pruning": True,
            "use_one_initial_population": False,
            "use_g3_instead_of_h3": False,
            "validate_ring_structure": True,
            "use_modified_graph_embedding": False,
            "random_sample_ratio": 0.0,
            "graph_embedding_attr": "graph_embedding",
        },
    ],
}
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)


def are_isomorphic(G1, G2, attr="feat"):
    """
    Check if two graphs are isomorphic, considering node attributes.
    Is more expansive than WL hashing.
    """
    nm = isomorphism.categorical_node_match(attr, None)
    GM = isomorphism.GraphMatcher(G1, G2, node_match=nm)
    return GM.is_isomorphic()


def eval_retrieval(ds: SupportedDataset, n_samples: int = 1):
    ds_config = ds.default_cfg
    base_dataset = ds_config.base_dataset
    for hv_dim in HV_DIMS[base_dataset]:
        for decoder_setting in DECODER_SETTINGS[base_dataset]:
            device = pick_device()
            print(f"Running on {device}")
            ds_config.hv_dim = hv_dim
            ds_config.device = device
            ds_config.name = f"DELETE_ME_LATER_{hv_dim}_{base_dataset}"
            hypernet: HyperNet = (
                load_or_create_hypernet(cfg=ds_config, use_edge_codebook=False).to(device, dtype=DTYPE).eval()
            )
            hypernet.base_dataset = base_dataset

            dataset = get_split(split="train", ds_config=ds.default_cfg, use_no_suffix=True)

            dataset = dataset[:n_samples]
            dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
            embedding = decoder_setting.get("graph_embedding_attr", "graph_embedding")

            edge_hits = []
            graph_hits = []
            best_sims = []
            ts = []
            results = []
            pbar = tqdm(dataloader)
            for data in pbar:
                original_nx_graph = DataTransformer.pyg_to_nx(data)

                # Real Data
                node_tuples = [tuple(i) for i in data.x.int().tolist()]
                edge_tuples = [(node_tuples[e[0]], node_tuples[e[1]]) for e in data.edge_index.t().int().cpu().tolist()]
                real_edge_counter = Counter(edge_tuples)

                forward = hypernet.forward(data)
                edge_terms = forward["edge_terms"]
                graph_terms = forward[embedding]
                ts_start = time.perf_counter()
                decoded_edges = hypernet.decode_order_one_no_node_terms(edge_terms[0].clone())
                decoded_edge_counter = Counter(decoded_edges)

                is_hit = decoded_edge_counter == real_edge_counter
                edge_hits.append(is_hit)

                decoded_graph_results: DecodingResult = hypernet.decode_graph_greedy(
                    edge_term=edge_terms[0].clone(),
                    graph_term=graph_terms[0].clone(),
                    decoder_settings=decoder_setting,
                )
                ts.append(time.perf_counter() - ts_start)
                if decoded_graph_results.nx_graphs:
                    best_graph = decoded_graph_results.nx_graphs[0]
                    best_sim = decoded_graph_results.cos_similarities[0]
                    graph_hits.append(are_isomorphic(best_graph, original_nx_graph))
                    best_sims.append(best_sim)
                else:
                    graph_hits.append(False)
                    best_sims.append(0)

                curr_edge_acc = sum(edge_hits) / len(edge_hits) if edge_hits else 0
                curr_graph_acc = sum(graph_hits) / len(graph_hits) if graph_hits else 0
                pbar.set_postfix(edge_acc=f"{curr_edge_acc:.4f}", graph_acc=f"{curr_graph_acc:.4f}")

            ts = np.array(ts)
            results.append(
                {
                    "n_samples": n_samples,
                    "dataset": ds_config.base_dataset,
                    "vsa": ds_config.vsa.value,
                    "hv_dim": ds_config.hv_dim,
                    "device": str(device),
                    "time_per_sample": ts.mean(),
                    "edge_accuracy": sum(edge_hits) / len(edge_hits),
                    "graph_accuracy": sum(graph_hits) / len(graph_hits),
                    "best_sims_avg": sum(best_sims) / len(best_sims),
                    **decoder_setting,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            pprint(results[-1])

            # --- save metrics to disk ---
            asset_dir = Path(__file__).parent / "retrieval"
            asset_dir.mkdir(parents=True, exist_ok=True)

            parquet_path = asset_dir / "decoding_ablations_new_settings.parquet"
            csv_path = asset_dir / "decoding_ablations_new_settings.csv"

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
        default=SupportedDataset.ZINC_SMILES_HRR_256_F64_5G1NG4.value,
        choices=[ds.value for ds in SupportedDataset],
    )
    p.add_argument("--n_samples", type=int, default=50)
    args = p.parse_args()
    ds = SupportedDataset(args.dataset)

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    pprint(args)
    eval_retrieval(n_samples=args.n_samples, ds=ds)
