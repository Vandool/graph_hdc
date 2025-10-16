import argparse
import os
import time
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import torchhd
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import (
    QM9_SMILES_HRR_1600_CONFIG_F64,
    ZINC_SMILES_HRR_7744_CONFIG_F64,
)
from src.encoding.decoder import new_decoder  # noqa: F401
from src.encoding.graph_encoders import HyperNet, load_or_create_hypernet
from src.utils.nx_utils import is_induced_subgraph_by_features
from src.utils.utils import GLOBAL_ARTEFACTS_PATH, GLOBAL_MODEL_PATH, DataTransformer, pick_device

HV_DIMS = {
    "qm9": [1024, 1280, 1536, 1600, 1792],
    "zinc": [5120, 5632, 6144, 7168, 7744, 8192],
}

DECODER_SETTINGS = {
    "qm9": [
        {
            "initial_limit": 2048,
            "limit": 1024,
            "beam_size": 256,
            "pruning_method": "negative_euclidean_distance",
            "use_size_aware_pruning": True,
            "use_one_initial_population": True,
            "use_g3_instead_of_h3": True,
        }
    ],
    "zinc": [
        {
            "initial_limit": 1024,
            "limit": 512,
            "beam_size": 32,
            "pruning_method": "cos_sim",
            "use_size_aware_pruning": True,
            "use_one_initial_population": False,
            "use_g3_instead_of_h3": True,
        },
        # {
        #     "initial_limit": 1024,
        #     "limit": 512,
        #     "beam_size": 32,
        #     "pruning_method": "negative_euclidean_distance",
        #     "use_size_aware_pruning": False,
        #     "use_one_initial_population": False,
        # },
    ],
}


def eval_retrieval(n_samples: int = 1, base_dataset: str = "qm9"):
    ds_config = QM9_SMILES_HRR_1600_CONFIG_F64 if base_dataset == "qm9" else ZINC_SMILES_HRR_7744_CONFIG_F64
    for hv_dim in HV_DIMS[base_dataset]:
        for d in [3, 4, 5, 6]:
            for decoder_setting in DECODER_SETTINGS[base_dataset]:
                device = pick_device()
                # device = torch.device("cpu")
                print(f"Running on {device}")
                # device = torch.device("cpu")
                ds_config = ds_config
                ds_config.hv_dim = hv_dim
                ds_config.device = device
                hypernet: HyperNet = (
                    load_or_create_hypernet(GLOBAL_MODEL_PATH, cfg=ds_config, use_edge_codebook=False).to(device).eval()
                )
                hypernet.depth = d

                dataset = QM9Smiles(split="train") if ds_config.base_dataset == "qm9" else ZincSmiles(split="train")

                nodes_set = set(map(tuple, dataset.x.long().tolist()))
                hypernet.limit_nodes_codebook(limit_node_set=nodes_set)

                dataset = dataset[:n_samples]
                dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

                hits = []
                finals = []
                sims = []
                ts = []
                results = []
                for data in tqdm(dataloader):
                    forward = hypernet.forward(data)
                    node_terms = forward["node_terms"]
                    edge_terms = forward["edge_terms"]
                    graph_terms = forward["graph_embedding"]

                    if decoder_setting.get("use_g3_instead_of_h3", False):
                        graph_terms = node_terms + edge_terms + graph_terms

                    t0 = time.perf_counter()
                    counters = hypernet.decode_order_zero_counter(node_terms)
                    try:
                        candidates, final_flags = hypernet.decode_graph(
                            node_counter=counters[0],
                            edge_term=edge_terms[0],
                            graph_term=graph_terms[0],
                            decoder_settings=decoder_setting,
                        )
                    except Exception as e:
                        hits.append(False)
                        finals.append(False)
                        sims.append(0.0)
                    else:
                        data_list = [DataTransformer.nx_to_pyg(c) for c in candidates]
                        batch = Batch.from_data_list(data_list)
                        enc_out = hypernet.forward(batch)
                        g_terms = enc_out["graph_embedding"]

                        q = graph_terms[0].to(g_terms.device, g_terms.dtype)
                        similarities = torchhd.cos(q, g_terms)

                        best_idx = int(torch.argmax(similarities))
                        best_g = candidates[best_idx]

                        ts.append(time.perf_counter() - t0)

                        is_hit = is_induced_subgraph_by_features(
                            best_g, DataTransformer.pyg_to_nx(data.to_data_list()[0])
                        )
                        hits.append(is_hit)
                        finals.append(final_flags[best_idx])
                        sims.append(float(similarities[best_idx].detach().cpu()))

                sims = np.array(sims)
                ts = np.array(ts)
                results.append(
                    {
                        "n_samples": n_samples,
                        "dataset": ds_config.base_dataset,
                        "vsa": ds_config.vsa.value,
                        "hv_dim": ds_config.hv_dim,
                        "depth": d,
                        "device": str(device),
                        "time_per_sample": ts.mean(),
                        "accuracy": sum(hits) / len(hits),
                        "final_flag": 100 * sum(finals) / len(finals),
                        "cos_sim_mean": sims.mean(),
                        "cos_sim_std": sims.std(),
                        "decoder_settings": decoder_setting,
                    }
                )
                pprint(results[-1])

                # --- save metrics to disk ---
                asset_dir = GLOBAL_ARTEFACTS_PATH / "retrieval"
                asset_dir.mkdir(parents=True, exist_ok=True)

                parquet_path = asset_dir / "edge_decoding.parquet"
                csv_path = asset_dir / "edge_decoding.csv"

                metrics_df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()

                new_row = pd.DataFrame(results)
                metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

                # write back out
                metrics_df.to_parquet(parquet_path, index=False)
                metrics_df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluation retrieval of full graph from encoded graph")
    p.add_argument("--dataset", type=str, default="qm9", choices=["zinc", "qm9"])
    p.add_argument("--n_samples", type=int, default=1000)
    args = p.parse_args()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    pprint(args)
    eval_retrieval(n_samples=args.n_samples, base_dataset=args.dataset)
