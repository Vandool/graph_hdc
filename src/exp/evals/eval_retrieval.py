import argparse
import os
import time

import numpy as np
import pandas as pd
import torchhd
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import ZINC_SMILES_HRR_7744_CONFIG_F64
from src.encoding.decoder import new_decoder  # noqa: F401
from src.encoding.graph_encoders import HyperNet, load_or_create_hypernet
from src.utils.nx_utils import is_induced_subgraph_by_features
from src.utils.utils import GLOBAL_ARTEFACTS_PATH, GLOBAL_MODEL_PATH, DataTransformer, pick_device


def eval_retrieval(n_samples: int = 100):
    for ds_config in [
        # QM9_SMILES_HRR_1600_CONFIG_F64,
        ZINC_SMILES_HRR_7744_CONFIG_F64
    ]:
        for hv_dim in [
            4096 + 0 * 512,
            4096 + 1 * 512,
            4096 + 2 * 512,
            4096 + 3 * 512,
            4096 + 4 * 512,
            4096 + 5 * 512,
            4096 + 6 * 512,
            7744,
        ]:
            for beam_size in [16, 32, 64]:
                device = pick_device()
                # device = torch.device("cpu")
                ds_config = ds_config
                ds_config.hv_dim = hv_dim
                hypernet: HyperNet = (
                    load_or_create_hypernet(GLOBAL_MODEL_PATH, cfg=ds_config, use_edge_codebook=False).to(device).eval()
                )

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
                decoder_settings = {"limit": 128, "beam_size": beam_size, "pruning_method": "cos_sim"}
                for data in tqdm(dataloader):
                    forward = hypernet.forward(data)
                    node_terms = forward["node_terms"]
                    edge_terms = forward["edge_terms"]
                    graph_terms = forward["graph_embedding"]
                    # node_counter = DataTransformer.get_node_counter_from_batch(0, data)
                    t0 = time.perf_counter()
                    counters = hypernet.decode_order_zero_counter(node_terms)
                    candidates, final_flags = hypernet.decode_graph(
                        node_counter=counters[0],
                        edge_term=edge_terms[0],
                        graph_term=graph_terms[0],
                        settings=decoder_settings,
                    )

                    data_list = [DataTransformer.nx_to_pyg(c) for c in candidates]
                    batch = Batch.from_data_list(data_list)
                    enc_out = hypernet.forward(batch)
                    g_terms = enc_out["graph_embedding"]

                    q = graph_terms[0].to(g_terms.device, g_terms.dtype)
                    similarities = torchhd.cos(q, g_terms)

                    best_idx = similarities.tolist().index(max(similarities))
                    best_g = candidates[best_idx]

                    ts.append(time.perf_counter() - t0)

                    is_hit = is_induced_subgraph_by_features(best_g, DataTransformer.pyg_to_nx(data.to_data_list()[0]))
                    hits.append(is_hit)
                    finals.append(final_flags[best_idx])
                    sims.append(max(similarities))

                sims = np.array(sims)
                ts = np.array(ts)
                results.append(
                    {
                        "n_samples": n_samples,
                        "dataset": ds_config.base_dataset,
                        "vsa": ds_config.vsa.value,
                        "hv_dim": ds_config.hv_dim,
                        "device": str(device),
                        "time_per_sample": ts.mean(),
                        "accuracy": sum(hits) / len(hits),
                        "final_flag": 100 * sum(finals) / len(finals),
                        "cos_sim_mean": sims.mean(),
                        "cos_sim_std": sims.std(),
                        "decoder_settings": decoder_settings,
                        "comments": "initial limit 4096",
                    }
                )

                # --- save metrics to disk ---
                asset_dir = GLOBAL_ARTEFACTS_PATH / "retrieval"
                asset_dir.mkdir(parents=True, exist_ok=True)

                parquet_path = asset_dir / "qm9_retrieval.parquet"
                csv_path = asset_dir / "qm9_retrieval.csv"

                metrics_df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()

                new_row = pd.DataFrame(results)
                metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

                # write back out
                metrics_df.to_parquet(parquet_path, index=False)
                metrics_df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluation retrieval of full graph from encoded graph")
    p.add_argument("--dataset", type=str, default="zinc", choices=["zinc", "qm9"])
    p.add_argument("--n_samples", type=int, default=1000)
    args = p.parse_args()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    eval_retrieval(n_samples=args.n_samples)
