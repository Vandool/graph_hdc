import argparse
import os
import time
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import torchhd
from matplotlib import pyplot as plt
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.datasets.utils import get_split
from src.encoding.configs_and_constants import (
    SupportedDataset,
)
from src.encoding.decoder import new_decoder  # noqa: F401
from src.encoding.graph_encoders import HyperNet, load_or_create_hypernet
from src.utils.nx_utils import is_induced_subgraph_by_features
from src.utils.utils import GLOBAL_ARTEFACTS_PATH, DataTransformer, pick_device
from src.utils.visualisations import draw_nx_with_atom_colorings

PLOT = False

HV_DIMS = {
    "qm9": [
        # 1024, 1280, 1536,
        1600,
        1792,
    ],
    "zinc": [
        # 5120, 5632,
        6144,
        # 7168, 7744, 8192
    ],
}

DECODER_SETTINGS = {
    "qm9": [{"max_solutions": 1000}],
    "zinc": [
        {"max_solutions": 1000},
    ],
}


def eval_retrieval(ds: SupportedDataset, n_samples: int = 1):
    ds_config = ds.default_cfg
    base_dataset = ds_config.base_dataset
    for hv_dim in HV_DIMS[base_dataset]:
        for d in [
            3,
            4,
            5,
            # 6
        ]:
            for decoder_setting in DECODER_SETTINGS[base_dataset]:
                device = pick_device()
                # device = torch.device("cpu")
                print(f"Running on {device}")
                ds_config.hv_dim = hv_dim
                ds_config.device = device
                hypernet: HyperNet = load_or_create_hypernet(cfg=ds_config, use_edge_codebook=False).to(device).eval()
                hypernet.depth = d
                hypernet.decoding_limit_for = ds

                dataset = get_split(split="train", ds_config=ds.default_cfg, use_no_suffix=True)

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
                    if PLOT:
                        nx_g = DataTransformer.pyg_to_nx(data)
                        draw_nx_with_atom_colorings(
                            nx_g, dataset="QM9Smiles" if dataset == "qm9" else "ZincSmiles", label="Original"
                        )
                        plt.show()

                    forward = hypernet.forward(data)
                    edge_terms = forward["edge_terms"]
                    graph_terms = forward["graph_embedding"]

                    t0 = time.perf_counter()
                    try:
                        res = hypernet.decode_graph_z3(
                            edge_term=edge_terms[0],
                            decoder_settings=decoder_setting,
                        )
                        candidates, final_flags = res.nx_graphs, res.final_flags
                    except Exception as e:
                        hits.append(False)
                        finals.append(False)
                        sims.append(0.0)
                    else:
                        if len(candidates) == 0 or (len(candidates) == 1 and candidates[0].number_of_nodes() == 0):
                            hits.append(False)
                            finals.append(False)
                            sims.append(0.0)
                            print("No candidates found")
                            continue
                        data_list = [DataTransformer.nx_to_pyg(c) for c in candidates]
                        batch = Batch.from_data_list(data_list)
                        enc_out = hypernet.forward(batch)
                        g_terms = enc_out["graph_embedding"]

                        q = graph_terms[0].to(g_terms.device, g_terms.dtype)
                        similarities = torchhd.cos(q, g_terms)

                        best_idx = int(torch.argmax(similarities))
                        best_g = candidates[best_idx]
                        if PLOT:
                            draw_nx_with_atom_colorings(
                                best_g,
                                dataset="QM9Smiles" if dataset == "qm9" else "ZincSmiles",
                                label=f"Decoded sim: {similarities[best_idx]:.2f}",
                            )
                            plt.show()

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

                parquet_path = asset_dir / "retrieval_z3_new_decoder_constraints.parquet"
                csv_path = asset_dir / "retrieval_z3_new_decoder_constraints.csv"

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
        default=SupportedDataset.QM9_SMILES_HRR_1600_F64_G1G3.value,
        choices=[ds.value for ds in SupportedDataset],
    )
    p.add_argument("--n_samples", type=int, default=1000)
    args = p.parse_args()
    ds = SupportedDataset(args.dataset)

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    pprint(args)
    eval_retrieval(n_samples=args.n_samples, ds=ds)
