import argparse
import os
from collections import Counter
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.datasets.utils import get_split
from src.encoding.configs_and_constants import (
    SupportedDataset,
)
from src.encoding.decoder import new_decoder  # noqa: F401
from src.encoding.graph_encoders import HyperNet, load_or_create_hypernet
from src.utils.utils import GLOBAL_ARTEFACTS_PATH, DataTransformer, pick_device
from src.utils.visualisations import draw_nx_with_atom_colorings

PLOT = False

HV_DIMS = {
    "qm9": [
        # 1024,
        # 1280,
        # 1536,
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
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)


def eval_retrieval(ds: SupportedDataset, n_samples: int = 1):
    ds_config = ds.default_cfg
    base_dataset = ds_config.base_dataset
    for hv_dim in HV_DIMS[base_dataset]:
        for decoder_setting in DECODER_SETTINGS[base_dataset]:
            device = pick_device()
            print(f"Running on {device}")
            ds_config.hv_dim = hv_dim
            ds_config.device = device
            hypernet: HyperNet = (
                load_or_create_hypernet(cfg=ds_config, use_edge_codebook=False).to(device, dtype=DTYPE).eval()
            )
            hypernet.decoding_limit_for = ds

            dataset = get_split(split="train", ds_config=ds.default_cfg, use_no_suffix=True)

            nodes_set = set(map(tuple, dataset.x.long().tolist()))
            hypernet.limit_nodes_codebook(limit_node_set=nodes_set)

            dataset = dataset[:n_samples]
            dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

            hits = []
            ts = []
            results = []
            pbar = tqdm(dataloader)
            for data in pbar:
                if PLOT:
                    nx_g = DataTransformer.pyg_to_nx(data)
                    draw_nx_with_atom_colorings(
                        nx_g, dataset="QM9Smiles" if dataset == "qm9" else "ZincSmiles", label="Original"
                    )
                    plt.show()

                # Real Data
                node_tuples = [tuple(i) for i in data.x.int().tolist()]
                edge_tuples = [(node_tuples[e[0]], node_tuples[e[1]]) for e in data.edge_index.t().int().cpu().tolist()]
                real_edge_counter = Counter(edge_tuples)

                forward = hypernet.forward(data)
                edge_terms = forward["edge_terms"]
                decoded_edges = hypernet.decode_order_one_no_node_terms(edge_terms[0].clone())
                decoded_edge_counter = Counter(decoded_edges)

                is_hit = decoded_edge_counter == real_edge_counter
                hits.append(is_hit)
                curr_acc = sum(hits) / len(hits) if hits else 0
                pbar.set_postfix(curr_acc=f"{curr_acc:.4f}")

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
                    "decoder_settings": decoder_setting,
                }
            )
            pprint(results[-1])

            # --- save metrics to disk ---
            asset_dir = GLOBAL_ARTEFACTS_PATH / "retrieval"
            asset_dir.mkdir(parents=True, exist_ok=True)

            parquet_path = asset_dir / "edge_decoder_direct.parquet"
            csv_path = asset_dir / "edge_decoder_direct.csv"

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
    p.add_argument("--n_samples", type=int, default=1000)
    args = p.parse_args()
    ds = SupportedDataset(args.dataset)

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    pprint(args)
    eval_retrieval(n_samples=args.n_samples, ds=ds)
