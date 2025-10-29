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
        128,
        256,
        512,
        1024,
        1280,
        1536,
        1600,
    ],
    "zinc": [
        128,
        256,
        512,
        1024,
        1280,
        1536,
        1600,
        2048,
    ],
}


DTYPE = torch.float32
torch.set_default_dtype(DTYPE)


def eval_retrieval(ds: SupportedDataset, n_samples: int = 1):
    for ds in [
        # SupportedDataset.QM9_SMILES_HRR_1600_F64_G1NG3,
        SupportedDataset.ZINC_SMILES_HRR_5120_F64_G1G3,
    ]:
        ds_config = ds.default_cfg
        base_dataset = ds_config.base_dataset
        for hv_dim in HV_DIMS[base_dataset]:
            device = pick_device()
            print(f"Running on {device}")
            ds_config.hv_dim = hv_dim
            ds_config.device = device
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
            for data in tqdm(dataloader):
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

                # Decoded
                edge_terms = hypernet.forward(data)["edge_terms"][0]
                decoded_edges = hypernet.decode_order_one_no_node_terms(edge_terms.clone())
                decoded_edge_counter = Counter(decoded_edges)

                is_hit = decoded_edge_counter == real_edge_counter
                if not is_hit:
                    deguggin_edges = hypernet.decode_order_one_no_node_terms(edge_terms.clone())
                    print(len(deguggin_edges))
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
                    "accuracy": sum(hits) / len(hits),
                    "dtype": "float32" if torch.float32 == DTYPE else "float64",
                }
            )
            pprint(results[-1])

            # --- save metrics to disk ---
            asset_dir = GLOBAL_ARTEFACTS_PATH / "edge_retrieval"
            asset_dir.mkdir(parents=True, exist_ok=True)

            parquet_path = asset_dir / "edge_retrieval.parquet"
            csv_path = asset_dir / "edge_retrieval.csv"

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
    p.add_argument("--n_samples", type=int, default=200)
    args = p.parse_args()
    ds = SupportedDataset(args.dataset)

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    pprint(args)
    eval_retrieval(n_samples=args.n_samples, ds=ds)
