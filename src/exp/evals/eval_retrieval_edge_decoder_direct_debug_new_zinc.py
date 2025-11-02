import argparse
import math
import os
from collections import Counter, OrderedDict
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import torchhd
from matplotlib import pyplot as plt
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import (
    DSHDCConfig,
    FeatureConfig,
    Features,
    IndexRange,
    SupportedDataset,
)
from src.encoding.decoder import new_decoder  # noqa: F401
from src.encoding.feature_encoders import CombinatoricIntegerEncoder
from src.encoding.graph_encoders import HyperNet, load_or_create_hypernet
from src.encoding.the_types import VSAModel
from src.utils.utils import GLOBAL_ARTEFACTS_PATH, DataTransformer, pick_device, pick_device_str
from src.utils.visualisations import draw_nx_with_atom_colorings

PLOT = False

HV_DIMS = {
    "qm9": [
        64,
        128,
        256,
        512,
        1024,
        1280,
        1536,
        1600,
        # 1792,
    ],
    "zinc": [
        64,
        128,
        256,
        512,
        1024,
        # 1600,
        # 1792,
        2048,
        # 3072,
        4096,
        5120,
        5632,
        6144,
        # 7168,
        # 7744,
        # 8192,
        # 9216,
        # 10240,
        # 11264,
        # 12288,
        # 13312,
        # 14336,
    ],
}

DECODER_SETTINGS = {
    "qm9": [{"max_solutions": 1000}],
    "zinc": [
        {"max_solutions": 1000},
    ],
}

DTYPE = torch.float32
torch.set_default_dtype(DTYPE)


def eval_retrieval(ds: SupportedDataset, n_samples: int = 1):
    ds_config = ds.default_cfg
    base_dataset = ds_config.base_dataset
    for hv_dim in HV_DIMS[base_dataset]:
        for decoder_setting in DECODER_SETTINGS[base_dataset]:
            device = pick_device()
            print(f"Running on {device}")
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
            hypernet: HyperNet = (
                load_or_create_hypernet(cfg=ds_config, use_edge_codebook=False).to(device=device, dtype=DTYPE).eval()
            )
            hypernet.decoding_limit_for = base_dataset

            dataset = ZincSmiles(split="train")

            nodes_set = set(map(tuple, dataset.x.long().tolist()))
            hypernet.limit_nodes_codebook(limit_node_set=nodes_set)

            dataset = dataset[:n_samples]
            dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

            hits = []
            ts = []
            results = []
            diffs = []
            all_oks = []
            # norms_edge_terms_data = []
            sims_progress_data_hit = []
            sims_progress_data_miss = []
            norms_progress_data_hit = []
            norms_progress_data_miss = []
            pbar = tqdm(dataloader, desc=f"Dataset: {ds.value}, HV Dim: {hv_dim}")
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

                # Manually compute the edge terms
                edges = []
                for a, b in edge_tuples:
                    idx_a = hypernet.nodes_indexer.get_idx(a)
                    idx_b = hypernet.nodes_indexer.get_idx(b)
                    hd_a = hypernet.nodes_codebook[idx_a]
                    hd_b = hypernet.nodes_codebook[idx_b]

                    # bind
                    edges.append(hd_a.bind(hd_b))
                    # edges.append(hd_b.bind(hd_a))

                t = torch.stack(edges)
                edge_terms_manul = torchhd.multibundle(t)

                edget_terms_m_copy = edge_terms_manul.clone()
                # Now just reverse to see if you get 0
                for a, b in edge_tuples:
                    idx_a = hypernet.nodes_indexer.get_idx(a)
                    idx_b = hypernet.nodes_indexer.get_idx(b)
                    hd_a = hypernet.nodes_codebook[idx_a]
                    hd_b = hypernet.nodes_codebook[idx_b]

                    # bind
                    edget_terms_m_copy -= hd_a.bind(hd_b)

                sum_elements = edget_terms_m_copy.abs().sum().item()
                assert sum_elements < 1e-2, f"sum_elements={sum_elements}"

                # Decoded
                edge_terms = hypernet.forward(data, normalize=True)["edge_terms"][0]

                eps = 1e-9
                ok_mask = (edge_terms - edge_terms_manul).abs() <= eps  # bool tensor
                all_ok = ok_mask.all()  # << GOOD
                all_oks.append(all_ok.item())

                decoded_edges, norms, sims = hypernet.decode_order_one_no_node_terms(
                    edge_terms_manul.clone(), debug=True
                )
                decoded_edge_counter = Counter(decoded_edges)
                is_hit = decoded_edge_counter == real_edge_counter
                hits.append(is_hit)

                if is_hit:
                    sims_progress_data_hit.append(sims)
                    norms_progress_data_hit.append(norms)
                else:
                    sims_progress_data_miss.append(sims)
                    norms_progress_data_miss.append(norms)

                if not is_hit:
                    print("MISS")
                # #######################################
                # ## Manual decode here
                # hypernet.populate_edges_codebook()
                # hypernet._populate_edges_indexer()
                # decoded_edges_man: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
                # # while not target_reached(decoded_edges):
                # norms = []
                # edge_term = edge_terms_manul.clone()
                # prev_edge_term = edge_terms_manul.clone()
                # similarities = []
                # acc_mans = []
                # while True:
                #     curr_norm = edge_term.norm().item()
                #     norms.append(curr_norm)
                #     if curr_norm <= 0.001:
                #         break
                #     sims = torchhd.cos(edge_term, hypernet.edges_codebook)
                #     # get top-2 indices and their similarity values
                #     topk = torch.topk(sims, k=2)
                #     top_idxs = topk.indices.tolist()
                #     top_vals = topk.values.tolist()
                #
                #     # record the best similarity
                #     similarities.append(top_vals[0])
                #
                #     # unpack the best and second-best matches
                #     a_found, b_found = hypernet.edges_indexer.get_tuple(top_idxs[0])
                #     hd_a_found: VSATensor = hypernet.nodes_codebook[a_found]
                #     hd_b_found: VSATensor = hypernet.nodes_codebook[b_found]
                #
                #     a_second, b_second = hypernet.edges_indexer.get_tuple(top_idxs[1])
                #     if (a_found != b_second or b_found != a_second) and a_found != b_found:
                #         print("what?")
                #
                #     if (
                #         hypernet.nodes_indexer.get_tuple(a_found),
                #         hypernet.nodes_indexer.get_tuple(b_found),
                #     ) not in edge_tuples:
                #         print("what?")
                #
                #     hd_a_second: VSATensor = hypernet.nodes_codebook[a_second]
                #     hd_b_second: VSATensor = hypernet.nodes_codebook[b_second]
                #     edge_term -= hd_a_found.bind(hd_b_found)
                #     edge_term -= hd_b_found.bind(hd_a_found)
                #     # if edge_term.norm().item() > prev_edge_term.norm().item():
                #     #     if not target_reached(decoded_edges_man) and target_reached(
                #     #         [
                #     #             *decoded_edges_man,
                #     #             (hypernet.nodes_indexer.get_tuple(a_found), hypernet.nodes_indexer.get_tuple(b_found)),
                #     #             (hypernet.nodes_indexer.get_tuple(b_found), hypernet.nodes_indexer.get_tuple(a_found)),
                #     #         ]
                #     #     ):
                #     #         decoded_edges_man.append(
                #     #             (hypernet.nodes_indexer.get_tuple(a_found), hypernet.nodes_indexer.get_tuple(b_found))
                #     #         )
                #     #         decoded_edges_man.append(
                #     #             (hypernet.nodes_indexer.get_tuple(b_found), hypernet.nodes_indexer.get_tuple(a_found))
                #     #         )
                #     #     break
                #     # prev_edge_term = edge_term.clone()
                #
                #     decoded_edges_man.append(
                #         (hypernet.nodes_indexer.get_tuple(a_found), hypernet.nodes_indexer.get_tuple(b_found))
                #     )
                #     decoded_edges_man.append(
                #         (hypernet.nodes_indexer.get_tuple(b_found), hypernet.nodes_indexer.get_tuple(a_found))
                #     )
                # decoded_edges_man_counter = Counter(decoded_edges_man)
                #
                # is_hit_man = decoded_edges_man_counter == real_edge_counter
                # acc_mans.append(is_hit_man)
                # #######################################

                diffs.append(len(decoded_edges) - len(real_edge_counter))
                curr_acc = sum(hits) / len(hits) if len(hits) > 0 else 0
                pbar.set_postfix(curr_acc=f"{curr_acc:.4f}")

            # ---- plotting ----
            from pathlib import Path

            import matplotlib.pyplot as plt

            def _savefig(fig, path: Path):
                fig.tight_layout()
                # fig.savefig(path, dpi=200)
                plt.show()
                plt.close(fig)

            # 2) Progress plots (list[list[float]]) — plot each sequence as a faint line
            def _plot_progress(progress: list[list[float]], title: str, fname: str):
                fig, ax = plt.subplots()
                max_len = 2 * max((len(seq) for seq in progress if seq), default=0)
                for seq in progress:
                    if not seq:
                        continue
                    ax.plot(seq, linewidth=0.8, alpha=0.6)

                ax.set_xlabel("decode step")
                ax.set_ylabel("L2 norm")
                ax.set_title(title)
                ax.set_xlim(0, max_len - 1)  # <-- fix truncated x-axis
                fig.tight_layout()
                plt.show()
                # _savefig(fig, base_dir / f"{fname}{out_suffix}.png")

            _plot_progress(norms_progress_data_hit, f"Decode progress HIT (data - {hv_dim})", "decode_progress_data")
            _plot_progress(norms_progress_data_miss, f"Decode progress MISS (data - {hv_dim})", "decode_progress_data")
            _plot_progress(sims_progress_data_hit, f"Cos Sim Progress HIT (data - {hv_dim})", "sims_progress_random")
            _plot_progress(sims_progress_data_miss, f"Cos Sim Progress MISS (data - {hv_dim})", "sims_progress_random")

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

            parquet_path = asset_dir / "new_zinc_edge_decoder.parquet"
            csv_path = asset_dir / "new_zinc_edge_decoder.csv"

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
        default=SupportedDataset.ZINC_SMILES_HRR_6144_F64_G1G3.value,
        choices=[ds.value for ds in SupportedDataset],
    )
    p.add_argument("--n_samples", type=int, default=100)
    args = p.parse_args()
    ds = SupportedDataset(args.dataset)

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    pprint(args)
    eval_retrieval(n_samples=args.n_samples, ds=ds)
