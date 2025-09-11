import random
import time
from collections import Counter

import networkx as nx
from statsmodels.distributions.tools import average_grid
from torch.utils.data import Subset

from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.decoder import greedy_oracle_decoder, is_induced_subgraph_by_features
from src.utils.utils import DataTransformer


def test_greedy_oracle_decoder_works_with_perfect_oracle():
    zinc_smiles = ZincSmiles(split="valid")
    s = random.sample(range(len(zinc_smiles)), 100)
    y = []
    dataset = Subset(zinc_smiles, s)
    dec_times = []
    for i, data in enumerate(dataset):
        print("================================================")
        full_graph_nx = DataTransformer.pyg_to_nx(data=data)

        node_multiset = Counter(tuple(map(int, row.tolist())) for row in data.x)

        print(f"Multiset Nodes {node_multiset.total()}")
        start = time.perf_counter()
        nx_GS: list[nx.Graph] = greedy_oracle_decoder(
            node_multiset=node_multiset,
            oracle=None,
            full_g_h=None,
            beam_size=1,
            oracle_threshold=0.5,
            strict=True,
            full_g_nx=full_graph_nx,
            draw=False,
            use_pair_feasibility=True,
            expand_on_n_anchors=1,
            use_perfect_oracle=True,
            report_cnf_matrix=False
        )
        nx_GS = list(filter(None, nx_GS))
        if len(nx_GS) == 0:
            y.append(0)
            continue

        sub_g_ys = [0]
        for j, g in enumerate(nx_GS):
            is_final = is_induced_subgraph_by_features(g, full_graph_nx, node_keys=["feat"])
            sub_g_ys.append(int(is_final))
        is_final_graph_ = int(sum(sub_g_ys) >= 1)
        y.append(is_final_graph_)
        dec_times.append(time.perf_counter() - start)

    print(f"Accuracy: {sum(y) / len(y)}")
    print(f"Min. Average Dec. Time: {sum(dec_times)/ len(dec_times):.2f} per graph")
    print(len(y))
