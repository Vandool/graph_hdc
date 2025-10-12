import pytest
import torch
import torchhd
from torch_geometric.data import Batch, DataLoader
from tqdm import tqdm

from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import QM9_SMILES_HRR_1600_CONFIG_F64
from src.encoding.decoder import new_decoder  # noqa: F401
from src.encoding.graph_encoders import load_or_create_hypernet
from src.utils.nx_utils import is_induced_subgraph_by_features
from src.utils.utils import GLOBAL_MODEL_PATH, DataTransformer, pick_device  # noqa: F401

"""
1000 Samples:
Accuracy: 0.994
Average sim:  0.9998896030902724
Average final:  0.998
"""
@pytest.mark.parametrize("ds_config", [QM9_SMILES_HRR_1600_CONFIG_F64])
def test_hypernet_hdc_decoder(ds_config):
    # device = pick_device()
    device = torch.device("cpu")
    hypernet = load_or_create_hypernet(GLOBAL_MODEL_PATH, cfg=ds_config, use_edge_codebook=False).to(device).eval()

    n_samples = 1000
    dataset = (
        QM9Smiles(split="train")[:n_samples]
        if ds_config.base_dataset == "qm9"
        else ZincSmiles(split="train")[:n_samples]
    )
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    hits = []
    finals = []
    sims = []
    for data in tqdm(dataloader):
        forward = hypernet.forward(data)
        node_terms = forward["node_terms"]
        edge_terms = forward["edge_terms"]
        graph_terms = forward["graph_embedding"]
        # node_counter = DataTransformer.get_node_counter_from_batch(0, data)
        counters = hypernet.decode_order_zero_counter(node_terms)
        candidates, final_flags = hypernet.decode_graph(
            node_counter=counters[0], edge_term=edge_terms[0], graph_term=graph_terms[0]
        )

        # candidates, final_flags = new_decoder(
        #     nodes_multiset=node_counter, edge_terms=edge_terms, graph_terms=graph_terms, encoder=hypernet
        # )

        data_list = [DataTransformer.nx_to_pyg(c) for c in candidates]
        batch = Batch.from_data_list(data_list)
        enc_out = hypernet.forward(batch)
        g_terms = enc_out["graph_embedding"]

        q = graph_terms[0].to(g_terms.device, g_terms.dtype)
        similarities = torchhd.cos(q, g_terms).tolist()

        best_idx = similarities.index(max(similarities))
        best_g = candidates[best_idx]
        is_hit = is_induced_subgraph_by_features(best_g, DataTransformer.pyg_to_nx(data.to_data_list()[0]))
        hits.append(is_hit)
        finals.append(final_flags[best_idx])
        sims.append(max(similarities))

    print(f"Accuracy: {sum(hits) / len(hits)}")
    print("Average sim: ", sum(sims) / len(sims))
    print("Average final: ", sum(finals) / len(finals))
