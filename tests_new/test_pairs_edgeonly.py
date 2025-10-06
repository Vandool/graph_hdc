import torch
import torchhd
from torch_geometric.data import Batch, Data, DataLoader

from src.datasets.zinc_pairs_v3 import ZincPairsV3
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import ZINC_SMILES_HRR_7744_CONFIG
from src.encoding.graph_encoders import load_or_create_hypernet
from src.utils.utils import GLOBAL_MODEL_PATH, pick_device


def test_edges_only():
    device = pick_device()
    base_dataset = ZincSmiles(split="train", enc_suffix="HRR7744")
    ds = ZincPairsV3(split="train", base_dataset=base_dataset, dev=True, edge_only=True)

    loader = DataLoader(ds, batch_size=1, shuffle=False)

    pos = []
    neg = []
    hypernet = load_or_create_hypernet(GLOBAL_MODEL_PATH, cfg=ZINC_SMILES_HRR_7744_CONFIG, use_edge_codebook=False)
    for item in loader:
        # g1 (candidate subgraph)
        g1 = Data(x=item.x1, edge_index=item.edge_index1)

        # g2 (condition) -> encode to cond
        g2 = Data(x=item.x2, edge_index=item.edge_index2)

        # Encode a single graph safely
        batch_g2 = Batch.from_data_list([g2]).to(device)
        h2 = hypernet.forward(batch_g2)["edge_terms"]  # [1, D] on device
        cond = h2.detach().cpu()  # let PL move the whole Batch later
        edge_terms = cond.as_subclass(torch.Tensor).to(device)

        # target/meta
        y = float(item.y.view(-1)[0].item())
        parent_idx = int(item.parent_idx.view(-1)[0].item()) if hasattr(item, "parent_idx") else -1

        # Attach fields to g1
        cond = base_dataset[parent_idx].graph_terms.detach().cpu().as_subclass(torch.Tensor).unsqueeze(0).to(device)
        g1.cond = cond
        g1.y = torch.tensor(y, dtype=torch.float32)
        g1.parent_idx = torch.tensor(parent_idx, dtype=torch.long)
        k = int(item.k.view(-1)[0].item())

        batch_g1 = Batch.from_data_list([g1]).to(device)
        # hypernet.depth = 3
        # h2_3 = hypernet.forward(batch_g1)["edge_terms"].as_subclass(torch.Tensor).to(device)
        # cos_3 = torchhd.cos(h2_3, edge_terms).item()
        #
        # hypernet.depth = 2
        # h2_2 = hypernet.forward(batch_g1)["edge_terms"].as_subclass(torch.Tensor).to(device)
        # cos_2 = torchhd.cos(h2_2, edge_terms).item()

        hypernet.depth = 1
        h2_1 = hypernet.forward(batch_g1)["edge_terms"].as_subclass(torch.Tensor).to(device)
        cos_1 = torchhd.cos(h2_1, edge_terms).item()

        if y == 0:
            neg.append(cos_1)
        else:
            pos.append(cos_1)

        # print(f"cos_3={cos_3}, cos_2={cos_2}, cos_1={cos_1}, y={y}")

    print(f"{sum(pos)/len(pos)=}")
    print(f"{max(pos)=}")
    print(f"{min(pos)=}")

    print(f"{sum(neg)/len(neg)=}")
    print(f"{max(neg)=}")
    print(f"{min(neg)=}")

    assert True
