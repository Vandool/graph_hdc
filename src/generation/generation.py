import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import rdkit.Chem
import torch
from torch_geometric.data import Data

from src.encoding.configs_and_constants import ZINC_SMILES_HRR_7744_CONFIG
from src.encoding.graph_encoders import load_or_create_hypernet
from src.encoding.the_types import VSAModel
from src.encoding.decoder import greedy_oracle_decoder
from src.encoding.oracles import Oracle, get_oracle_classifier_config_cls
from src.normalizing_flow.models import AbstractNFModel, RealNVPLightning, FlowConfig
from src.utils import visualisations
from src.utils.utils import GLOBAL_MODEL_PATH, DataTransformer

## For unpickling
setattr(sys.modules['__main__'], 'FlowConfig', FlowConfig)


class Generator:
    def __init__(self, gen_model: AbstractNFModel, oracle: Oracle, ds_config):
        device = torch.device('cpu')
        print(f"Using device: {device}")
        self.encoder = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, ds_name=ds_config.name, cfg=ds_config).to(device)
        self.gen_model: torch.nn.Module = gen_model
        self.oracle = oracle
        self.oracle.encoder = self.encoder
        self.vsa: VSAModel = ds_config.vsa

    def generate(self, n_samples: int = 16) -> list[nx.Graph]:
        node_terms, graph_terms, _ = self.gen_model.sample_split(n_samples)

        node_terms_hd = node_terms.as_subclass(self.vsa.tensor_class)
        graph_terms_hd = graph_terms.as_subclass(self.vsa.tensor_class)

        full_ctrs: dict[int, Counter[tuple[int, ...]]] = self.encoder.decode_order_zero_counter(node_terms_hd)

        for i, c in full_ctrs.items():
            print(f"{i}: {c.total()}")

        return [
            greedy_oracle_decoder(node_multiset=full_ctr, oracle=self.oracle, full_g_h=graph_terms_hd[i], beam_size=4,
                                  oracle_threshold=0)
            for i, full_ctr in enumerate(full_ctrs.values())]

    def generate_mols(self, n_samples: int = 16, validate: bool = True) -> list[rdkit.Chem.Mol]:
        Gs: list[nx.Graph] = self.generate(n_samples=n_samples)
        return [DataTransformer.nx_to_mol(g, sanitize=validate, kekulize=validate) for g in Gs]

    def generate_data(self, n_samples: int = 16) -> list[Data]:
        Gs: list[nx.Graph] = self.generate(n_samples=n_samples)
        return [DataTransformer.nx_to_pyg(g) for g in Gs]


def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


if __name__ == '__main__':
    device = torch.device('cpu')
    generate_model_path = Path(
        "/Users/arvandkaveh/Projects/kit/graph_hdc/_models/2_real_nvp/2025-08-27_15-11-23_qahf/models/last.ckpt")

    gen_model = RealNVPLightning.load_from_checkpoint(generate_model_path, map_location="cpu", strict=True).to(
        device).eval()

    classifier_model_path = Path(
        "/Users/arvandkaveh/Projects/kit/graph_hdc/_models/2_base_line_mlp_mirror/2025-08-26_22-08-39_qahf/models/last.pt")
    classifier_config_path = Path(
        "/Users/arvandkaveh/Projects/kit/graph_hdc/_models/2_base_line_mlp_mirror/2025-08-26_22-08-39_qahf/evaluations/run_config.json")
    classifier_cls = get_oracle_classifier_config_cls("mirror_mlp")
    cfg = read_json(classifier_config_path)
    classifier = classifier_cls(cfg.get("hv_dim"), cfg.get("hidden_dims")).to(device).eval()
    oracle = Oracle(model=classifier)

    generator = Generator(gen_model=gen_model, oracle=oracle, ds_config=ZINC_SMILES_HRR_7744_CONFIG)
    Gs = generator.generate(n_samples=16)
    print(len(Gs))
    print(Gs)
    GS = list(filter(None, Gs))
    for i, gs in enumerate(Gs):
        print(f"Graph Nr: {i}")
        for j, g in enumerate(gs):
            print(f"Graph Nr: {i}-{j}")
            visualisations.draw_nx_with_atom_colorings(g)
            plt.show()
