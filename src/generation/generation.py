import json
import sys
from collections import Counter
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import rdkit.Chem
import torch
from networkx import Graph
from rdkit.Chem import Mol
from torch_geometric.data import Data
from src.encoding.configs_and_constants import QM9_SMILES_HRR_1600_CONFIG, ZINC_SMILES_HRR_7744_CONFIG
from src.encoding.graph_encoders import load_or_create_hypernet
from src.encoding.oracles import Oracle, MLPClassifier
from src.encoding.the_types import VSAModel
from src.encoding.decoder import greedy_oracle_decoder
from src.normalizing_flow.models import AbstractNFModel, RealNVPLightning, FlowConfig
from src.utils import visualisations
from src.utils.registery import resolve_model
from src.utils.utils import GLOBAL_MODEL_PATH, DataTransformer

## For unpickling
setattr(sys.modules['__main__'], 'FlowConfig', FlowConfig)


class Generator:
    def __init__(self, gen_model: AbstractNFModel, oracle: Oracle, ds_config):
        device = torch.device('cpu')
        print(f"Using device: {device}")
        self.encoder = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, cfg=ds_config).to(device)
        self.gen_model: torch.nn.Module = gen_model
        self.oracle = oracle
        self.oracle.encoder = self.encoder
        self.vsa: VSAModel = ds_config.vsa

    def generate(self, n_samples: int = 16) -> list[list[Graph]]:
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

    def generate_mols(self, n_samples: int = 16, validate: bool = True) -> list[tuple[Mol, dict[int, int]]]:
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
    for gen_model_path in [
        Path("/Users/akaveh/projects/kit/graph_hdc/_models/results/0_real_nvp/2025-09-01_02-45-49_qahf/models/last.ckpt"),
        Path("/Users/akaveh/projects/kit/graph_hdc/_models/results/0_real_nvp/2025-09-01_06-43-25_qahf/models/last.ckpt"),
        Path("/Users/akaveh/projects/kit/graph_hdc/_models/results/0_real_nvp/2025-09-01_16-34-49_qahf/models/last.ckpt"),
        Path("/Users/akaveh/projects/kit/graph_hdc/_models/results/0_real_nvp/2025-09-02_02-22-22_qahf/models/last.ckpt"),
    ]:
        print("----")
        print(gen_model_path)

        gen_model = resolve_model(name="NVP").load_from_checkpoint(gen_model_path, map_location="cpu", strict=True).to(
            device).eval()

        ## Classifier
        for model_path in [
            Path("/Users/akaveh/projects/kit/graph_hdc/_models/classifier_mlp_baseline.pt"),
            Path("/Users/akaveh/projects/kit/graph_hdc/_models/classifier_mlp_baseline_laynorm.pt"),
        ]:
            classifier = None
            if "mlp" in model_path.as_posix():
                classifier = resolve_model(name="MLP")
            if "bah" in model_path.as_posix():
                classifier = resolve_model(name="BAH")

            classifier.load_from_checkpoint(model_path, map_location="cpu", strict=True).to(device).eval()
            oracle = Oracle(model=classifier, model_type="mlp")

            ds_config = ZINC_SMILES_HRR_7744_CONFIG
            if "qm9" in model_path.as_posix():
                ds_config = QM9_SMILES_HRR_1600_CONFIG

            generator = Generator(gen_model=gen_model, oracle=oracle, ds_config=ds_config)
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
