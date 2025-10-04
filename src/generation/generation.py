import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import torch
from networkx import Graph
from torch_geometric.data import Batch

from src.encoding.configs_and_constants import DatasetConfig, QM9_SMILES_HRR_1600_CONFIG, ZINC_SMILES_HRR_7744_CONFIG
from src.encoding.decoder import greedy_oracle_decoder_faster, greedy_oracle_decoder_voter_oracle
from src.encoding.graph_encoders import load_or_create_hypernet
from src.encoding.oracles import Oracle, SimpleVoterOracle
from src.encoding.the_types import VSAModel
from src.normalizing_flow.models import AbstractNFModel, FlowConfig
from src.utils import visualisations
from src.utils.registery import resolve_model
from src.utils.utils import GLOBAL_MODEL_PATH, DataTransformer

## For unpickling
sys.modules["__main__"].FlowConfig = FlowConfig


class Generator:
    def __init__(
        self,
        gen_model: AbstractNFModel,
        oracle: Oracle | SimpleVoterOracle,
        decoder_settings: dict,
        ds_config: DatasetConfig,
        device=None,
    ):
        device = torch.device("cpu") if device is None else device
        print(f"Using device: {device}")
        self.encoder = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, cfg=ds_config).to(device)
        self.gen_model: torch.nn.Module = gen_model
        self.oracle = oracle
        self.oracle.encoder = self.encoder.eval()
        self.vsa: VSAModel = ds_config.vsa
        self.decoder_settings = decoder_settings
        self.decoding_fn = (
            greedy_oracle_decoder_voter_oracle
            if isinstance(oracle, SimpleVoterOracle)
            else greedy_oracle_decoder_faster
        )
        base_dataset = "zinc" if "zinc" in ds_config.name else "qm9"
        self.decode_skip_n_nodes_threshold = 70 if base_dataset == "zinc" else 15

    def generate_all(
        self,
        n_samples: int = 16,
        *,
        only_final_graphs: bool = True,
    ) -> tuple[list[list[Graph]], list[bool]]:
        """
        Generate candidates for each sample and return *all* graphs.

        :param n_samples: Number of independent samples to draw.
        :param only_final_graphs: If ``True``, the decoder enforces final/valid outputs.
        :returns: A pair ``(graphs_per_sample, are_final_flags)`` where

                  - ``graphs_per_sample`` is a list of lists of candidate graphs
                    (one list per sample, may be empty).
                  - ``are_final_flags`` is a boolean list (len == ``n_samples``)
                    indicating whether *that* sample's candidate set was deemed final.

        Notes
        -----
        The previous API returned ``list[tuple[list[Graph], bool]]``. This method
        separates data from flags to avoid shadowing/tuple unpacking pitfalls.
        """
        node_terms, graph_terms, _ = self.gen_model.sample_split(n_samples)
        node_terms_hd = node_terms.as_subclass(self.vsa.tensor_class)
        graph_terms_hd = graph_terms.as_subclass(self.vsa.tensor_class)

        full_ctrs = self.encoder.decode_order_zero_counter(node_terms_hd)  # dict[int, Counter]

        graphs_per_sample: list[list[Graph]] = []
        are_final_flags: list[bool] = []

        for i, full_ctr in enumerate(full_ctrs.values()):
            # decoder returns (candidates, is_final) for a single sample
            candidates, is_final = self.decoding_fn(
                node_multiset=full_ctr,
                oracle=self.oracle,
                full_g_h=graph_terms_hd[i],
                strict=only_final_graphs,
                skip_n_nodes=self.decode_skip_n_nodes_threshold,
                **self.decoder_settings,
            )
            graphs_per_sample.append(candidates)
            are_final_flags.append(bool(is_final))

        return graphs_per_sample, are_final_flags

    def generate_most_similar(
        self,
        n_samples: int = 16,
        *,
        only_final_graphs: bool = True,
    ) -> tuple[list[Graph], list[bool], list[list[float]]]:
        node_terms, graph_terms, _ = self.gen_model.sample_split(n_samples)
        return self.decode(node_terms, graph_terms, only_final_graphs=only_final_graphs)

    def decode(self, node_terms: torch.Tensor, graph_terms: torch.Tensor, *, only_final_graphs: bool = True):
        n_samples = node_terms.shape[0]
        node_terms_hd = node_terms.as_subclass(self.vsa.tensor_class)
        graph_terms_hd = graph_terms.as_subclass(self.vsa.tensor_class)

        full_ctrs: dict[int, Counter[tuple[int, ...]]] = self.encoder.decode_order_zero_counter(node_terms_hd)

        best_graphs: list[Graph] = []
        are_final_flags: list[bool] = []
        all_similarities: list[list[float]] = []

        def _row_norm(x: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
            return x / (x.norm(dim=dim, keepdim=True) + eps)

        # --- important: iterate by requested index---
        for i in range(n_samples):
            full_ctr = full_ctrs.get(i)  # may be None if dedup/failed decode
            if full_ctr is None or sum(full_ctr.values()) == 0:
                print("[WARNING] full ctr is None or empty.")
                # nothing to decode for this sample → return empty
                best_graphs.append(nx.Graph())
                are_final_flags.append(False)
                all_similarities.append([0])
                continue

            candidates, is_final = self.decoding_fn(
                node_multiset=full_ctr,
                oracle=self.oracle,
                full_g_h=graph_terms_hd[i],
                strict=only_final_graphs,
                **self.decoder_settings,
            )
            are_final_flags.append(bool(is_final))

            if not candidates:
                best_graphs.append(nx.Graph())
                all_similarities.append([0])
                continue

            nonempty_idx = [k for k, g in enumerate(candidates) if g.number_of_nodes() > 0]
            if not nonempty_idx:
                best_graphs.append(nx.Graph())
                all_similarities.append([0])
                continue

            data_list = [DataTransformer.nx_to_pyg(candidates[k]) for k in nonempty_idx]

            try:
                batch = Batch.from_data_list(data_list)
                enc_out = self.encoder.forward(batch)
                g_terms = enc_out["graph_embedding"]  # [B, D]
            except Exception:
                best_graphs.append(nx.Graph())
                all_similarities.append([0])
                continue

            q = graph_terms_hd[i].to(g_terms.device, g_terms.dtype)  # [D]
            g_norm = _row_norm(g_terms, dim=1)  # [B, D]
            q_norm = q / (q.norm() + 1e-8)  # [D]
            sims_t = g_norm @ q_norm  # [B]
            sims = sims_t.tolist()

            full_sims = [-float("inf")] * len(candidates)
            for pos, k in enumerate(nonempty_idx):
                full_sims[k] = sims[pos]

            all_similarities.append(full_sims)
            best_idx = int(max(range(len(full_sims)), key=lambda j: full_sims[j]))
            best_graphs.append(candidates[best_idx])

        return best_graphs, are_final_flags, all_similarities


def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    device = torch.device("cpu")
    for gen_model_path in [
        Path(
            "/Users/akaveh/projects/kit/graph_hdc/_models/results/0_real_nvp/2025-09-01_02-45-49_qahf/models/last.ckpt"
        ),
        Path(
            "/Users/akaveh/projects/kit/graph_hdc/_models/results/0_real_nvp/2025-09-01_06-43-25_qahf/models/last.ckpt"
        ),
        Path(
            "/Users/akaveh/projects/kit/graph_hdc/_models/results/0_real_nvp/2025-09-01_16-34-49_qahf/models/last.ckpt"
        ),
        Path(
            "/Users/akaveh/projects/kit/graph_hdc/_models/results/0_real_nvp/2025-09-02_02-22-22_qahf/models/last.ckpt"
        ),
    ]:
        print("----")
        print(gen_model_path)

        gen_model = (
            resolve_model(name="NVP")
            .load_from_checkpoint(gen_model_path, map_location="cpu", strict=True)
            .to(device)
            .eval()
        )

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
