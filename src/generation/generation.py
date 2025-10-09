import abc
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import attr
import networkx as nx
import torch
import torchhd
from networkx import Graph
from torch_geometric.data import Batch
from tqdm.auto import tqdm

from src.encoding.configs_and_constants import DatasetConfig
from src.encoding.decoder import greedy_oracle_decoder_faster, greedy_oracle_decoder_voter_oracle
from src.encoding.graph_encoders import HyperNet, load_or_create_hypernet
from src.encoding.oracles import Oracle, SimpleVoterOracle
from src.encoding.the_types import VSAModel
from src.normalizing_flow.models import FlowConfig, RealNVPV2Lightning
from src.utils import registery
from src.utils.registery import get_model_type
from src.utils.utils import GLOBAL_MODEL_PATH, DataTransformer, find_files

## For unpickling
sys.modules["__main__"].FlowConfig = FlowConfig


def get_model_path(hint: str) -> Path | None:
    for p in find_files(
        start_dir=GLOBAL_MODEL_PATH,
        prefixes=("epoch",),
        desired_ending=".ckpt",
    ):
        if hint in str(p):
            return p
    return None


@attr.define(slots=True, kw_only=True)
class AbstractGenerator(abc.ABC):
    gen_model_hint: str | Path
    ds_config: DatasetConfig
    device: torch.device | None = None
    decoder_settings: dict[str, Any] = attr.Factory(dict)

    # Post init stuff
    gen_model: RealNVPV2Lightning = attr.field(init=False)
    hypernet: HyperNet = attr.field(init=False)
    vsa: VSAModel = attr.field(init=False)
    base_dataset: str = attr.field(init=False)

    def __attrs_post_init__(self) -> None:
        self.device = torch.device("cpu") if self.device is None else self.device
        print(f"[{self.__class__.__name__}] is using device: {self.device}")

        gen_ckpt_path = get_model_path(hint=self.gen_model_hint)
        print(f"Generator Checkpoint: {gen_ckpt_path}")

        model_type = get_model_type(gen_ckpt_path)
        assert model_type is not None, f"Model type not found for {gen_ckpt_path}"

        self.gen_model = (
            registery.retrieve_model(model_type)
            .load_from_checkpoint(gen_ckpt_path, map_location="cpu", strict=True)
            .to(self.device)
        )
        self.hypernet = load_or_create_hypernet(path=GLOBAL_MODEL_PATH, cfg=self.ds_config).to(self.device).eval()
        self.vsa = self.ds_config.vsa
        self.base_dataset = "zinc" if "zinc" in self.ds_config.name.lower() else "qm9"

    @abc.abstractmethod
    def generate_most_similar(self, n_samples: int = 16) -> dict: ...


@attr.define(slots=True, kw_only=True)
class HDCGenerator(AbstractGenerator):
    def generate_most_similar(self, n_samples: int = 16) -> dict:
        samples = self.gen_model.sample_split(n_samples)
        node_terms = samples["node_terms"].to(torch.float64).as_subclass(self.vsa.tensor_class)
        edge_terms = samples["edge_terms"].to(torch.float64).as_subclass(self.vsa.tensor_class)
        graph_terms = samples["graph_terms"].to(torch.float64).as_subclass(self.vsa.tensor_class)

        full_ctrs: dict[int, Counter[tuple[int, ...]]] = self.hypernet.decode_order_zero_counter(node_terms)

        best_graphs: list[Graph] = []
        are_final_flags: list[bool] = []
        all_similarities: list[list[float]] = []

        # --- important: iterate by requested index---
        for i in tqdm(range(n_samples), desc="Decoding", unit="sample"):
            full_ctr = full_ctrs.get(i)  # may be None if dedup/failed decode
            if full_ctr is None or sum(full_ctr.values()) == 0 or full_ctr.total() > 20:
                print("[WARNING] full ctr is None or empty.")
                # nothing to decode for this sample → return empty
                best_graphs.append(nx.Graph())
                are_final_flags.append(False)
                all_similarities.append([0])
                continue

            candidates, final_flags = self.hypernet.decode_graph(
                node_counter=full_ctr, edge_term=edge_terms[i], graph_term=graph_terms[i]
            )

            if len(candidates) == 1 and candidates[0].number_of_nodes() == 0:
                best_graphs.append(candidates[0])
                are_final_flags.append(final_flags[0])
                all_similarities.append([0])
                continue

            assert all(c.number_of_nodes() > 0 for c in candidates)
            data_list = [DataTransformer.nx_to_pyg(c) for c in candidates]
            try:
                batch = Batch.from_data_list(data_list)
                enc_out = self.hypernet.forward(batch)
                g_terms = enc_out["graph_embedding"]  # [B, D]
            except Exception:
                best_graphs.append(nx.Graph())
                all_similarities.append([0])
                are_final_flags.append(False)
                continue

            sampled_g_term = graph_terms[i].to(g_terms.device, g_terms.dtype)  # [D]
            sims = torchhd.cos(sampled_g_term, g_terms).tolist()

            all_similarities.append(sims)
            best_idx = sims.index(max(sims))
            best_graphs.append(candidates[best_idx])
            are_final_flags.append(final_flags[best_idx])

        return {
            "graphs": best_graphs,
            "final_flags": are_final_flags,
            "similarities": all_similarities,
        }


@attr.define(slots=True, kw_only=True)
class OracleGenerator(AbstractGenerator):
    oracle: Oracle | SimpleVoterOracle
    decode_skip_n_nodes_threshold: int = attr.field(init=False)
    decoding_fn: Any = attr.field(init=False)  # put the real callable type if you have one

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        # wire oracle
        self.oracle.encoder = self.hypernet.eval()
        # pick decoder
        self.decoding_fn = (
            greedy_oracle_decoder_voter_oracle
            if isinstance(self.oracle, SimpleVoterOracle)
            else greedy_oracle_decoder_faster
        )
        self.decode_skip_n_nodes_threshold = 70 if self.base_dataset == "zinc" else 15

    def generate_most_similar(
        self,
        n_samples: int = 16,
        *,
        only_final_graphs: bool = False,
    ) -> tuple[list[Graph], list[bool], list[list[float]]]:
        node_terms, graph_terms, _ = self.gen_model.sample_split(n_samples)
        return self.decode(node_terms, graph_terms, only_final_graphs=only_final_graphs)

    def decode(self, node_terms: torch.Tensor, graph_terms: torch.Tensor, *, only_final_graphs: bool = True):
        n_samples = node_terms.shape[0]
        node_terms_hd = node_terms.as_subclass(self.vsa.tensor_class)
        graph_terms_hd = graph_terms.as_subclass(self.vsa.tensor_class)

        full_ctrs: dict[int, Counter[tuple[int, ...]]] = self.hypernet.decode_order_zero_counter(node_terms_hd)

        best_graphs: list[Graph] = []
        are_final_flags: list[bool] = []
        all_similarities: list[list[float]] = []

        # --- important: iterate by requested index---
        for i in tqdm(range(n_samples), desc="Decoding", unit="sample"):
            full_ctr = full_ctrs.get(i)  # may be None if dedup/failed decode
            if full_ctr is None or sum(full_ctr.values()) == 0:
                print("[WARNING] full ctr is None or empty.")
                # nothing to decode for this sample → return empty
                best_graphs.append(nx.Graph())
                are_final_flags.append(False)
                all_similarities.append([0])
                continue

            candidates, final_flags = self.decoding_fn(
                node_multiset=full_ctr,
                oracle=self.oracle,
                full_g_h=graph_terms_hd[i],
                strict=only_final_graphs,
                skip_n_nodes=self.decode_skip_n_nodes_threshold,
                **self.decoder_settings,
            )

            if len(candidates) == 1 and candidates[0].number_of_nodes() == 0:
                best_graphs.append(candidates[0])
                are_final_flags.append(final_flags[0])
                all_similarities.append([0])
                continue

            assert all(c.number_of_nodes() > 0 for c in candidates)
            data_list = [DataTransformer.nx_to_pyg(c) for c in candidates]
            try:
                batch = Batch.from_data_list(data_list)
                enc_out = self.hypernet.forward(batch)
                g_terms = enc_out["graph_embedding"]  # [B, D]
            except Exception:
                best_graphs.append(nx.Graph())
                all_similarities.append([0])
                are_final_flags.append(False)
                continue

            q = graph_terms_hd[i].to(g_terms.device, g_terms.dtype)  # [D]
            sims = torchhd.cos(q, g_terms).tolist()

            all_similarities.append(sims)
            best_idx = sims.index(max(sims))
            best_graphs.append(candidates[best_idx])
            are_final_flags.append(final_flags[best_idx])

        return {
            "graphs": best_graphs,
            "final_flags": are_final_flags,
            "similarities": all_similarities,
        }
