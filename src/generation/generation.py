import abc
import sys
from collections import Counter
from pathlib import Path

import attr
import torch
from networkx import Graph
from torchhd import VSATensor
from tqdm.auto import tqdm

from src.encoding.configs_and_constants import DecoderSettings, DSHDCConfig
from src.encoding.graph_encoders import (
    MAX_ALLOWED_DECODING_NODES_QM9,
    MAX_ALLOWED_DECODING_NODES_ZINC,
    CorrectionLevel,
    DecodingResult,
    HyperNet,
    load_or_create_hypernet,
)
from src.encoding.the_types import VSAModel
from src.normalizing_flow.models import FlowConfig, RealNVPV2Lightning
from src.utils import registery
from src.utils.registery import get_model_type
from src.utils.utils import GLOBAL_BEST_MODEL_PATH, GLOBAL_MODEL_PATH, find_files

## For unpickling
sys.modules["__main__"].FlowConfig = FlowConfig


def get_model_path(hint: str) -> Path | None:
    for p in find_files(
        # start_dir=GLOBAL_MODEL_PATH,
        start_dir=GLOBAL_BEST_MODEL_PATH,
        prefixes=("epoch",),
        desired_ending=".ckpt",
    ):
        if hint in str(p):
            return p
    return None


@attr.define(slots=True, kw_only=True)
class AbstractGenerator(abc.ABC):
    gen_model_hint: str | Path
    ds_config: DSHDCConfig
    device: torch.device | None = None
    decoder_settings: DecoderSettings | None = None
    dtype: torch.dtype = torch.float32

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
            .to(device=self.device, dtype=self.dtype)
            .eval()
        )
        self.hypernet = (
            load_or_create_hypernet(path=GLOBAL_MODEL_PATH, cfg=self.ds_config)
            .to(device=self.device, dtype=self.dtype)
            .eval()
        )
        self.vsa = self.ds_config.vsa
        self.base_dataset = self.ds_config.base_dataset
        self.hypernet.base_dataset = self.base_dataset
        self.hypernet.normalize = self.ds_config.normalize

    def get_raw_samples(self, n_samples: int = 16) -> dict:
        return self.gen_model.sample_split(n_samples)

    def generate_most_similar(self, n_samples: int = 16) -> dict:
        samples = self.get_raw_samples(n_samples)
        nt = samples.get("node_terms")
        if nt is not None:
            nt = nt.as_subclass(self.vsa.tensor_class)
        et = samples.get("edge_terms")
        if et is not None:
            et = et.as_subclass(self.vsa.tensor_class)
        gt = samples.get("graph_terms")
        if gt is not None:
            gt = gt.as_subclass(self.vsa.tensor_class)
        return self.decode(node_terms=nt, edge_terms=et, graph_terms=gt)

    def decode(
        self,
        edge_terms: VSATensor,
        graph_terms: VSATensor,
        node_terms: VSATensor | None = None,
        *,
        most_similar: bool = True,
    ) -> dict:
        n_samples = graph_terms.shape[0]
        full_ctrs: dict[int, Counter[tuple[int, ...]]] | None = None
        if node_terms is not None:
            full_ctrs: dict[int, Counter[tuple[int, ...]]] = self.hypernet.decode_order_zero_counter(node_terms)

        best_graphs: list[Graph] = []
        are_final_flags: list[bool] = []
        all_similarities: list[float] = []
        intermediate_target_reached: list[bool] = []
        correction_levels: list[CorrectionLevel] = []

        for i in tqdm(range(n_samples), desc="Decoding", unit="sample"):
            full_ctr = None
            if full_ctrs is not None:
                full_ctr = full_ctrs.get(i)  # may be None if dedup/failed decode
                total_node_limit = (
                    MAX_ALLOWED_DECODING_NODES_QM9 if self.base_dataset == "qm9" else MAX_ALLOWED_DECODING_NODES_ZINC
                )
                if full_ctr is None or sum(full_ctr.values()) == 0 or full_ctr.total() > total_node_limit:
                    continue

            res = self._decode_single_graph(
                node_counter=full_ctr,
                edge_term=edge_terms[i],
                graph_term=graph_terms[i],
            )
            candidates, cos_sims, final_flags, target_reached, correction_level = (
                res.nx_graphs,
                res.cos_similarities,
                res.final_flags,
                res.target_reached,
                res.correction_level,
            )

            if not candidates:
                continue

            if most_similar:
                best_graphs.append(candidates[0])
                are_final_flags.append(final_flags[0])
                intermediate_target_reached.append(target_reached)
                all_similarities.append(cos_sims[0])
                correction_levels.append(correction_level)

        return {
            "graphs": best_graphs,
            "final_flags": are_final_flags,
            "similarities": all_similarities,
            "intermediate_target_reached": intermediate_target_reached,
            "correction_levels": correction_levels,
        }

    def decode_all_topk(
        self,
        edge_terms: VSATensor,
        graph_terms: VSATensor,
        node_terms: VSATensor | None = None,
    ) -> dict:
        """
        Decode samples and return ALL top-k results for each sample.
        Unlike decode(), this returns all k graphs from each DecodingResult, not just the best one.

        Args:
            edge_terms: Edge term hypervectors
            graph_terms: Graph term hypervectors
            node_terms: Optional node term hypervectors

        Returns:
            dict with keys:
            - 'all_graphs': List[List[Graph]] - top-k graphs for each sample
            - 'all_similarities': List[List[float]] - similarities for each graph
            - 'all_final_flags': List[List[bool]] - final flags for each graph
            - 'correction_levels': List[CorrectionLevel] - correction level per sample
            - 'intermediate_target_reached': List[bool] - target reached per sample
        """
        n_samples = graph_terms.shape[0]
        full_ctrs: dict[int, Counter[tuple[int, ...]]] | None = None
        if node_terms is not None:
            full_ctrs = self.hypernet.decode_order_zero_counter(node_terms)

        # Store ALL results per sample
        all_graphs_per_sample = []  # List[List[nx.Graph]]
        all_sims_per_sample = []  # List[List[float]]
        all_flags_per_sample = []  # List[List[bool]]
        correction_levels = []  # List[CorrectionLevel]
        intermediate_target_reached = []  # List[bool]

        for i in tqdm(range(n_samples), desc="Decoding (all top-k)", unit="sample"):
            full_ctr = None
            if full_ctrs is not None:
                full_ctr = full_ctrs.get(i)
                total_node_limit = (
                    MAX_ALLOWED_DECODING_NODES_QM9 if self.base_dataset == "qm9" else MAX_ALLOWED_DECODING_NODES_ZINC
                )
                if full_ctr is None or sum(full_ctr.values()) == 0 or full_ctr.total() > total_node_limit:
                    # Empty result for invalid samples
                    all_graphs_per_sample.append([])
                    all_sims_per_sample.append([])
                    all_flags_per_sample.append([])
                    correction_levels.append(CorrectionLevel.FAIL)
                    intermediate_target_reached.append(False)
                    continue

            res = self._decode_single_graph(
                node_counter=full_ctr,
                edge_term=edge_terms[i],
                graph_term=graph_terms[i],
            )

            # Store ALL top-k results (not just first)
            all_graphs_per_sample.append(res.nx_graphs)
            all_sims_per_sample.append(res.cos_similarities)
            all_flags_per_sample.append(res.final_flags)
            correction_levels.append(res.correction_level)
            intermediate_target_reached.append(res.target_reached)

        return {
            "all_graphs": all_graphs_per_sample,
            "all_similarities": all_sims_per_sample,
            "all_final_flags": all_flags_per_sample,
            "correction_levels": correction_levels,
            "intermediate_target_reached": intermediate_target_reached,
        }

    def extract_topn_all_decoder_outputs(
        self,
        edge_terms: VSATensor,
        graph_terms: VSATensor,
        sample_indices: list[int],
        decoder_k: int = 10,
        node_terms: VSATensor | None = None,
    ) -> dict:
        """
        Extract all decoder outputs for specific samples.
        Used after identifying top-n best molecules.

        Args:
            edge_terms: All edge term hypervectors
            graph_terms: All graph term hypervectors
            sample_indices: Indices of samples to extract
            decoder_k: Number of decoder outputs per sample
            node_terms: Optional node term hypervectors

        Returns:
            dict with keys:
            - 'graphs': Flattened list of all graphs
            - 'similarities': Flattened list of similarities
            - 'final_flags': Flattened list of final flags
            - 'sample_origins': Which original sample each graph came from
            - 'decoder_ranks': Decoder rank (1-indexed) for each graph
            - 'counts_per_sample': Dict mapping sample index to number of graphs decoded
        """
        # Temporarily set decoder top_k
        original_k = None
        if self.decoder_settings is not None:
            original_k = self.decoder_settings.top_k
            self.decoder_settings.top_k = decoder_k

        # Extract edge/graph terms for specified samples
        selected_edge_terms = edge_terms[sample_indices]
        selected_graph_terms = graph_terms[sample_indices]
        selected_node_terms = node_terms[sample_indices] if node_terms is not None else None

        # Decode all top-k for these samples
        results = self.decode_all_topk(
            edge_terms=selected_edge_terms,
            graph_terms=selected_graph_terms,
            node_terms=selected_node_terms,
        )

        # Flatten results for evaluation
        flattened_graphs = []
        flattened_sims = []
        flattened_flags = []
        sample_origins = []  # Track which original sample each came from
        decoder_ranks = []  # Track which decoder rank (1-indexed)

        for sample_idx, graphs_list in enumerate(results["all_graphs"]):
            original_idx = sample_indices[sample_idx]
            for rank, graph in enumerate(graphs_list):
                flattened_graphs.append(graph)
                flattened_sims.append(results["all_similarities"][sample_idx][rank])
                flattened_flags.append(results["all_final_flags"][sample_idx][rank])
                sample_origins.append(original_idx)
                decoder_ranks.append(rank + 1)  # 1-indexed

        # Restore original k
        if original_k is not None and self.decoder_settings is not None:
            self.decoder_settings.top_k = original_k

        return {
            "graphs": flattened_graphs,
            "similarities": flattened_sims,
            "final_flags": flattened_flags,
            "sample_origins": sample_origins,
            "decoder_ranks": decoder_ranks,
            "counts_per_sample": {
                sample_indices[i]: len(results["all_graphs"][i]) for i in range(len(sample_indices))
            },
        }

    @abc.abstractmethod
    def _decode_single_graph(
        self,
        node_counter: Counter[tuple[int, ...]],
        edge_term: torch.Tensor,
        graph_term: torch.Tensor,
    ) -> DecodingResult: ...


@attr.define(slots=True, kw_only=True)
class HDCGenerator(AbstractGenerator):
    def _decode_single_graph(
        self,
        edge_term: torch.Tensor,
        graph_term: torch.Tensor,
        node_counter: Counter[tuple[int, ...]] | None = None,
    ) -> DecodingResult:
        # TODO: Here we expand the 2D type
        return self.hypernet.decode_graph(
            edge_term=edge_term,
            graph_term=graph_term,
            decoder_settings=self.decoder_settings,
        )


@attr.define(slots=True, kw_only=True)
class HDCZ3Generator(AbstractGenerator):
    def _decode_single_graph(
        self,
        edge_term: torch.Tensor,
        graph_term: torch.Tensor,
        node_counter: Counter[tuple[int, ...]] | None = None,
    ) -> DecodingResult:
        # TODO: Here we expand the 2D type
        return self.hypernet.decode_graph_z3(
            node_counter=node_counter,
            edge_term=edge_term,
            graph_term=graph_term,
            decoder_settings=self.decoder_settings,
        )
