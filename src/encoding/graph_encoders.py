import enum
import itertools
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Final, Literal

import networkx as nx
import numpy as np
import pytorch_lightning as pl
import torch
import torchhd
from torch import Tensor
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.loader import DataLoader
from torchhd import VSATensor
from tqdm import tqdm

from src.datasets.utils import DatasetInfo, get_dataset_info
from src.encoding.configs_and_constants import (
    BaseDataset,
    DSHDCConfig,
    Features,
    IndexRange,
)
from src.encoding.correction_utilities import CorrectionResult, get_corrected_sets, get_node_counter, target_reached
from src.encoding.decoder import compute_sampling_structure, has_valid_ring_structure, try_find_isomorphic_graph
from src.encoding.feature_encoders import (
    AbstractFeatureEncoder,
    CategoricalIntegerEncoder,
    CategoricalLevelEncoder,
    CategoricalOneHotEncoder,
    CombinatoricIntegerEncoder,
    TrueFalseEncoder,
)
from src.encoding.the_types import Feat, VSAModel
from src.encoding.z3_decoder import enumerate_graphs
from src.utils.nx_utils import (
    _hash,
    add_node_and_connect,
    add_node_with_feat,
    anchors,
    connect_all_if_possible,
    leftover_features,
    order_leftovers_by_degree_distinct,
    powerset,
    residual_degree,
)
from src.utils.utils import (
    GLOBAL_MODEL_PATH,
    DataTransformer,
    TupleIndexer,
    cartesian_bind_tensor,
    flatten_counter,
    scatter_hd,
)

# === HYPERDIMENSIONAL MESSAGE PASSING NETWORKS ===

EncoderMap = dict[Features, tuple[AbstractFeatureEncoder, IndexRange]]
MAX_ALLOWED_DECODING_NODES_QM9: Final[int] = 18
MAX_ALLOWED_DECODING_NODES_ZINC: Final[int] = 60
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class CorrectionLevel(str, enum.Enum):
    FAIL = "failed to correct"
    ZERO = "not corrected"
    ONE = "edge added/removed"
    TWO = "edge added/removed then re-decoded"
    THREE = "edge added/removed of the level two results"


@dataclass
class DecodingResult:
    nx_graphs: list[nx.Graph] = field(default_factory=list)
    final_flags: list[bool] = field(default_factory=lambda: [False])
    target_reached: bool = False
    cos_similarities: list[float] = field(default_factory=lambda: [0.0])
    correction_level: CorrectionLevel = CorrectionLevel.ZERO


class AbstractGraphEncoder(pl.LightningModule):
    def __init__(self, **kwargs):
        pl.LightningModule.__init__(self, **kwargs)

    def forward_graphs(
        self,
        dataset: Dataset,
        batch_size: int = 128,
    ) -> list[dict[str, np.ndarray]]:
        """
        Given a list of ``Dataset`` this method will run the hypernet "forward" pass on all the data batches
        and return a list of dictionaries where each dict represents the result of the forward pass for
        each of the given graphs.

        :param dataset: A list of graph dict representations where each dict contains the information
            about the nodes, edges, and properties of the graph.
        :param batch_size: The batch size to use for the forward pass internally.

        :returns: A list of result dictionaries where each dict contains the same string keys as the
            result of the "forward" method.
        """

        # first of all we need to convert the graphs into a format that can be used by the hypernet.
        # For this task there is the utility function "data_list_from_graph_dicts" which will convert
        # the list of graph dicts into a list of torch_geometric Data objects.
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        result_list: list[dict[str, np.ndarray]] = []
        for data in data_loader:
            # The problem here is that the "data" object yielded by the data loader contains multiple
            # batched graphs but to return the results we would like to disentangle this information
            # back to the individual graphs.
            result: dict[str, torch.Tensor] = self.forward(data)

            # The "extract_graph_results" method will take the batched results and disentangle them
            # into a list of dictionaries with the same string keys as the batched results but where
            # the values are the numpy array representations of the tensors only for the specific graphs.
            results: list[dict[str, np.ndarray]] = self.extract_graph_results(data, result)
            result_list.extend(results)

        return result_list

    @staticmethod
    def extract_graph_results(
        data: Data,
        graph_results: dict[str, torch.Tensor],
    ) -> list[dict[str, np.ndarray]]:
        """
        Given an input ``data`` object and the ``graph_results`` dict that is returned by the "forward" method
        of the hyper net, this method will disentangle these *batched* results into a list of individual
        dictionaries where each dict contains the results of the individual graphs in the batch in the form
        of numpy arrays.

        This disentanglement is done dynamically based on the string key names that can be found in the results
        dict returned by the "forward" method. The following prefix naming conventions should be used when returning
        properties as part of the results:
        - "graph_": for properties that are related to the overall graph with a shape of (batch_size, ?)
        - "node_": for properties that are related to the individual nodes with a shape of (batch_size * num_nodes, ?)
        - "edge_": for properties that are related to the individual edges with a shape of (batch_size * num_edges, ?)

        :param data: The PyG Data object that represents the batch of graphs.
        :param graph_results: The dictionary that contains the results of the forward pass for the batch of
            graphs.

        :returns: A list of dictionaries where each dict contains the results of the individual graphs in
            the batch.
        """
        # The batch size as calculated from the data object
        batch_size = torch.max(data.batch).detach().numpy() + 1

        # In this list we will store the disentangled results for each of the individual graphs in the batch
        # in the form of a dictionary with the same keys as the batched dict results "graph_results" but
        # where the values are the numpy array representations of the tensors only for the specific graphs.
        results: list[dict[str, np.ndarray]] = []
        for index in range(batch_size):
            node_mask: torch.Tensor = data.batch == index
            edge_mask: torch.Tensor = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]

            result: dict[str, np.ndarray] = {}
            for key, tens in graph_results.items():
                if key.startswith("graph"):
                    result[key] = tens[index].cpu().detach().numpy()

                elif key.startswith("node"):
                    result[key] = tens[node_mask].cpu().detach().numpy()

                elif key.startswith("edge"):
                    result[key] = tens[edge_mask].cpu().detach().numpy()

            results.append(result)

        return results

    # == To be implemented ==

    def forward(
        self,
        data: Data,
    ) -> dict[str, torch.Tensor]:
        """
        This method accepts a PyG Data object which represents a *batch* of graphs and is supposed
        to implement the forward pass encoding of these graphs into the hyperdimensional vector.
        The method should return a dictionary which contains at least the key "graph_embedding"
        which should be the torch Tensor representation of the encoded graph embeddings for the
        various graphs in the batch.
        """
        raise NotImplementedError

    # Replacing the instance attributes with loaded state from a given path
    def load_from_path(self, path: str):
        """
        Given an existing absolute file ``path`` this method should implement the loading of the
        properties from that file to replace the current properties of the HyperNet object instance
        """
        raise NotImplementedError

    # Saving the instance attributes to a given path
    def save_to_path(self, path: str):
        """
        Given an absolute file ``path`` this method should implement the saving of the current properties
        of the HyperNet object instance to that file.
        """
        raise NotImplementedError

    def get_data_x(self, node_counter, node_encoder_map):
        """
        Build a `data.x` like tensor from a Counter of node tuples and an ordered map of encoders.

        Args:
          node_counter: Counter mapping each node tuple (t0, t1, …, t_{k-1}) → count (int).
                        Each tuple has length k, where k = number of encoders.
          node_encoder_map: OrderedDict whose values are (encoder, index_range, some_int).
                            We only care about `encoder`, which must implement
                            `decode_index(label: int) -> Tensor[D]`.  The order of these
                            encoders must match the positions of the key tuples.

        Returns:
          A Tensor of shape [N, k*D], where N = sum(node_counter.values()), D is the
          hypervector dimension returned by `encode_index(...)`, and k = len(encoders).
        """
        # 1) Extract encoders in a LIST in the same order as the map
        encoders: list[AbstractFeatureEncoder] = [v[0] for v in node_encoder_map.values()]

        k = len(encoders)
        if k == 0:
            # No encoders → no features → return empty [0,0] tensor
            return torch.empty((0, 0))

        if k == 1:
            # Special case we have only one combinatorial node encoder
            node_idx = flatten_counter(node_counter)
            return torch.tensor([node_idx], device=self.device, dtype=torch.float).squeeze()

        # 3) Prepare a list to accumulate each repeated block
        rows: list[torch.Tensor] = []

        # 4) For each (tuple_of_labels → count) in insertion order:
        # We're only re-constructing a single graph, no batches
        for key_tuple, count in node_counter.items():
            # b) Decode each component label to a [D]‐vector
            parts: list[torch.Tensor] = []
            for i, label in enumerate(key_tuple):
                enc = encoders[i] if len(encoders) > 1 else encoders[0]
                vec_i = enc.decode_index(label).view(-1)  # shape [D]
                parts.append(vec_i)

            # c) Concatenate these k vectors → one node hypervector of size [k*D]
            node_vec = torch.cat(parts, dim=0)  # shape [k*D]

            # d) Repeat that vector `count` times along a new batch dimension
            if count < 0:
                msg = f"Invalid count {count} for key {key_tuple}"
                raise ValueError(msg)

            if count > 0:
                repeated_block = node_vec.unsqueeze(0).repeat(count, 1)  # shape [count, k*D]
                rows.append(repeated_block)
            # if count == 0: skip (no rows to add)

        # 5) Concatenate all blocks along dim=0 to form the final [N, k*D] tensor
        if not rows:
            # Every count must have been zero (unlikely, but handle it)
            return torch.empty((0, k * self.hv_dim), device=self.device)

        # shape [sum(counts), k*D]
        return torch.cat(rows, dim=0)

    def get_edge_index_list(self, edge_counter: Counter[Any], node_counter: Counter[Any]) -> list[tuple[int, int]]:
        # ~ Build node information from constraints list
        # This data structure will contain a unique integer node index as the key and the value will
        # be the dictionary which contains the node properties that were originally decoded.
        nodes_idxs = self.nodes_indexer.get_idxs(flatten_counter(node_counter))

        # ~ Build edge information from constraints list
        edge_indices: set[tuple[int, int]] = set()
        for i in range(len(nodes_idxs)):
            for j in range(i + 1, len(nodes_idxs)):
                u = nodes_idxs[i]
                v = nodes_idxs[j]
                if (u, v) in edge_counter:
                    edge_indices.add((i, j))
                    edge_indices.add((j, i))
        return list(edge_indices)


class HyperNet(AbstractGraphEncoder):
    # MAP XOR operation is not differentiable
    # VTB does some in-place operations which breaks the gradient
    # FHRR is imaginary numbers and cannot be differentiated
    __allowed_vsa_models__: ClassVar[set[VSAModel]] = {VSAModel.MAP, VSAModel.FHRR, VSAModel.HRR}

    def __init__(
        self,
        config: DSHDCConfig | None = None,
        depth: int = 3,
        *,
        use_explain_away: bool = True,
        use_edge_codebook: bool = True,
    ):
        AbstractGraphEncoder.__init__(self)
        self.validate_ring_structure: bool = False
        self.use_explain_away = use_explain_away
        self.use_edge_codebook = use_edge_codebook
        self.depth = depth
        self.vsa = self._validate_vsa(config.vsa)
        self.hv_dim = config.hv_dim

        # Relate to decoding limits to prevent endless loops
        self._directed_decoded_edge_limit: int = 66  # Default for zinc
        self._max_step_delta: float | None = None  # will be set after first run

        self._cfg_device = torch.device(getattr(config, "device", "cpu"))

        self.node_encoder_map: EncoderMap = {
            feat: (
                cfg.encoder_cls(
                    num_categories=cfg.count,
                    dim=config.hv_dim,
                    vsa=config.vsa.value,
                    idx_offset=cfg.idx_offset,
                    device=self._cfg_device,
                    indexer=TupleIndexer(sizes=cfg.bins) if cfg.bins else TupleIndexer(sizes=[28, 6, config.nha_bins]),
                    dtype=config.dtype,
                ),
                cfg.index_range,
            )
            for feat, cfg in config.node_feature_configs.items()
        }
        self.edge_encoder_map: EncoderMap = {
            feat: (
                cfg.encoder_cls(
                    num_categories=cfg.count,
                    dim=config.hv_dim,
                    vsa=config.vsa.value,
                    idx_offset=cfg.idx_offset,
                    device=self._cfg_device,
                    dtype=config.dtype,
                ),
                cfg.index_range,
            )
            for feat, cfg in config.edge_feature_configs.items()
        }
        self.graph_encoder_map: EncoderMap = {
            feat: (
                cfg.encoder_cls(
                    num_categories=cfg.count,
                    dim=config.hv_dim,
                    vsa=config.vsa.value,
                    idx_offset=cfg.idx_offset,
                    device=self._cfg_device,
                    dtype=config.dtype,
                ),
                cfg.index_range,
            )
            for feat, cfg in config.graph_feature_configs.items()
        }
        self.seed = config.seed

        ### Attributes that will be populated after initialization
        self._init_lazy_fields()

    @property
    def decoding_limit_for(self) -> int:
        return self._directed_decoded_edge_limit

    @decoding_limit_for.setter
    def decoding_limit_for(self, base_dataset: str) -> None:
        if base_dataset == "qm9":
            self._directed_decoded_edge_limit = 50
        elif base_dataset == "zinc":
            # already the default
            self._directed_decoded_edge_limit = 122  # max 88 in train

    @property
    def dataset_info(self) -> DatasetInfo:
        return self._dataset_info

    @dataset_info.setter
    def dataset_info(self, info: DatasetInfo) -> None:
        self._dataset_info = info

    @property
    def base_dataset(self) -> BaseDataset:
        return self._base_dataset

    @base_dataset.setter
    def base_dataset(self, base_dataset: BaseDataset):
        self._base_dataset = base_dataset
        self.decoding_limit_for = base_dataset
        self._dataset_info = get_dataset_info(base_dataset)
        self.limit_nodes_codebook()

    def _init_lazy_fields(self) -> None:
        """Create attributes that __init__ normally sets but load() may bypass."""
        # Contains hypervectors that represents a Node (HV_node = bind(HV(f1), ..., HV(f2)))
        self.nodes_codebook = None
        self.nodes_indexer = None
        # Contains hypervectors that represents an edge regardless of its nodes (HV_node = bind(HV(f1), ..., HV(f2)))
        self.edge_feature_codebook = None
        self.edge_feature_indexer = None
        # Contains hypervectors that represents an edge including the nodes and edge features
        self.edges_codebook = None
        self.edges_indexer = None
        self.normalize: bool = False
        self._max_step_delta: float | None = None
        self._directed_decoded_edge_limit: int = 50  # Default for zinc
        self._base_dataset: BaseDataset = "qm9"
        self._dataset_info: DatasetInfo

    def to(self, device, dtype=None):
        # normalize + store; also move nn.Module state if any
        device = torch.device(device)
        super().to(device)  # safe even if there are no nn.Parameters
        if dtype is not None:
            super().to(dtype)

        self.populate_codebooks()  # ensure they exist before moving

        # move codebooks (non-parameter buffers)
        if self.nodes_codebook is not None:
            self.nodes_codebook = self.nodes_codebook.to(device=self.device, dtype=self.dtype)
        if getattr(self, "edge_feature_codebook", None) is not None:
            self.edge_feature_codebook = self.edge_feature_codebook.to(device=self.device, dtype=self.dtype)
        if self.use_edge_codebook and self.edges_codebook is not None:
            self.edges_codebook = self.edges_codebook.to(device=self.device, dtype=self.dtype)

        # move encoder codebooks & record their device
        for enc_map in (self.node_encoder_map, self.edge_encoder_map, self.graph_encoder_map):
            for enc, _ in enc_map.values():
                enc.device = self.device
                enc.dtype = self.dtype
                if getattr(enc, "codebook", None) is not None:
                    enc.codebook = enc.codebook.to(device=self.device, dtype=self.dtype)

        return self

    def _validate_vsa(self, vsa: VSAModel) -> VSAModel:
        if vsa in self.__allowed_vsa_models__:
            return vsa

        err_msg = (
            f"{vsa.value}Tensor is not supported by {self.__class__.__name__}. "
            f"Supported vsa models are: {self.__allowed_vsa_models__}"
        )
        raise ValueError(err_msg)

    def populate_codebooks(self) -> None:
        """Generates the codebooks and indexers for the nodes, edges, and graph features."""
        self.populate_nodes_codebooks()
        self._populate_nodes_indexer()
        self.populate_edge_feature_codebook()
        self._populate_edge_feature_indexer()
        if self.use_edge_codebook:
            self.populate_edges_codebook()
            self._populate_edges_indexer()

    # -- encoding
    # These methods handle the encoding of the graph structures into the graph embedding vector

    def encode_properties(self, data: Data) -> Data:
        """
        Encode node, edge and graph properties into hyper-vectors.

        :param data: a torch_geometric Data or Batch object, with attributes
            - x:  [num_nodes, total_node_feature_dims]
            - edge_attr: [num_edges, total_edge_feature_dims] or [num_edges]
            - y: [num_graphs, total_graph_feature_dims] or [num_graphs]

        :returns: same Data object with added attributes
            - node_hv: [num_nodes, D]
            - edge_hv: [num_edges, D]
            - graph_hv: [num_graphs, D]
        """

        # --- node‐level ---
        num_nodes = data.x.size(0)
        data.node_hv = self._slice_encode_bind(self.node_encoder_map, data.x, fallback_count=num_nodes)

        # --- edge‐level ---
        if self.use_edge_features():
            num_edges = data.edge_index.size(1)
            data.edge_hv = self._slice_encode_bind(self.edge_encoder_map, data.edge_attr, fallback_count=num_edges)

        # --- graph‐level ---
        if self.use_graph_features():
            num_graphs = data.y.size(0)
            data.graph_hv = self._slice_encode_bind(self.graph_encoder_map, data.y, fallback_count=num_graphs)

        return data

    def _slice_encode_bind(self, encoder_map: EncoderMap, tensor: Tensor, fallback_count: int) -> Tensor | None:
        """
        Generic helper to:
          1) split `tensor` into slices by each (start,end) in encoder_map
          2) run each encoder on its slice → list of [..., N?, D]
          3) multibind them into one [..., N?, D]
          4) if no encoders: return zeros([fallback_count, D])
        """
        if tensor is None or encoder_map is None:
            return None

        # ensure last‐axis is “feature” dim - with edge and graph features it can be [N,]
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(-1)  # [N,1]

        slices = []
        for encoder, (start, end) in encoder_map.values():
            # slice out exactly those features
            feat = tensor[..., start:end]  # e.g. [N, E_i]
            slices.append(encoder.encode(feat))  # [..., N?, D]

        if not slices:
            return torch.zeros(fallback_count, self.hv_dim, device=tensor.device, dtype=tensor.dtype)

        # stack on new “property” axis and bind
        stacked = torch.stack(slices, dim=0)  # [P, N, D]
        # move property axis to just before D for multibind on P dimension:
        to_bind = stacked.transpose(0, 1)  # [N, P, D]

        # Otherwise, bind across the P dimension
        return torchhd.multibind(to_bind)  # [..., N, D]

    def forward(
        self,
        data: Data | Batch,
        *,
        bidirectional: bool = False,  # PyG datasets come already bidirectional
        normalize: bool = False,
        separate_levels: bool = True,
    ) -> dict:
        """
        Performs a forward pass on the given PyG ``data`` object which represents a batch of graphs. Primarily
        this method will encode the graphs into high-dimensional graph embedding vectors.

        :param separate_levels:
        :param normalize:
        :param bidirectional: a flag indicating whether to use bidirectional encoding or not.
        :param data: The PyG Data object that represents the batch of graphs.

        :returns: A dict with string keys and torch Tensor values. The "graph_embedding" key should contain the
            high-dimensional graph embedding vectors for the input graphs with shape (batch_size, hv_dim)
        """

        # ~ mapping node & graph properties as hyper-vectors
        # The "encoder_properties" method will actually manage the encoding of the node and graph properties of
        # the graph (as represented by the Data object) into representative
        # Afterwards, the data object contains the additional properties "data.node_hv"
        # which represent the encoded hyper-vectors for the individual nodes
        data = self.encode_properties(data)

        # ~ handling edge bi-directionality
        # If the bidirectional flag is given we will duplicate each edge in the input graphs and reverse the
        # order of node indices such that each node of each edge is always considered as a source and a target
        # for the message passing operation.

        edge_index = torch.cat([data.edge_index, data.edge_index[[1, 0]]], dim=1) if bidirectional else data.edge_index
        srcs, dsts = edge_index

        # In this data structure we will stack all the intermediate node embeddings for the various message-passing
        # depths.
        # node_hv_stack: (num_layers + 1, batch_size * num_nodes, hv_dim)
        node_dim = data.x.size(0)
        node_hv_stack = data.node_hv.new_zeros(size=(self.depth + 1, node_dim, self.hv_dim))
        node_hv_stack[0] = data.node_hv  # Level 0 HV: Nodes are binding of their features

        # ~ message passing
        edge_terms = None
        node_terms = None
        for layer_index in range(self.depth):
            messages = node_hv_stack[layer_index][dsts]

            # aggregate (bundle) neighbor messages back into each node slot
            aggregated = scatter_hd(messages, srcs, dim_size=node_dim, op="bundle")

            prev_hv = node_hv_stack[layer_index].clone()

            # Level's HV
            hr = torchhd.bind(prev_hv, aggregated)  # [node_dim, D]

            # capture “edge terms” once at the very first layer
            if layer_index == 0:
                edge_terms = hr.clone()

            if normalize or self.normalize:
                # compute L2 norm along the last dimension (out‐of‐place)
                hr_norm = hr.norm(dim=-1, keepdim=True)  # [node_dim, 1]

                # divide by norm, also out‐of‐place
                node_hv_stack[layer_index + 1] = hr / (hr_norm + 1e-8)  # [node_dim, D]
            else:
                node_hv_stack[layer_index + 1] = hr

        # We calculate the final graph-level embedding as the sum of all the node embeddings over all the various (k-hop neighbourhood)
        # message passing depths and as the sum over all the nodes.
        node_hv_stack = node_hv_stack.transpose(0, 1)
        node_hv = torchhd.multibundle(node_hv_stack)  # This is bundle - [N, P, D] -> [N, D]
        readout = scatter_hd(src=node_hv, index=data.batch, op="bundle")
        embedding = readout

        if separate_levels:
            ## Prepare Level 0 Embeddings: Only node terms
            node_terms = scatter_hd(src=data.node_hv, index=data.batch, op="bundle")

            ## Prepare level 1 Embeddings: Only edge terms (not bounded with level 0)
            edge_terms = scatter_hd(src=edge_terms, index=data.batch, op="bundle")

        return {
            # This the main result of the forward pass which is the individual graph embedding vectors of the
            # input graphs.
            # graph_embedding: (batch_size, hv_dim)
            "graph_embedding": embedding,
            "node_terms": node_terms,
            "edge_terms": edge_terms,
        }

    # -- decoding
    # These methods handle the inverse operation -> The decoding of the graph embedding vectors back into
    # the original graph structure.

    def decode_order_zero(self, embedding: torch.Tensor) -> list[dict]:
        """
        Decodes the types and counts of nodes (order-zero information) from the given graph embedding vector.

        **Node Decoding**

        The aim of this method is to reconstruct the information about what kinds of nodes existed in the original
        graph based on the given graph embedding vector ``embedding``. The way in which this works is that for
        every possible combination of node properties we know the corresponding base hypervector encoding which
        is stored in the self.node_hv_combinations data structure. Multiplying each of these node hypervectors
        with the final graph embedding is essentially a projection along that node type's dimension. The magnitude
        of this projection should be proportional to the number of times that node type was present in the original
        graph.

        Therefore, we iterate over all the possible node property combinations and calculate the projection of the
        graph embedding along the direction of the node hypervector. If the magnitude of this projection is non-zero
        we can assume that this node type was present in the original graph and we derive the number of times it was
        present from the magnitude of the projection.

        :returns: A list of constraints where each constraint is represented by a dictionary with the keys:
            - src: A dictionary that represents the properties of the node as they were originally encoded
              by the node encoders. The keys in this dict are the same as the names of the node encoders
              given to the constructor.
            - num: The integer number of how many of these nodes are present in the graph.
        """

        self.populate_nodes_codebooks()
        self._populate_nodes_indexer()

        # Compute the dot product between the embedding and each node hypervector representation.
        d = torchhd.dot(embedding, self.nodes_codebook)
        if self.vsa in {VSAModel.FHRR, VSAModel.MAP}:
            # For FHRR and MAP VSA models, the dot product scales with the dimensionality.
            # Normalize by dividing by the hypervector dimension to approximate the original counts.
            d = d / self.hv_dim

        # Round the dimension normalized dot products to obtain integer counts.
        # [Batch, N]: where N is the number of possible combinatorial nodes.
        # i.e. A Graph with two categorical node features of size F1 and F2 has F1xF2 possible nodes
        return torch.round(d).int().clamp(min=0)

    def decode_order_zero_counter(self, embedding: torch.Tensor) -> dict[int, Counter]:
        dot_products_rounded = self.decode_order_zero(embedding)

        # Counts (feature1, feature2, ...) : #Occurrences
        return self.convert_to_counter(similarities=dot_products_rounded, indexer=self.nodes_indexer)

    @staticmethod
    def convert_to_counter(similarities: torch.Tensor, indexer: TupleIndexer) -> dict[int, Counter]:
        """
        Given a tensor `similarities` of shape [batch_size, num_tuples], where each entry
        is an integer count (or suitably rounded score) ≥ 0, return a dict mapping each
        batch index b to a Counter whose keys are the node‐tuples and whose values are
        the counts.

        This version uses defaultdict(Counter) and a single call to nonzero() to avoid
        looping over every row manually.
        """

        # `sim_nonzero` returns two 1-D LongTensors:
        #   - b_indices: indices of the batch dimension
        #   - t_indices: indices of the tuple dimension
        # such that similarities[b_indices[i], t_indices[i]] > 0.
        # if it’s a single graph, make it into a batch of size 1
        if similarities.dim() == 1:
            similarities = similarities.unsqueeze(0)  # now [1, num_tuples]
        b_indices, t_indices = torch.nonzero(similarities, as_tuple=True)  # both have shape [total_nonzero_entries]

        # Create a defaultdict so that counters[b] auto‐initializes to Counter() the first time we see b
        counters: dict[int, Counter] = defaultdict(Counter)

        # Iterate over each (b, t) where similarities[b, t] > 0
        for b, t in zip(b_indices.tolist(), t_indices.tolist(), strict=False):
            tup = indexer.get_tuple(t)  # tuple[int, ...]
            count = int(similarities[b, t].item())  # the nonzero count
            counters[b][tup] = count

        return counters

    def _populate_nodes_indexer(self) -> None:
        # Create an indexer to map between combinations and their corresponding indices.
        # Examples:
        # A graph with three categorical node features of size F1, F2, F3 would have a
        # tuple index of: ([0, F1-1], [0, F2-1], [0, F3-1])
        if not self.nodes_indexer:
            ## Edge case for having CombinatoricEncoder - Change impl later.
            if isinstance(enc := self.node_encoder_map[Features.NODE_FEATURES][0], CombinatoricIntegerEncoder):
                self.nodes_indexer = enc.indexer
            else:
                self.nodes_indexer = TupleIndexer([e.num_categories for e, _ in self.node_encoder_map.values()])

    def populate_nodes_codebooks(self, limit_node_set: set[tuple[int, ...]] | None = None) -> None:
        if self.is_not_initialized(tensor=self.nodes_codebook):
            # Retrieve the node encoders from the node_encoder_map.
            # Each node encoder is responsible for encoding on property of the node
            node_encoders = [enc for enc, _ in self.node_encoder_map.values()]

            # Extract the codebooks from each node encoder (codebook of each feature).
            cbs = [e.codebook for e in node_encoders]

            # Generate all possible combinations of node hypervectors using a Cartesian product.
            # The binding of encoded properties of a node represents a node
            # HVnode = bind(HV(feature_1), ..., HV(feature_n))
            # [N, D]
            self.nodes_codebook = cartesian_bind_tensor(cbs)

    def limit_nodes_codebook(self) -> None:
        self.populate_nodes_codebooks()
        self._populate_nodes_indexer()

        # Limit the codebook only to a subset of the nodes.
        sorted_limit_node_set = sorted(self.dataset_info.node_features)
        idxs = self.nodes_indexer.get_idxs(sorted_limit_node_set)
        self.nodes_indexer.idx_to_tuple = sorted_limit_node_set
        self.nodes_indexer.tuple_to_idx = {tup: idx for idx, tup in enumerate(sorted_limit_node_set)}
        self.nodes_codebook = self.nodes_codebook[idxs].as_subclass(type(self.nodes_codebook))

        for _, (enc, _) in self.node_encoder_map.items():
            enc.codebook = self.nodes_codebook
            enc.indexer = self.nodes_indexer

    @staticmethod
    def is_not_initialized(tensor: torch.Tensor | None) -> bool:
        return tensor is None or tensor.numel() == 0

    def decode_order_one(
        self,
        edge_term: torch.Tensor,
        node_counter: Counter[tuple[int, ...]],
        debug: bool = False,
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """
        Returns information about the kind and number of edges (order one information) that were contained in the
        original graph represented by the given ``embedding`` vector.

        **Edge Decoding**

        The aim of this method is to reconstruct the first order information about what kinds of edges existed in
        the original graph based on the given graph embedding vector ``embedding``. The way in which this works is
        that we already get the zero oder constraints (==informations about which nodes are present) passed as an
        argument. Based on that we construct all possible combinations of node pairs (==edges) and calculate the
        corresponding binding of the hypervector representations. Then we can multiply each of these edge hypervectors
        with the final graph embedding to get a projection along that edge type's dimension. The magnitude of this
        projection should be proportional to the number of times that edge type was present in the original graph
        (except for a correction factor).

        Therefore, we iterate over all the possible node pairs and calculate the projection of the graph embedding
        along the direction of the edge hypervector. If the magnitude of this projection is non-zero we can assume
        that this edge type was present in the original graph and we derive the number of times it was present from
        the magnitude of the projection.

        :param edge_term: Graph representation with HDC message passing depth 1.
        :param node_tuples: The list of constraints that represent the zero order information about the
            nodes that were present in the original graph.


        :returns: A list of edges represented as tuples of (u, v) where u and v are node tuples
        """
        all_edges = list(itertools.product(node_counter.keys(), node_counter.keys()))
        num_edges = sum([(k[1] + 1) * n for k, n in node_counter.items()])
        edge_count = num_edges

        # Get all indices at once
        node_tuples_a, node_tuples_b = zip(*all_edges, strict=False)
        idx_a = torch.tensor(self.nodes_indexer.get_idxs(node_tuples_a), dtype=torch.long, device=self.device)
        idx_b = torch.tensor(self.nodes_indexer.get_idxs(node_tuples_b), dtype=torch.long, device=self.device)

        # Gather all node hypervectors at once: [N*N, D]
        hd_a = self.nodes_codebook[idx_a]  # [N*N, D]
        hd_b = self.nodes_codebook[idx_b]  # [N*N, D]

        # Vectorized bind operation
        edges_hdc = hd_a.bind(hd_b)

        norms = []
        similarities = []
        decoded_edges: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for i in range(int(edge_count // 2)):
            norms.append(edge_term.norm().item())
            sims = torchhd.cos(edge_term, edges_hdc)
            idx_max = torch.argmax(sims).item()
            similarities.append(sims[idx_max].item())
            a_found, b_found = all_edges[idx_max]
            if not a_found or not b_found:
                break
            hd_a_found: VSATensor = self.nodes_codebook[self.nodes_indexer.get_idx(a_found)]
            hd_b_found: VSATensor = self.nodes_codebook[self.nodes_indexer.get_idx(b_found)]
            edge_term -= hd_a_found.bind(hd_b_found)
            edge_term -= hd_b_found.bind(hd_a_found)

            decoded_edges.append((a_found, b_found))
            decoded_edges.append((b_found, a_found))

        if debug:
            return decoded_edges, norms, similarities
        return decoded_edges

    def decode_order_one_no_node_terms(
        self, edge_term: torch.Tensor, debug: bool = False
    ) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """
        extracts the list of edge tuples from the given ``edge_term`` tensor.

        :param edge_term: Graph representation with HDC message passing depth 1.
        :returns: A list of edges represented as tuples of (u, v) where u and v are node tuples
        :rtype: list[tuple[tuple[int, ...], tuple[int, ...]]]]
        """
        self.populate_edges_codebook()
        self._populate_edges_indexer()

        # Determine the lowest energy (norm) required to extract an edge
        if not self._max_step_delta:
            norms = []
            for hd_a in self.nodes_codebook:
                for hd_b in self.nodes_codebook:
                    delta = hd_a.bind(hd_b) + hd_b.bind(hd_a)
                    norms.append(delta.norm().item())

            min_norm = min(norms)
            # print(f"[delta-norm] min={min_norm:.6f}")

            self._max_step_delta = min_norm

        eps = self._max_step_delta * 0.01  # small relative tolerance

        decoded_edges: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        # while not target_reached(decoded_edges):
        norms = []
        prev_edge_term = edge_term.clone()
        similarities = []
        while (
            # not target_reached(decoded_edges)
            # essentially by getting the norm of het edge_term we know how many edges we have
            # roughly for each ||edge_term|| // 2 tells us how many edges since the hv resulted as binding of
            # two nodes has norm ~1.0. We could prune earlier, but we might reach a target and be able to decode
            # something, so we keep going.
            # edge_term.norm().item() > eps
            # this will almost certainly cause an invalid setup, should be caught by the caller
            len(decoded_edges) <= self.decoding_limit_for
        ):
            curr_norm = edge_term.norm().item()
            norms.append(curr_norm)
            if curr_norm <= eps:
                break
            sims = torchhd.cos(edge_term, self.edges_codebook)
            idx_max = torch.argmax(sims).item()
            similarities.append(sims[idx_max].item())
            a_found, b_found = self.edges_indexer.get_tuple(idx_max)
            hd_a_found: VSATensor = self.nodes_codebook[a_found]
            hd_b_found: VSATensor = self.nodes_codebook[b_found]
            edge_term -= hd_a_found.bind(hd_b_found)
            edge_term -= hd_b_found.bind(hd_a_found)
            if edge_term.norm().item() > prev_edge_term.norm().item():
                if not target_reached(decoded_edges) and target_reached(
                    [
                        *decoded_edges,
                        (self.nodes_indexer.get_tuple(a_found), self.nodes_indexer.get_tuple(b_found)),
                        (self.nodes_indexer.get_tuple(b_found), self.nodes_indexer.get_tuple(a_found)),
                    ]
                ):
                    decoded_edges.append((self.nodes_indexer.get_tuple(a_found), self.nodes_indexer.get_tuple(b_found)))
                    decoded_edges.append((self.nodes_indexer.get_tuple(b_found), self.nodes_indexer.get_tuple(a_found)))
                break
            prev_edge_term = edge_term.clone()

            decoded_edges.append((self.nodes_indexer.get_tuple(a_found), self.nodes_indexer.get_tuple(b_found)))
            decoded_edges.append((self.nodes_indexer.get_tuple(b_found), self.nodes_indexer.get_tuple(a_found)))

        if debug:
            return decoded_edges, norms, similarities
        return decoded_edges

    def _populate_edge_feature_indexer(self) -> None:
        # Create an indexer to map between combinations and their corresponding indices.
        # Examples:
        # A graph with three categorical node features of size F1, F2, F3 would have a
        # tuple index of: ([0, F1-1], [0, F2-1], [0, F3-1])

        self.edge_feature_indexer = TupleIndexer([e.num_categories for e, _ in self.edge_encoder_map.values()])

    def populate_edge_feature_codebook(self) -> None:
        if self.is_not_initialized(self.edge_feature_codebook) and self.use_edge_features():
            # Retrieve the node encoders from the edge_encoder_map.
            # Each node encoder is responsible for encoding on property of the node
            edge_encoders = [enc for enc, _ in self.edge_encoder_map.values()]

            # Extract the codebooks from each edge encoder (codebook of each feature).
            edge_codebooks = [e.codebook for e in edge_encoders]

            # The binding of encoded properties of a node represents an edge
            # HV_edge = bind(HV(feature_1), ..., HV(feature_n))
            # [ |E|, D ]
            self.edge_feature_codebook = cartesian_bind_tensor(edge_codebooks)

    def _populate_edges_indexer(self) -> None:
        # Create an indexer to map between combinations and their corresponding indices.
        # Examples:
        # A graph with three categorical node features of size F1, F2, F3 and one edge feature of size EF1  would have a
        # tuple index of: ([0, (F1*F2*F3)], [0, EF1-1])
        self._populate_nodes_indexer()
        self._populate_edge_feature_indexer()
        if not self.edges_indexer:
            self.edges_indexer = TupleIndexer(
                [self.nodes_indexer.size(), self.nodes_indexer.size(), self.edge_feature_indexer.size()]
            )

    def populate_edges_codebook(self) -> None:
        # The binding of encoded properties of a nodes of an with with the edges HV represents an edge
        # HV_edge = bind(HV(src_node), HV(dst_node), HV(edge_feature))
        # [N*N*E, D]
        # For example in QM9 there is one categorical node feature of size 28, and one categorical edge feature of
        # size 4. That would make the shape of the edges_codebook [28*28*4, D]
        if self.is_not_initialized(self.edges_codebook):
            self.edges_codebook = cartesian_bind_tensor(
                [self.nodes_codebook, self.nodes_codebook, self.edge_feature_codebook]
            ).to(device=self.device, dtype=self.dtype)

        # -- saving and loading
        # methods that handle the storage of the HyperNet instance to and from a file.

    def save_to_path(self, path: str | Path) -> None:
        """Serialize the HyperNet to *path* in a pickle-safe, version-tolerant format."""

        def serialize_encoder_map(encoder_map):
            result: dict[str, Any] = {}
            for feat, (encoder, idx_range) in encoder_map.items():
                entry = {
                    "encoder_class": encoder.__class__.__name__,
                    "init_args": {
                        "dim": encoder.dim,
                        "vsa": encoder.vsa,  # ← store string
                        "device": str(encoder.device),
                        "seed": self.seed,
                        "num_categories": getattr(encoder, "num_categories", None),
                        "idx_offset": getattr(encoder, "idx_offset", 0),
                    },
                    "index_range": idx_range,
                    "codebook": encoder.codebook.cpu(),  # always CPU
                }
                # Special case: CombinatoricIntegerEncoder carries an indexer
                if hasattr(encoder, "indexer"):
                    entry["indexer_state"] = encoder.indexer.__dict__.copy()
                result[feat.name] = entry  # ← just the member name
            return result

        def serialize_codebook(codebook):
            return None if self.is_not_initialized(codebook) else codebook.cpu()

        state = {
            "version": 1,
            "attributes": {
                "depth": self.depth,
                "seed": self.seed,
                "vsa": self.vsa.value,
                "hv_dim": self.hv_dim,
                "use_explain_away": self.use_explain_away,
                "use_edge_codebook": self.use_edge_codebook,
                "device": str(self._cfg_device),  # for convenience
            },
            "node_encoder_map": serialize_encoder_map(self.node_encoder_map),
            "edge_encoder_map": serialize_encoder_map(self.edge_encoder_map),
            "graph_encoder_map": serialize_encoder_map(self.graph_encoder_map),
            # --- codebooks ---
            "nodes_codebook": serialize_codebook(self.nodes_codebook),
            "edge_feature_codebook": serialize_codebook(self.edge_feature_codebook),
            "edges_codebook": serialize_codebook(self.edges_codebook),
        }
        torch.save(state, path)

    def load_from_path(self, path: str | Path) -> None:
        """Inverse of *save_to_path* – tolerates both old and new checkpoints."""

        encoder_class_map = {
            "CategoricalOneHotEncoder": CategoricalOneHotEncoder,
            "CategoricalIntegerEncoder": CategoricalIntegerEncoder,
            "CombinatoricIntegerEncoder": CombinatoricIntegerEncoder,
            "TrueFalseEncoder": TrueFalseEncoder,
            "CategoricalLevelEncoder": CategoricalLevelEncoder,
        }

        def _ensure_enum(x: str | VSAModel) -> VSAModel:
            """Return *x* as VSAModel (idempotent)."""
            return x if isinstance(x, VSAModel) else VSAModel(x)

        def _feat_key_to_enum(key: str) -> Features:
            """
            Accept either `'ATOM_TYPE'` (new) or `'Features.ATOM_TYPE'` (legacy)
            and return the corresponding enum member.
            """
            if key.startswith("Features."):
                key = key.split(".", 1)[1]
            key = "NODE_FEATURES" if key == "ATOM_TYPE" else key
            return Features[key]

        def deserialize_encoder_map(
            serialized: dict[str, Any],
            override_device: str | torch.device | None = None,
            move_codebook: bool = True,
        ):
            result = {}
            for feat_key, entry in serialized.items():
                args = dict(entry["init_args"])  # copy; don't mutate original
                # normalize VSA if it was serialized as a string
                if isinstance(args.get("vsa"), str):
                    args["vsa"] = args["vsa"]

                # --- decide target device BEFORE calling the encoder ctor ---
                if override_device is not None:
                    target = torch.device(override_device)
                else:
                    saved = str(args.get("device", "cpu"))
                    if saved.startswith("mps") and not torch.backends.mps.is_available():
                        target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    elif saved.startswith("cuda") and not torch.cuda.is_available():
                        target = torch.device("cpu")
                    else:
                        target = torch.device(saved)
                args["device"] = str(target)

                enc_cls = encoder_class_map[entry["encoder_class"]]

                # rebuild optional TupleIndexer
                if "indexer_state" in entry:
                    indexer = TupleIndexer.__new__(TupleIndexer)
                    indexer.__dict__.update(entry["indexer_state"])
                    encoder = enc_cls(**args, indexer=indexer)
                else:
                    encoder = enc_cls(**args)

                # codebook was saved as CPU tensors; move if desired
                codebook = entry["codebook"]
                if move_codebook and target.type != "cpu":
                    codebook = codebook.to(target, non_blocking=True)

                encoder.codebook = codebook
                encoder.device = target

                result[_feat_key_to_enum(feat_key)] = (encoder, entry["index_range"])
            return result

        # With weights_only=True the “safe” un-pickler only accepts tensors, primitive types and whatever you
        # explicitly allow-list. We set it to False to allow unsafe un-pickler.
        state = torch.load(path, map_location="cpu", weights_only=False)

        # attributes (including flags like use_edge_codebook/use_explain_away if saved)
        attrs = state["attributes"]
        for k, v in attrs.items():
            if k == "vsa":
                setattr(self, k, _ensure_enum(v))
            elif k == "device":
                pass
            else:
                setattr(self, k, v)

        # encoder maps (encoders already carry their codebooks on CPU)
        self.node_encoder_map = deserialize_encoder_map(
            state["node_encoder_map"], override_device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.edge_encoder_map = deserialize_encoder_map(
            state["edge_encoder_map"], override_device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.graph_encoder_map = deserialize_encoder_map(
            state["graph_encoder_map"], override_device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # restore codebooks (may be None if checkpoint created before codebooks existed)
        self.nodes_codebook = state.get("nodes_codebook", None)
        self.edge_feature_codebook = state.get("edge_feature_codebook", None)
        self.edges_codebook = state.get("edges_codebook", None)

        # (re)build indexers from the encoder maps (sizes are inferable)
        self._populate_nodes_indexer()
        if self.use_edge_features():
            self._populate_edge_feature_indexer()
        if getattr(self, "use_edge_codebook", False):
            self._populate_edges_indexer()

        # move everything to target device (prefer saved device in attributes; fallback CPU)
        override_device = "cuda" if torch.cuda.is_available() else "cpu"

        def _to_dev(x):
            return None if x is None else x.to(override_device)

        self.nodes_codebook = _to_dev(self.nodes_codebook)
        self.edge_feature_codebook = _to_dev(self.edge_feature_codebook)
        self.edges_codebook = _to_dev(self.edges_codebook)

        for emap in (self.node_encoder_map, self.edge_encoder_map, self.graph_encoder_map):
            for enc, _ in emap.values():
                enc.device = override_device
                if getattr(enc, "codebook", None) is not None:
                    enc.codebook = enc.codebook.to(override_device)

    @classmethod
    def load(cls, path: str | Path) -> "HyperNet":
        """
        Given the absolute string ``path`` to an existing file, this will load the saved state that
        has been saved using the "save_to_path" method. This will overwrite the values of the
        current object instance.

        :param path: The absolute path to the file where a HyperNet instance has previously been
            saved to.

        :returns: A new instance of the HyperNet class with the loaded state.
        """
        instance = cls.__new__(cls)
        pl.LightningModule.__init__(instance)
        instance._init_lazy_fields()
        instance.load_from_path(path=path)
        return instance

    def encode_edge_multiset(self, edge_list: list[tuple[tuple, tuple]]) -> torch.Tensor:
        """Encode a multiset of edges into a single hypervector (vectorized for 10-100x speedup).
        This is used during greedy decoding with use_modified_graph_embedding to encode
        the leftover edges (edges not yet added to a partial graph) into a hypervector
        that can be subtracted from the target graph_term.

        Parameters
        ----------
        edge_list : list[tuple[tuple, tuple]]
            List of (src_node_tuple, dst_node_tuple) edges, where each node tuple
            contains the node features (atom_type, degree, formal_charge, ...).
            Edges are bidirectional, so typically each edge appears twice.

        Returns
        -------
        torch.Tensor
            Hypervector representation of the edge multiset (shape: [hv_dim]).
            Returns zero vector if edge_list is empty. Returns VSATensor type.
        """
        if not edge_list:
            return torch.zeros(self.hv_dim, device=self.device, dtype=self.dtype)

        # OPTIMIZATION: Use pre-computed edges_codebook if available (fastest)
        if self.use_edge_codebook and self.edges_codebook is not None:
            edge_indices = []
            for src_tuple, dst_tuple in edge_list:
                src_idx = self.nodes_indexer.get_idx(src_tuple)
                dst_idx = self.nodes_indexer.get_idx(dst_tuple)
                # Edge feature index is 0 (no edge features in current datasets)
                edge_idx = self.edges_indexer.get_idx((src_idx, dst_idx, 0))
                edge_indices.append(edge_idx)

            # Batch index into edges_codebook (preserves VSATensor type)
            edge_indices_tensor = torch.tensor(edge_indices, dtype=torch.long, device=self.device)
            edge_hvs = self.edges_codebook[edge_indices_tensor]  # [num_edges, D] VSATensor

            # Sum all edges at once (bundle operation, preserves VSATensor type)
            edge_term = edge_hvs.sum(dim=0)

            return edge_term

        # FALLBACK: Vectorized computation without edges_codebook
        # Batch convert tuples to indices
        src_indices = []
        dst_indices = []
        for src_tuple, dst_tuple in edge_list:
            src_indices.append(self.nodes_indexer.get_idx(src_tuple))
            dst_indices.append(self.nodes_indexer.get_idx(dst_tuple))

        src_indices_tensor = torch.tensor(src_indices, dtype=torch.long, device=self.device)
        dst_indices_tensor = torch.tensor(dst_indices, dtype=torch.long, device=self.device)

        # Batch index nodes_codebook (preserves VSATensor type)
        hv_src = self.nodes_codebook[src_indices_tensor]  # [num_edges, D] VSATensor
        hv_dst = self.nodes_codebook[dst_indices_tensor]  # [num_edges, D] VSATensor

        # Vectorized bind (VSATensor.bind() handles batch operations)
        edge_hvs = hv_src.bind(hv_dst)  # [num_edges, D] VSATensor

        # Sum all edges at once (preserves VSATensor type)
        edge_term = edge_hvs.sum(dim=0)

        return edge_term

    def decode_graph_greedy(
        self,
        edge_term: torch.Tensor,
        graph_term: torch.Tensor,
        node_counter: Counter | None = None,
        decoder_settings: dict | None = None,
    ) -> DecodingResult:
        """
        Greedy beam search decoder for graph reconstruction.

        This is a fallback decoder used when the main decoding strategy fails. It performs
        beam search in the graph construction space, incrementally adding nodes while
        maintaining the highest-scoring partial graphs according to cosine similarity
        with the target graph_term.

        Key improvements:
        - Returns top_k graphs sorted by cosine similarity (consistent with main decoder)
        - Final ranking ensures best candidates are returned first

        Parameters
        ----------
        edge_term : torch.Tensor
            Hypervector representing the edge structure.
        graph_term : torch.Tensor
            Hypervector representing the full graph for similarity comparison.
        node_counter : Counter, optional
            Pre-computed node count information (for 2D/3D vectors with G0 encoded).
        decoder_settings : dict, optional
            Configuration parameters:
            - top_k: int (default: 10)
              Number of top-scoring graphs to return.
            - beam_size: int
              Beam width during search.
            - initial_limit: int (default: 1024)
              Population size limit.
            - pruning_fn: str (default: "cos_sim")
              Similarity function for pruning.
            - use_g3_instead_of_h3: bool (default: False)
              Whether to use combined terms for similarity.

        Returns
        -------
        DecodingResult
            Object containing:
            - nx_graphs: Top-k NetworkX graphs sorted by similarity (descending)
            - final_flags: Completion status for each graph
            - target_reached: Whether valid graph was decoded
            - correction_level: Always CorrectionLevel.FAIL (greedy fallback)
        """
        # print("Using Greedy Decoder")
        if decoder_settings is None:
            decoder_settings = {}
        validate_ring_structure = decoder_settings.get("validate_ring_structure", False)
        random_sample_ratio = decoder_settings.get("random_sample_ratio", 0.0)
        use_modified_graph_embedding = decoder_settings.get("use_modified_graph_embedding", False)
        graph_embedding_attr = decoder_settings.get("graph_embedding_attr", "graph_embedding")

        # Case 2D/3D vectors with G0 encoded
        if node_counter:
            decoded_edges = self.decode_order_one(edge_term=edge_term, node_counter=node_counter)
            edge_count = sum([(e_idx + 1) * n for (_, e_idx, _, _), n in node_counter.items()])
        else:
            decoded_edges = self.decode_order_one_no_node_terms(edge_term=edge_term.clone())
            edge_count = len(decoded_edges) // 2  # bidirectional edges
            node_counter = get_node_counter_corrective(decoded_edges)
            if not target_reached(decoded_edges):
                decoded_edges = self.decode_order_one(edge_term=edge_term.clone(), node_counter=node_counter)

        node_limit = MAX_ALLOWED_DECODING_NODES_QM9 if self.base_dataset == "qm9" else MAX_ALLOWED_DECODING_NODES_ZINC
        if node_counter.total() > node_limit:
            # print(f"Skipping graph with {node_counter.total()} nodes for '{self.base_dataset}'")
            return DecodingResult(correction_level=CorrectionLevel.FAIL)

        node_count = node_counter.total()
        ## We have the multiset of nodes and the multiset of edges
        # OPTIMIZATION: Convert decoded_edges to Counter for O(1) lookups throughout
        decoded_edges_counter = Counter(decoded_edges)

        # OPTIMIZATION: Store Counter instead of list in population tuples for O(1) operations
        first_pop: list[tuple[nx.Graph, Counter]] = []
        global_seen: set = set()
        for k, (u_t, v_t) in enumerate(decoded_edges):
            G = nx.Graph()
            uid = add_node_with_feat(G, Feat.from_tuple(u_t))
            ok = add_node_and_connect(G, Feat.from_tuple(v_t), connect_to=[uid], total_nodes=node_count) is not None
            if not ok:
                continue
            key = _hash(G)
            if key in global_seen:
                continue
            global_seen.add(key)

            # OPTIMIZATION: Store Counter directly - no conversion to list needed
            remaining_edges_counter = decoded_edges_counter.copy()
            remaining_edges_counter[(u_t, v_t)] -= 1
            remaining_edges_counter[(v_t, u_t)] -= 1

            first_pop.append((G, remaining_edges_counter))

        pruning_fn = decoder_settings.get("pruning_fn", "cos_sim")

        def get_similarities(a, b):
            if pruning_fn != "cos_sim":
                diff = a[:, None, :] - b[None, :, :]
                return torch.sum(diff**2, dim=-1)
            return torchhd.cos(a, b)

        initial_limit = decoder_settings.get("initial_limit", 1024)
        use_size_aware_pruning = decoder_settings.get("use_size_aware_pruning", False)
        if decoder_settings.get("use_one_initial_population", False):
            # Start with a child both anchors free
            selected = [(G, l) for G, l in first_pop if len(anchors(G)) == 2]
            first_pop = selected[:1] if len(selected) >= 1 else first_pop[:1]
        population = first_pop

        for _ in tqdm(range(2, node_count)):
            children: list[tuple[nx.Graph, Counter]] = []

            # Expand the current population
            for gi, (G, edges_left) in enumerate(population):
                leftovers_ctr = leftover_features(node_counter, G)
                if not leftovers_ctr:
                    continue

                leftover_types = order_leftovers_by_degree_distinct(leftovers_ctr)
                ancrs = anchors(G)
                if not ancrs:
                    continue

                # Choose the first N anchors to expand on
                lowest_degree_ancrs = sorted(ancrs, key=lambda n: residual_degree(G, n))[:1]

                # Try to connect the left over nodes to the lowest degree anchors
                for a, lo_t in list(itertools.product(lowest_degree_ancrs, leftover_types)):
                    a_t = G.nodes[a]["feat"].to_tuple()
                    # OPTIMIZATION: Use Counter for O(1) lookup (no set conversion needed)
                    if edges_left[(a_t, lo_t)] == 0:
                        continue

                    # OPTIMIZATION: Use nx.Graph(G) instead of G.copy() for faster copying
                    C = nx.Graph(G)
                    nid = add_node_and_connect(C, Feat.from_tuple(lo_t), connect_to=[a], total_nodes=node_count)
                    if nid is None:
                        continue
                    if C.number_of_edges() > edge_count:
                        continue

                    keyC = _hash(C)
                    if keyC in global_seen:
                        continue

                    # # Early pruning of bad ring structures
                    if (
                        validate_ring_structure
                        and self.base_dataset == "zinc"
                        and not has_valid_ring_structure(
                            G=C,
                            processed_histogram=self.dataset_info.ring_histogram,
                            single_ring_atom_types=self.dataset_info.single_ring_features,
                            is_partial=True,  # Graph is still being constructed
                        )
                    ):
                        continue

                    # self._print_and_plot(g=C, graph_terms=graph_term)

                    # OPTIMIZATION: Use Counter arithmetic for O(1) edge removal
                    remaining_edges = edges_left.copy()
                    remaining_edges[(a_t, lo_t)] -= 1
                    remaining_edges[(lo_t, a_t)] -= 1
                    global_seen.add(keyC)
                    children.append((C, remaining_edges))

                    ancrs_rest = [a_ for a_ in ancrs if a_ != a]

                    # OPTIMIZATION: remaining_edges is already a Counter, use it directly
                    nid_t = C.nodes[nid]["feat"].to_tuple()

                    for subset in powerset(ancrs_rest):
                        if len(subset) == 0:
                            continue

                        # OPTIMIZATION: Build all_new_connection and validate using Counter
                        all_new_connection = []
                        subset_ts = [C.nodes[s]["feat"].to_tuple() for s in subset]
                        should_continue = False
                        for st in subset_ts:
                            ts = (nid_t, st)
                            # Check if edge exists in remaining_edges Counter (O(1))
                            if remaining_edges[ts] == 0:
                                should_continue = True
                                break
                            all_new_connection.append(ts)

                        if should_continue:
                            continue

                        # OPTIMIZATION: Validate edge counts using Counter
                        all_new_counter = Counter(all_new_connection)
                        # if both ends of an edge is the same tuple, it should be considered twice
                        for k, v in all_new_counter.items():
                            if k[0] == k[1]:
                                all_new_counter[k] = 2 * v

                        # Check if we have enough edges in remaining_edges
                        for k, v in all_new_counter.items():
                            if remaining_edges[k] < v:
                                should_continue = True
                                break

                        if should_continue:
                            continue

                        # OPTIMIZATION: Use nx.Graph(C) instead of C.copy()
                        H = nx.Graph(C)
                        new_nid = connect_all_if_possible(H, nid, connect_to=list(subset), total_nodes=node_count)
                        if new_nid is None:
                            continue
                        if H.number_of_edges() > edge_count:
                            continue

                        keyH = _hash(H)
                        if keyH in global_seen:
                            continue

                        # # Early pruning of bad ring structures
                        if (
                            validate_ring_structure
                            and self.base_dataset == "zinc"
                            and not has_valid_ring_structure(
                                G=H,
                                processed_histogram=self.dataset_info.ring_histogram,
                                single_ring_atom_types=self.dataset_info.single_ring_features,
                                is_partial=True,  # Graph is still being constructed
                            )
                        ):
                            continue

                        # OPTIMIZATION: Use Counter arithmetic for O(1) batch edge removal
                        remaining_edges_ = remaining_edges.copy()
                        for a_t, b_t in all_new_connection:
                            remaining_edges_[(a_t, b_t)] -= 1
                            remaining_edges_[(b_t, a_t)] -= 1

                        # self._print_and_plot(g=H, graph_terms=graph_term)

                        global_seen.add(keyH)
                        children.append((H, remaining_edges_))

            ## Collect the children with highest number of edges
            if not children:
                # Extract top_k parameter from decoder_settings
                top_k = decoder_settings.get("top_k", 10)

                graphs, edges_left = zip(*population, strict=True)
                # OPTIMIZATION: Use Counter.total() to check if empty
                are_final = [i.total() == 0 for i in edges_left]

                # Compute cosine similarities for all graphs
                batch = Batch.from_data_list([DataTransformer.nx_to_pyg(g) for g in graphs])
                enc_out = self.forward(batch)
                g_terms = enc_out[graph_embedding_attr]
                if decoder_settings.get("use_g3_instead_of_h3", False):
                    g_terms = enc_out["node_terms"] + enc_out["edge_terms"] + g_terms

                # Compute similarities and sort
                sims = torchhd.cos(graph_term, g_terms)
                sim_order = torch.argsort(sims, descending=True)

                # Select top_k graphs based on similarity
                top_k_indices = sim_order[: min(top_k, len(graphs))].cpu().numpy()
                top_k_graphs = [graphs[i] for i in top_k_indices]
                top_k_flags = [are_final[i] for i in top_k_indices]
                top_cos_sims = [sims[i].item() for i in top_k_indices]

                # Convert Counter to list for target_reached function
                decoded_edges_list = []
                for edge, count in decoded_edges_counter.items():
                    if count > 0:
                        decoded_edges_list.extend([edge] * count)

                return DecodingResult(
                    nx_graphs=top_k_graphs,
                    final_flags=top_k_flags,
                    cos_similarities=top_cos_sims,
                    target_reached=target_reached(decoded_edges_list),
                    correction_level=CorrectionLevel.FAIL,
                )

            if len(children) > initial_limit:
                initial_limit = decoder_settings.get("limit", initial_limit)
                beam_size = decoder_settings.get("beam_size")

                # Calculate split: top-k by similarity + random from rest
                keep = int((1 - random_sample_ratio) * beam_size)
                random_pick = int(random_sample_ratio * beam_size)

                if use_size_aware_pruning:
                    repo = defaultdict(list)

                    # Prune for each size separately
                    for c, l in children:
                        repo[c.number_of_edges()].append((c, l))

                    res = []
                    for ch in [v for _, v in repo.items()]:
                        # Encode and compute similaity
                        batch = Batch.from_data_list([DataTransformer.nx_to_pyg(c) for c, _ in ch])
                        enc_out = self.forward(batch)
                        g_terms = enc_out[graph_embedding_attr]
                        if decoder_settings.get("use_g3_instead_of_h3", False):
                            g_terms = enc_out["node_terms"] + enc_out["edge_terms"] + g_terms

                        if use_modified_graph_embedding:
                            # Modify graph_term for each child based on its leftover edges
                            # This provides fairer comparison by accounting for edges not yet added
                            sims_list = []

                            for idx, (c, leftover_edges_counter) in enumerate(ch):
                                if leftover_edges_counter and leftover_edges_counter.total() > 0:
                                    # Convert Counter to list for encode_edge_multiset
                                    leftover_edges = []
                                    for edge, count in leftover_edges_counter.items():
                                        if count > 0:
                                            leftover_edges.extend([edge] * count)
                                    # Encode the leftover edges into a hypervector
                                    leftover_edge_term = self.encode_edge_multiset(leftover_edges)
                                    # Subtract from target to get modified graph_term
                                    modified_graph_term = graph_term - leftover_edge_term
                                else:
                                    # No leftover edges, use original graph_term
                                    modified_graph_term = graph_term

                                # Compute element-wise similarity for this child
                                child_sim = get_similarities(modified_graph_term, g_terms[idx].unsqueeze(0))
                                sims_list.append(child_sim)

                            # Concatenate all similarities into a single tensor
                            sims = torch.cat(sims_list)
                        else:
                            # Original behavior: compare all to the same graph_term
                            sims = get_similarities(graph_term, g_terms)

                        # Sort by similarity first
                        sim_order = torch.argsort(sims, descending=True)

                        # Take top 'keep' by similarity
                        top_candidates = [ch[i.item()] for i in sim_order[:keep]]
                        res.extend(top_candidates)

                        # Randomly pick 'random_pick' from the rest
                        if random_pick > 0 and len(ch) > keep:
                            remaining_indices = sim_order[keep:].cpu().numpy()
                            if len(remaining_indices) > 0:
                                n_random = min(random_pick, len(remaining_indices))
                                random_indices = np.random.choice(remaining_indices, size=n_random, replace=False)
                                random_candidates = [ch[i] for i in random_indices]
                                res.extend(random_candidates)
                    children = res
                else:
                    # Encode and compute similaity
                    batch = Batch.from_data_list([DataTransformer.nx_to_pyg(c) for c, _ in children])
                    enc_out = self.forward(batch)
                    g_terms = enc_out[graph_embedding_attr]
                    if decoder_settings.get("use_g3_instead_of_h3", False):
                        g_terms = enc_out["node_terms"] + enc_out["edge_terms"] + g_terms

                    if use_modified_graph_embedding:
                        # Modify graph_term for each child based on its leftover edges
                        # This provides fairer comparison by accounting for edges not yet added
                        sims_list = []

                        for idx, (c, leftover_edges_counter) in enumerate(children):
                            if leftover_edges_counter and leftover_edges_counter.total() > 0:
                                # Convert Counter to list for encode_edge_multiset
                                leftover_edges = []
                                for edge, count in leftover_edges_counter.items():
                                    if count > 0:
                                        leftover_edges.extend([edge] * count)
                                # Encode the leftover edges into a hypervector
                                leftover_edge_term = self.encode_edge_multiset(leftover_edges)
                                # Subtract from target to get modified graph_term
                                modified_graph_term = graph_term - leftover_edge_term
                            else:
                                # No leftover edges, use original graph_term
                                modified_graph_term = graph_term

                            # Compute element-wise similarity for this child
                            child_sim = get_similarities(modified_graph_term, g_terms[idx].unsqueeze(0))
                            sims_list.append(child_sim)

                        # Concatenate all similarities into a single tensor
                        sims = torch.cat(sims_list)
                    else:
                        # Original behavior: compare all to the same graph_term
                        sims = get_similarities(graph_term, g_terms)

                    # Sort by similarity first
                    sim_order = torch.argsort(sims, descending=True)

                    # Take top 'keep' by similarity
                    top_candidates = [children[i.item()] for i in sim_order[:keep]]

                    # Randomly pick 'random_pick' from the rest
                    result = top_candidates.copy()
                    if random_pick > 0 and len(children) > keep:
                        remaining_indices = sim_order[keep:].cpu().numpy()
                        if len(remaining_indices) > 0:
                            n_random = min(random_pick, len(remaining_indices))
                            random_indices = np.random.choice(remaining_indices, size=n_random, replace=False)
                            random_candidates = [children[i] for i in random_indices]
                            result.extend(random_candidates)

                    children = result

            population = children

        # Extract top_k parameter from decoder_settings (consistent with main decode_graph)
        top_k = decoder_settings.get("top_k", 10)

        # Sort the final population by cosine similarity to graph_term
        graphs, edges_left = zip(*population, strict=True)
        # OPTIMIZATION: Use Counter.total() to check if empty
        are_final = [i.total() == 0 for i in edges_left]

        # Compute cosine similarities for all final graphs
        batch = Batch.from_data_list([DataTransformer.nx_to_pyg(g) for g in graphs])
        enc_out = self.forward(batch)
        g_terms = enc_out[graph_embedding_attr]
        if decoder_settings.get("use_g3_instead_of_h3", False):
            g_terms = enc_out["node_terms"] + enc_out["edge_terms"] + g_terms

        # Compute similarities and sort
        sims = torchhd.cos(graph_term, g_terms)
        sim_order = torch.argsort(sims, descending=True)

        # Select top_k graphs based on similarity
        top_k_indices = sim_order[: min(top_k, len(graphs))].cpu().numpy()
        top_k_graphs = [graphs[i] for i in top_k_indices]
        top_k_flags = [are_final[i] for i in top_k_indices]
        top_k_sims = [sims[i].item() for i in top_k_indices]

        # Convert Counter to list for target_reached function
        decoded_edges_list = []
        for edge, count in decoded_edges_counter.items():
            if count > 0:
                decoded_edges_list.extend([edge] * count)

        return DecodingResult(
            nx_graphs=top_k_graphs,
            final_flags=top_k_flags,
            cos_similarities=top_k_sims,
            target_reached=target_reached(decoded_edges_list),
            correction_level=CorrectionLevel.FAIL,
        )

    def ensure_vsa(self, t: torch.Tensor) -> VSATensor:
        # ensures vsa type of the tensor is correct
        if isinstance(t, self.vsa.tensor_class):
            return t
        return t.as_subclass(self.vsa.tensor_class)

    def _apply_edge_corrections(
        self,
        edge_term: VSATensor,
        initial_decoded_edges: list[tuple[tuple, tuple]],
    ) -> tuple[CorrectionResult, CorrectionLevel]:
        """
        Apply progressive correction strategies to decoded edges.

        When the initially decoded edges don't form a valid graph (node degrees don't match
        edge counts), this method applies up to three correction strategies in sequence:

        - Level 1: Simple add/remove corrections based on fractional node counts
        - Level 2: Re-decode with corrected node counter (ceiling method)
        - Level 3: Re-decode + corrections

        If all corrections fail, returns the initial edges with FAIL level.

        Parameters
        ----------
        edge_term : VSATensor
            The edge term hypervector for re-decoding if needed.
        initial_decoded_edges : list[tuple[tuple, tuple]]
            The initially decoded edges that need correction.

        Returns
        -------
        tuple[list[list[tuple[tuple, tuple]]], CorrectionLevel]
            - List of corrected edge multisets (each multiset is a list of edges)
            - Correction level achieved (ONE, TWO, THREE, or FAIL)
        """
        # Level 1: Try simple add/remove corrections
        # print(f"Target not reached. Attempting edge corrections {CorrectionLevel.ONE}")
        node_counter_fp = get_node_counter_fp(initial_decoded_edges)
        correction_result: CorrectionResult = get_corrected_sets(
            node_counter_fp, initial_decoded_edges, valid_edge_tuples=self.dataset_info.edge_features
        )

        if correction_result.add_sets or correction_result.remove_sets:
            return correction_result, CorrectionLevel.ONE

        # Level 2: Corrective re-decoding with ceiling method
        # The ceiling method assumes we missed some edges and tries to find more
        # print(f"Target not reached. Attempting edge corrections {CorrectionLevel.TWO}")
        node_counter_corrective = get_node_counter_corrective(initial_decoded_edges, method="ceil")
        decoded_edges_corrective = self.decode_order_one(
            edge_term=edge_term.clone(), node_counter=node_counter_corrective
        )

        if target_reached(decoded_edges_corrective):
            return CorrectionResult(add_sets=[decoded_edges_corrective]), CorrectionLevel.TWO

        # Level 3: Re-decode + corrections
        # print(f"Target not reached. Attempting edge corrections {CorrectionLevel.THREE}")
        correction_result = get_corrected_sets(
            node_counter_fp=get_node_counter_fp(decoded_edges_corrective),
            decoded_edges_s=decoded_edges_corrective,
            valid_edge_tuples=self.dataset_info.edge_features,
        )

        if correction_result.add_sets or correction_result.remove_sets:
            return correction_result, CorrectionLevel.THREE

        # All corrections failed: return initial edges with FAIL level
        # print(f"Target not reached. {CorrectionLevel.FAIL}")
        return CorrectionResult(add_sets=[initial_decoded_edges]), CorrectionLevel.FAIL

    def _find_top_k_isomorphic_graphs(
        self,
        edge_multiset: list[tuple[tuple, tuple]],
        graph_term: VSATensor,
        iteration_budget: int,
        max_graphs_per_iter: int,
        top_k: int,
        sim_eps: float,
        use_early_stopping: bool,
    ) -> list[tuple[nx.Graph, float]]:
        """
        Find top-k graph candidates through pattern matching and similarity ranking.

        This method enumerates valid graph structures from the given edge multiset,
        re-encodes each candidate to HDC, and ranks them by cosine similarity to
        the target graph_term.

        Parameters
        ----------
        edge_multiset : list[tuple[tuple, tuple]]
            Valid edge multiset from which to generate graph candidates.
        graph_term : VSATensor
            Target graph hypervector for similarity comparison.
        iteration_budget : int
            Number of pattern matching iterations to perform.
        max_graphs_per_iter : int
            Maximum number of graph candidates to generate per iteration.
        top_k : int
            Number of top-scoring graphs to retain from each iteration.
        sim_eps : float
            Early stopping threshold: stop if similarity >= 1.0 - sim_eps.
        use_early_stopping : bool
            Whether to enable early stopping when near-perfect match is found.

        Returns
        -------
        list[tuple[nx.Graph, float]]
            List of (graph, similarity_score) tuples from all iterations.
        """
        top_k_graphs = []
        top_k_in_eps_range_found = 0
        for _ in range(iteration_budget):
            # Compute sampling structure from edge multiset
            node_counter = get_node_counter(edge_multiset)
            matching_components, id_to_type = compute_sampling_structure(
                nodes_multiset=[k for k, v in node_counter.items() for _ in range(v)],
                edges_multiset=edge_multiset,
            )

            # Enumerate valid graph structures via isomorphism
            decoded_graphs_iter = try_find_isomorphic_graph(
                matching_components=matching_components,
                id_to_type=id_to_type,
                max_samples=max_graphs_per_iter,
                ring_histogram=self.dataset_info.ring_histogram if self.validate_ring_structure else None,
                single_ring_atom_types=self.dataset_info.single_ring_features if self.validate_ring_structure else None,
            )

            if not decoded_graphs_iter:
                # print("NO isomorphic Graph found")
                continue

            # Batch encode candidate graphs back to HDC
            pyg_graphs = [DataTransformer.nx_to_pyg_with_type_attr(g) for g in decoded_graphs_iter]
            batch = Batch.from_data_list(pyg_graphs)
            graph_hdcs_batch = self.forward(batch)["graph_embedding"]

            # Compute similarities to target graph_term
            sims = torchhd.cos(graph_term, graph_hdcs_batch)

            # Get top k from this iteration
            top_k_sims, top_k_indices = torch.topk(sims, k=min(top_k, len(sims)))
            top_k_sims_cpu = top_k_sims.cpu().numpy()

            # Extend results with (graph, similarity) pairs
            top_k_graphs.extend([(decoded_graphs_iter[top_k_indices[i]], sim) for i, sim in enumerate(top_k_sims_cpu)])

            # Early stopping: if we found the perfect match, or top_k number of near-perfect matches, stop iterating
            should_break = False
            if use_early_stopping:
                for sim in top_k_sims_cpu:
                    if sim == 1.0:
                        print("[EARLY STOPPING] One exact match found!!!")
                        should_break = True
                    # if abs(sim - 1.0) <= sim_eps:
                    #     top_k_in_eps_range_found += 1

                # if top_k_in_eps_range_found >= top_k:
                #     # print(f"[EARLY STOPPING] {top_k} almost exact matches found!!")
                #     break
            if should_break:
                break

        return top_k_graphs

    def _is_feasible_set(self, edge_multiset) -> bool:
        # Compute sampling structure from edge multiset
        node_counter = get_node_counter(edge_multiset)
        matching_components, id_to_type = compute_sampling_structure(
            nodes_multiset=[k for k, v in node_counter.items() for _ in range(v)],
            edges_multiset=edge_multiset,
        )

        # Enumerate valid graph structures via isomorphism with a limited budget to prune the statistically impossible
        # ones
        decoded_graphs_iter = try_find_isomorphic_graph(
            matching_components=matching_components,
            id_to_type=id_to_type,
            max_samples=100,
            ring_histogram=self.dataset_info.ring_histogram if self.validate_ring_structure else None,
            single_ring_atom_types=self.dataset_info.single_ring_features if self.validate_ring_structure else None,
        )
        is_feasible = len(decoded_graphs_iter) > 0
        print(f"Feasibility test {'passed' if is_feasible else 'NOT passed'}")
        return is_feasible

    def decode_graph(
        self, edge_term: VSATensor, graph_term: VSATensor, decoder_settings: dict | None = None
    ) -> DecodingResult:
        """
        Decode a graph from its hyperdimensional representation (edge_term and graph_term).

        This function implements a multi-phase decoding strategy:

        1. **Edge Decoding**: Extract edge multiset from edge_term using cosine similarity
           and iterative unbinding.

        2. **Correction**: If the decoded edges don't form a valid graph (node degrees
           don't match edges), apply progressive correction strategies:
           - Level 0: No correction needed
           - Level 1: Simple add/remove operations on initial decoding
           - Level 2: Re-decode with corrected node counter (ceiling method)
           - Level 3: Re-decode + corrections
           - Fail: Fall back to greedy decoding

        3. **Pattern Matching**: Enumerate valid graph structures from the corrected edge
           multiset, encode them back to HDC, and rank by cosine similarity to graph_term.

        4. **Fallback**: If all corrections fail, use the greedy beam search decoder
           with the same top_k parameter for consistency.

        Parameters
        ----------
        edge_term : VSATensor
            Hypervector representing the edge structure (output from forward() at depth=1).
        graph_term : VSATensor
            Hypervector representing the full graph (output from forward() graph_embedding).
        decoder_settings : dict, optional
            Configuration parameters:
            - iteration_budget: int (default: 1 for QM9, 10 for ZINC)
              Number of pattern matching iterations per edge multiset.
            - max_graphs_per_iter: int (default: 1024)
              Maximum graph candidates to generate per iteration.
            - top_k: int (default: 10)
              Number of top-scoring graphs to return.
            - sim_eps: float (default: 0.0001)
              Early stopping threshold (if similarity >= 1.0 - sim_eps).
            - early_stopping: bool (default: False)
              Whether to stop when a near-perfect match is found.
            - fallback_decoder_settings: dict, optional
              Settings for the greedy decoder fallback. If not provided or if top_k
              is not specified in fallback_decoder_settings, the main top_k value
              will be inherited.

        Returns
        -------
        DecodingResult
            Object containing:
            - nx_graphs: list of NetworkX graphs (top-k candidates)
            - final_flags: list of bools (True if successfully decoded)
            - target_reached: bool (True if correction succeeded)
            - correction_level: CorrectionLevel enum value

        Notes
        -----
        The decoding process is probabilistic due to:
        - Cosine similarity-based edge selection (greedy argmax)
        - Stochastic graph isomorphism enumeration
        - Multiple correction strategies

        For deterministic results, set the same random seed before calling.

        Examples
        --------
        >>> result = hypernet.decode_graph(
        ...     edge_term=edge_terms[0],
        ...     graph_term=graph_terms[0],
        ...     decoder_settings={"top_k": 10, "early_stopping": True},
        ... )
        >>> print(result.correction_level)  # CorrectionLevel.ONE
        >>> best_graph = result.nx_graphs[0]
        """
        edge_term = self.ensure_vsa(edge_term)
        graph_term = self.ensure_vsa(graph_term)

        if decoder_settings is None:
            decoder_settings = {}

        # Validate decoder_settings
        if "top_k" in decoder_settings and decoder_settings["top_k"] < 1:
            raise ValueError("top_k must be >= 1")
        if "iteration_budget" in decoder_settings and decoder_settings["iteration_budget"] < 1:
            raise ValueError("iteration_budget must be >= 1")
        if "max_graphs_per_iter" in decoder_settings and decoder_settings["max_graphs_per_iter"] < 1:
            raise ValueError("max_graphs_per_iter must be >= 1")

        # Extract settings with defaults
        iteration_budget: int = decoder_settings.get("iteration_budget", 1 if self.base_dataset == "qm9" else 10)
        max_graphs_per_iter: int = decoder_settings.get("max_graphs_per_iter", 1024)
        top_k: int = decoder_settings.get("top_k", 10)
        sim_eps: float = decoder_settings.get("sim_eps", 0.0001)
        use_early_stopping: bool = decoder_settings.get("early_stopping", False)
        fallback_decoder_settings = decoder_settings.get("fallback_decoder_settings")
        prefer_smaller_corrective_edits = decoder_settings.get("prefer_smaller_corrective_edits", False)
        self.validate_ring_structure = decoder_settings.get("fallback_decoder_settings", {}).get(
            "validate_ring_structure", False
        )

        # Phase 1: Decode edge multiset from edge_term using greedy unbinding
        initial_decoded_edges = self.decode_order_one_no_node_terms(edge_term.clone())
        if len(initial_decoded_edges) > self.decoding_limit_for:
            print(f"Too many edges to decode: {len(initial_decoded_edges)}")
            return DecodingResult()

        # Phase 2: Check if edges form a valid graph (node degrees match edge counts)
        correction_level = CorrectionLevel.ZERO
        decoded_edges = [initial_decoded_edges]

        if not target_reached(initial_decoded_edges):
            # Edges don't form valid graph → apply progressive correction strategies
            correction_results, correction_level = self._apply_edge_corrections(
                edge_term=edge_term, initial_decoded_edges=initial_decoded_edges
            )

        # Phase 3: If corrections succeeded, enumerate and rank valid graphs via pattern matching
        if correction_level != CorrectionLevel.FAIL:
            top_k_graphs: list[tuple[nx.Graph, float]] = []

            if correction_level == CorrectionLevel.ZERO:
                # We only have the initial set
                decoded_edges = list(filter(self._is_feasible_set, decoded_edges))
            elif prefer_smaller_corrective_edits:
                if correction_results.add_edit_count <= correction_results.remove_edit_count:
                    # Prefer ADD, fallback to REMOVE
                    decoded_edges = list(filter(self._is_feasible_set, correction_results.add_sets))
                    if not decoded_edges:
                        decoded_edges = list(filter(self._is_feasible_set, correction_results.remove_sets))
                else:
                    # Prefer REMOVE, fallback to ADD
                    decoded_edges = list(filter(self._is_feasible_set, correction_results.remove_sets))
                    if not decoded_edges:
                        decoded_edges = list(filter(self._is_feasible_set, correction_results.add_sets))
            else:
                # No preference, use all feasible sets
                feasible_add_sets = list(filter(self._is_feasible_set, correction_results.add_sets))
                feasible_remove_sets = list(filter(self._is_feasible_set, correction_results.remove_sets))
                decoded_edges = feasible_add_sets + feasible_remove_sets

            if len(decoded_edges) == 0:
                # print("[WARNING] all the corrected edge multisets are infeasible")
                return self._fallback_greedy(edge_term, fallback_decoder_settings, graph_term, top_k)

            # print(f"Pruned {before_pruning - len(decoded_edges)}/{before_pruning} of the corrected edge multisets")
            iteration_budget = max(1, iteration_budget // len(decoded_edges))
            print(
                f"[{correction_level.value}] Corrected decoded edges length: {len(decoded_edges)}, Allocated iteration budget per set: {iteration_budget}"
            )

            # For each valid edge multiset, sample and rank graph candidates
            for edge_multiset in decoded_edges:
                graphs_from_multiset = self._find_top_k_isomorphic_graphs(
                    edge_multiset=edge_multiset,
                    graph_term=graph_term,
                    iteration_budget=iteration_budget,
                    max_graphs_per_iter=max_graphs_per_iter,
                    top_k=top_k,
                    sim_eps=sim_eps,
                    use_early_stopping=use_early_stopping,
                )
                top_k_graphs.extend(graphs_from_multiset)

            if len(top_k_graphs) == 0:
                # print("[WARNING] even with correction no valid graphs was enumerated")
                return self._fallback_greedy(edge_term, fallback_decoder_settings, graph_term, top_k)

            # Sort all candidates by similarity (descending) and take top k
            top_k_graphs = sorted(top_k_graphs, key=lambda x: x[1], reverse=True)[:top_k]
            nx_graphs, cos_sims = [], []
            if top_k_graphs:
                nx_graphs, cos_sims = zip(*top_k_graphs, strict=False)

            return DecodingResult(
                nx_graphs=nx_graphs,
                final_flags=[True] * len(top_k_graphs),
                cos_similarities=cos_sims,
                target_reached=True,
                correction_level=correction_level,
            )

        return self._fallback_greedy(edge_term, fallback_decoder_settings, graph_term, top_k)

    # Phase 4: Fallback to greedy decoder if all corrections failed
    # Ensure top_k is passed to greedy decoder (use fallback_decoder_settings if provided,
    # but inherit top_k from main settings if not specified in fallback settings)
    def _fallback_greedy(
        self, edge_term: VSATensor, fallback_decoder_settings: Any | None, graph_term: VSATensor, top_k: int
    ) -> DecodingResult:
        # print("Fall back greedy..")
        greedy_settings = fallback_decoder_settings if fallback_decoder_settings is not None else {}
        if "top_k" not in greedy_settings:
            greedy_settings["top_k"] = top_k
        return self.decode_graph_greedy(edge_term=edge_term, graph_term=graph_term, decoder_settings=greedy_settings)

    def decode_graph_z3(
        self,
        edge_term: torch.Tensor,
        graph_term: torch.Tensor | None = None,
        node_counter: Counter | None = None,
        decoder_settings: dict | None = None,
    ):
        if decoder_settings is None:
            decoder_settings = {}
        max_solutions = decoder_settings.get("max_solutions", 1000)

        # Case 2D/3D vectors with G0 encoded
        if node_counter:
            decoded_edges = self.decode_order_one(edge_term=edge_term, node_counter=node_counter)
        else:
            decoded_edges = self.decode_order_one_no_node_terms(edge_term=edge_term)
            # Only using the edges and the degree of the nodes we can count the number of nodes
            node_counter = get_node_counter(decoded_edges)

        if node_counter.total() > 20:
            # print(f"Skipping graph with {node_counter.total()} nodes")
            # TODO: Correct this for ZINC
            return [nx.Graph()], [False]

        try:
            candidates = enumerate_graphs(
                nodes_multiset=list(node_counter.elements()), edges_multiset=decoded_edges, max_solutions=max_solutions
            )
        except ValueError as err:
            print(f"[ERROR] While decoding graph: {err}")
            return [nx.Graph()], [False]

        return DecodingResult(
            nx_graphs=[
                DataTransformer.z3_res_to_nx(ordered_nodes=c["ordered_nodes"], edge_indexes=c["associated_edge_idxs"])
                for c in candidates
            ],
            final_flags=[True] * len(candidates),  # z3 only produces final graphs
            target_reached=True,
        )

    def use_edge_features(self) -> bool:
        return len(self.edge_encoder_map) > 0

    def use_graph_features(self) -> bool:
        return len(self.graph_encoder_map) > 0


def get_node_counter_corrective(
    edges: list[tuple[tuple, tuple]], method: Literal["ceil", "round", "max_round"] = "ceil"
) -> Counter[tuple]:
    # Only using the edges and the degree of the nodes we can count the number of nodes
    node_degree_counter = Counter(u for u, _ in edges)
    node_counter = Counter()
    for k, v in node_degree_counter.items():
        # By dividing the number of outgoing edges to the node degree, we can count the number of nodes
        if method == "ceil":
            node_counter[k] = math.ceil(v / (k[1] + 1))  # Performs best
        if method == "round":
            node_counter[k] = round(v / (k[1] + 1))
        if method == "max_round":
            node_counter[k] = max(1, round(v / (k[1] + 1)))
    return node_counter


def get_node_counter_fp(edges: list[tuple[tuple, tuple]]) -> Counter[tuple]:
    # Only using the edges and the degree of the nodes we can count the number of nodes
    node_degree_counter = Counter(u for u, _ in edges)
    # having the number nodes as floating point, we can determine how many nodes are missing
    # or how many nodes are too many with regards to the edges
    return Counter({k: v / (k[1] + 1) for k, v in node_degree_counter.items()})


def load_or_create_hypernet(
    cfg: DSHDCConfig,
    path: Path = GLOBAL_MODEL_PATH,
    depth: int = 3,
    *,
    use_edge_codebook: bool = False,
) -> HyperNet:
    dtype_sfx = "-f64" if cfg.dtype == "float64" else ""
    path = (
        path
        / f"hypernet_{cfg.name}_{cfg.vsa.value}_dim{cfg.hv_dim}_s{cfg.seed}_depth{depth}_ecb{int(use_edge_codebook)}{dtype_sfx}.pt"
    )
    if path.exists():
        encoder = HyperNet.load(path=path)
        encoder.depth = cfg.hypernet_depth
        encoder.decoding_limit_for = cfg
        encoder.normalize = cfg.normalize
        print(f"Loaded from existing HyperNet from {path}")
    else:
        print("Creating new HyperNet instance.")
        encoder = HyperNet(config=cfg, depth=depth, use_edge_codebook=use_edge_codebook)
        encoder.populate_codebooks()
        encoder.save_to_path(path)
        encoder.normalize = cfg.normalize
        encoder.depth = cfg.hypernet_depth
    return encoder
