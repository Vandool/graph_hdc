import itertools
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, ClassVar

import jsonpickle
import numpy as np
import pytorch_lightning as pl
import torch
import torchhd
from torch import Tensor
from torch.nn.functional import sigmoid
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from graph_hdc.utils import shallow_dict_equal
from src.encoding.configs_and_constants import DatasetConfig, Features, IndexRange, SupportedDataset
from src.encoding.feature_encoders import (
    AbstractFeatureEncoder,
    CategoricalIntegerEncoder,
    CategoricalLevelEncoder,
    CategoricalOneHotEncoder,
    CombinatoricIntegerEncoder,
    TrueFalseEncoder,
)
from src.encoding.the_types import VSAModel
from src.utils.utils import TupleIndexer, cartesian_bind_tensor, flatten_counter, scatter_hd

# === HYPERDIMENSIONAL MESSAGE PASSING NETWORKS ===

EncoderMap = dict[Features, tuple[AbstractFeatureEncoder, IndexRange]]


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
        config: DatasetConfig | None = None,
        depth: int = 3,
        *,
        use_explain_away: bool = True,
        use_edge_codebook: bool = True,
    ):
        AbstractGraphEncoder.__init__(self)
        self.use_explain_away = use_explain_away
        self.use_edge_codebook = use_edge_codebook
        self.depth = depth
        self.vsa = self._validate_vsa(config.vsa)
        self.hv_dim = config.hv_dim
        self.node_encoder_map: EncoderMap = {
            feat: (
                cfg.encoder_cls(
                    num_categories=cfg.count,
                    dim=config.hv_dim,
                    vsa=config.vsa.value,
                    idx_offset=cfg.idx_offset,
                    indexer=TupleIndexer(sizes=cfg.bins) if cfg.bins else TupleIndexer(sizes=[28,6, config.nha_bins]),
                ),
                cfg.index_range,
            )
            for feat, cfg in config.node_feature_configs.items()
        }
        self.edge_encoder_map: EncoderMap = {
            feat: (
                cfg.encoder_cls(
                    num_categories=cfg.count, dim=config.hv_dim, vsa=config.vsa.value, idx_offset=cfg.idx_offset
                ),
                cfg.index_range,
            )
            for feat, cfg in config.edge_feature_configs.items()
        }
        self.graph_encoder_map: EncoderMap = {
            feat: (
                cfg.encoder_cls(
                    num_categories=cfg.count, dim=config.hv_dim, vsa=config.vsa.value, idx_offset=cfg.idx_offset
                ),
                cfg.index_range,
            )
            for feat, cfg in config.graph_feature_configs.items()
        }
        self.seed = config.seed

        ### Attributes that will be populated after initialisation
        self.nodes_indexer: TupleIndexer | None = None

        # Contains hypervectors that represents a Node (HV_node = bind(HV(f1), ..., HV(f2)))
        self.nodes_codebook: Tensor | None = None
        self.edges_indexer: TupleIndexer | None = None

        # Contains hypervectors that represents an edge regardless of its nodes (HV_node = bind(HV(f1), ..., HV(f2)))
        self.edge_feature_codebook: Tensor | None = None
        self.edges_feature_indexer: TupleIndexer | None = None

        # Contains hypervectors that represents an edge including the nodes and edge features
        self.edges_codebook: Tensor | None = None
        self.edges_indexer: TupleIndexer | None = None

    def to(self, device):
        self.populate_codebooks()
        self.nodes_codebook = self.nodes_codebook.to(device)
        self.edges_codebook = self.edges_codebook.to(device)
        # self.graph_codebook = self.graph_codebook.to(device)
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
            return torch.zeros(fallback_count, self.hv_dim, device=self.device)

        # stack on new “property” axis and bind
        stacked = torch.stack(slices, dim=0)  # [P, N, D]
        # move property axis to just before D for multibind on P dimension:
        to_bind = stacked.transpose(0, 1)  # [N, P, D]

        # Otherwise, bind across the P dimension
        return torchhd.multibind(to_bind)  # [..., N, D]

    def forward(
        self,
        data: Data,
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
        # Afterwards, the data object contains the additional properties "data.node_hv" and "data.graph_hv"
        # which represent the encoded hyper-vectors for the individual nodes or for the overall graphs respectively.
        data = self.encode_properties(data)

        # ~ handling continuous edge weights
        # Optionally it is possible for the input graph structures to also define a "edge_weight" property which
        # should be a continuous value that represents the weight of the edge. This weight will later be used
        # to weight/gate the message passing over the corresponding edge during the message-passing steps.
        # Specifically, the values in the "edge_weight" property should be the edge weight LOGITS, which will
        # later be transformed into a [0, 1] range using the sigmoid function!

        ## We don't have edge_weights, candidate for deletion (keep the default behaviour??)
        if hasattr(data, "edge_weight") and data.edge_weight is not None:
            edge_weight = data.edge_weight
        else:
            # If the given graphs do not define any edge weights we set the default values to 10 for all edges
            # because sigmoid(10) ~= 1.0 which will effectively be the same as discrete edges.
            # edge_weight should have the same dtype as the hypervectors
            edge_weight = 100 * torch.ones(
                data.edge_index.shape[1],
                1,
                device=self.device,
                # dtype=data.node_hv.dtype
            )

        # ~ handling edge bi-directionality
        # If the bidirectional flag is given we will duplicate each edge in the input graphs and reverse the
        # order of node indices such that each node of each edge is always considered as a source and a target
        # for the message passing operation.
        # Similarly we also duplicate the edge weights such that the same edge weight is used for both edge
        # "directions".

        ## We don't have edge_weights
        if bidirectional:
            edge_index = torch.cat([data.edge_index, data.edge_index[[1, 0]]], dim=1)
            edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
        else:
            edge_index = data.edge_index

        srcs, dsts = edge_index

        # In this data structure we will stack all the intermediate node embeddings for the various message-passing
        # depths.
        # node_hv_stack: (num_layers + 1, batch_size * num_nodes, hv_dim)
        node_dim = data.x.size(0)
        node_hv_stack = data.node_hv.new_zeros(size=(self.depth + 1, node_dim, self.hv_dim), device=self.device)
        node_hv_stack[0] = data.node_hv  # Level 0 HV: Nodes are binding of their features

        # ~ message passing
        edge_terms = None
        node_terms = None
        for layer_index in range(self.depth):
            # messages are gated with the corresponding edge weights! Pick the neighbours
            messages = node_hv_stack[layer_index][dsts] * sigmoid(edge_weight)

            # aggregate (bundle) neighbor messages back into each node slot
            aggregated = scatter_hd(messages, srcs, dim_size=node_dim, op="bundle")

            prev_hv = node_hv_stack[layer_index].clone()

            # Level's HV
            hr = torchhd.bind(prev_hv, aggregated)  # [node_dim, D]

            # capture “edge terms” once at the very first layer
            if layer_index == 0:
                edge_terms = hr.clone()

            if normalize:
                # compute L2 norm along the last dimension (out‐of‐place)
                hr_norm = hr.norm(dim=-1, keepdim=True)  # [node_dim, 1]

                # divide by norm, also out‐of‐place
                node_hv_stack[layer_index + 1] = hr / (hr_norm + 1e-8)  # [node_dim, D]
            else:
                node_hv_stack[layer_index + 1] = hr

        # We calculate the final graph-level embedding as the sum of all the node embeddings over all the various
        # message passing depths and as the sum over all the nodes.
        node_hv_stack = node_hv_stack.transpose(0, 1)
        node_hv = torchhd.multibundle(node_hv_stack)  # This is bundle - [N, P, D] -> [N, D]
        readout = scatter_hd(src=node_hv, index=data.batch, op="bundle")
        # TODO: Research to see how often the hv should be normalised? should the final embedding be normalised?
        # We should not normalise the embedding, otherwise all the information regarding the frequency of the encoded
        # properties will be lost
        embedding = readout

        if separate_levels:
            ## Prepare Level 0 Embeddings: Only node terms
            node_terms = scatter_hd(src=data.node_hv, index=data.batch, op="bundle")

            ## Prepare level 1 Embeddings: Only edge terms (not bounded with level 0)
            ## Prepare Level 0 Embedding: Only node terms
            edge_terms = scatter_hd(src=edge_terms, index=data.batch, op="bundle")

        return {
            # This the main result of the forward pass which is the individual graph embedding vectors of the
            # input graphs.
            # graph_embedding: (batch_size, hv_dim)
            "graph_embedding": embedding,
            # As additional information that might be useful we also pass the stack of the node embeddings across
            # the various convolutional depths.
            # node_hv_stack: (batch_size * num_nodes, num_layers + 1, hv_dim)
            "node_hv_stack": node_hv_stack,
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
            if isinstance(enc := self.node_encoder_map[Features.ATOM_TYPE][0], CombinatoricIntegerEncoder):
                self.nodes_indexer = enc.indexer
            else:
                self.nodes_indexer = TupleIndexer([e.num_categories for e, _ in self.node_encoder_map.values()])

    def populate_nodes_codebooks(self) -> None:
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

    @staticmethod
    def is_not_initialized(tensor: torch.Tensor | None) -> bool:
        return tensor is None or tensor.numel() == 0

    def decode_order_one_counter_explain_away(
        self,
        embedding: torch.Tensor,  # [B, D] or [D]
        unique_decode_nodes_batch: list[set[tuple[int, int]]],  # length B
        max_iters: int = 50,
        threshold: float = 0.5,
        *,
        use_break: bool = True,
    ) -> list[Counter]:
        """
        Peel off top edges iteratively exactly as in your script,
        but do it for each graph in the batch.
        """
        self.populate_codebooks()

        # 1) ensure batch dimension
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        B, D = embedding.shape

        # populate all codebooks/indexers
        # self._populate_nodes_codebooks()
        # self._populate_node_indexer()

        results: list[Counter] = []

        for b in range(B):
            # — build per‐graph data structures —
            graph_hv = embedding[b].clone()
            # which nod‐indices are present?
            node_idxs = self.nodes_indexer.get_idxs(unique_decode_nodes_batch[b])
            # flatten+sort for reproducibility
            nodes_decoded_flat = sorted(node_idxs)

            # all possible directed pairs over those nodes
            edge_to_check = list(itertools.product(nodes_decoded_flat, nodes_decoded_flat))

            found = set()

            iteration = 1
            should_break = False
            while iteration <= max_iters and edge_to_check and not should_break:
                # 2) score every candidate
                scores = []
                for u, v in edge_to_check:
                    eh = self.nodes_codebook[u].bind(self.nodes_codebook[v])
                    s = torchhd.dot(graph_hv, eh)
                    # if MAP, normalize
                    if self.vsa == VSAModel.MAP:
                        s = s / D
                    scores.append(s.item())

                # 3) stop if even the top score is <= 0
                if max(scores) <= 0.0:
                    break

                # 4) sort descending, pick every‐other slot
                idxs_sorted = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

                for idx in idxs_sorted[::2]:
                    u, v = edge_to_check[idx]
                    if scores[idx] > threshold:
                        found.add((u, v))
                        found.add((v, u))
                        graph_hv -= self.nodes_codebook[u].bind(self.nodes_codebook[v])
                        graph_hv -= self.nodes_codebook[v].bind(self.nodes_codebook[u])
                        edge_to_check = [e for e in edge_to_check if e not in {(u, v), (v, u)}]
                        if use_break:
                            break
                    elif scores[idx] <= 0:
                        should_break = True

                iteration += 1

            results.append(Counter(found))

        return results

    def decode_order_one_counter_explain_away_faster(
        self,
        embedding: torch.Tensor,  # [B, D] or [D]
        unique_decode_nodes_batch: list[set[tuple[int, int]]],  # length B
        max_iters: int = 50,
        threshold: float = 0.5,
        *,
        use_break: bool = False,
    ) -> list[Counter]:
        self.populate_nodes_codebooks()
        self._populate_nodes_indexer()
        self.populate_edge_feature_codebook()
        self._populate_edge_feature_indexer()
        self.populate_edges_codebook()
        self._populate_edges_indexer()

        # 1) ensure batch dimension
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        B, D = embedding.shape

        results = []
        for b in range(B):
            graph_hv = embedding[b].clone()

            # 2) build the list of candidate edge *flat* indices once
            node_idxs = self.nodes_indexer.get_idxs(unique_decode_nodes_batch[b])
            assert len(node_idxs) > 0
            pairs = list(itertools.product(node_idxs, node_idxs))
            cand_idxs = torch.tensor(self.edges_indexer.get_idxs(pairs), dtype=torch.long)
            assert cand_idxs.shape[0] == len(node_idxs) ** 2

            found = set()
            iteration, should_break = 1, False

            while iteration <= max_iters and len(cand_idxs) > 0 and not should_break:
                # 2) score all candidates at once
                slice_hvs = self.edges_codebook[cand_idxs]  # [E, D]
                sims = slice_hvs.matmul(graph_hv)  # [E]
                if self.vsa == VSAModel.MAP:
                    sims = sims / D

                # 3) break if top ≤ 0
                if sims.max() <= 0:
                    break

                # 4) sort descending indices
                idxs_sorted = sims.argsort(descending=True)  # [E]

                # inner loop, every‐other slot
                to_remove = set()
                for idx in idxs_sorted[::2].tolist():
                    score = sims[idx].item()
                    candidates_idx = cand_idxs[idx].item()
                    u, v = self.edges_indexer.get_tuple(candidates_idx)

                    if score > threshold:
                        # exactly your original body:
                        found.add((u, v))
                        found.add((v, u))

                        graph_hv = (
                            graph_hv
                            - self.nodes_codebook[u].bind(self.nodes_codebook[v])
                            - self.nodes_codebook[v].bind(self.nodes_codebook[u])
                        )

                        # remove both directions
                        to_remove.update({self.edges_indexer.get_idx((u, v)), self.edges_indexer.get_idx((v, u))})
                        if use_break:
                            break
                    elif score <= 0:
                        should_break = True
                        break
                    # else continue searching

                # Remove the found ones
                mask = [(i.item() not in to_remove) for i in cand_idxs]
                cand_idxs = cand_idxs[mask]

                iteration += 1

            results.append(Counter(found))

        return results

    def decode_order_one_counter_explain_away_faster_lazy(
        self,
        embedding: torch.Tensor,  # [B, D] or [D]
        unique_decode_nodes_batch: list[set[tuple[int, int]]],  # length B
        max_iters: int = 50,
        threshold: float = 0.5,
        *,
        use_break: bool = False,
    ) -> list[Counter]:
        # Use no edge codebook
        self.populate_nodes_codebooks()
        self._populate_nodes_indexer()

        # 1) ensure batch dimension
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        B, D = embedding.shape

        results = []
        for b in range(B):
            graph_hv = embedding[b].clone()

            # 2) build the list of candidate edge *flat* indices once
            node_idxs = self.nodes_indexer.get_idxs(unique_decode_nodes_batch[b])
            assert len(node_idxs) > 0
            pairs = list(itertools.product(node_idxs, node_idxs))
            bound_hvs = torch.stack(
                [self.nodes_codebook[u].bind(self.nodes_codebook[v]) for u, v in pairs],
                dim=0,
            )  # [E, D]
            active = torch.ones(len(pairs), dtype=torch.bool, device=bound_hvs.device)

            found = set()
            iteration, should_break = 1, False

            while iteration <= max_iters and active.any() and not should_break:
                cand_hvs = bound_hvs[active]  # [E_active, D]
                sims = cand_hvs.matmul(graph_hv)  # [E_active]
                if self.vsa == VSAModel.MAP:
                    sims = sims / D

                top_val = sims.max()
                if top_val <= 0:  # step 3 – nothing left
                    break

                idxs_sorted = sims.argsort(descending=True)  # step 4
                # Map back to *global* pair indices
                global_idxs = active.nonzero(as_tuple=False).flatten()

                active_modified = False
                for loc_idx in idxs_sorted[::2].tolist():  # inner loop
                    g_idx = global_idxs[loc_idx].item()
                    score = sims[loc_idx].item()
                    u, v = pairs[g_idx]

                    if score > threshold:
                        # record & explain-away
                        found.update({(u, v), (v, u)})
                        graph_hv = (
                            graph_hv
                            - bound_hvs[g_idx]  # (u, v)
                            - bound_hvs[pairs.index((v, u))]  # (v, u)
                        )
                        # mark both directions as inactive
                        active[g_idx] = False
                        active[pairs.index((v, u))] = False
                        active_modified = True
                        if use_break:
                            break
                    #
                    elif 0 < score <= threshold:
                        if not active_modified:
                            should_break = True
                        break
                    else:
                        should_break = True
                        break
                iteration += 1

            results.append(Counter(found))

        return results


    def decode_order_one_counter(
        self,
        embedding: torch.Tensor,
        unique_decode_nodes_batch: list[set[tuple[int, int]]],  # [unique_nodes], len = batch_size
    ) -> list[dict]:
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

        :param unique_decode_nodes_batch:
        :param embedding: The high-dimensional graph embedding vector that represents the graph.
        :param constraints_order_zero: The list of constraints that represent the zero order information about the
            nodes that were present in the original graph.
        :param correction_factor_map: A dictionary that contains correction factors for the number of shared core
            properties between the nodes that constitute the edge. The keys are the number of shared core properties
            and the values are the correction factors that should be applied to the calculated edge count.

        :returns: A list of constraints where each constraint is represented by a dictionary with the keys:
            - src: A dictionary that represents the properties of the source node as they were originally encoded
                by the node encoders. The keys in this dict are the same as the names of the node encoders given to
                the constructor.
            - dst: A dictionary that represents the properties of the destination node as they were originally
                encoded by the node encoders. The keys in this dict are the same as the names of the node encoders
                given to the constructor.
            - num: The integer number of how many of these edges are present in the graph.
        """

        # 1) Make sure we have a true batch of embeddings
        if embedding.dim() == 1:
            # single graph case → treat it as batch of size 1
            embedding = embedding.unsqueeze(0)

        self.populate_nodes_codebooks()
        self._populate_nodes_indexer()
        self.populate_edge_feature_codebook()
        self._populate_edge_feature_indexer()
        self.populate_edges_codebook()
        self._populate_edges_indexer()

        # # Compute the dot product between the embedding and each node hypervector representation.
        # d = torchhd.dot(embedding, self.edges_codebook)
        # # Round the dimension normalized dot products to obtain integer counts.
        # # [Batch, N]: where N is the number of possible combinatorial edges (nodes*nodes*edges).
        # dot_product_rounded = torch.round(d).int().clamp(min=0)
        # return self.convert_to_counter(similarities=dot_product_rounded, indexer=self.edges_indexer)
        ## Let's optimize the edge decoder

        results: list[dict] = []

        # our indexer expects tuples of (src_idx, dst_idx) over node‐index space:
        edge_indexer = self.edges_indexer

        batches = embedding.shape[0]
        for b in range(batches):
            # 1) collect this batch’s unique node‐tuples, turn them into their integer indices
            node_idxs = self.nodes_indexer.get_idxs(unique_decode_nodes_batch[b])

            pairs = [(u, v) for u in node_idxs for v in node_idxs]

            if not pairs:
                # no candidate edges
                results.append({})
                continue

            # 3) map those to positions in edges_codebook
            edge_idxs = edge_indexer.get_idxs(pairs)  # shape [E]

            # 4) gather their hypervectors and compute similarities
            # edges_codebook: [#all_possible_edges, D]
            codebook_slice = self.edges_codebook[edge_idxs]  # [E, D]
            sims = torchhd.dot(embedding[b : b + 1], codebook_slice).flatten()  # [E]

            # 5) round to counts, clamp ≥0
            counts = sims.tolist()

            # 6) assemble a Counter mapping the original tuple‐of‐node‐tuples → count
            #    map back from idx to tuple
            ctr = Counter()
            for i, edge_idx in enumerate(edge_idxs):
                if counts[i] > 0.0:
                    ctr[edge_indexer.get_tuple(edge_idx)] += 1

            results.append(ctr)

        # Returns (node_idx, node_idx) Counter - node_idx based on the nodes_indexer
        return results

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
            )

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
                        "seed": getattr(encoder, "seed", None),
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

        state = {
            "attributes": {
                "depth": self.depth,
                "seed": self.seed,
                "vsa": self.vsa.value,  # ← store string
                "hv_dim": self.hv_dim,
            },
            "node_encoder_map": serialize_encoder_map(self.node_encoder_map),
            "edge_encoder_map": serialize_encoder_map(self.edge_encoder_map),
            "graph_encoder_map": serialize_encoder_map(self.graph_encoder_map),
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
            return Features[key]

        def deserialize_encoder_map(serialized: dict[str, Any]):
            from src.utils.utils import TupleIndexer

            result = {}
            for feat_key, entry in serialized.items():
                args = entry["init_args"]
                enc_cls = encoder_class_map[entry["encoder_class"]]

                # -- indexer (only for CombinatoricIntegerEncoder) ---------------
                if "indexer_state" in entry:
                    indexer = TupleIndexer.__new__(TupleIndexer)
                    indexer.__dict__.update(entry["indexer_state"])
                    encoder = enc_cls(**args, indexer=indexer)
                else:
                    encoder = enc_cls(**args)

                # restore codebook / device
                encoder.codebook = entry["codebook"]
                encoder.device = torch.device(args["device"])

                result[_feat_key_to_enum(feat_key)] = (encoder, entry["index_range"])
            return result

        # With weights_only=True the “safe” un-pickler only accepts tensors, primitive types and whatever you
        # explicitly allow-list. We set it to False to allow unsafe un-pickler.
        state = torch.load(path, map_location="cpu", weights_only=False)
        for k, v in state["attributes"].items():
            setattr(self, k, v)
        self.node_encoder_map = deserialize_encoder_map(state["node_encoder_map"])
        self.edge_encoder_map = deserialize_encoder_map(state["edge_encoder_map"])
        self.graph_encoder_map = deserialize_encoder_map(state["graph_encoder_map"])

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
        instance.load_from_path(path=path)
        return instance

    def possible_graph_from_constraints(
        self,
        zero_order_constraints: list[dict],
        first_order_constraints: list[dict],
    ) -> tuple[dict, list]:
        # ~ Build node information from constraints list
        # This data structure will contain a unique integer node index as the key and the value will
        # be the dictionary which contains the node properties that were originally decoded.
        index_node_map: dict[int, dict] = {}
        index: int = 0
        for nc in zero_order_constraints:
            num = nc["num"]
            for _ in range(num):
                index_node_map[index] = nc["src"]
                index += 1

        # ~ Build edge information from constraints list
        edge_indices: set[tuple[int, int]] = set()
        for ec in first_order_constraints:
            src = ec["src"]
            dst = ec["dst"]

            # Now we need to find all the node indices which match the description of the edge source
            # and destination. This is done by iterating over the index_node_map and checking if the
            # node properties match the source and destination properties of the edge.
            # For each matching pair, we insert an edge into the edge_indices list.
            for i, node_i in index_node_map.items():
                if shallow_dict_equal(node_i, src):
                    for j, node_j in index_node_map.items():
                        if shallow_dict_equal(node_j, dst) and i != j:
                            hi = max(i, j)
                            lo = min(i, j)
                            edge_indices.add((hi, lo))

        return index_node_map, list(edge_indices)

    def reconstruct(
        self,
        graph_hv: torch.Tensor,
        node_terms: torch.Tensor | None = None,
        edge_terms: torch.Tensor | None = None,
        num_iterations: int = 25,
        learning_rate: float = 1.0,
        batch_size: int = 10,
        low: float = 0.0,
        high: float = 1.0,
        alpha: float = 1.0,
        lambda_l1: float = 0.0,
        *,
        use_node_degree: bool = False,
        is_undirected: bool = True,
    ) -> dict:
        """
        Reconstructs a graph dict representation from the given graph hypervector by first decoding
        the order constraints for nodes and edges to build an initial guess and then refining the
        structure using gradient descent optimization.

        Now, instead of optimizing a single candidate, a whole batch of candidates are optimized.
        The edge weights are randomly initialized between low and high and, after optimization,
        are discretized. The candidate with the best similarity to graph_hv is selected.
        """
        log = False
        # ~ Decode node constraints
        node_terms = node_terms if node_terms is not None else graph_hv
        ## Nodes are represented as tuple of their features
        node_counters = self.decode_order_zero_counter(node_terms)

        # ~ Decode Edge terms
        edge_terms = edge_terms if edge_terms is not None else graph_hv
        decode_order_one_fn = (
            self.decode_order_one_counter_explain_away_faster
            if self.use_explain_away
            else self.decode_order_one_counter
        )
        # Edges are (Node_idx_u, Node_idx_v)
        edge_counters = decode_order_one_fn(edge_terms, node_counters)
        if log:
            print("node counter", node_counters)
            print("edge counter", edge_counters)

        # node_keys = list(node_counters[0]["src"].keys())

        # Given the node and edge constraints, this method will assemble a first guess of the graph
        # structure by inserting all the nodes that were defined by the node constraints and inserting
        # all possible edges that match any of the given edge constraints.
        # index_node_map, edge_indices = self.possible_graph_from_constraints(node_counters, edge_counters)

        # node_counters counts the number of nodes per batch
        # We have no batch of graph, the incoming data belongs to one graph
        __x__ = self.get_data_x(node_counters[0], self.node_encoder_map)
        edge_indices = self.get_edge_index_list(edge_counter=edge_counters[0], node_counter=node_counters[0])

        data = Data()
        data.edge_index = torch.tensor(list(edge_indices), dtype=torch.long).t()
        data.batch = torch.tensor([0] * __x__.shape[0], dtype=torch.long, device=__x__.device)
        # data.x = torch.zeros(__x__.shape[0], self.hv_dim)
        data.x = __x__
        # data = self.encode_properties(data)

        # Create batch of identical templates
        data_list = [data.clone() for _ in range(batch_size)]
        batch = Batch.from_data_list(data_list)

        # CHANGED: initialize trainable raw_weights for unique edges per candidate
        edges_per_graph = data.edge_index.size(1)
        if is_undirected:
            assert edges_per_graph % 2 == 0, "Expected each undirected edge twice"
            num_unique = edges_per_graph // 2
            raw_weights = torch.nn.Parameter(
                torch.empty(batch_size * num_unique, 1, device=self.device).uniform_(low, high)
            )
        else:
            raw_weights = torch.nn.Parameter(
                torch.empty(batch_size * edges_per_graph, 1, device=self.device).uniform_(low, high)
            )
        optimizer = torch.optim.Adam([raw_weights], lr=learning_rate)

        # Precompute constants
        num_nodes = __x__.shape[0] * batch_size
        node_keys = [x.value for x in self.node_encoder_map]

        # Optimization loop over candidate batch
        for it in range(num_iterations):
            optimizer.zero_grad()
            # CHANGED: expand raw_weights into full per-edge logits, tying for undirected
            if is_undirected:
                full_logits = raw_weights.repeat_interleave(2, dim=0)
            else:
                full_logits = raw_weights
            batch.edge_weight = full_logits

            result = self.forward(batch)
            embedding = result["graph_embedding"]  # shape (candidate_batch_size, hv_dim)

            # Compute mean squared error loss for each candidate (compare each to graph_hv)
            losses = torch.square(embedding - graph_hv.expand_as(embedding)).mean(dim=1)
            loss_embed = losses.mean()  # loss should have grad_fn, it works only when the grad_fn is not None

            if use_node_degree and ("node_degree" in node_keys or "node_degrees" in node_keys):
                _, (start, end) = self.node_encoder_map[Features.NODE_DEGREE]
                true_degree = batch.x[:, start:end]

                _edge_weight = torch.sigmoid(2 * full_logits)
                _edges_src = scatter(
                    torch.ones_like(_edge_weight), batch.edge_index[0], dim_size=num_nodes, reduce="sum"
                )
                _edges_dst = scatter(
                    torch.ones_like(_edge_weight), batch.edge_index[1], dim_size=num_nodes, reduce="sum"
                )
                _num_edges = _edges_src + _edges_dst

                # _edge_weight = torch.where(_edge_weight > 0.5, _edge_weight, _edge_weight * 0.001)
                # _edge_weight = torch.where(_edge_weight > 0.2, torch.ones_like(_edge_weight), torch.zeros_like(_edge_weight))
                scatter_src = scatter(_edge_weight, batch.edge_index[0], dim_size=num_nodes, reduce="sum")
                scatter_dst = scatter(_edge_weight, batch.edge_index[1], dim_size=num_nodes, reduce="sum")
                # Calculate the actual node degree by summing over the edge weights of all the in and out going edges of a node
                pred_degree = scatter_src + scatter_dst  # shape [batch_size * num_nodes]
                # CHANGED: add L2 degree‐matching loss
                loss_degree = (pred_degree - true_degree.view_as(pred_degree)).pow(2).mean()  # CHANGED
                alpha = alpha  # CHANGED: tradeoff weight
            else:
                loss_degree = 0.0  # CHANGED
                alpha = 0.0  # CHANGED

            # # 6) Sparsity (L1 on the raw logits)
            sparsity = torch.abs(batch.edge_weight).sum()  # CHANGED
            lambda_l1 = lambda_l1  # CHANGED: tradeoff for sparsity

            # 7) Total loss
            loss = loss_embed + alpha * loss_degree + lambda_l1 * sparsity  # CHANGED
            # loss = loss_embed
            loss.backward()
            optimizer.step()

            if log:
                print(f"Iter {it}: loss={loss.item()}")

        # After optimization, assemble final logits
        if is_undirected:
            final_logits = raw_weights.repeat_interleave(2, dim=0).detach()
        else:
            final_logits = raw_weights.detach()

        # Assemble final logits
        if is_undirected:
            full_logits = raw_weights.repeat_interleave(2, dim=0).detach()
        else:
            full_logits = raw_weights.detach()
        edges = edges_per_graph
        final_logits = full_logits.view(batch_size, edges, 1)

        # Recompute embeddings to pick best candidate
        batch.edge_weight = final_logits.view(batch_size * edges, 1)
        out = self.forward(batch)
        emb = out["graph_embedding"]
        losses = torch.square(emb - graph_hv.expand_as(emb)).mean(dim=1)
        best_idx = int(torch.argmin(losses).item())

        # Extract best logits and build final Data
        best_logits = final_logits[best_idx]  # (edges,1)
        # select edges above threshold
        mask = torch.sigmoid(best_logits) > 0.0
        kept_edges = data.edge_index[:, mask.flatten()]

        # Build best Data and attach edge_weight
        best_data = Data(x=__x__, edge_index=kept_edges)
        best_data.edge_weight = best_logits[mask]

        # Compute final degree sums for logs
        num_nodes_final = __x__.shape[0]
        if best_data.edge_index.numel() > 0:
            scatter_src = scatter(
                best_data.edge_weight, best_data.edge_index[0], dim_size=num_nodes_final, reduce="sum"
            )
            scatter_dst = scatter(
                best_data.edge_weight, best_data.edge_index[1], dim_size=num_nodes_final, reduce="sum"
            )
            final_degrees = scatter_src + scatter_dst
        else:
            # No edges: degrees are zero for all nodes
            final_degrees = torch.zeros(num_nodes_final, device=self.device)

        if log:
            print(f"Final batch.edge_weight (flattened): {batch.edge_weight.flatten()}")
            print(f"Final losses per candidate: {losses.detach().cpu().numpy()}")
            print(f"Chosen index: {best_idx}, loss: {losses[best_idx].item():.4f}")
            print(f"Final edge weights: {best_data.edge_weight.flatten().cpu().numpy()}")
            print(f"Final degrees: {(scatter_src + scatter_dst).cpu().numpy()}")
            print(f"Final edge index: {kept_edges.detach().cpu().numpy().T}")

        return best_data, node_counters, edge_counters

    def use_edge_features(self) -> bool:
        return len(self.edge_encoder_map) > 0

    def use_graph_features(self) -> bool:
        return len(self.graph_encoder_map) > 0


def load_or_create_hypernet(
    path: Path, ds: SupportedDataset, depth: int = 3, *, use_edge_codebook: bool = False
) -> HyperNet:
    cfg = ds.default_cfg
    ds_name = ds.name
    if cfg.nha_depth:
        ds_name = f"{ds_name}-d{cfg.nha_depth}"
    if cfg.nha_bins:
        ds_name = f"{ds_name}-b{cfg.nha_bins}"
    path = path / f"hypernet_ds{ds_name}_{cfg.vsa.value}_d{cfg.hv_dim}_s{cfg.seed}_dpth{depth}_ecb{int(use_edge_codebook)}.pt"
    if path.exists():
        print(f"Loading existing HyperNet from {path}")
        encoder = HyperNet(config=cfg, depth=depth, use_edge_codebook=use_edge_codebook)
        encoder.load(path)
    else:
        print("Creating new HyperNet instance.")
        encoder = HyperNet(config=cfg, depth=depth, use_edge_codebook=use_edge_codebook)
        encoder.populate_codebooks()
        encoder.save_to_path(path)
        print(f"Saved new HyperNet to {path}")
    return encoder
