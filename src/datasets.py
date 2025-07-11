import random
from typing import Any, Optional

import networkx as nx
import torch
from torch_geometric.data import Batch, Data, InMemoryDataset
from torch_geometric.utils import degree, to_undirected

from graph_hdc.utils import AbstractEncoder
from src.encoding.pca_encoder import PCAEncoder

IDX2COLOR = {
    0: ("R", "red"),
    1: ("G", "green"),
    2: ("B", "blue"),
    3: ("W", "white"),
    4: ("X", "gray"),
}

COLOR2IDX = {color_word: idx for idx, (color_char, color_word) in IDX2COLOR.items()}


class HashableData(Data):
    def __eq__(self, other):
        if not isinstance(other, Data):
            return False
        return torch.equal(self.x, other.x) and torch.equal(self.edge_index, other.edge_index)

    def __hash__(self):
        x_hash = hash(self.x.cpu().numpy().tobytes())
        ei_hash = hash(self.edge_index.cpu().numpy().tobytes())
        return hash((x_hash, ei_hash))


torch.serialization.add_safe_globals([HashableData])


class ColorGraphDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        num_graphs=100,
        min_nodes=5,
        max_nodes=10,
        edge_p=0.15,
        transform=None,
        pre_transform=None,
        *,
        use_rgb=True,
    ):
        self.num_graphs, self.min_nodes, self.max_nodes = num_graphs, min_nodes, max_nodes
        self.edge_p = edge_p
        self.use_rgb = use_rgb
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        motif_centric_collections = []
        motif_centrics = set()
        motifs = [self.triangle_motif, self.ring_motif]
        for _, motif_func in enumerate(motifs):
            # Ensure Evenly distributed motifs
            while len(motif_centrics) < self.num_graphs // len(motifs):
                base_colors, base_edges = motif_func()
                extra_nodes = random.randint(1, 5)
                total_colors, total_edges = self.expand_graph_planar(base_colors, base_edges, extra_nodes)

                # Add node features
                x = torch.tensor([COLOR2IDX[c] for c in total_colors], dtype=torch.float)

                # Add edges
                edge_index = torch.tensor(total_edges, dtype=torch.long).t().contiguous()
                edge_index = to_undirected(edge_index, num_nodes=len(total_colors))

                data = HashableData(x=x, edge_index=edge_index)
                data.validate(raise_on_error=True)
                motif_centrics.add(data)

            motif_centric_collections.append(motif_centrics)
            motif_centrics = set()

        motif_centric_collections = [g for mfs in motif_centric_collections for g in mfs]
        if self.pre_filter is not None:
            motif_centric_collections = [
                graph for part in motif_centric_collections for graph in part if self.pre_filter(graph)
            ]
        if self.pre_transform is not None:
            motif_centric_collections = [self.pre_transform(d) for d in motif_centric_collections]

        self.save(motif_centric_collections, self.processed_paths[0])

    @staticmethod
    def ring_motif() -> tuple[list[str], list[tuple]]:
        colors = ["red", "green", "red", "green", "red"]
        nodes = list(range(5))
        edges = [(i, (i + 1) % 5) for i in nodes]
        return colors, edges

    @staticmethod
    def triangle_motif() -> tuple[list[str], list[tuple]]:
        colors = ["red", "green", "blue"]
        edges = [(0, 1), (1, 2), (2, 0)]
        return colors, edges

    @staticmethod
    def expand_graph_planar(base_colors, base_edges, num_extra_nodes, max_attempts=100) -> tuple[list, list]:
        n0 = len(base_colors)
        G = nx.Graph()
        G.add_nodes_from(range(n0))
        G.add_edges_from(base_edges)

        colors = base_colors[:]
        edges = base_edges[:]

        for i in range(num_extra_nodes):
            new_node_id = n0 + i
            G.add_node(new_node_id)
            new_color = random.choice(list(set(COLOR2IDX.keys()).difference(set(base_colors))))
            colors.append(new_color)

            attempts = 0
            while attempts < max_attempts:
                # Connect to 1–3 existing nodes
                k = random.randint(1, min(3, n0 + i))
                targets = random.sample(sorted(G.nodes - {new_node_id}), k)
                trial_edges = [(new_node_id, t) for t in targets]

                G.add_edges_from(trial_edges)
                is_planar, _ = nx.check_planarity(G)
                if is_planar:
                    edges.extend(trial_edges)
                    break
                G.remove_edges_from(trial_edges)
                attempts += 1

        return colors, edges


class AddNodeDegree:
    """
    A PyG style transform that computes each node's (undirected) degree
    and appends it as an extra feature column to data.x.
    """

    def __call__(self, data: Data) -> Data:
        # data.edge_index: shape [2, num_edges]
        # data.num_nodes: number of nodes in this graph
        # data.x: existing node features, shape [num_nodes, num_orig_features]

        # 1) Compute node degree.  We’ll treat the graph as undirected:
        #    - edge_index[0] = source nodes
        #    - edge_index[1] = target nodes
        #    If the graph is already undirected, you can choose either row.
        row, col = data.edge_index

        # degree(col) counts how many times each node appears as a "destination".
        # For an undirected graph, you typically want degree = in‐degree + out‐degree.
        # If edge_index is symmetric (i↔j appears twice), then degree(col) already equals full degree.
        # If edge_index is not symmetric, you can do degree(row) + degree(col) to get full undirected degree.
        deg_out = degree(row, data.num_nodes, dtype=torch.float)
        deg_in = degree(col, data.num_nodes, dtype=torch.float)
        # Since the undirected edges would count twice
        node_deg = (deg_out + deg_in) // 2

        # 2) Turn that into a column vector of shape [num_nodes, 1]
        node_deg = node_deg.view(-1, 1)  # ensures shape [num_nodes, 1]

        # 3) If data.x doesn’t exist (maybe your graphs have no node features to begin with),
        #    create data.x = node_deg.  Otherwise, concatenate onto the existing feature matrix.
        if data.x is None:
            data.x = node_deg
        else:
            # data.x: [num_nodes, orig_dim]
            data.x = torch.cat([data.x, node_deg], dim=1)  # → shape [num_nodes, orig_dim+1]

        return data


import torch
from torch.utils.data import Dataset


class EncodedPCADataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        graph_encoder: AbstractEncoder,
        pca_encoder: PCAEncoder | None = None,
    ):
        """
        Wraps a base PyG dataset, encodes each graph via `graph_encoder`, then
        optionally reduces its embedding via `pca_encoder`.

        :param base_dataset: any torch Dataset yielding PyG Data objects
        :param graph_encoder: object with .forward(data=Batch) → dict of tensors
        :param pca_encoder: optional PCAEncoder; if None, PCA is skipped
        """
        self.base_dataset = base_dataset
        self.graph_encoder = graph_encoder
        self.pca_encoder = pca_encoder

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # 1) get a single graph and batch it
        data = self.base_dataset[idx]
        batch = Batch.from_data_list([data])

        # 2) encode node/edge/graph terms
        res = self.graph_encoder.forward(data=batch)
        # squeeze out the batch dim
        node_terms = res["node_terms"].squeeze(0)
        edge_terms = res["edge_terms"].squeeze(0)
        graph_embedding = res["graph_embedding"].squeeze(0)

        # 3) stack into [3, D] tensor
        x = torch.stack([node_terms, edge_terms, graph_embedding], dim=0)

        # 4) optionally apply PCAEncoder
        if self.pca_encoder is not None:
            # PCAEncoder.transform returns a tensor of shape [..., new_dim]
            # Here x has shape [3, D] → we want [3, D_pca]
            x_reduced = self.pca_encoder.transform(x)
            return x_reduced

        # 5) fallback: return raw stack
        return x
