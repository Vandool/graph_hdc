import hashlib
import random
from collections.abc import Callable
from typing import Literal

import networkx as nx
import torch
from torch.utils.data import Subset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import degree, k_hop_subgraph, to_undirected

from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import DSHDCConfig

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
        # For an undirected graph, you typically want degree = in‐degree plus out‐degree.
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


def stable_hash(tensor: torch.Tensor, bins: int) -> int:
    """
    Map a feature tensor to a stable integer in [0, bins-1], such that small changes in features produce different
    (but deterministic) outputs. This is better than a naive .sum() since it’s less prone to collisions.

    :param tensor:
    :param bins:
    :return:
    """
    byte_str = tensor.numpy().tobytes()
    h = hashlib.sha256(byte_str).hexdigest()
    return int(h, 16) % bins


class AddNeighbourhoodEncodings:
    """
    A PyG-style transform that adds neighborhood encoding to data.x.
    Each node receives a hash derived from the summed features of its k-hop neighbors,
    hashed and modded into `bins` buckets to provide permutation-invariant node IDs.
    """

    def __init__(self, depth: int = 3, bins: int = 3):
        self.depth = depth
        self.bins = bins

    def __call__(self, data: Data) -> Data:
        x = data.x
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        hash_features = []
        for node_idx in range(num_nodes):
            node_ids, _, _, _ = k_hop_subgraph(node_idx, self.depth, edge_index, relabel_nodes=False)
            # Remove self
            mask = node_ids != node_idx
            node_ids = node_ids[mask]
            # neighbor_feats = x[node_ids].sum(dim=0)  # Aggregate neighborhood
            # We're hashing
            neighbor_feats = x[node_ids]  # Aggregate neighborhood
            hashed_value = stable_hash(neighbor_feats.cpu(), self.bins)
            hash_features.append([hashed_value])

        nha = torch.tensor(hash_features, dtype=torch.float32)  # [num_nodes, 1]

        data.x = torch.cat([data.x, nha], dim=1)

        return data


class Compose:
    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(self, data: Data) -> Data:
        for t in self.transforms:
            data = t(data)
        return data


def get_split(split: Literal["train", "valid", "test", "simple"], ds_config: DSHDCConfig):
    if ds_config.base_dataset == "qm9":
        ds = QM9Smiles(split=split, enc_suffix=ds_config.name)

        # --- Filter known disconnected molecules ---
        if split == "train":
            disconnected_graphs_idxs = set(qm9_train_dc_list)
        elif split == "valid":
            disconnected_graphs_idxs = set(qm9_valid_dc_list)
        elif split == "test":
            disconnected_graphs_idxs = set(qm9_test_dc_list)
        else:
            disconnected_graphs_idxs = set()

        if disconnected_graphs_idxs:
            keep_idx = [i for i in range(len(ds)) if i not in disconnected_graphs_idxs]
            ds = ds.index_select(keep_idx)
            print(
                f"[QM9:{split}] filtered {len(disconnected_graphs_idxs)} disconnected molecules → kept {len(keep_idx)}"
            )

        return ds

    return ZincSmiles(split=split, enc_suffix=ds_config.name)


qm9_train_dc_list = [
    103,
    1251,
    1593,
    1851,
    1995,
    2295,
    4099,
    4531,
    5216,
    5221,
    5834,
    6145,
    8286,
    8491,
    8949,
    9125,
    9999,
    11232,
    12131,
    12542,
    12740,
    13217,
    13876,
    14195,
    14485,
    14558,
    16087,
    16570,
    17058,
    17153,
    17628,
    17836,
    17909,
    18422,
    18466,
    18561,
    18971,
    19381,
    19426,
    19564,
    19832,
    19974,
    20572,
    20809,
    20834,
    21226,
    21576,
    22407,
    24078,
    24171,
    25407,
    25458,
    25886,
    26227,
    26466,
    26496,
    26944,
    27140,
    27460,
    27518,
    27741,
    30253,
    30839,
    32067,
    32967,
    33555,
    34331,
    34804,
    35030,
    35529,
    35781,
    36068,
    36764,
    37067,
    37070,
    37358,
    37987,
    38571,
    41050,
    41652,
    41713,
    41962,
    43185,
    44361,
    44818,
    45095,
    45294,
    45322,
    45984,
    46272,
    46345,
    46633,
    47233,
    47950,
    48911,
    48936,
    50163,
    51300,
    51823,
    52411,
    52847,
    53366,
    53487,
    53862,
    55836,
    56449,
    58667,
    59069,
    59243,
    61063,
    61196,
    61961,
    62382,
    63267,
    63276,
    64004,
    64281,
    64647,
    64868,
    64974,
    65499,
    66551,
    66632,
    66768,
    67243,
    69640,
    70128,
    70464,
    71456,
    72377,
    72457,
    72630,
    74138,
    76129,
    76215,
    76336,
    76437,
    76641,
    77011,
    77281,
    77298,
    77547,
    78068,
    78611,
    78709,
    80060,
    82092,
    83239,
    83577,
    83666,
    83778,
    85104,
    85956,
    87216,
    87461,
    88635,
    88957,
    89248,
    90275,
    90377,
    92239,
    92292,
    93117,
    94613,
    95209,
    95255,
    97026,
    97440,
    97717,
    98032,
    98115,
    98344,
    99317,
    99326,
    100052,
    101280,
    101519,
    101830,
    102806,
    103334,
    104274,
    104781,
    104876,
    106238,
    106269,
    106490,
    106619,
    107151,
    107274,
    107502,
    109765,
    110736,
    113316,
    115211,
    115226,
    115757,
    116116,
    117135,
    117266,
    117800,
    117948,
    118700,
]
qm9_valid_dc_list = [
    1242,
    1407,
    1570,
    1950,
    2256,
    2286,
    2574,
    2899,
    2950,
    3681,
    3955,
    3969,
    4134,
    4147,
    4182,
    5702,
    5838,
    6375,
    7791,
]
qm9_test_dc_list = [489, 1097, 1495, 1757, 1988, 2532, 4164, 4738]
