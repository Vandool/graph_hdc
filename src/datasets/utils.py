import hashlib
import pickle
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset

from src.datasets.qm9_smiles_generation import QM9Smiles
from src.datasets.zinc_smiles_generation import ZincSmiles
from src.encoding.configs_and_constants import BaseDataset, DSHDCConfig


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


class Compose:
    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(self, data: Data) -> Data:
        for t in self.transforms:
            data = t(data)
        return data


def get_split(
    split: Literal["train", "valid", "test", "simple"], ds_config: DSHDCConfig, use_no_suffix: bool = False
) -> InMemoryDataset:
    enc_suffix = ds_config.name if not use_no_suffix else ""
    if ds_config.base_dataset == "qm9":
        ds = QM9Smiles(split=split, enc_suffix=enc_suffix)

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
    return ZincSmiles(split=split, enc_suffix=enc_suffix)


@dataclass
class DatasetInfo:
    """Holds aggregated information about an entire graph dataset."""

    # Set of all the features appearing in the whole Dataset
    node_features: set[tuple]
    # Set of all the edge tuples appearing in the whole dataset
    # Stored as sorted( (feat_u, feat_v) ) for undirected graphs
    edge_features: set[tuple[tuple, tuple]]
    # Ring Histogram. Features -> ring size -> count
    ring_histogram: dict[tuple, dict[int, int]] | None
    # Single Ring set: These features appear only once in a ring, never part of multiple rings
    single_ring_features: set[tuple] | None


def get_dataset_info(base_dataset: BaseDataset) -> DatasetInfo:
    """
    Analyzes a dataset ('qm9' or 'zinc') to extract global information about
    node features, edge features, and ring structures (for ZINC).

    Results are cached as a raw dictionary to the dataset's processed_dir.
    """

    # --- 1. Setup and Cache Check ---

    if base_dataset == "qm9":
        dataset_cls = QM9Smiles
    elif base_dataset == "zinc":
        dataset_cls = ZincSmiles
    else:
        raise ValueError(f"Unknown base_dataset: {base_dataset}")

    dataset_info_file = Path(dataset_cls(split="test").processed_dir) / "dataset_info.pkl"

    # First try to read the raw dictionary if we've already saved it
    if dataset_info_file.is_file():
        print(f"Loading existing dataset info from {dataset_info_file}...")
        try:
            with open(dataset_info_file, "rb") as f:
                # Load the raw dictionary
                info_dict = pickle.load(f)

            # Check if it's a dict and has at least one expected key
            if isinstance(info_dict, dict) and "node_features" in info_dict:
                # Cast the loaded dictionary to the DatasetInfo object
                return DatasetInfo(**info_dict)
            print("Warning: Cached file is corrupted or not a dict. Regenerating...")
        except (pickle.UnpicklingError, EOFError, TypeError) as e:
            # TypeError catches errors during unpacking (e.g., missing keys)
            print(f"Warning: Could not load cache file ({e}). Regenerating...")

    # --- 2. Initialization ---

    print("Generating new dataset info...")
    never_multiple_rings_counter = Counter()
    atom_tuple_total_counts = Counter()
    ring_histogram = defaultdict(Counter)
    edge_features = set()
    node_features = set()

    # --- 3. Iterate over all splits and graphs ---

    for split in ["train", "test", "valid"]:
        ds = dataset_cls(split=split)
        print(f"Processing {split} split with {len(ds)} graphs...")

        for data in ds:
            if "." in data.smiles:
                print(f"Broken smiles found: {data.smiles}. Skipping ...")
                continue

            # 3a. Get Node Features
            current_node_features = {tuple(feat.tolist()) for feat in data.x.int()}
            node_features.update(current_node_features)

            # 3b. Get Edge Features
            node_idx_to_tuple = {i: tuple(data.x[i].int().tolist()) for i in range(data.x.size(0))}

            for u, v in data.edge_index.T.tolist():
                feat_u = node_idx_to_tuple[u]
                feat_v = node_idx_to_tuple[v]
                edge_tuple = tuple(sorted((feat_u, feat_v)))
                edge_features.add(edge_tuple)

            # 3c. Get Ring Information (ZINC only)
            if base_dataset == "qm9":
                continue

            mol = Chem.MolFromSmiles(data.smiles)
            if mol is None:
                print(f"Warning: Could not parse SMILES: {data.smiles}")
                continue

            instance_multiple_rings = set()
            atom_tuples_in_rings = {}

            # First pass: Populate histogram and find multi-ring atoms
            for atom in mol.GetAtoms():
                if atom.IsInRing():
                    atom_idx = atom.GetIdx()
                    if atom_idx >= len(node_idx_to_tuple):
                        continue

                    atom_tuple = node_idx_to_tuple[atom_idx]
                    atom_tuples_in_rings[atom_idx] = atom_tuple
                    atom_tuple_total_counts[atom_tuple] += 1

                    ring_count = 0
                    for ring_size in range(3, 21):
                        if atom.IsInRingSize(ring_size):
                            ring_histogram[atom_tuple][ring_size] += 1
                            ring_count += 1

                    if ring_count > 1:
                        instance_multiple_rings.add(atom_tuple)

            # Second pass
            for atom_tuple in atom_tuples_in_rings.values():
                if atom_tuple not in instance_multiple_rings:
                    never_multiple_rings_counter[atom_tuple] += 1

    # --- 4. Finalize and Save Results ---

    final_ring_histogram: dict[tuple, dict[int, int]] | None = None
    single_ring_features: set[tuple] | None = None

    if base_dataset == "zinc":
        single_ring_features = set()
        for atom_tuple, total_count in atom_tuple_total_counts.items():
            if never_multiple_rings_counter[atom_tuple] == total_count:
                single_ring_features.add(atom_tuple)

        final_ring_histogram = {k: dict(v) for k, v in ring_histogram.items()}

    # Create the final dictionary to be saved
    saved_dict = {
        "node_features": node_features,
        "edge_features": edge_features,
        "ring_histogram": final_ring_histogram,
        "single_ring_features": single_ring_features,
    }

    # Save the raw dictionary in the processed dir
    print(f"Saving new dataset info (as dict) to {dataset_info_file}...")
    dataset_info_file.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_info_file, "wb") as f:
        pickle.dump(saved_dict, f)

    # Return the object casted to DatasetInfo
    return DatasetInfo(**saved_dict)


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

if __name__ == "__main__":
    dataset_info_qm9 = get_dataset_info("qm9")
    dataset_info_zinc = get_dataset_info("zinc")
    print(dataset_info_qm9)
    print(dataset_info_zinc)
