import hashlib
import pickle
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
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
    split: Literal["train", "valid", "test", "simple"],
    ds_config: DSHDCConfig | None = None,
    use_no_suffix: bool = False,
    base_dataset: str = "",
) -> InMemoryDataset:
    enc_suffix = ds_config.name if ds_config else ""
    base_dataset = ds_config.base_dataset if ds_config else base_dataset
    if base_dataset == "qm9":
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
    Analyzes a dataset ('qm9', 'zinc', or 'zinc_ring_count') to extract global information about
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
        ds = get_split(split=split, base_dataset=base_dataset)
        print(f"Processing {split} split with {len(ds)} graphs...")

        for data in ds:
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


@dataclass(frozen=True)
class DatasetProps:
    """
    Holds lists of all molecular properties for an entire dataset,
    where each list's index corresponds to the data's index.
    """

    # Properties read directly from the data object
    smiles: list[str]
    logp: list[float]
    qed: list[float]
    sa_score: list[float]
    max_ring_size_data: list[float]  # Renamed to avoid clash
    pen_logp: list[float]

    # Properties calculated from SMILES
    mw: list[float]
    tpsa: list[float]
    num_atoms: list[int]
    num_bonds: list[int]
    num_rings: list[int]
    num_rotatable_bonds: list[int]
    num_hba: list[int]
    num_hbd: list[int]
    num_aliphatic_rings: list[int]
    num_aromatic_rings: list[int]
    max_ring_size_calc: list[int]  # Renamed to avoid clash
    bertz_ct: list[float]

    def __len__(self):
        # All lists should have the same length
        return len(self.smiles)


def calculate_molecular_properties(mol) -> dict[str, float | int]:
    """
    Calculate comprehensive molecular properties for evaluation.
    (Cleaned to map directly to DatasetProps attributes)

    Args:
        mol: RDKit molecule object (must be valid, not None)

    Returns:
        Dictionary of property names to values

    Raises:
        AttributeError: If mol is None or invalid
        RuntimeError: If any property calculation fails
    """
    # Import here to avoid circular import
    from src.generation.evaluator import rdkit_max_ring_size

    if mol is None:
        raise ValueError("Cannot calculate properties for None molecule")

    props = {
        "mw": Descriptors.MolWt(mol),
        "tpsa": Descriptors.TPSA(mol),
        "num_atoms": mol.GetNumAtoms(),
        "num_bonds": mol.GetNumBonds(),
        "num_rings": Chem.Descriptors.RingCount(mol),
        "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "num_hba": Chem.Descriptors.NumHAcceptors(mol),
        "num_hbd": Chem.Descriptors.NumHDonors(mol),
        "num_aliphatic_rings": Descriptors.NumAliphaticRings(mol),
        "num_aromatic_rings": Chem.Descriptors.NumAromaticRings(mol),
        "max_ring_size_calc": rdkit_max_ring_size(mol),
        "bertz_ct": Descriptors.BertzCT(mol),
    }
    return props


def get_dataset_props(base_dataset: BaseDataset, splits: list[str] | None = None) -> DatasetProps:
    """
    Analyzes a dataset ('qm9' or 'zinc') to extract lists of all
    molecular properties, preserving the dataset order.

    Results are cached as a raw dictionary mapping:
    property_name -> list_of_values,
    which matches the DatasetProps dataclass structure.
    """

    # --- 1. Setup and Cache Check ---

    if base_dataset == "qm9":
        dataset_cls = QM9Smiles
    elif base_dataset == "zinc":
        dataset_cls = ZincSmiles
    else:
        raise ValueError(f"Unknown base_dataset: {base_dataset}")

    if not splits:
        splits = ["train"]

    split_signature = "_".join(splits)
    dataset_props_file = Path(dataset_cls(split="test").processed_dir) / f"dataset_props_{split_signature}.pkl"

    # 2) Return them as the class when requested if we have them saved
    if dataset_props_file.is_file():
        print(f"Loading existing dataset props from {dataset_props_file}...")
        try:
            with open(dataset_props_file, "rb") as f:
                # Load the raw dictionary: {'mw': [v1, v2, ...], ...}
                info_dict = pickle.load(f)

            # Check if it's a dict, has a key, and the value is a list
            if isinstance(info_dict, dict) and "mw" in info_dict and isinstance(info_dict["mw"], list):
                # Cast the loaded dictionary directly to the DatasetProps object
                return DatasetProps(**info_dict)
            print("Warning: Cached file is corrupted or not a valid dict. Regenerating...")
        except (pickle.UnpicklingError, EOFError, TypeError) as e:
            print(f"Warning: Could not load cache file ({e}). Regenerating...")

    # --- 2. Initialization (if not saved) ---

    print("Generating new dataset properties (property lists)...")
    # We use a defaultdict of lists to store the property lists
    all_props_lists = defaultdict(list)

    # Get all expected attribute names from the dataclass definition
    # This ensures all lists are created, even if empty
    all_expected_keys = DatasetProps.__annotations__.keys()
    for key in all_expected_keys:
        all_props_lists[key] = []

    # --- 3. Iterate over all splits and graphs ---

    total_graphs = 0
    for split in splits:
        ds = get_split(split=split, base_dataset=base_dataset)
        print(f"Processing {split} split with {len(ds)} graphs...")

        for i in range(len(ds)):
            data = ds[i]

            # --- A. Read properties directly from data object ---
            try:
                all_props_lists["smiles"].append(data.smiles)
                all_props_lists["logp"].append(data.logp.item())
                all_props_lists["qed"].append(data.qed.item())
                all_props_lists["sa_score"].append(data.sa_score.item())
                # Renamed key to avoid clash with calculated one
                all_props_lists["max_ring_size_data"].append(data.max_ring_size.item())
                all_props_lists["pen_logp"].append(data.pen_logp.item())
            except (AttributeError, TypeError) as e:
                print(f"Error reading pre-computed prop for {data.smiles}: {e}. Skipping.")
                # We can't continue if pre-computed props fail, as lists
                # would go out of sync. Or we could append None.
                # For now, let's skip this graph entirely.
                continue

            # --- B. Calculate other properties from SMILES ---
            mol = Chem.MolFromSmiles(data.smiles)
            if mol is None:
                print(f"Warning: Could not parse SMILES: {data.smiles}. Skipping this molecule.")
                # Remove the pre-computed properties we just added (keep lists in sync)
                all_props_lists["smiles"].pop()
                all_props_lists["logp"].pop()
                all_props_lists["qed"].pop()
                all_props_lists["sa_score"].pop()
                all_props_lists["max_ring_size_data"].pop()
                all_props_lists["pen_logp"].pop()
                continue

            # Get calculated props
            try:
                calc_props = calculate_molecular_properties(mol)
            except Exception as e:
                print(f"Error calculating properties for {data.smiles}: {e}. Skipping this molecule.")
                # Remove the pre-computed properties we just added (keep lists in sync)
                all_props_lists["smiles"].pop()
                all_props_lists["logp"].pop()
                all_props_lists["qed"].pop()
                all_props_lists["sa_score"].pop()
                all_props_lists["max_ring_size_data"].pop()
                all_props_lists["pen_logp"].pop()
                continue

            for key, value in calc_props.items():
                all_props_lists[key].append(value)

            total_graphs += 1

    print(f"Processed a total of {total_graphs} graphs.")

    # 3) Save them as python dict

    # Convert defaultdict to a standard dict for saving
    final_props_dict = dict(all_props_lists)

    print(f"Saving new dataset props (as dict of lists) to {dataset_props_file}...")
    dataset_props_file.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_props_file, "wb") as f:
        pickle.dump(final_props_dict, f)

    return DatasetProps(**final_props_dict)


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

    qm9_props = get_dataset_props("qm9")
    zinc_props = get_dataset_props("zinc")
    print(qm9_props)
    print(zinc_props)
