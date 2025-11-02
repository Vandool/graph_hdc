"""
qm9_smiles.py
=============

Light-weight PyG `InMemoryDataset` for QM9-style SMILES lists, aligned with ZincSmiles.

Folder layout expected
----------------------
<root>/
├── raw/
│   ├── train_smile.txt
│   ├── valid_smile.txt
│   └── test_smile.txt
└── processed/          # auto-created on first run

Example
-------
#>>> from pathlib import Path
#>>> ds = QM9Smiles(Path("_datasets/QM9Smiles"), split="train")
#>>> len(ds), ds[0]

Notes
-----
- Atom feature layout is identical to ZincSmiles:
  [atom_type_idx, degree_minus_1, formal_charge_mapped, total_num_Hs]
- QM9 atoms are a strict subset: {H, C, N, O, F}. Hydrogens are often implicit
  in RDKit graphs but are still counted via `GetTotalNumHs()`.
  For the sake of comparability, we consider the Hs implicit analogue to Zinc dataset.

"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import QED, Crippen
from rdkit.Contrib.SA_Score import sascorer
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from src.encoding.graph_encoders import HyperNet
from src.utils.chem import eval_key_from_data
from src.utils.utils import GLOBAL_DATASET_PATH


# ─────────────────────────────── SMILES iterator ──────────────────────────────
def iter_smiles(fp: Path):
    """Yield SMILES strings, skipping an optional header line 'smiles'."""
    with fp.open() as fh:
        for i, line in enumerate(fh):
            if i == 0 and line.strip().lower() == "smiles":
                continue
            if line := line.strip().split()[0]:
                yield line


def _count_smiles_lines(fp: Path) -> int:
    """Count non-empty SMILES lines, skipping an optional first-line header 'smiles'."""
    total = 0
    with fp.open() as fh:
        for i, line in enumerate(fh):
            if i == 0 and line.strip().lower() == "smiles":
                continue
            if line.strip():
                total += 1
    return total


# QM9 atoms: C, N, O, F
QM9_SMILE_ATOM_TO_IDX: dict[str, int] = {
    "C": 0,
    "N": 1,
    "O": 2,
    "F": 3,
}
QM9_SMILE_IDX_TO_ATOM: dict[int, str] = {v: k for k, v in QM9_SMILE_ATOM_TO_IDX.items()}


def largest_ring_size(mol: Chem.Mol) -> int:
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    if not atom_rings:
        return 0
    return max(len(r) for r in atom_rings)


def mol_to_data(mol: Chem.Mol) -> Data:
    """
    Encode an RDKit Mol into a PyG `Data` with features aligned to ZincSmiles.

    Atom features (per node)
    ------------------------
    - atom_type_idx: int in {0..3} for {C,N,O,F} (4 atom types)
    - degree_minus_1: degree-1 mapped to {0,1,2,3,4} (5 degree values)
    - formal_charge_mapped: {0,1,-1} -> {0,1,2} (3 formal charge values)
    - total_num_Hs: explicit+implicit hydrogens in {0,1,2,3,4} (5 hydrogen count values)

    Molecule features (per graph)
    ------------------------
    - smiles: canonical SMILES string
    - eval_smiles: Smiles generated using our decoding pipeline
    - logp: RDKit cLogP (Wildman-Crippen); deterministic for a given SMILES
    """
    x = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in QM9_SMILE_ATOM_TO_IDX:
            raise ValueError(f"Unexpected atom '{sym}' for QM9 encoding.")
        x.append(
            [
                float(QM9_SMILE_ATOM_TO_IDX[sym]),
                float(max(0, atom.GetDegree() - 1)),
                float(atom.GetFormalCharge() if atom.GetFormalCharge() >= 0 else 2),
                float(atom.GetTotalNumHs()),
            ]
        )

    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]

    eval_smiles = eval_key_from_data(
        data=Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor([src, dst], dtype=torch.long),
        ),
        dataset="qm9",
    )
    max_ring_size = largest_ring_size(mol)
    sa_score = sascorer.calculateScore(mol)
    logp = Crippen.MolLogP(mol)
    penalized_logp = float(logp) - float(sa_score) - max(max_ring_size - 6, 0)

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        smiles=Chem.MolToSmiles(mol, canonical=True),
        eval_smiles=eval_smiles,
        logp=torch.tensor([float(logp)], dtype=torch.float32),
        qed=torch.tensor([float(QED.qed(mol))], dtype=torch.float32),
        sa_score=torch.tensor([float(sa_score)], dtype=torch.float32),
        max_ring_size=torch.tensor([float(max_ring_size)], dtype=torch.float32),
        pen_logp=torch.tensor([penalized_logp], dtype=torch.float32),
    )


class QM9Smiles(InMemoryDataset):
    """
    Minimal `InMemoryDataset` that reads ``<split>_smile.txt`` from *root/raw/*
    and caches a collated ``data_<split>.pt`` under *root/processed/*.

    Node Feature Configuration: bins=[4, 5, 3, 5]
    - Atom types: 4 values
      ['C', 'N', 'O', 'F']
    - Degrees: 5 values (after degree-1 transformation)
      {0, 1, 2, 3, 4}
    - Formal Charges: 3 values
      {0, 1, -1} mapped to {0, 1, 2}
    - Total Hs: 5 values
      {0, 1, 2, 3, 4}

    Combinatorial space: 4 × 5 × 3 × 5 = 300 possible node types
    Actual dataset: ~39 unique node types appear in QM9 dataset
    (HyperNet limits codebook to these 39 during decoding)

    Parameters
    ----------
    root:
        Dataset root directory (defaults to ``GLOBAL_DATASET_PATH / "QM9Smiles"``).
    split:
        One of ``{"train","valid","test","simple"}``.
    transform, pre_transform, pre_filter:
        Standard PyG callables.
    enc_suffix:
        Optional suffix for processed filename (e.g., ``_HRR7744``), producing
        ``data_<split>_<enc_suffix>.pt``.
    """

    def __init__(
        self,
        root: str | Path = GLOBAL_DATASET_PATH / "QM9Smiles",
        split: str = "train",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        enc_suffix: str = "",
    ) -> None:
        self.split = split.lower()
        self.enc_suffix = enc_suffix
        assert self.split in {"train", "valid", "test", "simple"}
        super().__init__(root, transform, pre_transform, pre_filter)

        # PyTorch ≥ 2.6 defaults to weights-only unpickler; disable explicitly.
        with open(self.processed_paths[0], "rb") as f:
            self.data, self.slices = torch.load(f, map_location="cpu", weights_only=False)

    # ---------- filenames ------------------------------------------------------
    @property
    def raw_file_names(self) -> list[str]:
        return [f"{self.split}_smile.txt"]

    @property
    def processed_file_names(self) -> list[str]:
        suffix = f"_{self.enc_suffix}" if self.enc_suffix else ""
        return [f"data_{self.split}{suffix}.pt"]

    # ---------- no remote download needed --------------------------------------
    def download(self):
        pass

    # ---------- create `processed/…` -------------------------------------------
    def process(self):
        data_list: list[Data] = []

        src = Path(self.raw_paths[0])
        total = _count_smiles_lines(src)
        for s in tqdm(
            iter_smiles(src),
            total=total,
            desc=f"QM9Smiles[{self.split}]",
            unit="mol",
            dynamic_ncols=True,
        ):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            data = mol_to_data(mol)
            if self.pre_filter and not self.pre_filter(data):
                continue
            if self.pre_transform:
                data = self.pre_transform(data)
            data_list.append(data)

        data, slices = self.collate(data_list)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])


@torch.no_grad()
def precompute_encodings(
    base_ds: QM9Smiles,
    hypernet: HyperNet,
    *,
    normalize: bool = False,
    batch_size: int = 1024,
    device: torch.device | None = None,
    out_suffix: str = "enc",  # writes data_<split>_enc.pt
) -> Path:
    """
    Batch-encode all graphs and write an augmented processed file.

    Parameters
    ----------
    base_ds:
        A `QM9Smiles` dataset instance.
    hypernet:
        Module with `forward(batch)` returning dict keys: ``graph_embedding``, ``node_terms``.
    batch_size:
        Mini-batch size for encoding.
    device:
        Torch device; defaults to CUDA if available else CPU.
    out_suffix:
        Suffix for the augmented processed file.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    loader = DataLoader(base_ds, batch_size=batch_size, shuffle=False)
    hypernet = hypernet.to(device)

    aug: list[Data] = []
    for batch in tqdm(loader, desc=f"encode[{base_ds.split}]", unit="batch", dynamic_ncols=True):
        batch = batch.to(device)
        out = hypernet.forward(batch, normalize=normalize)  # expects batch.batch

        graph_terms = out["graph_embedding"].detach().cpu()
        edge_terms = out["edge_terms"].detach().cpu()
        # node_terms = out["node_terms"].detach().cpu()

        per_graph = batch.to_data_list()
        assert len(per_graph) == graph_terms.size(0)
        for i, d in enumerate(per_graph):
            d = d.clone()
            d.graph_terms = graph_terms[i]  # [Dg]
            d.edge_terms = edge_terms[i]  # [Dg]
            # d.node_terms = node_terms[i]  # [Dn]
            aug.append(d)

    data, slices = InMemoryDataset.collate(aug)
    out_path = Path(base_ds.processed_dir) / f"data_{base_ds.split}_{out_suffix}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((data, slices), out_path)
    return out_path


def copy_graph_edge_terms_from_backup(
    source_root: str | Path = GLOBAL_DATASET_PATH / "QM9Smiles_bk4",
    target_root: str | Path = GLOBAL_DATASET_PATH / "QM9Smiles",
    suffix: str = "QM9SmilesHRR1600F64G1NG3",
) -> None:
    """
    Copy graph_terms and edge_terms from backup dataset files to current dataset files.

    This function reads the precomputed graph_terms and edge_terms from files in
    QM9Smiles_bk4/processed/ and either updates existing files in QM9Smiles/processed/
    or creates new files with the added terms.

    If target file doesn't exist, it will load the base dataset (data_{split}.pt)
    and add the graph_terms and edge_terms to create the new file.

    Parameters
    ----------
    source_root:
        Root directory containing the source files (default: QM9Smiles_bk4)
    target_root:
        Root directory where files will be updated/created (default: QM9Smiles)
    suffix:
        Suffix of the processed files (default: QM9SmilesHRR1600F64G1NG3)

    Example
    -------
    >>> copy_graph_edge_terms_from_backup()  # Uses defaults
    >>> # Or specify custom paths:
    >>> copy_graph_edge_terms_from_backup(
    ...     source_root="/path/to/backup", target_root="/path/to/target", suffix="QM9SmilesHRR1600F64G1NG3"
    ... )
    """
    from pathlib import Path

    source_root = Path(source_root)
    target_root = Path(target_root)

    source_dir = source_root / "processed"
    target_dir = target_root / "processed"

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "valid", "test"]

    for split in splits:
        source_file = source_dir / f"data_{split}_{suffix}.pt"
        target_file = target_dir / f"data_{split}_{suffix}.pt"
        base_file = target_dir / f"data_{split}.pt"  # Fallback to base dataset

        print(f"\n{'=' * 80}")
        print(f"Processing split: {split}")
        print(f"Source: {source_file}")
        print(f"Target: {target_file}")

        if not source_file.exists():
            print(f"⚠️  Source file not found, skipping: {source_file}")
            continue

        # Load source data (weights_only=False for backward compatibility)
        print("Loading source data...")
        with open(source_file, "rb") as f:
            source_data, source_slices = torch.load(f, map_location="cpu", weights_only=False)

        # Verify source has required terms
        if not hasattr(source_data, "graph_terms"):
            print("⚠️  Source data missing 'graph_terms', skipping")
            continue
        if not hasattr(source_data, "edge_terms"):
            print("⚠️  Source data missing 'edge_terms', skipping")
            continue

        # Load target data (or base dataset if target doesn't exist)
        if target_file.exists():
            print("Loading existing target data...")
            with open(target_file, "rb") as f:
                target_data, target_slices = torch.load(f, map_location="cpu", weights_only=False)
        elif base_file.exists():
            print(f"Target file not found. Loading base dataset from: {base_file}")
            with open(base_file, "rb") as f:
                target_data, target_slices = torch.load(f, map_location="cpu", weights_only=False)
            print("Creating new target file with added graph_terms and edge_terms")
        else:
            print("⚠️  Neither target file nor base file found:")
            print(f"    - {target_file}")
            print(f"    - {base_file}")
            print(f"  Skipping split: {split}")
            continue

        # Copy graph_terms and edge_terms
        print("Copying graph_terms and edge_terms...")
        target_data.graph_terms = source_data.graph_terms.clone()
        target_data.edge_terms = source_data.edge_terms.clone()

        # Copy slices if they exist
        if "graph_terms" in source_slices:
            target_slices["graph_terms"] = source_slices["graph_terms"].clone()
        if "edge_terms" in source_slices:
            target_slices["edge_terms"] = source_slices["edge_terms"].clone()

        # Save updated target data
        print(f"Saving target file: {target_file}")
        torch.save((target_data, target_slices), target_file)

        print(f"✓ Successfully copied graph_terms and edge_terms for {split}")
        print(f"  graph_terms shape: {target_data.graph_terms.shape}")
        print(f"  edge_terms shape: {target_data.edge_terms.shape}")

    print(f"\n{'=' * 80}")
    print("✓ All splits processed successfully!")


if __name__ == "__main__":
    train_ds = QM9Smiles(split="train")
    valid_ds = QM9Smiles(split="valid")
    test_ds = QM9Smiles(split="test")
    copy_graph_edge_terms_from_backup()
