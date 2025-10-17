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
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from src.datasets.sa_score import calculateScore
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


def mol_to_data(mol: Chem.Mol) -> Data:
    """
    Encode an RDKit Mol into a PyG `Data` with features aligned to ZincSmiles.

    Atom features (per node)
    ------------------------
    - atom_type_idx: int in {0..4} for {H,C,N,O,F}
    - degree_minus_1: degree-1 mapped to {0,1,2,3,4,...}
    - formal_charge_mapped: {0,1,-1} -> {0,1,2}
    - total_num_Hs: explicit+implicit hydrogens as integer (0..n)

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

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        smiles=Chem.MolToSmiles(mol, canonical=True),
        eval_smiles=eval_smiles,
        logp=torch.tensor([float(Crippen.MolLogP(mol))], dtype=torch.float32),
        qed=torch.tensor([float(QED.qed(mol))], dtype=torch.float32),
        sa_score=torch.tensor([float(calculateScore(mol))], dtype=torch.float32),
    )


class QM9Smiles(InMemoryDataset):
    """
    Minimal `InMemoryDataset` that reads ``<split>_smile.txt`` from *root/raw/*
    and caches a collated ``data_<split>.pt`` under *root/processed/*.

    Atom types size: 4
    Atom types: ['C', 'F', 'N', 'O']
    Degrees size: 5
    Degrees: {0, 1, 2, 3, 4}
    Formal Charges size: 3
    Formal Charges: {0, 1, -1}
    Explicit Hs size: 5
    Explicit Hs: {0, 1, 2, 3, 4}

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
        node_terms = out["node_terms"].detach().cpu()

        per_graph = batch.to_data_list()
        assert len(per_graph) == graph_terms.size(0)
        for i, d in enumerate(per_graph):
            d = d.clone()
            d.graph_terms = graph_terms[i]  # [Dg]
            d.node_terms = node_terms[i]  # [Ni, Dn]
            aug.append(d)

    data, slices = InMemoryDataset.collate(aug)
    out_path = Path(base_ds.processed_dir) / f"data_{base_ds.split}_{out_suffix}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((data, slices), out_path)
    return out_path


if __name__ == "__main__":
    train_ds = QM9Smiles(split="train")
    valid_ds = QM9Smiles(split="valid")
    test_ds = QM9Smiles(split="test")
