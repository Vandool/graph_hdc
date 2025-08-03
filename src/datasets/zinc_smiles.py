"""
zinc_smiles.py
==============

Light-weight PyG `InMemoryDataset` for ZINC-style SMILES lists.

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
#>>> ds = ZincSmiles(Path("_datasets/ZincSmiles"), split="train")
#>>> len(ds), ds[0]
"""


from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import torch
from rdkit import Chem
from rdkit.Chem import Atom, ValenceType
from torch_geometric.data import Data, InMemoryDataset

from src.utils.utils import GLOBAL_DATASET_PATH


# ─────────────────────────────── SMILES iterator ──────────────────────────────
def iter_smiles(fp: Path):
    """Yield SMILES strings, skipping an optional header line “smiles”."""
    with fp.open() as fh:
        for i, line in enumerate(fh):
            if i == 0 and line.strip().lower() == "smiles":
                continue
            if line := line.strip().split()[0]:
                yield line

ZINC_SMILE_ATOM_TO_IDX: dict[str, int] = {
    "Br": 0,
    "C": 1,
    "Cl": 2,
    "F": 3,
    "I": 4,
    "N": 5,
    "O": 6,
    "P": 7,
    "S": 8,
}
ZINC_SMILE_IDX_TO_ATOM: dict[int, str] = {v: k for k, v in ZINC_SMILE_ATOM_TO_IDX.items()}

def mol_to_data(mol: Chem.Mol) -> Data:
    """Very small node/edge feature recipe, tweak as you like."""
    x = [
        [
            float(ZINC_SMILE_ATOM_TO_IDX[atom.GetSymbol()]),
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            float(atom.GetTotalNumHs()),
        ]
        for atom in mol.GetAtoms()
    ]
    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        smiles=Chem.MolToSmiles(mol, canonical=True),
    )

class ZincSmiles(InMemoryDataset):
    """
    Minimal `InMemoryDataset` that reads ``<split>_smile.txt`` from *root/raw/*
    and caches a collated ``data_<split>.pt`` under *root/processed/*.

    Pass `transform`, `pre_transform`, or `pre_filter` per the usual PyG API.
    """

    def __init__(
        self,
        root: str | Path = GLOBAL_DATASET_PATH / "ZincSmiles",
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.split = split.lower()
        assert self.split in {"train", "valid", "test"}
        super().__init__(root, transform, pre_transform, pre_filter)

        # PyTorch ≥ 2.6 defaults to weights-only un-pickler → disable explicitly
        with open(self.processed_paths[0], "rb") as f:
            self.data, self.slices = torch.load(f, map_location="cpu", weights_only=False)

    # ---------- filenames ------------------------------------------------------
    @property
    def raw_file_names(self) -> list[str]:
        return [f"{self.split}_smile.txt"]

    @property
    def processed_file_names(self) -> list[str]:
        return [f"data_{self.split}.pt"]

    # ---------- no remote download needed --------------------------------------
    def download(self):
        pass

    # ---------- create `processed/…` -------------------------------------------
    def process(self):
        data_list: list[Data] = []

        for s in iter_smiles(Path(self.raw_paths[0])):
            if (mol := Chem.MolFromSmiles(s)) is None:
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

if __name__ == '__main__':
    train_ds = ZincSmiles(split="train")
    valid_ds = ZincSmiles(split="valid")
    test_ds = ZincSmiles(split="test")

    print(len(train_ds), len(valid_ds), len(test_ds))