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
    """Yield SMILES strings, skipping an optional header line “smiles”."""
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


def largest_ring_size(mol: Chem.Mol) -> int:
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    if not atom_rings:
        return 0
    return max(len(r) for r in atom_rings)


def mol_to_data(mol: Chem.Mol) -> Data:
    """
    Encode an RDKit Mol into a PyG `Data` with ZINC node features.

    Node Feature Configuration: bins=[9, 6, 3, 4, 2]
    - Atom types: 9 values
      ['Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S']
    - Degrees: 6 values (after degree-1 transformation)
      Original degrees: {1, 2, 3, 4, 5, 6} → Transformed: {0, 1, 2, 3, 4, 5}
    - Formal Charges: 3 values
      {0, 1, -1} mapped to {0, 1, 2}
    - Total Hs: 4 values
      {0, 1, 2, 3}
    - Is in Ring: 2 values
      {0 (not in ring), 1 (is in ring)}

    Combinatorial space: 9 × 6 × 3 × 4 × 2 = 1,296 possible node types
    Actual dataset: ~79 unique node types appear in ZINC dataset
    (HyperNet limits codebook to these 79 during decoding)
    """
    x = [
        [
            float(ZINC_SMILE_ATOM_TO_IDX[atom.GetSymbol()]),
            float(max(0, atom.GetDegree() - 1)),  # [1, 2, 3, 4, 5] -> [0, 1, 2, 3, 4]
            float(atom.GetFormalCharge() if atom.GetFormalCharge() >= 0 else 2),  # [0, 1, -1] -> [0, 1, 2]
            float(atom.GetTotalNumHs()),
            float(atom.IsInRing()),
        ]
        for atom in mol.GetAtoms()
    ]
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
        dataset="zinc",
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


class ZincSmiles(InMemoryDataset):
    """
    Minimal `InMemoryDataset` that reads ``<split>_smile.txt`` from *root/raw/*
    and caches a collated ``data_<split>.pt`` under *root/processed/*.

    Pass `transform`, `pre_transform`, or `pre_filter` per the usual PyG API.

    Statistics:
    ---------train----------ZincSmiles(train, enc_suffix='HRR7744')
    .. rubric:: Encoded tensor statistics

    **Node terms**
    - min: -1.0664  max: +1.0895  mean: +0.0008 Median: +0.0006
    **Graph terms**
    - min: -29267.9609  max: +28255.9824  mean: +0.0145 Median: +0.3219


    ---------valid----------ZincSmiles(valid, enc_suffix='HRR7744')
    .. rubric:: Encoded tensor statistics

    **Node terms**
    - min: -0.9017  max: +0.9158  mean: +0.0008 Median: +0.0006
    **Graph terms**
    - min: -25233.6641  max: +25318.3516  mean: +0.0135 Median: +0.3074

    ---------test----------ZincSmiles(test, enc_suffix='HRR7744')
    .. rubric:: Encoded tensor statistics

    **Node terms**
    - min: -0.7992  max: +0.9377  mean: +0.0008 Median: +0.0006
    **Graph terms**
    - min: -16596.3379  max: +16393.8242  mean: +0.0165 Median: +0.3019
    """

    def __init__(
        self,
        root: str | Path = GLOBAL_DATASET_PATH / "ZincSmiles",
        split: str = "train",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        enc_suffix: str = "",
    ):
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
            desc=f"ZincSmiles[{self.split}]",
            unit="mol",
            dynamic_ncols=True,
        ):
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


@torch.no_grad()
def precompute_encodings(
    base_ds: ZincSmiles,
    hypernet: HyperNet,
    *,
    normalize: bool = False,
    batch_size: int = 1024,
    device: torch.device | None = None,
    out_suffix: str = "enc",  # writes data_<split>_enc.pt
) -> Path:
    """Batch-encode all graphs and write an augmented processed file."""
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    loader = DataLoader(base_ds, batch_size=batch_size, shuffle=False)
    hypernet = hypernet.to(device)

    # Collect augmented Data objects
    aug: list[Data] = []
    for batch in tqdm(loader, desc=f"encode[{base_ds.split}]", unit="batch", dynamic_ncols=True):
        batch = batch.to(device)
        out = hypernet.forward(batch, normalize=normalize)  # expects batch.batch

        graph_terms = out["graph_embedding"].detach().cpu()
        # node_terms = out["node_terms"].detach().cpu()
        edge_terms = out["edge_terms"].detach().cpu()

        # Unbatch the underlying Data objects and attach encodings
        # `batch.to_data_list()` returns per-graph Data in batch order
        per_graph = batch.to_data_list()
        assert len(per_graph) == graph_terms.size(0)
        for i, d in enumerate(per_graph):
            d = d.clone()
            # d.node_terms = node_terms[i]  # [Dg]
            d.edge_terms = edge_terms[i]  # [Dg]
            d.graph_terms = graph_terms[i]  # [Dg]
            aug.append(d)

    # Collate and save under a distinct processed filename
    data, slices = InMemoryDataset.collate(aug)
    out_path = Path(base_ds.processed_dir) / f"data_{base_ds.split}_{out_suffix}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((data, slices), out_path)
    return out_path


if __name__ == "__main__":
    train_ds = ZincSmiles(split="train")
    valid_ds = ZincSmiles(split="valid")
    test_ds = ZincSmiles(split="test")
    # simple_ds = ZincSmiles(split="simple")
