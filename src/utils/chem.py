from typing import Literal

import networkx as nx
import torch
from IPython.display import display
from rdkit import Chem
from rdkit.Chem import SanitizeFlags
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import Data

from src.utils.utils import DataTransformer

QM9_SMILE_ATOM_TO_IDX: dict[str, int] = {
    "C": 0,
    "N": 1,
    "O": 2,
    "F": 3,
}

BaseDataset = Literal["zinc", "qm9"]


def draw_mol(
    mol: Chem.Mol,
    save_path: str | None = None,
    size: tuple[int, int] = (300, 300),
    fmt: str = "svg",
) -> None:
    """
    Draw an RDKit molecule with appealing style (like Jupyter's display).

    Parameters
    ----------
    mol : Chem.Mol
        Molecule to render.
    save_path : str | None
        If provided, saves the drawing to this path.
        - Use '.svg' for vector output
        - Use '.png' for raster output
    size : tuple[int, int]
        Width, height in pixels.
    fmt : str
        "svg" (vector, pretty) or "png" (bitmap).
    """
    if fmt == "svg":
        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    elif fmt == "png":
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    else:
        raise ValueError(f"Unsupported fmt {fmt}, choose 'svg' or 'png'.")

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    data = drawer.GetDrawingText()

    if fmt == "svg":
        if save_path:
            with open(save_path, "w") as f:
                f.write(data)
    elif save_path:
        with open(save_path, "wb") as f:
            f.write(data)
    else:
        # inline display for PNG in notebooks
        from IPython.display import Image

        display(Image(data=data))


def mol_to_data(mol: Chem.Mol) -> Data:
    x = [
        [
            float(QM9_SMILE_ATOM_TO_IDX[atom.GetSymbol()]),
            float(max(0, atom.GetDegree() - 1)),  # [1, 2, 3, 4, 5] -> [0, 1, 2, 3, 4]
            float(atom.GetFormalCharge() if atom.GetFormalCharge() >= 0 else 2),  # [0, 1, -1] -> [0, 1, 2]
            float(atom.GetTotalNumHs()),
        ]
        for atom in mol.GetAtoms()
    ]
    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]

    # Make a sanitized, canonical reference SMILES for storage
    m_ref = Chem.Mol(mol)  # copy to avoid mutating caller
    # (Optional but common) Clear any explicit H baggage in the textual representation:
    m_ref = Chem.RemoveHs(m_ref)
    # Ensure sanitization; MolFromSmiles does this too, but do it explicitly if you prefer:
    Chem.SanitizeMol(m_ref)

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        smiles=Chem.MolToSmiles(mol, canonical=True),
        smiles_canon=Chem.MolToSmiles(m_ref, canonical=True, isomericSmiles=True),
    )


def reconstruct_for_eval(nx: nx.Graph, *, dataset="qm9"):
    # Variant A: aromatic flags kept (preferred for QM9-like SMILES)
    try:
        mol, _ = DataTransformer.nx_to_mol_v2(nx, dataset=dataset, infer_bonds=True, sanitize=True, kekulize=False)
        if is_valid_molecule(mol):
            return mol
    except Exception:
        pass

    # Variant B: allow Kekulé localization (sometimes required around N–O / fused rings)
    try:
        mol, _ = DataTransformer.nx_to_mol_v2(nx, dataset=dataset, infer_bonds=True, sanitize=True, kekulize=True)
        if is_valid_molecule(mol):
            return mol
    except Exception:
        pass

    # Variant C: partial sanitize → set aromaticity first, then full sanitize
    # Build unsanitized, then drive sanitize ops explicitly
    mol, _ = DataTransformer.nx_to_mol_v2(nx, dataset=dataset, infer_bonds=True, sanitize=False, kekulize=False)
    # light ops first
    Chem.SanitizeMol(
        mol,
        sanitizeOps=(
            SanitizeFlags.SANITIZE_CLEANUP | SanitizeFlags.SANITIZE_SYMMRINGS | SanitizeFlags.SANITIZE_SETAROMATICITY
        ),
    )
    # now full pass (will no-op if OK)
    Chem.SanitizeMol(mol)
    return mol


def canonical_key(mol: Chem.Mol) -> str:
    """
    Stable string key for uniqueness/novelty.
    - remove Hs for a text key
    - keep aromatic notation (not kekulé)
    - canonical+isomeric
    """
    m = Chem.RemoveHs(Chem.Mol(mol))
    Chem.SanitizeMol(m)  # idempotent if already sanitized
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True, kekuleSmiles=False)


def is_valid_molecule(mol: Chem.Mol) -> bool:
    # Return True if all sanitize ops succeed and mol is not empty
    if mol.GetNumAtoms() == 0:
        return False
    err = Chem.SanitizeMol(mol, catchErrors=True)
    return err == SanitizeFlags.SANITIZE_NONE


def eval_key_from_data(data: Data, dataset: BaseDataset) -> str:
    return eval_key_from_nx(nx_g=DataTransformer.pyg_to_nx(data), dataset=dataset)


def eval_key_from_nx(nx_g: nx.Graph, dataset: BaseDataset) -> str:
    mol = reconstruct_for_eval(nx_g, dataset=dataset)  # infer_bonds=True, sanitize=True, kekulize=False
    assert is_valid_molecule(mol), f"mol is invalid: {mol}"
    return canonical_key(mol)  # RemoveHs + canonical, isomeric, aromatic
