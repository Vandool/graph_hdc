from dataclasses import dataclass

import networkx as nx
import torch
from rdkit import Chem
from rdkit.Chem import QED, SanitizeFlags, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import Data

from src.encoding.configs_and_constants import BaseDataset
from src.utils.utils import DataTransformer

QM9_SMILE_ATOM_TO_IDX: dict[str, int] = {
    "C": 0,
    "N": 1,
    "O": 2,
    "F": 3,
}


def draw_mol(
    mol: Chem.Mol,
    save_path: str | None = None,
    size: tuple[int, int] = (300, 300),
    fmt: str = "svg",
    # --- Tweaked defaults for "Pretty but Colorful" ---
    bond_width: float = 2.0,  # Thicker lines (standard is ~1.0)
    bw_palette: bool = False,  # KEPT FALSE to keep colors!
    font_size: int = 14,  # Readable font size
    transparent: bool = True,  # Transparent background
) -> None:
    """
    Draw an RDKit molecule with publication-quality styling (Thick lines, High Res).
    Keeps standard element colors (O=Red, N=Blue) by default.
    """
    if mol is None:
        return

    # 1. Layout: Ensure molecule isn't "squashed"
    try:
        if mol.GetNumConformers() == 0:
            rdDepictor.Compute2DCoords(mol)
        Chem.NormalizeDepiction(mol)
    except Exception:
        pass

    # 2. Init Drawer
    if fmt == "svg":
        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    elif fmt == "png":
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    else:
        raise ValueError(f"Unsupported fmt {fmt}")

    # 3. Apply "Paper" Aesthetics (Thicker, Cleaner, but COLORFUL)
    opts = drawer.drawOptions()

    opts.bondLineWidth = bond_width  # Critical for visibility in papers
    opts.minFontSize = font_size  # Ensure text is readable
    opts.padding = 0.05  # prevent clipping
    opts.multipleBondOffset = 0.15  # distinct double bonds

    # Use RDKit standard colors (Oxygen=Red, Nitrogen=Blue),
    # but make the font look sharper.
    if bw_palette:
        opts.useBWAtomPalette()

    if transparent:
        opts.clearBackground = False
        opts.setBackgroundColour((1, 1, 1, 0))

    # 4. Draw
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)

        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        data = drawer.GetDrawingText()

        if save_path:
            mode = "w" if fmt == "svg" else "wb"
            with open(save_path, mode) as f:
                f.write(data)

    except Exception as e:
        print(f"Error drawing molecule: {e}")


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


@dataclass
class ReconstructionResult:
    """Result of molecule reconstruction with diagnostics.

    Attributes:
        mol: Reconstructed RDKit molecule
        strategy: Name of the fallback strategy that succeeded
        confidence: Confidence score (1.0 = best, lower = more permissive)
        warnings: List of diagnostic messages or warnings
    """

    mol: Chem.Mol
    strategy: str
    confidence: float
    warnings: list[str]


# Confidence scores for each reconstruction strategy
RECONSTRUCTION_STRATEGY_CONFIDENCE = {
    "standard": 1.0,
    "kekulized": 0.95,
    "single_bonds": 0.7,
    "partial_sanitize": 0.6,
}


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


def reconstruct_for_eval_v2(
    nx_graph: nx.Graph, *, dataset="qm9", return_diagnostics=False
) -> Chem.Mol | ReconstructionResult:
    """
    Reconstruct RDKit molecule from NetworkX graph with enhanced diagnostics.

    This is an improved version of `reconstruct_for_eval` that provides:
    - Additional fallback strategies (single bonds variant)
    - Diagnostic information (which strategy succeeded, confidence score)
    - Enhanced error messages with graph statistics
    - Backward compatible when return_diagnostics=False

    The function tries progressively more permissive reconstruction strategies:
    1. Standard aromatic (preferred for QM9/ZINC)
    2. Kekulized (handles N-O bonds, fused rings)
    3. Single bonds only (bypasses valence inference issues)
    4. Partial sanitization (staged sanitize operations)

    Args:
        nx_graph: NetworkX graph with node features (type or feat attributes)
        dataset: Dataset name ("qm9" or "zinc") - determines atom symbol mapping
        return_diagnostics: If True, returns ReconstructionResult with metadata

    Returns:
        If return_diagnostics=False: RDKit Mol object (backward compatible)
        If return_diagnostics=True: ReconstructionResult with mol + diagnostics

    Raises:
        ValueError: If all reconstruction strategies fail

    Example:
        >>> # Backward compatible usage
        >>> mol = reconstruct_for_eval_v2(graph, dataset="zinc")
        >>>
        >>> # With diagnostics
        >>> result = reconstruct_for_eval_v2(graph, dataset="zinc", return_diagnostics=True)
        >>> print(f"Strategy: {result.strategy}, Confidence: {result.confidence}")
    """
    warnings = []
    n_nodes = nx_graph.number_of_nodes()
    n_edges = nx_graph.number_of_edges()

    # Variant A: Standard aromatic (preferred for both QM9 and ZINC)
    try:
        mol, _ = DataTransformer.nx_to_mol_v3(
            nx_graph, dataset=dataset, infer_bonds=True, sanitize=True, kekulize=False
        )
        # Handle empty graph or structural issues
        if mol is None:
            warnings.append("Standard aromatic failed: empty graph or structural issues")
        elif is_valid_molecule(mol):
            if return_diagnostics:
                return ReconstructionResult(
                    mol=mol,
                    strategy="standard",
                    confidence=RECONSTRUCTION_STRATEGY_CONFIDENCE["standard"],
                    warnings=warnings,
                )
            return mol
    except Exception as e:
        warnings.append(f"Standard aromatic failed: {type(e).__name__}")

    # Variant B: Kekulized (handles N-O bonds, fused rings with unusual electron distribution)
    try:
        mol, _ = DataTransformer.nx_to_mol_v3(nx_graph, dataset=dataset, infer_bonds=True, sanitize=True, kekulize=True)
        # Handle empty graph or structural issues
        if mol is None:
            warnings.append("Kekulized failed: empty graph or structural issues")
        elif is_valid_molecule(mol):
            if return_diagnostics:
                return ReconstructionResult(
                    mol=mol,
                    strategy="kekulized",
                    confidence=RECONSTRUCTION_STRATEGY_CONFIDENCE["kekulized"],
                    warnings=warnings,
                )
            return mol
    except Exception as e:
        warnings.append(f"Kekulized failed: {type(e).__name__}")

    # Variant C: Single bonds only (bypasses valence inference heuristics)
    # Useful when bond order inference fails on unusual structures
    try:
        mol, _ = DataTransformer.nx_to_mol_v3(
            nx_graph, dataset=dataset, infer_bonds=False, sanitize=True, kekulize=False
        )
        # Handle empty graph or structural issues
        if mol is None:
            warnings.append("Single bonds failed: empty graph or structural issues")
        elif is_valid_molecule(mol):
            warnings.append("Used single bonds only (no bond order inference)")
            if return_diagnostics:
                return ReconstructionResult(
                    mol=mol,
                    strategy="single_bonds",
                    confidence=RECONSTRUCTION_STRATEGY_CONFIDENCE["single_bonds"],
                    warnings=warnings,
                )
            return mol
    except Exception as e:
        warnings.append(f"Single bonds failed: {type(e).__name__}")

    # Variant D: Partial sanitization (staged ops: cleanup → rings → aromaticity → full)
    # Last resort for molecules that fail standard sanitization
    try:
        mol, _ = DataTransformer.nx_to_mol_v3(
            nx_graph, dataset=dataset, infer_bonds=True, sanitize=False, kekulize=False
        )
        # Handle empty graph or structural issues
        if mol is None:
            warnings.append("Partial sanitize failed: empty graph or structural issues")
        else:
            # Stage 1: Light sanitization operations
            Chem.SanitizeMol(
                mol,
                sanitizeOps=(
                    SanitizeFlags.SANITIZE_CLEANUP
                    | SanitizeFlags.SANITIZE_SYMMRINGS
                    | SanitizeFlags.SANITIZE_SETAROMATICITY
                ),
            )
            # Stage 2: Full sanitization
            Chem.SanitizeMol(mol)

            if is_valid_molecule(mol):
                warnings.append("Used partial sanitization strategy")
                if return_diagnostics:
                    return ReconstructionResult(
                        mol=mol,
                        strategy="partial_sanitize",
                        confidence=RECONSTRUCTION_STRATEGY_CONFIDENCE["partial_sanitize"],
                        warnings=warnings,
                    )
                return mol
    except Exception as e:
        warnings.append(f"Partial sanitize failed: {type(e).__name__}")

    # All strategies failed - raise informative error
    error_msg = (
        f"All reconstruction strategies failed for graph:\n"
        f"  Nodes: {n_nodes}, Edges: {n_edges}, Dataset: {dataset}\n"
        f"  Attempted strategies: standard, kekulized, single_bonds, partial_sanitize\n"
        f"  Failure details: {'; '.join(warnings)}"
    )
    raise ValueError(error_msg)


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
    mol = reconstruct_for_eval_v2(nx_g, dataset=dataset)  # infer_bonds=True, sanitize=True, kekulize=False
    assert is_valid_molecule(mol), f"mol is invalid: {mol}"
    return canonical_key(mol)  # RemoveHs + canonical, isomeric, aromatic


def compute_qed(mol: Chem.Mol) -> float:
    try:
        return float(QED.qed(mol))
    except Exception:
        return float("nan")
