"""
HOMO-LUMO Gap Comparison: PySCF DFT vs tblite xTB

This script compares HOMO-LUMO gap values computed by two physics-based methods:
1. PySCF DFT: On-the-fly B3LYP/6-31G* calculation (slow but ab initio)
2. tblite xTB: GFN2-xTB semi-empirical method (fast, physics-based)

Key differences:
- PySCF: ~5-10 seconds per molecule, ab initio DFT
- tblite: ~50-200 milliseconds per molecule, semi-empirical

Usage:
    pixi run -e local python src/datasets/humo_lumo_gap_analysis.py
"""

from __future__ import annotations

import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Constants
HARTREE_TO_EV = 27.2114  # 1 Hartree = 27.2114 eV
N_MOLECULES = 10


def smiles_to_3d_coords(smiles: str) -> tuple[list[int], np.ndarray]:
    """
    Generate 3D coordinates from SMILES using RDKit.

    Args:
        smiles: SMILES string of the molecule

    Returns:
        Tuple of (atomic_numbers, positions) where positions is [N, 3] array in Angstrom
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    mol = Chem.AddHs(mol)

    # Embed molecule to get 3D coordinates
    embed_result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if embed_result == -1:
        embed_result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if embed_result == -1:
            raise ValueError(f"Could not generate 3D structure for: {smiles}")

    # Optimize geometry with MMFF
    AllChem.MMFFOptimizeMolecule(mol)

    # Extract atomic numbers and positions
    atomic_nums = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(mol.GetNumAtoms())]
    coords = mol.GetConformer().GetPositions()

    return atomic_nums, coords


def compute_pyscf_gap(smiles: str) -> float:
    """
    Compute HOMO-LUMO gap using PySCF DFT (B3LYP/6-31G*).

    Args:
        smiles: SMILES string of the molecule

    Returns:
        HOMO-LUMO gap in eV
    """
    from pyscf import dft, gto

    atomic_nums, coords = smiles_to_3d_coords(smiles)

    # Build PySCF geometry string
    atom_symbols = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
    atoms = [atom_symbols.get(z, Chem.GetPeriodicTable().GetElementSymbol(z)) for z in atomic_nums]
    geom_str = "; ".join(f"{a} {x:.6f} {y:.6f} {z:.6f}" for a, (x, y, z) in zip(atoms, coords, strict=False))

    # PySCF DFT calculation
    pyscf_mol = gto.Mole()
    pyscf_mol.atom = geom_str
    pyscf_mol.basis = "6-31g*"
    pyscf_mol.verbose = 0
    pyscf_mol.build()

    # Run DFT with B3LYP functional
    mf = dft.RKS(pyscf_mol)
    mf.xc = "B3LYP"
    mf.verbose = 0
    mf.kernel()

    # Extract HOMO and LUMO energies
    occ = mf.mo_occ
    mo_energy = mf.mo_energy

    homo = mo_energy[occ > 0][-1]
    lumo = mo_energy[occ == 0][0]

    gap = (lumo - homo) * HARTREE_TO_EV
    return gap


def compute_tblite_gap(smiles: str) -> float:
    """
    Compute HOMO-LUMO gap using tblite GFN2-xTB semi-empirical method.

    Args:
        smiles: SMILES string of the molecule

    Returns:
        HOMO-LUMO gap in eV
    """
    from tblite.interface import Calculator

    atomic_nums, coords = smiles_to_3d_coords(smiles)

    # Convert to numpy arrays (tblite expects Bohr, RDKit gives Angstrom)
    # 1 Angstrom = 1.8897259886 Bohr
    ANGSTROM_TO_BOHR = 1.8897259886
    coords_bohr = coords * ANGSTROM_TO_BOHR

    # Create calculator and run single point
    calc = Calculator("GFN2-xTB", np.array(atomic_nums), coords_bohr)
    res = calc.singlepoint()

    # Get orbital energies and occupations
    orbital_energies = res.get("orbital-energies")
    orbital_occupations = res.get("orbital-occupations")

    if orbital_energies is None or orbital_occupations is None:
        raise ValueError("Could not get orbital energies from tblite")

    # Find HOMO and LUMO
    occupied = orbital_occupations > 0.1
    homo_idx = np.where(occupied)[0][-1]
    lumo_idx = homo_idx + 1

    homo = orbital_energies[homo_idx]
    lumo = orbital_energies[lumo_idx]

    # tblite returns energies in Hartrees
    gap = (lumo - homo) * HARTREE_TO_EV
    return gap


def main():
    """Run HOMO-LUMO gap comparison between PySCF and tblite."""
    from src.datasets.utils import get_split

    print("=" * 95)
    print("HOMO-LUMO Gap Comparison: PySCF DFT vs tblite xTB")
    print("=" * 95)
    print()
    print("Method 1 (PySCF):  Ab initio DFT - B3LYP/6-31G* (slow, physics-based)")
    print("Method 2 (tblite): GFN2-xTB semi-empirical (fast, physics-based)")
    print()

    # Load dataset
    print("Loading QM9Smiles dataset (train split)...")
    ds = get_split(split="train", base_dataset="qm9")
    print(f"Loaded {len(ds)} molecules")
    print()

    results = []

    print("-" * 95)
    header = f"{'Idx':>4} | {'SMILES':<25} | {'PySCF (eV)':>10} | {'tblite (eV)':>11} | {'Diff (eV)':>10} | {'PySCF-t':>8} | {'tblite-t':>10}"
    print(header)
    print("-" * 95)

    for i in range(N_MOLECULES):
        data = ds[i]
        smiles = data.smiles

        pyscf_gap, tblite_gap = None, None
        pyscf_time, tblite_time = None, None

        # Method 1: PySCF DFT
        t0 = time.time()
        try:
            pyscf_gap = compute_pyscf_gap(smiles)
            pyscf_time = time.time() - t0
        except Exception as e:
            pyscf_time = time.time() - t0
            print(f"{i:>4} | {smiles:<25} | PySCF Error: {e}")

        # Method 2: tblite xTB
        t0 = time.time()
        try:
            tblite_gap = compute_tblite_gap(smiles)
            tblite_time = time.time() - t0
        except Exception as e:
            tblite_time = time.time() - t0
            print(f"      tblite Error: {e}")

        results.append({
            "idx": i,
            "smiles": smiles,
            "pyscf_gap": pyscf_gap,
            "tblite_gap": tblite_gap,
            "pyscf_time": pyscf_time,
            "tblite_time": tblite_time,
        })

        # Format output
        smiles_disp = smiles[:22] + "..." if len(smiles) > 25 else smiles
        pyscf_str = f"{pyscf_gap:>10.3f}" if pyscf_gap is not None else f"{'ERR':>10}"
        tblite_str = f"{tblite_gap:>11.3f}" if tblite_gap is not None else f"{'ERR':>11}"

        if pyscf_gap is not None and tblite_gap is not None:
            diff = abs(pyscf_gap - tblite_gap)
            diff_str = f"{diff:>10.3f}"
        else:
            diff_str = f"{'N/A':>10}"

        pyscf_t = f"{pyscf_time:>7.2f}s" if pyscf_time else f"{'N/A':>8}"
        tblite_t = f"{tblite_time * 1000:>8.1f}ms" if tblite_time else f"{'N/A':>10}"

        print(f"{i:>4} | {smiles_disp:<25} | {pyscf_str} | {tblite_str} | {diff_str} | {pyscf_t} | {tblite_t}")

    # Summary
    print()
    print("=" * 95)
    print("SUMMARY STATISTICS (all values in eV, times as shown)")
    print("=" * 95)

    pyscf_results = [r for r in results if r["pyscf_gap"] is not None]
    tblite_results = [r for r in results if r["tblite_gap"] is not None]
    both_results = [r for r in results if r["pyscf_gap"] is not None and r["tblite_gap"] is not None]

    print(f"\n{'Method':<20} | {'Count':>6} | {'Mean Gap':>10} | {'Std Gap':>10} | {'Mean Time':>12} | {'Total Time':>12}")
    print("-" * 85)

    if pyscf_results:
        gaps = [r["pyscf_gap"] for r in pyscf_results]
        times = [r["pyscf_time"] for r in pyscf_results]
        print(f"{'PySCF (B3LYP)':<20} | {len(gaps):>6} | {np.mean(gaps):>10.3f} | {np.std(gaps):>10.3f} | {np.mean(times):>10.2f}s | {sum(times):>10.2f}s")

    if tblite_results:
        gaps = [r["tblite_gap"] for r in tblite_results]
        times = [r["tblite_time"] for r in tblite_results]
        print(f"{'tblite (GFN2-xTB)':<20} | {len(gaps):>6} | {np.mean(gaps):>10.3f} | {np.std(gaps):>10.3f} | {np.mean(times)*1000:>9.1f}ms | {sum(times)*1000:>9.1f}ms")

    # Correlation and error analysis
    if both_results:
        pyscf_gaps = np.array([r["pyscf_gap"] for r in both_results])
        tblite_gaps = np.array([r["tblite_gap"] for r in both_results])
        diff = np.abs(pyscf_gaps - tblite_gaps)
        correlation = np.corrcoef(pyscf_gaps, tblite_gaps)[0, 1]

        print()
        print("-" * 85)
        print("ERROR ANALYSIS (tblite vs PySCF as reference)")
        print("-" * 85)
        print(f"  Mean Absolute Error (MAE): {np.mean(diff):.3f} eV")
        print(f"  Max Absolute Error:        {np.max(diff):.3f} eV")
        print(f"  Pearson Correlation:       {correlation:.4f}")

    # Speed comparison
    if pyscf_results and tblite_results:
        speedup = np.mean([r["pyscf_time"] for r in pyscf_results]) / np.mean([r["tblite_time"] for r in tblite_results])
        print()
        print("-" * 85)
        print(f"SPEEDUP: tblite is ~{speedup:.0f}x faster than PySCF")
        print("-" * 85)

    print()
    print("=" * 95)
    print("METHOD COMPARISON:")
    print("=" * 95)
    print("| Feature          | PySCF (DFT)                | tblite (xTB)               |")
    print("|------------------|----------------------------|----------------------------|")
    print("| Method           | Ab initio DFT              | Semi-empirical QM          |")
    print("| Functional/Basis | B3LYP/6-31G*               | GFN2-xTB parameters        |")
    print("| Speed            | ~5-10 sec/mol              | ~50-200 ms/mol             |")
    print("| Accuracy         | High (reference standard)  | Good (~0.3-0.5 eV MAE)     |")
    print("| Use case         | Accurate reference calcs   | Fast screening             |")
    print("=" * 95)


if __name__ == "__main__":
    main()
