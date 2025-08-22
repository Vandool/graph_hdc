import random
from collections import Counter
from pathlib import Path

import torch
from torch_geometric.data import Data

from rdkit import Chem

root: Path = Path("/Users/arvandkaveh/Projects/kit/graph_hdc")
smiles_dir: Path = root / "Smiles/ZINC_smile"
files: list[Path] = [
    smiles_dir / "train_smile.txt",
    smiles_dir / "test_smile.txt",
    smiles_dir / "valid_smile.txt",
    smiles_dir / "debug_smile.txt",
]
N: int = 100  # target count

def atom_key(a):
    """
    Create the categorical 'atom type' key similar to the one
    used in the PyG ZINC pickles: element + formal charge +
    aromatic flag + #implicit Hs.

    -> Its not fully aligned with ZINC yet since it generates 45 categories, where ZINC250 has 28 distinct atom types
    """
    return (
        a.GetSymbol(),              # element
        a.GetFormalCharge(),        # e.g. +1 in [NH3+]
        a.GetIsAromatic(),          # boolean
        a.GetTotalNumHs(includeNeighbors=True),
    )

def iter_smiles(file: Path):
    """Yield SMILES strings from each file, skipping an optional header."""
    with open(file) as fh:
        for line_no, line in enumerate(fh):
            if line_no == 0 and line.strip().lower() == "smiles":
                continue
            s = line.strip().split()[0]
            if s:
                yield s

def build_lookup_pyg_aligned(mols):
    keys = sorted({atom_key(a)
                   for m in mols
                   for a in m.GetAtoms()})
    atom2idx = {k: i for i, k in enumerate(keys)}
    idx2atom = {i: k for k, i in atom2idx.items()}
    return atom2idx, idx2atom


def build_lookup(mols: set[Chem.rdchem.Mol]):
    symbols = sorted({a.GetSymbol() for m in mols for a in m.GetAtoms()})
    atom2idx = {s: i for i, s in enumerate(symbols)}
    idx2atom = {i: s for s, i in atom2idx.items()}
    return atom2idx, idx2atom


def mol_to_data_pyg_aligned(m, atom2idx):
    x, ei_src, ei_dst = [], [], []
    for a in m.GetAtoms():
        k = atom_key(a)
        x.append([
            float(atom2idx[k]),                 # categorical index
            float(a.GetDegree()),               # degree
            float(a.GetTotalNumHs(includeNeighbors=True))  # #H
        ])
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ei_src += [i, j];  ei_dst += [j, i]
    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor([ei_src, ei_dst], dtype=torch.long),
        smiles=Chem.MolToSmiles(m)
    )

def mol_to_data(mol: Chem.rdchem.Mol, atom2idx: dict):
    x = [
        [
            float(atom2idx[a.GetSymbol()]),
            float(a.GetDegree()),
            float(a.GetTotalNumHs(includeNeighbors=True)),
        ]
        for a in mol.GetAtoms()
    ]
    ei_src, ei_dst = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ei_src += [i, j]
        ei_dst += [j, i]
    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor([ei_src, ei_dst], dtype=torch.long),
        smiles=Chem.MolToSmiles(mol),
    )


if __name__ == "__main__":
    # ## Debugging
    # debug_smiles = list(iter_smiles(files[3]))
    # debug_mols = [Chem.MolFromSmiles(s) for s in debug_smiles]
    # atom2idx_debug, idx2atom_debug = build_lookup_pyg_aligned(set(debug_mols))
    # debug_set = [mol_to_data_pyg_aligned(m, atom2idx_debug) for m in debug_mols]
    # print(len(debug_set))

    train_smiles = list(iter_smiles(files[0]))
    test_smiles = list(iter_smiles(files[1]))
    valid_smiles = list(iter_smiles(files[2]))


    train_mols = [Chem.MolFromSmiles(s) for s in train_smiles]
    test_mols = [Chem.MolFromSmiles(s) for s in test_smiles]
    valid_mols = [Chem.MolFromSmiles(s) for s in valid_smiles]

    atom_types = set()
    degrees = set()
    formal_charges = set()
    num_explicit_Hs = set()
    ring_ctrs = set()
    for m in train_mols + test_mols + valid_mols:
        rings = m.GetRingInfo().AtomRings()
        for r in rings:
            counter = Counter([m.GetAtomWithIdx(i).GetSymbol() for i in r])
            ring_ctrs.add(frozenset(counter.items()))
        for a in m.GetAtoms():
            atom_types.add(a.GetSymbol())
            degrees.add(a.GetDegree())
            formal_charges.add(a.GetFormalCharge())
            num_explicit_Hs.add(a.GetTotalNumHs())

    print("Atom types size:", len(atom_types))
    print("Atom types:", sorted(list(atom_types)))
    print("Degrees size:", len(degrees))
    print("Degrees:", degrees)
    print("Formal Charges size:", len(formal_charges))
    print("Formal Charges:", formal_charges)
    print("Explicit Hs size:", len(num_explicit_Hs))
    print("Explicit Hs:", num_explicit_Hs)
    print("Distinct Rings size:", len(ring_ctrs))
    print("Distinct Rings :", ring_ctrs)
    print("Distinct Ring Sizes", {sum(v for _, v in fs) for fs in ring_ctrs})
"""
Atom types size: 9
Atom types: ['Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S']
Degrees size: 5
Degrees: {1, 2, 3, 4, 5}
Formal Charges size: 3
Formal Charges: {0, 1, -1}
Explicit Hs size: 4
Explicit Hs: {0, 1, 2, 3}
Distinct Ring Sizes {3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 18, 24} len = 14
Distinct Rings size: 110
Distinct Rings : {frozenset({('O', 1), ('C', 11)}), frozenset({('C', 7), ('N', 2)}), frozenset({('S', 1), ('C', 5), ('O', 1)}),
                  frozenset({('N', 1), ('C', 4), ('O', 1)}), frozenset({('C', 13), ('O', 1)}), frozenset({('C', 3), ('O', 1)}), 
                  frozenset({('N', 2), ('S', 1), ('C', 5)}), frozenset({('N', 2), ('C', 3), ('S', 1)}), frozenset({('N', 1), ('C', 2), ('P', 1), ('O', 1)}), 
                  frozenset({('O', 2), ('C', 3)}), frozenset({('C', 7), ('N', 1)}), frozenset({('C', 1), ('N', 2), ('S', 1), ('O', 1)}), 
                  frozenset({('O', 1), ('N', 1), ('C', 6)}), frozenset({('C', 4), ('O', 1)}), frozenset({('N', 1), ('C', 5), ('O', 1)}), 
                  frozenset({('O', 8), ('C', 16)}), frozenset({('N', 4)}), frozenset({('C', 2), ('N', 2)}), frozenset({('C', 12), ('S', 1), ('O', 5)}), 
                  frozenset({('S', 3), ('C', 6)}), frozenset({('O', 3), ('C', 3)}), frozenset({('S', 4), ('C', 10)}), 
                  frozenset({('C', 12), ('O', 6)}), frozenset({('N', 1), ('C', 3), ('P', 1)}), frozenset({('O', 2), ('C', 4)}), 
                  frozenset({('O', 1), ('C', 6)}), frozenset({('O', 2), ('C', 8)}), frozenset({('C', 7), ('S', 1)}), 
                  frozenset({('C', 3), ('N', 4)}), frozenset({('N', 1), ('S', 1), ('C', 5), ('O', 1)}), frozenset({('C', 10), ('O', 5)}), 
                  frozenset({('C', 1), ('N', 4)}), frozenset({('S', 1), ('C', 6)}), frozenset({('C', 1), ('N', 2)}), 
                  frozenset({('S', 1), ('C', 5)}), frozenset({('N', 1), ('C', 8)}), frozenset({('N', 2), ('C', 4), ('S', 2)}), 
                  frozenset({('O', 2), ('C', 3), ('P', 1)}), frozenset({('C', 3), ('N', 3)}), frozenset({('N', 2), ('C', 4), ('O', 1)}), 
                  frozenset({('C', 2), ('N', 4)}), frozenset({('N', 2), ('C', 8)}), frozenset({('C', 4)}), frozenset({('N', 1), ('C', 2), ('S', 2)}), 
                  frozenset({('N', 1), ('C', 6)}), frozenset({('S', 1), ('N', 2), ('C', 4)}), frozenset({('O', 3), ('N', 1), ('C', 8)}), 
                  frozenset({('N', 1), ('C', 5)}), frozenset({('S', 2), ('C', 4), ('N', 1), ('O', 1)}), frozenset({('O', 2), ('C', 2), ('S', 1)}), 
                  frozenset({('N', 1), ('C', 3), ('S', 1), ('O', 1)}), frozenset({('N', 2), ('C', 5)}), frozenset({('C', 3)}), 
                  frozenset({('N', 1), ('C', 2), ('S', 1)}), frozenset({('N', 2), ('C', 6)}), frozenset({('N', 1), ('C', 4), ('S', 1)}), 
                  frozenset({('C', 1), ('N', 3), ('S', 1)}), frozenset({('C', 2), ('N', 2), ('O', 1)}), frozenset({('O', 2), ('C', 5)}), 
                  frozenset({('N', 1), ('C', 3), ('O', 1)}), frozenset({('N', 2), ('C', 2), ('S', 1)}), frozenset({('N', 1), ('C', 3), ('P', 1), ('O', 1)}), 
                  frozenset({('C', 5), ('S', 2)}), frozenset({('C', 3), ('S', 2)}), frozenset({('N', 1), ('C', 3), ('S', 1)}), 
                  frozenset({('C', 2), ('S', 1)}), frozenset({('N', 2), ('C', 3), ('O', 1)}), frozenset({('N', 3), ('C', 4)}), 
                  frozenset({('C', 2), ('N', 3)}), frozenset({('N', 1), ('O', 4), ('C', 10)}), frozenset({('C', 3), ('S', 1), ('O', 1)}), 
                  frozenset({('N', 2), ('P', 1), ('C', 4)}), frozenset({('S', 4), ('C', 4)}), frozenset({('C', 5)}), 
                  frozenset({('C', 12), ('N', 2), ('O', 4)}), frozenset({('N', 6), ('C', 12)}), frozenset({('C', 13), ('O', 2), ('N', 3)}), 
                  frozenset({('N', 1), ('P', 1), ('C', 4)}), frozenset({('C', 12)}), frozenset({('N', 1), ('C', 3)}), 
                  frozenset({('O', 2), ('P', 1), ('C', 4)}), frozenset({('O', 2), ('C', 2), ('P', 1)}), frozenset({('C', 4), ('S', 1)}), 
                  frozenset({('N', 1), ('S', 1), ('C', 5)}), frozenset({('N', 3), ('C', 2), ('P', 1)}), frozenset({('O', 2), ('C', 3), ('N', 1)}), 
                  frozenset({('C', 8), ('O', 4)}), frozenset({('C', 5), ('O', 1)}), frozenset({('C', 9)}), frozenset({('C', 7)}), 
                  frozenset({('O', 2), ('C', 2), ('N', 2)}), frozenset({('C', 2), ('N', 2), ('P', 1)}), frozenset({('C', 4), ('S', 2)}), 
                  frozenset({('N', 2), ('C', 3)}), frozenset({('C', 12), ('N', 1), ('O', 5)}), frozenset({('C', 3), ('S', 1)}), 
                  frozenset({('C', 10)}), frozenset({('N', 1), ('C', 2)}), frozenset({('C', 6)}), frozenset({('N', 1), ('C', 10), ('S', 2)}), 
                  frozenset({('N', 1), ('C', 4)}), frozenset({('N', 1), ('C', 9)}), frozenset({('S', 1), ('C', 4), ('O', 1)}), 
                  frozenset({('N', 2), ('C', 3), ('P', 1)}), frozenset({('C', 2), ('O', 1)}), frozenset({('O', 2), ('C', 10)}), 
                  frozenset({('S', 3), ('C', 3)}), frozenset({('C', 8)}), frozenset({('N', 2), ('C', 4)}), frozenset({('N', 1), ('S', 1), ('C', 6)})}


The must-haves (you cannot guess bonds without them)

Attribute (Atom method)	Why it’s indispensable chemically	How it constrains the edge search
Element / atomic number (GetSymbol() / GetAtomicNum())	Tells you the default valence (C ≈ 4, N ≈ 3, O ≈ 2…).	Gives an upper bound on how many bonds the node can accept.
Formal charge (GetFormalCharge())	A +1 N will usually want 4 neighbours, a neutral N only 3.	Adjusts the target valence.
Number of explicit or total hydrogens (GetNumExplicitHs() or GetTotalNumHs())	Every H already occupies one σ-bond.	Subtract H-count from the target valence ⇒ “open slots” left for heavy-atom edges.

With just those three you can compute each atom’s remaining valence budget and run a (NP-hard but tractable) matching/optimization to add bonds until every budget hits zero and the whole graph is connected.

⸻

Very useful “nice-to-haves” (prune the search space fast)

Attribute	Chemical meaning	Benefit for graph construction
GetExplicitValence() / GetImplicitValence()	RDKit’s own guess after property cache update.	Directly tells you how many σ-bonds RDKit thinks are legal → fewer candidate graphs.
Hybridization (GetHybridization())	sp, sp², sp³.	Limits max degree (sp ≤ 2, sp² ≤ 3, …) and bond orders.
Aromatic flag (GetIsAromatic())	Part of a π-delocalised ring.	Forces certain atoms to be in a continuous ring of alternating single/double bonds; huge pruning.
Ring participation (IsInRing(), IsInRingSize())	Whether atom must be cyclic.	Prevents you from adding bonds that would give impossible ring counts or sizes.
Coordinates (if you have them in Conformer)	3-D distances.	Simple cut-off based bond assignment; near-certain for organic molecules.


⸻

Attributes that don’t help with connectivity
	•	Isotope (GetIsotope()), exact mass, monomer info – they matter for spectra, not for bonding.
	•	Per-atom properties you add ad-hoc (SetProp) unless you encode extra chemical rules in them.

⸻

Minimal algorithm sketch with just the must-haves

for each atom:
    target_valence = default_valence(element, formal_charge)
    open_slots = target_valence - explicit_Hs

while any open_slots > 0:
    pick two atoms with open_slots > 0
    add single bond
    decrement each open_slots
    if a bond order upgrade (e.g. single→double) satisfies both sooner, try that
    back-track if someone’s valence is exceeded

Add the “nice-to-haves” as constraints and the search collapses from astronomically large to something solvable for normal drug-like molecules.
"""

    # ## Pyg Aligned
    # atom2idx_pyg, idx2atom_pyg = build_lookup_pyg_aligned(set(train_mols + valid_mols + test_mols))
    # train_dataset = [mol_to_data_pyg_aligned(m, atom2idx_pyg) for m in train_mols]
    # test_dataset = [mol_to_data_pyg_aligned(m, atom2idx_pyg) for m in test_mols]
    # valid_dataset = [mol_to_data_pyg_aligned(m, atom2idx_pyg) for m in valid_mols]
    #
    # print(len(debug_set))
    # ---- results ----
    # dataset : list[Data] ready for PyG loaders
    # idx2atom: {int: str}  reverse mapping for atom indices
