from typing import Iterable

from rdkit import Chem
from rdkit.Chem import Mol


def add_indices(mol: Mol):
    """Label atoms in a molecule with their index"""
    # from https://www.rdkit.org/docs/Cookbook.html
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


def mol_from_str(mol: str) -> Mol:
    if mol.startswith('InChi'):
        mol = Chem.MolFromInchi(mol)
    else:
        mol = Chem.MolFromSmiles(mol)
    return mol


def get_combined(mols: Iterable[Mol]) -> Mol:
    """Turn multiple molecules into a single one. This does not add bonds between fragments."""
    if len(mols) == 0:
        return Mol()
    _combo = mols[0]
    for mol in mols[1:]:
        _combo: Mol = Chem.CombineMols(_combo, mol)
    return _combo


def bump_bond_order(rw_mol, i, j):
    bond = rw_mol.GetBondBetweenAtoms(i, j)
    if bond is None:
        # no bond yet: create a single bond
        rw_mol.AddBond(i, j, Chem.BondType.SINGLE)
        return

    bt = bond.GetBondType()
    if bt == Chem.BondType.SINGLE:
        bond.SetBondType(Chem.BondType.DOUBLE)
    elif bt == Chem.BondType.DOUBLE:
        bond.SetBondType(Chem.BondType.TRIPLE)
    else:
        # AROMATIC or TRIPLE (or others) can't be simply "incremented"
        raise ValueError(f"Cannot increase bond order from {bt}")

    # optional: update caches/sanitize if you’re done editing
    rw_mol.UpdatePropertyCache(strict=False)
