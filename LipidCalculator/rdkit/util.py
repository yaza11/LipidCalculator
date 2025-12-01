from typing import Iterable

from rdkit import Chem
from rdkit.Chem import Mol, Atom, RWMol
import logging

logger = logging.getLogger(__name__)


def optional_copy_rwmol(mol: Mol, copy: bool) -> Mol:
    if copy:
        mol_new = Mol(mol)
    else:
        logger.warning('not copying not properly tested! expect the unexpected')
        mol_new = mol
    return Chem.RWMol(mol_new)


def check_hs_treated_as_neighbors(mol) -> bool:
    """Check that AddHs was called on molecule"""
    mol.UpdatePropertyCache(strict=False)
    no_hs = [(at.GetNumExplicitHs() + at.GetNumImplicitHs() == 0) for at in mol.GetAtoms()]
    if all(no_hs):
        return True
    logger.warning(f'found atoms with explicit/implicit Hs (instead of neighbors): {no_hs.index(False)}')
    return False


def get_num_hs(atom: Atom) -> int:
    return sum([at.GetAtomicNum() == 1 for at in atom.GetNeighbors()])


def remove_one_h_for_atom(mol: Mol, idx: int, copy: bool = True) -> Mol:
    assert check_hs_treated_as_neighbors(mol)
    rw_mol = optional_copy_rwmol(mol, copy=copy)

    atom = rw_mol.GetAtomWithIndex(idx)
    assert get_num_hs(atom) > 0
    for at in atom.GetNeighbors():
        if at.GetSymbol() == "H":
            jdx = at.GetIdx()
            break
    rw_mol.RemoveAtom(jdx)
    return rw_mol.GetMol()


def add_one_h_for_atom(mol: Mol, idx: int, copy: bool = True) -> tuple[Mol, int]:
    assert check_hs_treated_as_neighbors(mol)
    rw_mol = optional_copy_rwmol(mol, copy=copy)

    at = Atom(1)
    h_idx = rw_mol.AddAtom(at)
    rw_mol.AddBond(idx, h_idx, Chem.BondType.SINGLE)
    if copy:
        return rw_mol, h_idx
    return rw_mol.GetMol(), h_idx


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
    elif len(mols) == 1:
        return mols[0]
    _combo = Mol(mols[0])
    for mol in mols[1:]:
        _combo: Mol = Chem.CombineMols(_combo, mol)  #
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
