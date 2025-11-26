from typing import Literal

from rdkit import Chem
from rdkit.Chem import Mol, rdMolDescriptors

from LipidCalculator.cleaving.util import SUPPORTED_HETEROATOMS
from LipidCalculator.rdkit.plotting import mplt_mol, plt_indices_bond

PERIODIC_TABLE = Chem.GetPeriodicTable()
SUPPORTED_ADDUCT_TYPES = ['[M+H]+', '[M+NH4]+', '[M+Na]+', '[M+K]+']


def _get_single_atom_adduct(atom_type: str, charge: int = 1) -> Mol:
    rw_mol = Chem.RWMol()

    at = Chem.Atom(PERIODIC_TABLE.GetAtomicNumber(atom_type))
    at.SetFormalCharge(charge)  # protonated hydrogen

    rw_mol.AddAtom(at)

    mol_result = rw_mol.GetMol()
    Chem.SanitizeMol(mol_result)  # recomputes valences, implicit H counts etc.
    return mol_result


def _get_h_plus_adduct() -> Mol:
    return _get_single_atom_adduct('H', 1)


def _get_na_plus_adduct() -> Mol:
    return _get_single_atom_adduct('Na', 1)


def _get_k_plus_adduct() -> Mol:
    return _get_single_atom_adduct('K', 1)


def _get_ammonium_adduct() -> Mol:
    mol = Chem.MolFromSmiles('[NH3]')
    rw_mol = Chem.RWMol(mol)

    rw_mol.GetAtomWithIdx(0).SetFormalCharge(1)  # protonated hydrogen
    # bind H+ with hydrogen bond
    at = Chem.Atom(1)

    h_idx = rw_mol.AddAtom(at)
    rw_mol.AddBond(0, h_idx, Chem.BondType.SINGLE)

    mol_result = rw_mol.GetMol()
    Chem.SanitizeMol(mol_result)

    return mol_result


def _get_adduct_as_mol(adduct_type: Literal[*SUPPORTED_ADDUCT_TYPES]) -> Mol:
    if adduct_type == '[M+H]+':
        return _get_h_plus_adduct()
    elif adduct_type == '[M+Na]+':
        return _get_na_plus_adduct()
    elif adduct_type == '[M+K]+':
        return _get_k_plus_adduct()
    elif adduct_type == '[M+NH4]+':
        return _get_ammonium_adduct()
    else:
        raise ValueError(f'Unknown adduct type: {adduct_type}')


def add_formal_charge_for_atom(mol: Mol, atom_idx: int, add_H: bool = False):
    """increment charge for atom at specified index by 1 (making it more positive, so taking away an electron)"""
    rw_mol = Chem.RWMol(mol)
    at = rw_mol.GetAtomWithIdx(atom_idx)
    at.SetFormalCharge(at.GetFormalCharge() + 1)
    if not add_H:
        at.SetNumRadicalElectrons(at.GetNumRadicalElectrons() + 1)

    mol_result = rw_mol.GetMol()
    Chem.SanitizeMol(mol_result)
    return mol_result


def _set_m_plus_adduct(mol: Mol, atom_idx: int) -> Mol:
    return add_formal_charge_for_atom(mol=mol, atom_idx=atom_idx, add_H=False)


def _find_available_bond_locations_for_adduct(mol) -> list[int]:
    """Consider all heteroatoms as available bond locations."""
    heteroatoms: list[int] = [
        idx for idx, atom in enumerate(mol.GetAtoms())
        if atom.GetAtomicNum() not in [6, 1]
           and atom.GetFormalCharge() <= 0  # avoid overcharged species
    ]
    return heteroatoms


def _add_adduct_at_idx(mol: Mol, add: Mol, atom_idx: int, plts=False) -> Mol:
    n_atoms: int = len(mol.GetAtoms())
    mol = Chem.CombineMols(mol, add)
    rw_mol = Chem.RWMol(mol)
    # in combined molecule, indices are continued
    if plts:
        print(f'attempting to form bond between {atom_idx} and {n_atoms}')
        plt_indices_bond([rw_mol])

    rw_mol.AddBond(atom_idx, n_atoms, Chem.BondType.ZERO)
    mol_result = rw_mol.GetMol()
    Chem.SanitizeMol(mol_result)
    return mol_result


def _add_adduct_to_heteroatom(
        mol: Mol,
        adduct_type: Literal[*SUPPORTED_ADDUCT_TYPES] = '[M+H]+',
        return_mode: Literal['first', 'all', 'index'] = 'all',
        idx: int = None
) -> Mol | list[Mol]:
    # TODO: support for M+

    # Find all heteroatoms (not C or H)
    available_indices: list[int] = _find_available_bond_locations_for_adduct(mol)
    if len(available_indices) == 0:
        raise ValueError(f'No available bond locations found for molecule with SMILES={Chem.MolToSmiles(mol)}')

    add = _get_adduct_as_mol(adduct_type)

    if return_mode == 'index':
        assert idx in available_indices
        return _add_adduct_at_idx(mol, add, idx)
    if return_mode == 'first':
        return _add_adduct_at_idx(mol, add, available_indices[0])
    return [_add_adduct_at_idx(mol, add, idx) for idx in available_indices]


def _increase_formal_charge(
        mol: Mol,
        return_mode: Literal['first', 'all', 'index'] = 'all',
        idx: int = None
) -> Mol | list[Mol]:
    """mimics M+ adduct"""
    available_indices: list[int] = _find_available_bond_locations_for_adduct(mol)

    if return_mode == 'index':
        assert idx in available_indices
        return _set_m_plus_adduct(mol, idx)

    res: list[Mol] = []
    for idx in available_indices:
        try:
            mol_with_add: Mol = _set_m_plus_adduct(mol, idx)
            if return_mode == 'first':
                return mol_with_add
            res.append(mol_with_add)
        except Exception as e:
            print(f'Exception encountered in increasing formal charge: {e}')
    # did not return from loop
    if (return_mode == 'first') or (len(res) == 0):
        raise ValueError(f'Could not find position to increase formal charge for: {Chem.MolToSmiles(mol)}')

    return res


def _check_idx_available(
        available_indices: list[int], idx: int
) -> int:
    ...


def get_mol_with_adduct(
        mol: Mol,
        add: str,
        return_mode: Literal['first', 'all', 'index'] = 'all',
        idx: int = None
) -> Mol | list[Mol]:
    assert (add in SUPPORTED_ADDUCT_TYPES) or (add == 'M+'), f'Invalid adduct type: {add}'
    if return_mode == 'index':
        assert idx is not None

    if add == 'M+':
        return _increase_formal_charge(mol, return_mode=return_mode, idx=idx)
    return _add_adduct_to_heteroatom(mol, add, return_mode=return_mode, idx=idx)


def steal_charge_from_adduct(mol_with_adduct: Mol, keep_h: bool, plts=False) -> Mol:
    """
    Break the hydrogen bond, modify charge and radical of heteroatom, remove rest of adduct.
    Warning: This function will add an h atom, regardless of whether one is present in the adduct if keep_h is set to True
    """
    rw_mol = Chem.RWMol(mol_with_adduct)

    # find the hydrogen bond between molecule and adduct
    for bond in mol_with_adduct.GetBonds():
        if bond.GetBondType() == Chem.BondType.ZERO:
            break
    else:
        raise ValueError(f'Could not find adduct for {mol_with_adduct}')

    # break the bond
    rw_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    # begin atom is always that of the molecule
    heteroatom = rw_mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
    assert heteroatom.GetSymbol() not in ('C', 'H')

    heteroatom.SetFormalCharge(heteroatom.GetFormalCharge() + 1)

    if keep_h:
        heteroatom.SetNumExplicitHs(heteroatom.GetNumExplicitHs() + 1)
    else:
        heteroatom.SetNumRadicalElectrons(heteroatom.GetNumRadicalElectrons() + 1)

    if plts:
        plt_indices_bond(rw_mol)

    frags = Chem.GetMolFrags(rw_mol, asMols=True, sanitizeFrags=False)

    # here we are assuming that fragment 0 is always the molecule, TODO: is this always the case?
    mol = frags[0]
    Chem.SanitizeMol(mol)
    return mol


def find_adduct_positions(mol: Mol) -> list[int]:
    """
    Atoms amendable to protonization should
    - be heteroatoms
    - not be charged
    """
    positions: list[int] = []
    for idx, atom in enumerate(mol.GetAtoms()):
        if (atom.GetFormalCharge() == 0) and (atom.GetSymbol() in SUPPORTED_HETEROATOMS) and (
                len(atom.GetNeighbors()) < 4):
            positions.append(idx)
    return positions


def _test_adducts():
    mol = Chem.MolFromSmiles('CCOCCO')

    for add in SUPPORTED_ADDUCT_TYPES + ['M+']:
        mols_with_add = get_mol_with_adduct(mol, add, return_mode='all')
        for mol_with_add in mols_with_add:
            mplt_mol(mol_with_add)


if __name__ == '__main__':
    # _test_adducts()
    from LipidCalculator.compound_groups.intact_polar_lipids.generate_ipl import ipl_automatic_bonds

    name = 'PG DAG C32:0'
    mol = ipl_automatic_bonds(name.split(), plts=False, idx_plt=False, split_chain=False)
    plt_indices_bond(mol)

    adduct_pos = find_adduct_positions(mol)
    print(adduct_pos)

    pass
