import logging
from typing import Literal

from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Mol, AddHs
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.rdmolops import RemoveHs

from LipidCalculator import CompoundDict
from LipidCalculator.rdkit.cleaving.util import SUPPORTED_HETEROATOMS
from LipidCalculator.rdkit.plotting import mplt_mol, plt_indices_bond
from LipidCalculator.rdkit.util import add_one_h_for_atom, check_hs_treated_as_neighbors, get_num_hs, \
    optional_copy_rwmol

PERIODIC_TABLE = Chem.GetPeriodicTable()
SUPPORTED_ADDUCT_TYPES = ['[M+H]+', '[M+NH4]+', '[M+Na]+', '[M+K]+']

logger = logging.getLogger(__name__)


def _get_single_atom_adduct(atom_type: str, charge: int = 1) -> Mol:
    rw_mol = Chem.RWMol()

    at = Chem.Atom(PERIODIC_TABLE.GetAtomicNumber(atom_type))
    at.SetFormalCharge(charge)  # protonated hydrogen

    rw_mol.AddAtom(at)

    mol_result = rw_mol.GetMol()
    Chem.SanitizeMol(mol_result)  # recomputes valences, implicit H counts etc.
    return AddHs(mol_result)


def _get_h_plus_adduct() -> Mol:
    return _get_single_atom_adduct('H', 1)


def _get_na_plus_adduct() -> Mol:
    return _get_single_atom_adduct('Na', 1)


def _get_k_plus_adduct() -> Mol:
    return _get_single_atom_adduct('K', 1)


def _get_ammonium_adduct() -> Mol:
    mol = Chem.MolFromSmiles('[NH4+]')
    rw_mol = Chem.RWMol(mol)  #

    mol_result = rw_mol.GetMol()
    Chem.SanitizeMol(mol_result)
    return AddHs(mol_result)


def _get_adduct_as_mol(adduct_type: Literal[*SUPPORTED_ADDUCT_TYPES]) -> Mol:
    """Create molecule fragments with H atoms as neighbours."""
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


def add_formal_charge_for_atom(mol: Mol, atom_idx: int, copy: bool = True):
    """
    Increment charge for atom at specified index by 1 (making it more positive,
    so taking away an electron). Returns copy of original molecule.
    """
    assert check_hs_treated_as_neighbors(mol)
    rw_mol = optional_copy_rwmol(mol, copy=copy)

    at = rw_mol.GetAtomWithIdx(atom_idx)
    at.SetFormalCharge(at.GetFormalCharge() + 1)
    at.SetNumRadicalElectrons(at.GetNumRadicalElectrons() + 1)

    mol_result = rw_mol.GetMol()
    return mol_result


def add_h_plus_for_atom(mol: Mol, atom_idx: int, charge_at_hetero: bool, copy: bool = True):
    assert check_hs_treated_as_neighbors(mol)
    rw_mol = optional_copy_rwmol(mol, copy=copy)

    at = rw_mol.GetAtomWithIdx(atom_idx)
    h_idx = add_one_h_for_atom(rw_mol, atom_idx)
    if charge_at_hetero:
        at.SetFormalCharge(at.GetFormalCharge() + 1)
    else:
        rw_mol.GetAtomWithIdx(h_idx).SetFormalCharge(1)

    mol_result = rw_mol.GetMol()
    return mol_result


def swap_charge_h_hetero(mol: Mol, heteroatom_idx: int, from_hetero: bool, copy: bool = True) -> Mol:
    # TODO: swap radicals
    assert check_hs_treated_as_neighbors(mol)
    rw_mol = optional_copy_rwmol(mol, copy=copy)
    het_atm = rw_mol.GetAtomWithIdx(heteroatom_idx)

    if from_hetero:
        assert het_atm.GetFormalCharge() != 0
        for h_atm in het_atm.GetNeighbors():
            if (h_atm.GetSymbol() == 'H') and h_atm.GetFormalCharge() == 0:
                break
        else:
            raise ValueError(f'Atom with index {het_atm.GetIdx()} does not have a neutral H atom neighbor!')
        c = het_atm.GetFormalCharge()
        het_atm.SetFormalCharge(0)
        h_atm.SetFormalCharge(c)
    else:  # exactly of the neighboring H atoms must be charged
        idcs_charged_h = [
            n.GetIdx()
            for n in het_atm.GetNeighbors()
            if (n.GetFormalCharge() != 0) and (n.GetSymbol() == 'H')
        ]
        assert len(idcs_charged_h) == 1, \
            (f'found {len(idcs_charged_h)} charged H atoms neighboring atom with '
             f'idx {heteroatom_idx}, need exactly one charged')
        h_atm = rw_mol.GetAtomWithIdx(idcs_charged_h[0])
        c = h_atm.GetFormalCharge()
        het_atm.SetFormalCharge(c)
        h_atm.SetFormalCharge(0)

    return rw_mol.GetMol()


def find_available_bond_locations_for_adduct(mol) -> list[int]:
    """Consider all heteroatoms as available bond locations."""
    assert check_hs_treated_as_neighbors(mol)

    heteroatoms: list[int] = [
        idx for idx, atom in enumerate(mol.GetAtoms())
        if atom.GetAtomicNum() not in [6, 1]
           and atom.GetFormalCharge() <= 0  # avoid overcharged species
           and (len(atom.GetNeighbors()) - get_num_hs(atom)) < 4  # sterically hindered
    ]
    return heteroatoms


def _add_adduct_at_idx(mol: Mol, add: Mol, atom_idx: int, plts=False) -> Mol:
    """Always returns a copy"""
    n_atoms: int = len(mol.GetAtoms())

    mol = Chem.CombineMols(Mol(mol), add)
    rw_mol = Chem.RWMol(mol)  #
    # in combined molecule, indices are continued
    if plts:
        print(f'attempting to form bond between {atom_idx} and {n_atoms}')
        plt_indices_bond([rw_mol])

    rw_mol.AddBond(atom_idx, n_atoms, Chem.BondType.ZERO)
    return rw_mol.GetMol()


def _add_adduct_to_heteroatom(
        mol: Mol,
        adduct_type: Literal[*SUPPORTED_ADDUCT_TYPES] = '[M+H]+',
        return_mode: Literal['first', 'all', 'index'] = 'all',
        idx: int = None
) -> Mol | list[Mol]:
    # Find all heteroatoms (not C or H)
    available_indices: list[int] = find_available_bond_locations_for_adduct(mol)
    if len(available_indices) == 0:
        raise ValueError(f'No available bond locations found for molecule with SMILES={Chem.MolToSmiles(mol)}')

    add: Mol = _get_adduct_as_mol(adduct_type)

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
    available_indices: list[int] = find_available_bond_locations_for_adduct(mol)

    if return_mode == 'index':
        assert idx in available_indices
        return add_formal_charge_for_atom(mol, idx)

    res: list[Mol] = []
    for idx in available_indices:
        mol_with_add: Mol = add_formal_charge_for_atom(mol, idx)
        if return_mode == 'first':
            return mol_with_add
        res.append(mol_with_add)
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
    assert check_hs_treated_as_neighbors(mol)

    if return_mode == 'index':
        assert idx is not None

    if add == 'M+':
        return _increase_formal_charge(mol, return_mode=return_mode, idx=idx)
    return _add_adduct_to_heteroatom(mol, add, return_mode=return_mode, idx=idx)


def steal_pos_charge_from_adduct(mol_with_adduct: Mol, keep_h: bool, plts=False) -> Mol:
    """
    Break the hydrogen bond, modify charge and radical of heteroatom, remove rest of adduct.
    Warning: This function will add an h atom, regardless of whether one is present in the adduct if keep_h is set to True

    Always returns a copy.
    """
    assert check_hs_treated_as_neighbors(mol_with_adduct)
    rw_mol = Chem.RWMol(Mol(mol_with_adduct))

    if plts:
        fig, axs = plt.subplots(nrows=5)
        plt_indices_bond(rw_mol, ax=axs[0], remove_hs=False)
        axs[0].set_title('Original')

    # find the hydrogen bond between molecule and adduct
    for bond in mol_with_adduct.GetBonds():
        if bond.GetBondType() == Chem.BondType.ZERO:
            break
    else:
        raise ValueError(f'Could not find adduct for {mol_with_adduct}')
    logger.info(f'found hydrogen bond between {bond.GetBeginAtomIdx()} and {bond.GetEndAtomIdx()}')

    # break the bond
    rw_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    if plts:
        plt_indices_bond(rw_mol, ax=axs[1], remove_hs=False)
        axs[1].set_title('hydrogen bond broken')

    # begin atom is always that of the molecule
    heteroatom = rw_mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
    assert heteroatom.GetSymbol() not in ('C', 'H')

    heteroatom.SetNoImplicit(True)
    heteroatom.SetFormalCharge(heteroatom.GetFormalCharge() + 1)

    if plts:
        plt_indices_bond(rw_mol, ax=axs[2], remove_hs=False)
        axs[2].set_title('formal charge set')

    if keep_h:  # need to balance valence by either adding an H atom or creating a radical
        rw_mol = add_one_h_for_atom(rw_mol, heteroatom.GetIdx())[0]
    else:
        heteroatom.SetNumRadicalElectrons(heteroatom.GetNumRadicalElectrons() + 1)

    if plts:
        plt_indices_bond(rw_mol, ax=axs[3], remove_hs=False)
        axs[3].set_title('hydrogen or radical added')

    frags = Chem.GetMolFrags(rw_mol, asMols=True, sanitizeFrags=False)

    # here we are assuming that fragment 0 is always the molecule, TODO: is this always the case?
    mol = frags[0]
    add = frags[1]
    if not any([at.GetSymbol() == 'H' for at in add.GetAtoms()]):
        logger.warning('adding hydrogen for adduct even though adduct does not contain hydrogen!')

    if plts:
        plt_indices_bond(mol, ax=axs[4], remove_hs=False)
        axs[4].set_title('final molecule')

    return mol


def _test_adducts():
    mol = AddHs(Chem.MolFromSmiles('CCOCCO'))
    for add in SUPPORTED_ADDUCT_TYPES + ['M+']:
        mols_with_add = get_mol_with_adduct(mol, add, return_mode='all')
        for mol_with_add in mols_with_add:
            mol_cleaned = Mol(mol_with_add)
            Chem.SanitizeMol(mol_cleaned)

            # this does not yield the same molecule for H+ adducts!
            mol_red = RemoveHs(mol_with_add)

            f = CalcMolFormula(mol_with_add)
            f_cleaned = CalcMolFormula(mol_cleaned)
            f_red = CalcMolFormula(mol_red)

            fig, axs = plt.subplots(nrows=3)
            mplt_mol(mol_with_add, ax=axs[0], remove_hs=False)
            axs[0].set_title(f'{add}, formula: {f}')
            mplt_mol(mol_cleaned, ax=axs[1], remove_hs=False)
            axs[1].set_title(f'cleaned, formula: {f_cleaned}')
            mplt_mol(mol_red, ax=axs[2], remove_hs=False)
            axs[2].set_title(f'reduced, formula: {f_red}')
            plt.show()
            # check that cleaning does not add/remove H atoms
            cd_explicit = CompoundDict(f)
            cd_cleaned = CompoundDict(f_cleaned)
            assert cd_explicit == cd_cleaned, f'{f} != {f_cleaned} for {add}'


def _test_find_bond_locations():
    mol = AddHs(Chem.MolFromSmiles('CCOCCO'))
    assert find_available_bond_locations_for_adduct(mol) == [2, 5]


def _test_m_plus():
    fig, axs = plt.subplots(nrows=2)
    mol = AddHs(Chem.MolFromSmiles('CCOCCO'))
    mol_plus = _increase_formal_charge(mol, return_mode='index', idx=2)
    plt_indices_bond(mol, ax=axs[0])
    plt_indices_bond(mol_plus, ax=axs[1])
    plt.show()


def test_charge_transfer():
    mol = AddHs(Chem.MolFromSmiles('CCOCCO'))
    mol_with_add = get_mol_with_adduct(mol, add='M+', return_mode='index', idx=2)
    mol_with_add_merged = steal_pos_charge_from_adduct(mol_with_add, keep_h=True, plts=False)
    mol_with_charge = swap_charge_h_hetero(mol_with_add_merged, heteroatom_idx=2, from_hetero=True)

    fig, axs = plt.subplots(nrows=2, ncols=2)
    plt_indices_bond(mol, ax=axs[0, 0], remove_hs=False)
    plt_indices_bond(mol_with_add, ax=axs[0, 1], remove_hs=False)
    plt_indices_bond(mol_with_add_merged, ax=axs[1, 0], remove_hs=False)
    plt_indices_bond(mol_with_charge, ax=axs[1, 1], remove_hs=False)
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    _test_adducts()

    """
    from LipidCalculator.compound_groups.intact_polar_lipids.generate_ipl import ipl_automatic_bonds

    name = 'PG DAG C32:0'
    mol = ipl_automatic_bonds(name.split(), plts=False, idx_plt=False, split_chain=False)
    plt_indices_bond(mol)

    adduct_pos = find_available_bond_locations_for_adduct(mol)
    print(adduct_pos)
    """
    pass
