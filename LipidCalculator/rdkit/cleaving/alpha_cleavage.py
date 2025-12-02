"""Radical on functional group initiates sigma bond cleavage on alpha carbon"""
from rdkit import Chem
from rdkit.Chem import Mol, Atom, AddHs

from LipidCalculator.rdkit.cleaving.util import SUPPORTED_HETEROATOMS
from LipidCalculator.rdkit.plotting import plt_indices_bond
from LipidCalculator.rdkit.util import bump_bond_order, check_hs_treated_as_neighbors


def find_alpha_cleavage_positions(mol: Chem.Mol) -> list[tuple[int, int]]:
    # need heteroatom with radical and charge
    # need alpha carbon next to it
    # need another carbon next to alpha carbon
    assert check_hs_treated_as_neighbors(mol)

    possible_alpha_cleavage_positions: list[tuple[int, int]] = []
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        # atom with charge must be heteroatom with radical
        if (atom.GetNumRadicalElectrons() == 0) or (atom.GetFormalCharge() <= 0) or (
                atom.GetSymbol() not in SUPPORTED_HETEROATOMS):
            continue
        # next atom must be C atom
        for atom_c in atom.GetNeighbors():
            if atom_c.GetSymbol() != 'C':
                continue
            # atom after that must be C atom as well
            for atom_acc in atom_c.GetNeighbors():
                # exclude the first atom (shouldn't be necessary since the
                # start atom is a heteroatom
                # if (atom_acc_idx := atom_acc.GetIdx()) == atom_idx:
                #     continue
                if atom_acc.GetSymbol() == 'C':
                    possible_alpha_cleavage_positions.append(
                        (atom_idx, atom_acc.GetIdx())
                    )

    return possible_alpha_cleavage_positions


def get_alpha_cleaved(
        mol: Mol,
        atom_charged_radical_idx: int,
        atom_accepting_radical_idx: int
):
    assert check_hs_treated_as_neighbors(mol)

    rw_mol = Chem.RWMol(Mol(mol))

    atom_charged_radical = rw_mol.GetAtomWithIdx(atom_charged_radical_idx)
    atom_accepting_radical = rw_mol.GetAtomWithIdx(atom_accepting_radical_idx)
    assert atom_charged_radical.GetNumRadicalElectrons() > 0, \
        'Atom should have radical'
    assert atom_charged_radical.GetSymbol() in SUPPORTED_HETEROATOMS
    assert atom_charged_radical.GetFormalCharge() > 0

    # determine index of alpha carbon
    # alpha Carbon is between charged radical and atom accepting the radical
    idcs_neighbours_charged: set[int] = set(nbr.GetIdx() for nbr in atom_charged_radical.GetNeighbors())
    idcs_neighbours_acceptor: set[int] = set(nbr.GetIdx() for nbr in atom_accepting_radical.GetNeighbors())
    idcs_shared: list[int] = sorted(idcs_neighbours_charged & idcs_neighbours_acceptor)

    assert len(idcs_shared) == 1, \
        f'Assuming exactly one atom is bonded to the atoms with specified indices, found {len(idcs_shared)}'
    atom_alpha_carbon: Atom = rw_mol.GetAtomWithIdx(idcs_shared[0])
    atom_alpha_carbon_idx = atom_alpha_carbon.GetIdx()

    # radical moves from functional group to one of the carbon atoms bonded to the alpha Carbon
    atom_charged_radical.SetNumRadicalElectrons(atom_charged_radical.GetNumRadicalElectrons() - 1)
    atom_accepting_radical.SetNumRadicalElectrons(atom_accepting_radical.GetNumRadicalElectrons() + 1)
    # bond between accepting and alpha breaks
    rw_mol.RemoveBond(atom_accepting_radical_idx, atom_alpha_carbon_idx)
    # bond order between charged radical and carbon increases
    bump_bond_order(rw_mol, atom_alpha_carbon_idx, atom_charged_radical_idx)

    Chem.SanitizeMol(rw_mol)
    # rw_mol = Chem.AddHs(rw_mol)
    # Get fragments as separate Mols
    frags = Chem.GetMolFrags(rw_mol, asMols=True, sanitizeFrags=True)
    assert len(frags) == 2
    return [AddHs(f) for f in frags]


if __name__ == "__main__":
    # reproducing example from https://en.wikipedia.org/wiki/Fragmentation_(mass_spectrometry)
    mol = AddHs(Chem.MolFromSmiles("CCC([OH+])CC"))
    plt_indices_bond([mol], remove_hs=True)

    print(find_alpha_cleavage_positions(mol))

    frags = get_alpha_cleaved(mol, 3, 1)
    plt_indices_bond(frags, remove_hs=True)
