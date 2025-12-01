from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Mol, Atom, AddHs, BondType
from rdkit.Chem.Descriptors import ExactMolWt

from LipidCalculator.rdkit.cleaving.util import post_cleavage_fill_radicals_with_h
from LipidCalculator.rdkit.plotting import plt_indices_bond, mplt_mol
from LipidCalculator.rdkit.util import check_hs_treated_as_neighbors, add_one_h_for_atom


def find_sigma_cleavage_positions(mol: Chem.Mol) -> list[tuple[int, int]]:
    # need charged and uncharged atom
    assert check_hs_treated_as_neighbors(mol)
    possible_sigma_cleavage_positions = []
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        if atom.GetFormalCharge() == 0:
            continue
        for atom_b in atom.GetNeighbors():
            if atom_b.GetFormalCharge() != 0:
                continue
            if mol.GetBondBetweenAtoms(atom_idx, atom_b.GetIdx()).GetBondType() != BondType.SINGLE:
                continue
            possible_sigma_cleavage_positions.append((atom_idx, atom_b.GetIdx()))
    return possible_sigma_cleavage_positions


def get_sigma_cleaved(
        mol, charged_atom_idx, neutral_atom_idx, plts: bool = False
) -> list[Mol]:
    """Break sigma bond between two atoms. It is assumed that one atom is
    charged (otherwise we would not care about this bond in the fragment formation).
    """
    assert check_hs_treated_as_neighbors(mol)
    # Clone the molecule
    rw_mol = Chem.RWMol(Mol(mol))

    charged_atom: Atom = rw_mol.GetAtomWithIdx(charged_atom_idx)
    # assert charged_atom.GetNumRadicalElectrons() > 0, 'charged atom must have a radical'
    neutral_atom: Atom = rw_mol.GetAtomWithIdx(neutral_atom_idx)

    charged_atom.SetNumRadicalElectrons(charged_atom.GetNumRadicalElectrons() + 1)
    neutral_atom.SetNumRadicalElectrons(neutral_atom.GetNumRadicalElectrons() + 1)

    # Remove the bond
    rw_mol.RemoveBond(charged_atom_idx, neutral_atom_idx)

    if plts:
        plt_indices_bond([rw_mol])

    # Get fragments as separate Mols
    frags = Chem.GetMolFrags(rw_mol, asMols=True, sanitizeFrags=True)
    assert len(frags) == 2
    return [AddHs(f) for f in frags]


if __name__ == "__main__":
    # this is a bit weird, we would create two radicals at the ion side ...
    fig, axs = plt.subplots(nrows=3)
    mol = AddHs(Chem.MolFromSmiles('CCCC[CH+]C'))
    # add hydrogen, add radical not allowed :(
    # mol = Chem.RWMol(mol)
    # carb = mol.GetAtomWithIdx(4)
    # carb.SetNumRadicalElectrons(1)
    # mol, h_idx = add_one_h_for_atom(mol.GetMol(), 4)
    plt_indices_bond(mol, remove_hs=False, ax=axs[0])
    Chem.SanitizeMol(mol)
    # print(Chem.GetFormalCharge(mol))
    axs[0].set_title(f'M: {ExactMolWt(mol):.4f}')

    # smiles = Chem.MolToSmiles(mol)

    print(find_sigma_cleavage_positions(mol))

    frags = get_sigma_cleaved(mol, 4, 5)
    plt_indices_bond(frags, remove_hs=False, ax=axs[1])
    axs[1].set_title(f'M: {ExactMolWt(frags[0]):.4f}')

    compensated = post_cleavage_fill_radicals_with_h(frags[0])
    plt_indices_bond(compensated, remove_hs=False, ax=axs[2])
    plt.show()
