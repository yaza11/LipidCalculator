from rdkit import Chem
from rdkit.Chem import Mol, Atom

from LipidCalculator.rdkit.plotting import plt_indices_bond, mplt_mol


def find_sigma_cleavage_positions(mol: Chem.Mol) -> list[tuple[int, int]]:
    # need charged and uncharged atom
    possible_sigma_cleavage_positions = []
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        if (atom.GetFormalCharge() == 0) or (atom.GetNumRadicalElectrons() == 0):
            continue
        for atom_b in atom.GetNeighbors():
            if atom_b.GetFormalCharge() == 0:
                possible_sigma_cleavage_positions.append((atom_idx, atom_b.GetIdx()))
    return possible_sigma_cleavage_positions


def get_sigma_cleaved(
        mol, charged_atom_idx, neutral_atom_idx, plts: bool = False
) -> tuple[Mol, Mol]:
    """Break sigma bond between two atoms. It is assumed that one atom is
    charged (otherwise we would not care about this bond in the fragment formation).
    """
    # Clone the molecule
    rw_mol = Chem.RWMol(mol)

    charged_atom: Atom = rw_mol.GetAtomWithIdx(charged_atom_idx)
    assert charged_atom.GetNumRadicalElectrons() > 0, 'charged atom must have a radical'
    neutral_atom: Atom = rw_mol.GetAtomWithIdx(neutral_atom_idx)
    # assert charged_atom.GetFormalCharge() > 0, \
    #     f'atom with {charged_atom_idx} does not have positive charge'
    # assert neutral_atom.GetFormalCharge() == 0, \
    #     f'atom with {neutral_atom_idx} is not neutral'

    # reduce valence of charged mol
    initial_radicals = charged_atom.GetNumRadicalElectrons()
    charged_atom.SetNumRadicalElectrons(initial_radicals - 1)
    charged_atom.SetNumExplicitHs(2)
    neutral_atom.SetNumRadicalElectrons(neutral_atom.GetNumRadicalElectrons() + 1)

    # Remove the bond
    rw_mol.RemoveBond(charged_atom_idx, neutral_atom_idx)

    if plts:
        plt_indices_bond([rw_mol])

    # Sanitize before fragmenting
    Chem.SanitizeMol(rw_mol)

    # cleaning can mess with radicals and explicit Hs, explictly setting number of explicit Hs is not very safe
    assert rw_mol.GetAtomWithIdx(charged_atom_idx).GetNumRadicalElectrons() == (initial_radicals - 1)

    # Get fragments as separate Mols
    frags = Chem.GetMolFrags(rw_mol, asMols=True, sanitizeFrags=True)
    return frags


if __name__ == "__main__":
    mol = Chem.MolFromSmiles('CCC[C+]C')
    # print(Chem.GetFormalCharge(mol))
    plt_indices_bond([mol])

    # smiles = Chem.MolToSmiles(mol)

    print(find_sigma_cleavage_positions(mol))

    # frags = get_sigma_cleaved(mol, 3, 4)
    # plt_indices_bond(frags)
