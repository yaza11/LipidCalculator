from rdkit import Chem
from rdkit.Chem import Mol, Atom

from LipidCalculator.cleaving.util import SUPPORTED_HETEROATOMS
from LipidCalculator.rdkit.plotting import plt_indices_bond

SUPPORTED_OTHER_ATOMS = SUPPORTED_HETEROATOMS.copy()
SUPPORTED_OTHER_ATOMS.add('C')


def find_inductive_cleavage_positions(mol: Chem.Mol) -> list[tuple[int, int]]:
    # need heteroatom with charge
    # need atom with single bond to heteroatom
    possible_inductive_cleavage_positions: list[tuple[int, int]] = []
    for atom in mol.GetAtoms():
        if (atom.GetFormalCharge() <= 0) or (atom.GetSymbol() not in SUPPORTED_HETEROATOMS):
            continue
        for atom_nghbr in atom.GetNeighbors():
            if atom_nghbr.GetSymbol() in SUPPORTED_OTHER_ATOMS:
                possible_inductive_cleavage_positions.append((atom_nghbr.GetIdx(), atom.GetIdx()))
    return possible_inductive_cleavage_positions


def get_inductively_cleaved(
        mol: Mol, other_atom_idx: int, hetero_atom_idx: int, plts: bool = False
) -> tuple[Mol, Mol]:
    """
    Split molecules at the specified indices. Assumes that atom 1 is a C and
    atom 2 a heteroatom. Heteroatom will get H atom bonded and a formal charge
    (in reality, it is assumed that heteroatom was protonated before cleavage)

    The other part (with the heteroatom) is considered the neutral loss
    """
    # Clone the molecule
    rw_mol = Chem.RWMol(Mol(mol))

    # Ensure atom1 is carbon, atom2 is heteroatom
    c_atom: Atom = rw_mol.GetAtomWithIdx(other_atom_idx)
    assert c_atom.GetSymbol() in SUPPORTED_OTHER_ATOMS, \
        f'idx provided for {other_atom_idx=} should be in {SUPPORTED_OTHER_ATOMS} but is {c_atom.GetSymbol()}'

    hetero_atom: Atom = rw_mol.GetAtomWithIdx(hetero_atom_idx)
    assert hetero_atom.GetSymbol() in SUPPORTED_HETEROATOMS, \
        f'Heteroatom of type {hetero_atom.GetSymbol()} not supported (only {SUPPORTED_HETEROATOMS} are supported)'
    assert hetero_atom.GetFormalCharge() > 0, f'Formal charge of heteroatom should be greater than 0'

    hetero_atom.SetNoImplicit(True)
    c_atom.SetNoImplicit(True)

    print(f'Hs before inductive cleavage: het: {hetero_atom.GetNumExplicitHs()}, c: {c_atom.GetNumExplicitHs()}')
    print(
        f'radicals before inductive cleavage: het: {hetero_atom.GetNumRadicalElectrons()}, c: {c_atom.GetNumExplicitHs()}')

    # heteroatom gains an electron from the double bond
    hetero_atom.SetFormalCharge(hetero_atom.GetFormalCharge() - 1)
    # C atom gives one electron
    c_atom.SetFormalCharge(c_atom.GetFormalCharge() + 1)
    # Remove the bond
    rw_mol.RemoveBond(c_atom.GetIdx(), hetero_atom.GetIdx())

    if plts:
        plt_indices_bond([rw_mol])

    # Sanitize before fragmenting
    Chem.SanitizeMol(rw_mol)

    print(f'Hs after inductive cleavage: het: {hetero_atom.GetNumExplicitHs()}, c: {c_atom.GetNumExplicitHs()}')
    print(
        f'radicals after inductive cleavage: het: {hetero_atom.GetNumRadicalElectrons()}, c: {c_atom.GetNumExplicitHs()}')

    # Get fragments as separate Mols
    frags = Chem.GetMolFrags(rw_mol, asMols=True, sanitizeFrags=True)
    return frags


if __name__ == "__main__":
    # this is the exact result from https://en.wikipedia.org/wiki/Fragmentation_(mass_spectrometry)
    mol = Chem.MolFromSmiles('CC[O+]CC')
    # draw with indices
    plt_indices_bond([mol])

    print(find_inductive_cleavage_positions(mol))

    frags = get_inductively_cleaved(mol, 1, 2)
    plt_indices_bond(frags)
