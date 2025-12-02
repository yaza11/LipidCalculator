from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdmolops import SanitizeMol

from LipidCalculator.rdkit.util import add_one_h_for_atom

PERIODIC_TABLE = Chem.GetPeriodicTable()
SUPPORTED_HETEROATOMS = {
    'O', 'N', 'P', 'S'
}
SUPPORTED_HETEROATOM_NUMS = set(PERIODIC_TABLE.GetAtomicNumber(atom_abbr) for atom_abbr in SUPPORTED_HETEROATOMS)


def post_cleavage_fill_radicals_with_h(mol: Mol):
    rw_mol = Chem.RWMol(mol)
    for atm in rw_mol.GetAtoms():
        n_hs_missing = atm.GetNumRadicalElectrons()
        if n_hs_missing == 0:
            continue
        print('Radicals before:', n_hs_missing)
        for _ in range(n_hs_missing):
            rw_mol, _ = add_one_h_for_atom(mol, atm.GetIdx())
        atm.SetNumRadicalElectrons(0)
        print('Radicals after:', atm.GetNumRadicalElectrons())
    mol_new = rw_mol.GetMol()
    SanitizeMol(mol_new)
    return mol_new
