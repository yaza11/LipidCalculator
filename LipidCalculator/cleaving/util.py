from rdkit import Chem

PERIODIC_TABLE = Chem.GetPeriodicTable()
SUPPORTED_HETEROATOMS = {
    'O', 'N', 'P', 'S'
}
SUPPORTED_HETEROATOM_NUMS = set(PERIODIC_TABLE.GetAtomicNumber(atom_abbr) for atom_abbr in SUPPORTED_HETEROATOMS)
