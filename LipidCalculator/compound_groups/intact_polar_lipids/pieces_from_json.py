import os
import json

import pandas as pd


def smiles_to_formula(smiles):
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return rdMolDescriptors.CalcMolFormula(mol)


PLACEHOLDER_ELEMENTS: list[str] = ['Fr', 'Cs', 'Rb']
# placeholders at which chains are attached (Fr is used for head groups)
CHAIN_PLACEHOLDERS: tuple[str, ...] = ('Cs', 'Rb')


def implicit_chain(mol, placeholder_idx: int) -> tuple[int, int]:
    """
    Carbon atoms and C=C double bonds of a chain that are contained in the piece: all carbons connected to the chain
    placeholder through C-C bonds, e.g. the carbonyl carbon of an acyl chain (1, 0) or C1-C5 of sphingosine (5, 1).
    Chain lengths (lipid shorthand) include these carbons.
    """
    from rdkit import Chem

    start = mol.GetAtomWithIdx(placeholder_idx).GetNeighbors()[0]
    carbons: list[int] = []
    previous, current = placeholder_idx, start
    while current is not None and current.GetSymbol() == 'C':
        assert not current.IsInRing(), f'chain placeholder is connected to a ring carbon'
        carbons.append(current.GetIdx())
        following = [n for n in current.GetNeighbors()
                     if n.GetSymbol() == 'C' and n.GetIdx() not in (previous, *carbons)]
        assert len(following) <= 1, f'carbons at the chain placeholder are branched'
        previous, current = current.GetIdx(), (following[0] if following else None)
    n_double_bonds = sum(
        bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetBeginAtomIdx() in carbons and bond.GetEndAtomIdx() in carbons
        for bond in mol.GetBonds()
    )
    return len(carbons), n_double_bonds


path_file_blocks = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'package.json')
with open(path_file_blocks, 'r', encoding='utf-8') as f:
    IPL_PIECES: dict[str, str | list] = json.load(f)

ABBREVIATIONS: list[str] = []
GROUPS: list[str] = []
_entries = []
for name_type, members in IPL_PIECES.items():
    if name_type == 'group':
        continue
    for struct in members:
        ABBREVIATIONS.append(struct['abbreviation'])
        GROUPS.append(name_type)
        _entries.append(struct | dict(group=name_type))
DATA_FRAME_IPL_PIECES = pd.DataFrame(_entries)

ABBREVIATION_TO_GROUP: dict[str, str] = dict(zip(ABBREVIATIONS, GROUPS))


def _implicit_chains() -> dict[tuple[str, str], tuple[int, int]]:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')

    out = {}
    for entry in _entries:
        if not any(f'[{p}]' in entry['SMILES'] for p in CHAIN_PLACEHOLDERS):
            continue
        mol = Chem.MolFromSmiles(entry['SMILES'])
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in CHAIN_PLACEHOLDERS:
                out[(entry['abbreviation'], atom.GetSymbol())] = implicit_chain(mol, atom.GetIdx())
    return out


# carbons of chains contained in the pieces for the chain placeholders, e.g. ('CER', 'Rb'): (5, 1)
CORE_IMPLICIT_CHAIN: dict[tuple[str, str], tuple[int, int]] = _implicit_chains()

# groups whose members can serve as the core of a molecule (exactly one per molecule)
CORE_GROUPS: list[str] = ['core lipids', 'quinones', 'carotenoids']
# cores for which isoprenoid chains are specified in isoprene units instead of carbon atoms (e.g. MK(7:7))
CHAINS_IN_ISOPRENE_UNITS: set[str] = {
    e['abbreviation'] for e in _entries if e.get('chain_notation') == 'isoprene units'
}


def get_entry_by_abbreviation(name: str) -> pd.Series:
    match = DATA_FRAME_IPL_PIECES.abbreviation == name
    assert match.sum() == 1, f'got {match.sum()} matches for {name}'
    return DATA_FRAME_IPL_PIECES.loc[match, :]


def get_smiles(name: str) -> str:
    e = get_entry_by_abbreviation(name)
    return e.SMILES.values[0]


if __name__ == '__main__':
    DATA_FRAME_IPL_PIECES.loc[:, 'formula'] = DATA_FRAME_IPL_PIECES.SMILES.apply(smiles_to_formula)

    print(get_entry_by_abbreviation('DAG'))
    print(get_smiles('DAG'))
