"""
Core lipids (without head groups) of the ArchLips archaeal lipid library (Zheng et al., Nat. Commun. 2025,
doi:10.1038/s41467-025-67286-3) under their ArchLips names, e.g. GDGT-2a, OH-GMGT-1, Di-uns-GDGT-0a.

ArchLips labels regioisomers (positions of rings, hydroxyl groups, double bonds) by suffixes (GDGT-2a, GDGT-2b,
2OH-GDD-2_3), which are only defined by its structures, so the structures are stored in archlips_cores.json and
can be used as core pieces (head groups bond to the free primary OH groups).

The table is created from the ArchLips spectral library (msp) with
    python -m LipidCalculator.compound_groups.intact_polar_lipids.archlips_cores path/to/Archlips_Full_spectral_library.msp
"""
import json
import os
import re
import sys
from functools import lru_cache

PATH_TABLE: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'archlips_cores.json')
UNSATURATION: dict[str, int] = {'Mono': 1, 'Di': 2, 'Tri': 3, 'Tetra': 4, 'Penta': 5, 'Hexa': 6, 'Hepta': 7, 'Octa': 8}
TETRAETHER_FAMILIES: list[str] = ['GDGT', 'BDGT', 'PDGT', 'GDD', 'BDD', 'PDD', 'GTGT', 'GMGT']
# e.g. Di-uns-Homo-Me-OH-GDGT-2a: unsaturation, methylation, hydroxylation, family, number of rings and isomer
CORE_NAME = re.compile(
    r'(?:(?P<uns>' + '|'.join(UNSATURATION) + r')-uns-)?(?P<homo>Homo-\d?Me-)?(?P<oh>\d?OH-)?'
    r'(?P<family>' + '|'.join(TETRAETHER_FAMILIES) + r')-(?P<rings>\d|[Cc]ren)(?P<isomer>[a-z]?(?:_\d+)?)'
)


@lru_cache(maxsize=None)
def archlips_cores() -> dict[str, str]:
    """ArchLips core name --> SMILES"""
    if not os.path.exists(PATH_TABLE):
        return {}
    with open(PATH_TABLE, encoding='utf-8') as f:
        return json.load(f)


def is_archlips_core(name: str) -> bool:
    return name in archlips_cores()


@lru_cache(maxsize=None)
def archlips_core_smiles(name: str) -> str:
    """SMILES with Fr placeholders (marked by atom map numbers) at the free primary OH groups, those of the polyols
    first (head groups), then those at chain ends (GDDs)"""
    from rdkit import Chem

    mol = Chem.RWMol(Chem.MolFromSmiles(archlips_cores()[name]))

    def is_polyol_oh(o) -> bool:
        # O-CH2-CH(-O-chain): sn-1 of glycerol, C1 of butanetriol/pentanetriol
        c = o.GetNeighbors()[0]
        return any(nb.GetSymbol() == 'C' and any(o2.GetSymbol() == 'O' and o2.GetDegree() == 2
                                                 for o2 in nb.GetNeighbors())
                   for nb in c.GetNeighbors() if nb.GetIdx() != o.GetIdx())

    primary_ohs = [a for a in mol.GetAtoms() if a.GetSymbol() == 'O' and a.GetDegree() == 1
                   and a.GetNeighbors()[0].GetSymbol() == 'C' and a.GetNeighbors()[0].GetTotalNumHs() == 2]
    primary_ohs.sort(key=lambda o: (not is_polyol_oh(o), o.GetIdx()))
    for k, o in enumerate(primary_ohs, 1):
        fr = mol.AddAtom(Chem.Atom(87))
        mol.GetAtomWithIdx(fr).SetAtomMapNum(k)
        mol.AddBond(o.GetIdx(), fr, Chem.BondType.SINGLE)
    out = mol.GetMol()
    Chem.SanitizeMol(out)
    # rdkit writes O bonded to Fr as [O], which would keep no H once the placeholder is removed
    return Chem.MolToSmiles(out).replace('[O]', 'O')


def build_table(path_msp: str, path_out: str = PATH_TABLE) -> dict[str, str]:
    """Collect the head-group-free tetraether cores of an ArchLips msp library (canonical SMILES without
    stereochemistry)"""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')

    table: dict[str, str] = {}
    name = None
    with open(path_msp, encoding='utf-8', errors='replace') as f:
        for line in f:
            key, _, value = line.partition(':')
            key = key.strip().lower()
            if key == 'name':
                name = value.strip()
            elif key == 'smiles' and name and name not in table and CORE_NAME.fullmatch(name):
                mol = Chem.MolFromSmiles(value.strip())
                Chem.RemoveStereochemistry(mol)
                table[name] = Chem.MolToSmiles(mol)
    with open(path_out, 'w', encoding='utf-8') as f:
        json.dump(dict(sorted(table.items())), f, indent=1)
    archlips_cores.cache_clear()
    return table


if __name__ == '__main__':
    print(f'{len(build_table(sys.argv[1]))} cores written to {PATH_TABLE}')
