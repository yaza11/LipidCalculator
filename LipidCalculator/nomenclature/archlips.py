"""
Names of the ArchLips archaeal lipid library (Zheng et al., Nat. Commun. 2025, doi:10.1038/s41467-025-67286-3), the
standard for archaeal lipids:
- head groups are prefixed with -, a second head group (other glycerol of tetraethers) follows in parentheses, e.g.
  Gly-GDGT-0, 2Gly-(PG-)GDGT-0, glycosyl head groups are Gly, 2Gly, 3Gly, ..., phosphatidic acid is PA
- diethers in LIPID MAPS style with the chains (C atoms:double bonds, modification (OH) or (Me)) in parentheses,
  e.g. Archaeol(20:0_25:0), PG-Archaeol(20:0(OH)_20:0), cardiolipin analogues likewise, e.g. DGD-PA(20:0_20:0_20:0_20:0)
- tetraethers with traditional names: [<n>-uns-][Homo-<n>Me-][<n>OH-]<family>-<rings><isomer>, e.g. GDGT-0,
  OH-GDGT-1, Di-uns-GDGT-0a, families GDGT, BDGT, PDGT, GDD, BDD, PDD, GTGT, GMGT, crenarchaeol as -Cren/-cren,
  regioisomers labeled by a suffix (GDGT-2a, GDGT-2b, 2OH-GDD-2_3)
- quinones with the side chain in isoprene units, e.g. MK(7:7)
"""
import re
from functools import lru_cache

from LipidCalculator.compound_groups.intact_polar_lipids.archlips_cores import (
    CORE_NAME, UNSATURATION, TETRAETHER_FAMILIES, archlips_cores, is_archlips_core
)
from LipidCalculator.compound_groups.intact_polar_lipids.isoprenoid_blocks import is_block_core, parse_block_core
from LipidCalculator.compound_groups.intact_polar_lipids.pieces_from_json import (
    ABBREVIATIONS, ABBREVIATION_TO_GROUP, CORE_GROUPS
)
from LipidCalculator.nomenclature.lipid import Lipid, NotSupported

# ArchLips head group -> LipidCalculator abbreviation (all others are the same)
HEAD_MAP: dict[str, str] = {'Gly': '1G', '2Gly': '2G', '3Gly': '3G', 'PA': 'P', 'BL': 'TMHS'}
HEAD_MAP_REVERSE: dict[str, str] = {v: k for k, v in HEAD_MAP.items()}
# other spellings that are only parsed
HEAD_ALIASES: dict[str, str] = {'AC-2Gly': 'Ac-2Gly'}
# cardiolipin analogues and other dimeric archaeols
DIMERS: list[str] = ['S-TeGD-PA', 'TeGD-PA', 'S-TGD-PA', 'S-DGD-PA', 'S-MGD-PA', 'TGD-PA', 'DGD-PA', 'MGD-PA', 'BPG',
                     'PGPGP']
QUINONES: list[str] = ['MK', 'DMK', 'MMK', 'MTK', 'BDTQ', 'CQ', 'UQ', 'MP', 'SQ', 'PQ']
# ArchLips quinone -> LipidCalculator abbreviation
QUINONE_MAP: dict[str, str] = {'SQ': 'SQ-quinone'}
UNSATURATION_PREFIX: dict[int, str] = {v: k for k, v in UNSATURATION.items()}
TETRAETHER: str = (r'(?:(?:' + '|'.join(UNSATURATION) + r')-uns-)?(?:Homo-\d?Me-)?(?:\d?OH-)?'
                                                        r'(?:' + '|'.join(TETRAETHER_FAMILIES) + r')(?:-|$)')
TETRAETHER_NAME = re.compile(r'^(?P<head>.*?)-?(?P<core>' + TETRAETHER + r'.*)$')
# tetraether pieces of package.json, e.g. GDGT0, OH-GDGT1, Homo-Me-GDGT-Cren (GDGT5' is not named)
TETRAETHER_PIECE = re.compile(r'(?P<homo>Homo-\d?Me-)?(?P<oh>\d?OH-)?(?P<family>' + '|'.join(TETRAETHER_FAMILIES)
                              + r')-?(?P<rings>\d|Cren)')
# archaeol and its hydroxylated forms in LipidCalculator with their isoprenoid chains in ArchLips notation
ARCHAEOL_CHAINS: dict[str, str] = {'AR': '20:0_20:0', 'OH-AR': '20:0(OH)_20:0', '2OH-AR': '20:0(OH)_20:0(OH)'}


def head_abbreviation(head: str) -> str:
    head = HEAD_ALIASES.get(head, HEAD_MAP.get(head, head))
    if head not in ABBREVIATIONS:
        raise NotSupported(f'head {head} missing')
    return head


def heads_of(head: str) -> list[str]:
    """'' --> [], 'Gly' --> ['1G'], '2Gly-(PG-)' --> ['2G', 'PG'] (second head group in parentheses)"""
    if not head:
        return []
    if m := re.fullmatch(r'(?P<h1>.*?)-\((?P<h2>.*)-\)', head):
        return [head_abbreviation(m['h1']), head_abbreviation(m['h2'])]
    return [head_abbreviation(head)]


def tetraether_piece(core: str) -> tuple[str, int]:
    """LipidCalculator core piece for a tetraether name and number of double bonds in the biphytanyl chains:
    the ArchLips structure if the library has it (exact isomer), otherwise the piece with the same formula
    (isomers like 2a and 2b share a piece), e.g. Di-uns-GDGT-2a --> (GDGT-2a, 2) or (GDGT2, 2)"""
    if is_archlips_core(core):
        return core, 0
    n_double_bonds = 0
    if m := re.match(r'(' + '|'.join(UNSATURATION) + r')-uns-', core):
        n_double_bonds, core = UNSATURATION[m.group(1)], core[m.end():]
        if is_archlips_core(core):
            return core, n_double_bonds
    if re.search(r'(?i)-cren(?:[a-z]|_\d+)?$', core):
        core = re.sub(r'(?i)-cren(?:[a-z]|_\d+)?$', '-Cren', core)
    elif m := re.fullmatch(r'(.*)-(\d+)(?:[a-z]|_\d+)?', core):
        core = f'{m.group(1)}{m.group(2)}'
    if core == 'GDGT-Cren':
        core = 'GDGT5'  # the GDGT5 piece is crenarchaeol
    if core not in ABBREVIATIONS:
        raise NotSupported(f'core {core} missing')
    return core, n_double_bonds


def parse(name: str) -> Lipid:
    """Building blocks for an ArchLips name"""
    # isotopologues have the same structure
    name = re.sub(r'\[M\+\d\]$', '', name.strip())
    if m := re.fullmatch(r'(?P<head>.*?)-?Archaeol\s*\((?P<chains>.*)\)', name):
        # modifications (OH, Me) are added to AR at the positions of the database
        return Lipid(tuple(heads_of(m['head'])), 'AR', core_chains=m['chains'], family='archaeol')
    if m := re.fullmatch(r'(?P<head>.*?)-?(?P<cls>' + '|'.join(map(re.escape, DIMERS)) + r')\((?P<chains>.*)\)', name):
        return Lipid(tuple(heads_of(m['head'])), m['cls'], core_chains=m['chains'], family='dimeric archaeol')
    if m := re.fullmatch(r'(?P<cls>' + '|'.join(QUINONES) + r')\((?P<chains>\d+:\d+(?:\(OH\))?)\)', name):
        return Lipid(core=QUINONE_MAP.get(m['cls'], m['cls']), core_chains=m['chains'], family='quinone')
    if m := TETRAETHER_NAME.match(name):
        heads = heads_of(m['head'])
        core, n_double_bonds = tetraether_piece(m['core'])
        # isomers that are not in the core table (e.g. position of the head group in GDDs) keep their label
        isomer = None
        if not is_archlips_core(core) and (m_core := CORE_NAME.fullmatch(m['core'])):
            isomer = m_core['isomer'] or None
        return Lipid(tuple(heads), core, core_double_bonds=n_double_bonds, isomer=isomer,
                     family='tetraether two heads' if len(heads) == 2 else 'tetraether')
    if ABBREVIATION_TO_GROUP.get(name) in CORE_GROUPS:
        return Lipid(core=name, family='carotenoid / other')
    raise NotSupported(f'{name} is not an ArchLips name')


# ---------- formatting ----------
@lru_cache(maxsize=None)
def _core_names_by_structure() -> dict[str, str]:
    """canonical SMILES of the core --> ArchLips name (the shortest name if several share a structure)"""
    out: dict[str, str] = {}
    for name, smiles in sorted(archlips_cores().items(), key=lambda kv: (len(kv[0]), kv[0])):
        out.setdefault(smiles, name)
    return out


def _canonical_core_smiles(lipid: Lipid) -> str | None:
    from rdkit import Chem
    try:
        mol = lipid.core_only().mol()
    except Exception:
        return None
    mol = Chem.Mol(mol)
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def _species_tetraether_name(lipid: Lipid) -> str | None:
    """[<n>-uns-][Homo-<n>Me-][<n>OH-]<family>-<rings> without isomer suffix"""
    core, n_double_bonds = lipid.core, lipid.core_double_bonds
    if is_archlips_core(core):
        m = CORE_NAME.fullmatch(core)
        n_double_bonds += UNSATURATION.get(m['uns'], 0) if m['uns'] else 0
        homo, oh, family, rings = m['homo'] or '', m['oh'] or '', m['family'], m['rings']
    elif is_block_core(core):
        family, chains = parse_block_core(core)
        if family not in TETRAETHER_FAMILIES:
            return None
        n_cp, n_ch = sum(c.n_cyclopentanes for c in chains), sum(c.n_cyclohexanes for c in chains)
        n_oh = sum(c.n_hydroxy for c in chains)
        n_double_bonds += sum(c.n_double_bonds for c in chains)
        if n_ch and (n_ch, n_cp) != (1, 4):
            return None
        homo, oh, rings = '', {0: '', 1: 'OH-'}.get(n_oh, f'{n_oh}OH-'), 'Cren' if n_ch else str(n_cp)
    elif m := TETRAETHER_PIECE.fullmatch(core):
        homo, oh, family, rings = m['homo'] or '', m['oh'] or '', m['family'], m['rings']
        if (family, rings) == ('GDGT', '5') and not homo and not oh:
            rings = 'Cren'  # the GDGT5 piece is crenarchaeol
    else:
        return None
    uns = f'{UNSATURATION_PREFIX[n_double_bonds]}-uns-' if n_double_bonds else ''
    return f'{uns}{homo}{oh}{family}-{rings}'


def tetraether_name(lipid: Lipid) -> str | None:
    """ArchLips name of the core, with isomer suffix if the structure is in the ArchLips library"""
    if lipid.isomer and (species := _species_tetraether_name(lipid)) is not None:
        # ArchLips writes crenarchaeol isomers in lower case (GDD-crena)
        return species.replace('-Cren', '-cren') + lipid.isomer
    if is_archlips_core(lipid.core) and not lipid.core_double_bonds:
        return lipid.core
    smiles = _canonical_core_smiles(lipid)
    if smiles is not None and (name := _core_names_by_structure().get(smiles)):
        return name
    return _species_tetraether_name(lipid)


@lru_cache(maxsize=None)
def _default_isoprenoid_chains(core: str) -> str:
    """Isoprenoid chains of a quinone piece in isoprene units, e.g. '3:0'"""
    from LipidCalculator.compound_groups.intact_polar_lipids.generate_ipl import get_struct
    from LipidCalculator.compound_groups.intact_polar_lipids.isoprenoid_chains import describe_isoprenoid_chains
    from LipidCalculator.rdkit.util import mol_from_str
    return describe_isoprenoid_chains(mol_from_str(get_struct(core)), in_units=True)


def _archaeol_chains(lipid: Lipid) -> str | None:
    if lipid.core in ARCHAEOL_CHAINS:
        return lipid.core_chains or ARCHAEOL_CHAINS[lipid.core]
    if is_block_core(lipid.core):
        core_type, chains = parse_block_core(lipid.core)
        if core_type != 'AR' or any(c.n_cyclopentanes or c.n_cyclohexanes or c.n_hydroxy > 1 for c in chains):
            return None
        return '_'.join(f'{c.n_carbons}:{c.n_double_bonds}' + ('(OH)' if c.n_hydroxy else '') for c in chains)
    return None


def format_heads(heads: tuple[str, ...]) -> str:
    names = [HEAD_MAP_REVERSE.get(h, h) for h in heads]
    if len(names) == 2:
        return f'{names[0]}-({names[1]}-)'
    return f'{names[0]}-' if names else ''


def name(lipid: Lipid) -> str | None:
    """ArchLips name, None for lipids that are not covered by ArchLips (bacterial/eukaryotic lipids, cores from
    isoprene units that ArchLips does not name, ...)"""
    if lipid.chains:
        return None
    heads = format_heads(lipid.heads)
    if (chains := _archaeol_chains(lipid)) is not None:
        return f'{heads}Archaeol({chains})'
    if lipid.core in DIMERS:
        return f'{heads}{lipid.core}({lipid.core_chains or "20:0_20:0_20:0_20:0"})'
    quinone = {v: k for k, v in QUINONE_MAP.items()}.get(lipid.core, lipid.core)
    if quinone in QUINONES:
        if lipid.heads:
            return None
        return f'{quinone}({lipid.core_chains or _default_isoprenoid_chains(lipid.core)})'
    if (is_archlips_core(lipid.core) or is_block_core(lipid.core) and parse_block_core(lipid.core)[0]
            in TETRAETHER_FAMILIES or TETRAETHER_PIECE.fullmatch(lipid.core)):
        core = tetraether_name(lipid)
        return None if core is None else f'{heads}{core}'
    if ABBREVIATION_TO_GROUP.get(lipid.core) == 'carotenoids' and not lipid.heads:
        return lipid.core
    return None
