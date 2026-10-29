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
# tetraether pieces of package.json, e.g. GDGT0, GDD-Cren, GDGT5' (crenarchaeol with a parallel glycerol)
TETRAETHER_PIECE = re.compile(r'(?P<family>' + '|'.join(TETRAETHER_FAMILIES) + r')-?(?P<rings>\d|Cren)\'?')
# isoprenoid chains of archaeol in ArchLips notation
ARCHAEOL_CHAINS: str = '20:0_20:0'


def _count(prefix: str | None, default: int = 1) -> int:
    """'2OH-' --> 2, 'OH-' --> 1, 'Homo-2Me-' --> 2, None --> 0"""
    if not prefix:
        return 0
    digits = re.search(r'\d', prefix)
    return int(digits.group(0)) if digits else default


def _modification_prefix(n_double_bonds: int, n_methyls: int, n_hydroxy: int) -> str:
    """e.g. Di-uns-Homo-Me-OH-"""
    uns = f'{UNSATURATION_PREFIX[n_double_bonds]}-uns-' if n_double_bonds else ''
    homo = f'Homo-{n_methyls if n_methyls > 1 else ""}Me-' if n_methyls else ''
    oh = f'{n_hydroxy if n_hydroxy > 1 else ""}OH-' if n_hydroxy else ''
    return uns + homo + oh


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


def tetraether_core(name: str) -> dict:
    """Core piece and modifications (keyword arguments of Lipid) for a tetraether name: the ArchLips structure if the
    library has it (exact isomer), otherwise the unmodified core with double bonds, methyl and hydroxyl groups added
    by rules (see isoprenoid_chains.py), its isomer label is kept, e.g.
    Di-uns-GDGT-2a --> dict(core='GDGT-2a', core_double_bonds=2), OH-GDGT-9 --> NotSupported"""
    if is_archlips_core(name):
        return dict(core=name)
    m = CORE_NAME.fullmatch(name)
    if m is None:
        raise NotSupported(f'{name} is not a tetraether core')
    n_double_bonds = UNSATURATION[m['uns']] if m['uns'] else 0
    if n_double_bonds and is_archlips_core(name[m.end('uns') + len('-uns-'):]):
        return dict(core=name[m.end('uns') + len('-uns-'):], core_double_bonds=n_double_bonds)
    family, rings, isomer = m['family'], m['rings'], m['isomer']
    # unmodified core: the same isomer of ArchLips, a piece of package.json or another isomer of ArchLips
    piece = 'GDGT5' if (family, rings.lower()) == ('GDGT', 'cren') else \
        f'{family}-Cren' if rings.lower() == 'cren' else f'{family}{rings}'
    candidates = [f'{family}-{rings}{isomer}', piece, f'{family}-{rings}', f'{family}-{rings.capitalize()}']
    base = next((c for c in candidates if is_archlips_core(c) or c in ABBREVIATIONS), None)
    if base is None:
        raise NotSupported(f'core {name} missing')
    # the label of the name identifies the isomer (it may refer to the position of the head group, e.g. Gly-2OH-GDD-2a)
    return dict(core=base, core_double_bonds=n_double_bonds, core_methyls=_count(m['homo']),
                core_hydroxy=_count(m['oh']), isomer=isomer or None)


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
        # isomers that are not in the core table (e.g. position of the head group in GDDs) keep their label
        return Lipid(tuple(heads), **tetraether_core(m['core']),
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
    core = lipid.core
    n_double_bonds, n_methyls, n_hydroxy = lipid.core_double_bonds, lipid.core_methyls, lipid.core_hydroxy
    if is_archlips_core(core):
        m = CORE_NAME.fullmatch(core)
        n_double_bonds += UNSATURATION[m['uns']] if m['uns'] else 0
        n_methyls, n_hydroxy = n_methyls + _count(m['homo']), n_hydroxy + _count(m['oh'])
        family, rings = m['family'], m['rings']
    elif is_block_core(core):
        family, chains = parse_block_core(core)
        if family not in TETRAETHER_FAMILIES:
            return None
        n_cp, n_ch = sum(c.n_cyclopentanes for c in chains), sum(c.n_cyclohexanes for c in chains)
        n_hydroxy += sum(c.n_hydroxy for c in chains)
        n_double_bonds += sum(c.n_double_bonds for c in chains)
        if n_ch and (n_ch, n_cp) != (1, 4):
            return None
        rings = 'Cren' if n_ch else str(n_cp)
    elif m := TETRAETHER_PIECE.fullmatch(core):
        family, rings = m['family'], m['rings']
        if (family, rings) == ('GDGT', '5'):
            rings = 'Cren'  # the GDGT5 piece is crenarchaeol
    else:
        return None
    if n_double_bonds and n_double_bonds not in UNSATURATION_PREFIX:
        return None
    return f'{_modification_prefix(n_double_bonds, n_methyls, n_hydroxy)}{family}-{rings}'


def tetraether_name(lipid: Lipid) -> str | None:
    """ArchLips name of the core, with isomer suffix if the structure is in the ArchLips library"""
    if lipid.isomer and (species := _species_tetraether_name(lipid)) is not None:
        # ArchLips writes crenarchaeol isomers in lower case (GDD-crena)
        return species.replace('-Cren', '-cren') + lipid.isomer
    if is_archlips_core(lipid.core) and not (lipid.core_double_bonds or lipid.core_methyls or lipid.core_hydroxy):
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
    """Isoprenoid chains of archaeols in ArchLips notation, e.g. 20:0(OH)_25:0"""
    if lipid.core == 'AR':
        tokens = (lipid.core_chains or ARCHAEOL_CHAINS).split('_')
    elif is_block_core(lipid.core):
        core_type, chains = parse_block_core(lipid.core)
        if core_type != 'AR' or any(c.n_cyclopentanes or c.n_cyclohexanes or c.n_hydroxy > 1 for c in chains):
            return None
        tokens = [f'{c.n_carbons}:{c.n_double_bonds}' + ('(OH)' if c.n_hydroxy else '') for c in chains]
    else:
        return None
    if lipid.core_methyls:
        return None
    # hydroxyl groups added by the rule: ArchLips lists the hydroxylated chains first
    for _ in range(lipid.core_hydroxy):
        free = [i for i, t in enumerate(tokens) if '(' not in t]
        if not free:
            return None
        tokens[free[0]] += '(OH)'
    if lipid.core_hydroxy:
        tokens.sort(key=lambda t: '(OH)' not in t)
    return '_'.join(tokens)


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
