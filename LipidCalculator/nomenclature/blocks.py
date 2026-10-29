"""
Names of the building blocks of LipidCalculator (package.json abbreviations and cores from isoprene units), e.g.
PC-DAG 16:0/18:1, 2G-(PG-)GDGT0, PE-AR(20:0_25:1), Di-uns-GDGT0, 1G-GMGT(40:0_40:0(2cp)). They identify the pieces
used to build a molecule, but are not meant for communication (see names.standard_name).
"""
import re

from LipidCalculator.compound_groups.intact_polar_lipids.archlips_cores import is_archlips_core
from LipidCalculator.compound_groups.intact_polar_lipids.isoprenoid_blocks import is_block_core
from LipidCalculator.compound_groups.intact_polar_lipids.pieces_from_json import (
    ABBREVIATIONS, ABBREVIATION_TO_GROUP, CORE_GROUPS
)
from LipidCalculator.nomenclature.lipid import Lipid, NotSupported

UNSATURATION_PREFIXES: list[str] = ['', 'Mono', 'Di', 'Tri', 'Tetra', 'Penta', 'Hexa', 'Hepta', 'Octa']
_CHAINS = re.compile(r'\d+:\d+(?:;[^/ ]+)*(?:/\d+:\d+(?:;[^/ ]+)*)*')


def ipl_name(head: str | None, core: str, chains: list[str], head2: str | None = None,
             core_chains: str | None = None, core_double_bonds: int = 0, core_hydroxy: int = 0,
             core_methyls: int = 0) -> str:
    """e.g. PE-AR(20:0_20:1), 2G-(PG-)Di-uns-Homo-Me-OH-GDGT0 or PC-DAG 34:1"""
    name = core if not core_chains else f'{core}({core_chains})'
    if core_hydroxy:
        name = f'{core_hydroxy if core_hydroxy > 1 else ""}OH-{name}'
    if core_methyls:
        name = f'Homo-{core_methyls if core_methyls > 1 else ""}Me-{name}'
    if core_double_bonds:
        prefix = UNSATURATION_PREFIXES[core_double_bonds] if core_double_bonds < len(UNSATURATION_PREFIXES) \
            else str(core_double_bonds)
        name = f'{prefix}-uns-{name}'
    if head2:
        name = f'({head2}-){name}'
    if head:
        name = f'{head}-{name}'
    if chains:
        name += ' ' + '/'.join(chain[1:] for chain in chains)
    return name


def name(lipid: Lipid) -> str:
    heads = list(lipid.heads) + [None] * (2 - len(lipid.heads))
    return ipl_name(heads[0], lipid.core, list(lipid.chains), head2=heads[1], core_chains=lipid.core_chains,
                    core_double_bonds=lipid.core_double_bonds, core_hydroxy=lipid.core_hydroxy,
                    core_methyls=lipid.core_methyls)


def _is_core(core: str) -> bool:
    return ABBREVIATION_TO_GROUP.get(core) in CORE_GROUPS or is_block_core(core) or is_archlips_core(core)


def _parse_core(core: str) -> dict:
    """Keyword arguments of Lipid, e.g. 'Di-uns-GDGT0' --> dict(core='GDGT0', core_double_bonds=2),
    'AR(20:0_25:0)' --> dict(core='AR', core_chains='20:0_25:0')"""
    out = {}
    while True:
        if _is_core(core):
            return out | dict(core=core)
        if (m := re.fullmatch(r'(.+?)\((.+)\)', core)) and _is_core(m[1]):
            return out | dict(core=m[1], core_chains=m[2])
        # modifications in the order [<n>-uns-][Homo-<n>Me-][<n>OH-]
        if (m := re.match(r'(\w+)-uns-', core)) and 'core_double_bonds' not in out:
            if m[1] not in UNSATURATION_PREFIXES[1:] and not m[1].isdigit():
                raise NotSupported(f'unknown unsaturation {m[1]}')
            out['core_double_bonds'] = int(m[1]) if m[1].isdigit() else UNSATURATION_PREFIXES.index(m[1])
        elif (m := re.match(r'Homo-(\d?)Me-', core)) and 'core_methyls' not in out:
            out['core_methyls'] = int(m[1] or 1)
        elif (m := re.match(r'(\d?)OH-', core)) and 'core_hydroxy' not in out:
            out['core_hydroxy'] = int(m[1] or 1)
        else:
            raise NotSupported(f'{core} is not a core')
        core = core[m.end():]


def parse(name: str) -> Lipid:
    name = name.strip()
    chains: tuple[str, ...] = ()
    if (m := re.fullmatch(r'(.+) (' + _CHAINS.pattern + ')', name)) and not _is_core(name):
        name, chains = m[1], tuple(f'C{c}' for c in m[2].split('/'))
    # head groups contain hyphens, so try all splits: <head>-(<head2>-)<core> or <head>-<core> or <core>
    candidates = [(None, None, name)]
    for i in (i for i, char in enumerate(name) if char == '-'):
        head, rest = name[:i], name[i + 1:]
        if head not in ABBREVIATIONS:
            continue
        candidates.append((head, None, rest))
        if m := re.match(r'\((.+?)-\)', rest):
            candidates.append((head, m[1], rest[m.end():]))
    for head, head2, core in reversed(candidates):
        if head2 is not None and head2 not in ABBREVIATIONS:
            continue
        try:
            core_kwargs = _parse_core(core)
        except NotSupported:
            continue
        heads = tuple(h for h in (head, head2) if h)
        return Lipid(heads, chains=chains, **core_kwargs, family='LipidCalculator')
    raise NotSupported(f'{name} is not a LipidCalculator name')
