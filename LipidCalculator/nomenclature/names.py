"""
Lipid names in several conventions: any of them can be used to find a lipid (parse_name), but names for
communication are standardized (standard_name):
- archaeal lipids: ArchLips (Zheng et al., Nat. Commun. 2025), e.g. 2Gly-GDGT-0, PG-Archaeol(20:0_20:0)
- bacterial and eukaryotic lipids: LIPID MAPS shorthand (also used by RefMet), e.g. PC 16:0_18:1
- otherwise trivial names (e.g. Phytane) or the building blocks of LipidCalculator (e.g. MAR(40:0))
"""
from typing import Callable

from LipidCalculator.nomenclature import archlips, blocks, lipidmaps, trivial
from LipidCalculator.nomenclature.lipid import Lipid, NotSupported

# convention -> (parse, format), MS-DIAL uses LIPID MAPS shorthand
CONVENTIONS: dict[str, tuple[Callable[[str], Lipid], Callable[[Lipid], str | None]]] = {
    'trivial': (trivial.parse, trivial.name),
    'archlips': (archlips.parse, archlips.name),
    'lipidmaps': (lipidmaps.parse, lipidmaps.name),
    'blocks': (blocks.parse, blocks.name),
}
# conventions tried for the standard name, in this order
STANDARD_ORDER: list[str] = ['archlips', 'lipidmaps', 'trivial', 'blocks']


def parse_name(name: str, convention: str | None = None) -> Lipid:
    """Building blocks of a lipid name, conventions are tried in the order of CONVENTIONS if not given"""
    if convention is not None:
        return CONVENTIONS[convention][0](name)
    errors = []
    for key, (parse, _) in CONVENTIONS.items():
        try:
            return parse(name)
        except NotSupported as e:
            errors.append(f'{key}: {e}')
    raise NotSupported(f'could not parse "{name}" ({"; ".join(errors)})')


def lipid_name(lipid: Lipid, convention: str = 'standard') -> str | None:
    """Name in a convention ('standard', 'archlips', 'lipidmaps', 'trivial', 'blocks'), None if the convention does
    not cover the lipid"""
    if convention == 'standard':
        return standard_name(lipid)
    return CONVENTIONS[convention][1](lipid)


def standard_name(lipid: Lipid) -> str:
    """Name for communication: ArchLips for archaeal lipids, LIPID MAPS shorthand for others"""
    for convention in STANDARD_ORDER:
        if (name := CONVENTIONS[convention][1](lipid)) is not None:
            return name
    raise NotSupported(f'{lipid} has no name')  # blocks names every lipid, so this is not reached


def all_names(lipid: Lipid) -> dict[str, str]:
    """Names in all conventions that cover the lipid, the standard name first"""
    out = {'standard': standard_name(lipid)}
    for convention, (_, fmt) in CONVENTIONS.items():
        if (name := fmt(lipid)) is not None:
            out[convention] = name
    return out
