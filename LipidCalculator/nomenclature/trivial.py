"""Trivial names (literature, LIPID MAPS common names) mapped to names of other conventions."""
from LipidCalculator.nomenclature.lipid import Lipid, NotSupported

# trivial name -> (convention, name), the first trivial name of a structure is used when formatting
TRIVIAL_NAMES: dict[str, tuple[str, str]] = {
    # diethers
    'Archaeol': ('archlips', 'Archaeol(20:0_20:0)'),
    'Hydroxyarchaeol': ('archlips', 'Archaeol(20:0(OH)_20:0)'),
    'OH-Archaeol': ('archlips', 'Archaeol(20:0(OH)_20:0)'),
    'Dihydroxyarchaeol': ('archlips', 'Archaeol(20:0(OH)_20:0(OH))'),
    'Extended archaeol': ('archlips', 'Archaeol(20:0_25:0)'),
    'C20-C25 archaeol': ('archlips', 'Archaeol(20:0_25:0)'),
    'Macrocyclic archaeol': ('blocks', 'MAR(40:0)'),
    # tetraethers
    'Caldarchaeol': ('archlips', 'GDGT-0'),
    'Crenarchaeol': ('archlips', 'GDGT-Cren'),
    # hydrocarbons
    'Phytane': ('blocks', 'HT(20:0)'),
    'Pristane': ('blocks', 'HT(19:0)'),
    'Norpristane': ('blocks', 'HT(18:0)'),
    'Farnesane': ('blocks', 'HT(15:0)'),
    'Biphytane': ('blocks', 'HH(40:0)'),
    'Squalane': ('blocks', 'TT(30:0)'),
    'Lycopane': ('blocks', 'TT(40:0)'),
    'Crocetane': ('blocks', 'TT(20:0)'),
    'PMI': ('blocks', 'TT(25:0)'),
    '2,6,10,15,19-Pentamethylicosane': ('blocks', 'TT(25:0)'),
}
_BY_LOWER: dict[str, str] = {k.lower(): k for k in TRIVIAL_NAMES}


def parse(name: str) -> Lipid:
    from LipidCalculator.nomenclature.names import parse_name
    key = _BY_LOWER.get(name.strip().lower())
    if key is None:
        raise NotSupported(f'{name} is not a known trivial name')
    convention, other = TRIVIAL_NAMES[key]
    return parse_name(other, convention=convention)


def name(lipid: Lipid) -> str | None:
    from LipidCalculator.nomenclature.names import parse_name
    for trivial, (convention, other) in TRIVIAL_NAMES.items():
        if parse_name(other, convention=convention) == lipid:
            return trivial
    return None
