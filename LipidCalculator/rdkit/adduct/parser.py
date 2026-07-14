"""Different tools use varying formats to report adducts"""
from typing import Literal

from LipidCalculator.compound_creation.formula_parser import CompoundDict
from LipidCalculator.consts import m_e


def metaboscape_extract_adduct_and_charge(ipt: str) -> tuple[str, str]:
    """e.g. ION=[M+H+H]2+ or ION=[M+H]+"""
    pre = ipt.split('M')[1]
    add, charge = pre.split(']')
    return add, charge


def parse_charge(c: str):
    assert (c.count('+') == 1) ^ (c.count('-') == 1), 'Expecting exactly one + or -'
    sign = 1 if '+' in c else -1

    c = c.strip('+-')
    c = 1 if len(c) == 0 else int(c)

    return sign * c


def get_molecule_clustering(ipt: str) -> int:
    """e.g. [M2+H]+ --> 2"""
    assert 'M' in ipt, f'adduct expression must include "M" to denote molecule mass'
    l = ipt.split('M')[1]
    if not l[0].isdigit():
        return 1
    assert not l[1].isdigit(), \
        'clustering of more than 9 is unrealistic, next letter must not be digit'
    return int(l[0])


def get_adduct_mass_and_charge(
        ipt: str,
        format_template: Literal['metaboscape', 'simple', 'square_bracket'] = 'metaboscape'
) -> tuple[float, int, int]:
    if format_template in ('metaboscape', 'square_bracket'):
        add, charge = metaboscape_extract_adduct_and_charge(ipt)
    elif format_template == 'simple':
        add, charge = parse_charge(ipt)
    else:
        raise NotImplementedError()

    # check for clustered molecules
    molecule_multiplicity = ''
    while (len(add) > 0) and add[0].isdigit():
        molecule_multiplicity += add[0]
        add = add[1:]
    molecule_multiplicity = int(molecule_multiplicity) if len(molecule_multiplicity) > 0 else 1

    # convert add to mass
    #  make use of parser
    parts: list[CompoundDict] = []
    current_el = ''
    sign: int = 0
    for sym in add:
        if sym in '+-':  # start new el
            sign = 1 if sym == '+' else -1
            if len(current_el) == 0:
                continue
            parts.append(CompoundDict(current_el) * sign)
            current_el = ''
            continue
        current_el += sym
    if len(current_el) > 0:
        parts.append(CompoundDict(current_el) * sign)

    cd = CompoundDict()
    for p in parts:
        cd = cd + p

    add_c = parse_charge(charge)
    add_mass = cd.mass - add_c * m_e
    return add_mass, add_c, molecule_multiplicity


def convert_molecule_mass_to_mz(M: float, adduct_mass: float, adduct_charge: int,
                                molecule_multiplicity: int = 1) -> float:
    """Need to add adduct mass before dividing by charge"""
    m = M * molecule_multiplicity + adduct_mass
    return m / adduct_charge


def get_mz_from_M_and_adduct(M: float, adduct: str):
    add_mz, add_c, mul = get_adduct_mass_and_charge(adduct)
    return convert_molecule_mass_to_mz(M, add_mz, add_c, mul)


if __name__ == '__main__':
    ipts = ['ION=[M+Na]+', 'ION=[M+H]+', 'ION=[M+H+H]2+', 'ION=[M-H2O]+', 'ION=[M2+H]+']
    # ipts = ['ION=[M2+H]+']
    M = 1116.70911
    for ipt in ipts:
        print(get_adduct_mass_and_charge(ipt), get_mz_from_M_and_adduct(M, ipt), get_molecule_clustering(ipt))
