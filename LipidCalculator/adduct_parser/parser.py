"""Different tools use varying formats to report adducts"""
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


def get_adduct_mass_and_charge(ipt: str, format_template='metaboscape') -> tuple[float, int]:
    if format_template == 'metaboscape':
        add, charge = metaboscape_extract_adduct_and_charge(ipt)
    else:
        raise NotImplementedError()

    # convert add to mass
    #  make use of parser

    parts: list[CompoundDict] = []
    current_el = ''
    sign: int = None
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
    return add_mass, add_c


def convert_molecule_mass_to_mz(M: float, adduct_mass: float, adduct_charge: int):
    """Need to add adduct mass before dividing by charge"""
    m = M + adduct_mass
    return m / adduct_charge


def get_mz_from_M_and_adduct(M: float, adduct: str):
    add_mz, add_c = get_adduct_mass_and_charge(adduct)
    return convert_molecule_mass_to_mz(M, add_mz, add_c)


if __name__ == '__main__':
    ipts = ['ION=[M+Na]+', 'ION=[M+H]+', 'ION=[M+H+H]2+', 'ION=[M-H2O]+']
    M = 1116.70911
    for ipt in ipts:
        print(get_adduct_mass_and_charge(ipt), get_mz_from_M_and_adduct(M, ipt))
