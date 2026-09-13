"""Different tools use varying formats to report adducts"""
from typing import Literal

from LipidCalculator.compound_creation.formula_parser import CompoundDict, parse_equation, NEG_SIGN
from LipidCalculator.consts import m_e


def _split_adduct_and_charge(ipt: str) -> tuple[str, str]:
    """e.g. ION=[M+H+H]2+ or ION=[M+H]+ or [2M+H]+"""
    assert not ipt.startswith('ION=')

    *adduct_formula, charge = ipt.split(']')
    adduct_formula = ']'.join(adduct_formula)
    adduct_formula = adduct_formula.lstrip('[')
    return adduct_formula, charge


def _parse_charge(c: str):
    """e.g. 2+ or - or +"""
    assert (c.count('+') == 1) ^ (c.count('-') == 1), 'Expecting exactly one + or -'
    sign = 1 if '+' in c else -1

    c = c.rstrip('+-')
    if len(c) > 0:
        assert c.isdigit(), 'charge should be a whole number followed by the sign or just the sign for singly charged molecules'
        return sign * int(c)
    return sign


def _strip_molecule_multiplicity(ipt: str) -> tuple[int, str]:
    """e.g. M2+H --> 2"""
    # assert ('[' not in ipt) and (']' not in ipt), 'adduct expression must not contain brackets'
    assert 'M' in ipt, f'adduct expression must include "M" to denote molecule mass'
    l, add_composition = ipt.split('M', 1)
    if len(l) == 0:
        return 1, add_composition
    return int(l), add_composition


def _get_adduct_composition(ipt: str) -> CompoundDict:
    """It is assumed that +/-... is provided to this function, e.g. -H2O+H"""
    assert ('[' not in ipt) and (']' not in ipt), 'adduct expression must not contain brackets'
    assert 'M' not in ipt, f'adduct expression must not include "M" as molecule multiplcity has been stripped away already'

    #  make use of parser
    # '-e' is redundant, remove it
    if '-e' in ipt:
        ipt = ipt.replace('-e', '')
    return parse_equation(ipt)


class Adduct:
    charge: int = None
    multiplicity: int = None
    composition: CompoundDict = None
    isotope: int = 0

    def __init__(self, adduct: str = None):
        """
        Examples:
          different notations
        - ION=[M+H]+
        - ION=[M-H]-
        - [M+H]+
        - [M-e]+
        - [M]+
        - H+
          multiply charged
        - [M+H+H]2+
        - [M+H2]2+
        - [M+H+Na]2+
          n-mers
        - [2M+H]+
          mixed
        - [2M+H2]2+
        - [M-H2O+H]+
          wildcard
        - [M+?]

        :param adduct: string to parse
        """
        if adduct is None:
            return

        # handle simplified notation
        if '[' not in adduct:
            c = adduct[-1]
            f = adduct[:-1]
            adduct = f'[M+{f}]{c}'
        if adduct.startswith('ION='):  # metaboscape notation
            adduct = adduct[4:]
        # adduct notation is now standardized (at least enough for our purposes)
        self.adduct = adduct

        if adduct == '[M+?]':
            return

        adduct_composition, charge_str = _split_adduct_and_charge(adduct)
        self.charge: int = _parse_charge(charge_str)
        # multiplicity, adduct_composition = _strip_molecule_multiplicity(adduct_composition)
        # adduct_composition: CompoundDict = _get_adduct_composition(adduct_composition)
        composition: CompoundDict = parse_equation(adduct_composition, check_elements=False)
        M_key: list[str] = [
            el for el in composition.composition.keys()
            if el.split('[')[0] == 'M'
        ]
        assert len(M_key) == 1, f'adduct composition must have exactly one M'
        M_key: str = M_key[0]
        self.multiplicity = composition.composition.pop(M_key)
        self.isotope = int(M_key.split('[')[1].rstrip(']')) if '[' in M_key else 0

        if 'C[13]' in composition.composition:
            assert '[' not in M_key, 'if isotopes are provided for the composition, isotope index must not be provided for M'
            self.isotope += composition.composition['C[13]']
        if 'S[34]' in composition.composition:
            assert '[' not in M_key, 'if isotopes are provided for the composition, isotope index must not be provided for M'
            self.isotope += composition.composition['S[34]'] * 2

        # charge is part of composition so that we get correct mass
        self.composition: CompoundDict = composition + CompoundDict({"+": self.charge})
        if (self.isotope > 0) and ('C[13]' not in composition.composition) and ('S[34]' not in composition.composition):
            self.composition += (CompoundDict('C[13]') - CompoundDict('C[12]')) * self.isotope

    @classmethod
    def from_props(cls, multiplicity: int, charge: int, composition: CompoundDict, adduct: str | None = None):
        new = cls(None)
        new.multiplicity = multiplicity
        new.charge = charge
        new.composition = composition
        new.adduct = adduct
        return new

    def copy(self):
        return Adduct.from_props(self.multiplicity, self.charge, self.composition.copy(), self.adduct)

    def __eq__(self, other) -> bool:
        return (self.multiplicity == other.multiplicity) and (self.charge == other.charge) and (
                self.composition == other.composition)

    def __repr__(self) -> str:
        eta: str = str(self.multiplicity) if self.multiplicity > 1 else ''
        i = f'[{self.isotope}]' if self.isotope > 0 else ''
        comp: CompoundDict = self.composition.copy()
        if '-' in comp.composition:
            comp.composition.pop('-')
        elif '+' in comp.composition:
            comp.composition.pop('+')
        comp += -(CompoundDict('C[13]') - CompoundDict('C[12]')) * self.isotope
        s_comp = '+' if comp.mass > 0 else '-'
        if s_comp == '-':
            comp *= -1
        comp: str = comp.formula.replace(NEG_SIGN, '-')
        n_charges: int = abs(self.charge)
        s_charge: str = '+' if self.charge > 0 else '-'
        charge: str = f'{str(n_charges) if n_charges > 1 else ''}{s_charge}'
        return f'[{eta}M{i}{s_comp}{comp}]{charge}'

    @property
    def mass(self) -> float:
        return self.composition.mass

    def mass_to_mz(self, mass: float):
        return (mass * self.multiplicity + self.composition.mass - m_e * self.charge) / abs(self.charge)

    def mz_to_mass(self, mz: float):
        return (mz * self.charge - self.composition.mass + m_e * self.charge) / self.multiplicity

    def isopattern_molecule_mass_to_mz(self, mass: float):
        return (mass - m_e * self.charge) / abs(self.charge)


if __name__ == '__main__':
    # check against https://fiehnlab.ucdavis.edu/staff/kind/metabolomics/ms-adduct-calculator/
    ipts = ['ION=[M+Na]+', 'ION=[M[1]+Na]+', 'ION=[M+H]+', 'ION=[M+H+H]2+', 'ION=[M-H2O]+', 'ION=[2M+H]+']
    # ipts = ['ION=[M2+H]+']
    M = 853.33089
    mz = 876.32
    for ipt in ipts:
        add = Adduct(ipt)
        print(add.adduct, str(add), round(add.mass_to_mz(M), 6), round(add.mz_to_mass(mz), 6))
