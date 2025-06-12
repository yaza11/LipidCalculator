"""This module implements functions for parsing strings for (complex) formulae,
making use of a custom syntax and exposing the CompoundDict class, which allows
arithmetic operations on Compounds defined by their elemental counts."""
from __future__ import annotations

from typing import ItemsView, Iterable
from typing import Self

from rdkit.Chem import Mol, GetFormalCharge
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from LipidCalculator.isotopes.isotopes import isotope_mass
from LipidCalculator.isotopes.elements import elements

ALPHABET: str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
NUMBERS: str = '0123456789'
VALID_CHARACTERS: set[str] = set(ALPHABET.lower() + ALPHABET.upper() + NUMBERS + '.()[]')


def get_charge_from_str(formula: str) -> int:
    assert not (('+' in formula) and ('-' in formula)), \
        'string cannot contain both plus and minus'
    charge_p: int = formula.count('+')
    charge_m: int = -formula.count('-')
    return charge_m + charge_p


def remove_charge(formula: str) -> str:
    return formula.strip('+-')


def lexer(s: str) -> str:
    """
    Turn a sum formula into an algebraic expression by adding + and *
    operators and brackets.
    """
    assert set(s).issubset(VALID_CHARACTERS), \
        'formula contains invalid characters'

    def get_type(char_: str) -> None | str:
        if char_.isalpha() and char_.isupper():
            return 'el_start'
        elif char_.isalpha() and char_.islower():
            return 'el'
        elif char_.isdigit() or (char_ == '.'):
            return 'num'
        elif char_ == '(':
            return 'par_start'
        elif char_ == ')':
            return 'par_end'
        elif char_ == '[':
            return 'ignore_start'
        elif char_ == ']':
            return 'ignore_end'
        else:
            return

    s_new = ''
    current_type = None
    last_type = None
    flag_ignore = False
    for char in s:
        current_type = get_type(char)
        if current_type == 'ignore_start':
            flag_ignore = True
        if current_type == 'ignore_end':
            flag_ignore = False
        # compound dict object ends at element
        if (last_type in ('el_start', 'el', 'ignore_end')) and (current_type != 'el') and (not flag_ignore):
            s_new += '}'
        # plus between elements and ) element
        if current_type in ('el_start', 'par_start') and (
                last_type in ('el', 'el_start', 'num', 'par_end', 'ignore_end')) and (not flag_ignore):
            s_new += '+'
        elif (current_type == 'num') and (last_type not in ('num')) and (not flag_ignore):
            s_new += '*'
        if (current_type == 'el_start') and (not flag_ignore):
            s_new += '{'

        last_type = current_type
        s_new += char
    if current_type in ('el_start', 'el', 'ignore_end'):
        s_new += '}'

    return s_new


def parse_formula(s: str) -> CompoundDict:
    if len(s) == 0:
        return CompoundDict({})

    # handle charge
    if (sym := '+') in s:
        s, charge = s.split('+', 1)
        charge = int(charge) if charge.isdigit() else len(charge) + 1
    elif (sym := '-') in s:
        s, charge = s.split('-', 1)
        charge = int(charge) if charge.isdigit() else len(charge) + 1
    else:
        charge = 0

    s_new = lexer(s)
    to_eval = s_new.replace('{', 'CompoundDict({"').replace('}', '": 1})')
    cd = eval(
        to_eval
    )

    if charge != 0:
        cd += CompoundDict({sym: charge})
    return cd


class CompoundDict:
    """Wrapper around dicts."""

    def __init__(self, compound_dict: dict[str, float | int] | str):
        assert isinstance(compound_dict, dict | str)

        if isinstance(compound_dict, str):
            self._composition: dict[str, float | int] = parse_formula(
                compound_dict
            ).composition
        else:
            self._composition: dict[str, float | int] = compound_dict

        assert all([
            key.split('[')[0] in (elements + ['+', '-'])
            for key in self.composition
        ]), "provided dict contains invalid element(s)"

        self._clean()

    @classmethod
    def from_mol(cls, mol: Mol) -> Self:
        assert isinstance(mol, Mol), f'provided Molecule must be instance of {type(Mol)}, but got {type(mol)}'
        f_mol = CalcMolFormula(mol)
        charge = GetFormalCharge(mol)
        f_charge = ('+' if charge > 0 else '-') + str(abs(charge))
        cd = cls(f'{f_mol}{f_charge}')
        return cd

    def _clean_charge(self) -> None:
        """Make sure that only either + or - entry exists and that their values or positive or zero"""
        # cant use -= since + or - entries may not exist
        if ('+' in self.composition) and ('-' in self.composition):
            self.composition['+'] = self.composition.get('+', 0) - self.composition.pop('-')
        if self.composition.get('+', 0) < 0:
            self.composition['-'] = self.composition.get('-', 0) - self.composition.pop('+')
        if self.composition.get('-', 0) < 0:
            self.composition['+'] = self.composition.get('+', 0) - self.composition.pop('-')

    def _clean(self) -> None:
        self._clean_charge()
        new_dict = {}
        for k, v in self.composition.items():
            if v != 0:
                new_dict[k] = v
        self._composition = new_dict

    @property
    def composition(self) -> dict[str, float | int]:
        return self._composition

    @property
    def formula(self) -> str:
        def el_ct(el: str) -> tuple[str, float]:
            ct = round(self.composition[el], 2)
            if ct == 0:
                return '', 0.
            if abs(ct) == 1:
                s = el
            else:
                s = el + str(abs(ct))
            return s, ct

        self._clean()

        out_pos = ''  # part of the formular with positive atom counts
        out_neg = ''  # part of the formula with negative atom counts
        sort = 'C H O N P S'.split()
        # first, use common elements
        for el in sort:
            if el not in self.composition:
                continue
            inc, ct = el_ct(el)
            if ct > 0:
                out_pos += inc
            elif ct < 0:
                out_neg += inc
        for el, ct in self.composition.items():  # all other elements except charge
            if (el in sort) or (el in '+', '-'):  # charge last
                continue
            inc, ct = el_ct(el)
            if ct > 0:
                out_pos += inc
            elif ct < 0:
                out_neg += inc

        # use heavy minus sign to mark negative atom counts
        out = f'{out_pos}{chr(0x2796)}{out_neg}' if len(out_neg) > 0 else out_pos
        if '+' in self.composition:
            c = el_ct('+')[0]
            out += c
        elif '-' in self.composition:
            c = el_ct('-')[0]  # exclude minus sign
            out += c
        return out

    def __add__(self, other: Self) -> Self:
        new_dict = self.composition.copy()

        for key, value in other.composition.items():
            if key in new_dict:
                new_dict[key] += value
            else:
                new_dict[key] = value

        # self._composition = new_dict
        return type(self)(new_dict)

    def __sub__(self, other: Self) -> Self:
        new_dict: dict[str, int | float] = self.composition.copy()
        for key, value in other.composition.items():
            if key in new_dict:
                new_dict[key] -= value
            else:
                new_dict[key] = -value

        # self._composition = new_dict
        return type(self)(new_dict)

    def __mul__(self, other: float | int) -> Self:
        new_dict = self.composition.copy()
        for key, value in self.composition.items():
            if key in new_dict:
                new_dict[key] *= other

        # self._composition = new_dict
        return type(self)(new_dict)

    def __neg__(self) -> Self:
        return self.__mul__(-1)

    def __hash__(self) -> int:
        return hash(self.composition)

    def __eq__(self, other: Self) -> bool:
        return self.composition == other.composition

    def __repr__(self) -> str:
        return str(self.composition)

    def __getitem__(self, item: str) -> int | float:
        return self.composition[item]

    def items(self) -> ItemsView:
        return self.composition.items()

    def values(self) -> Iterable:
        return self.composition.values()

    def keys(self) -> Iterable:
        return self.composition.keys()

    @property
    def mass(self) -> float:
        s = 0
        for key, value in self.composition.items():
            s += value * isotope_mass(key)
        return s

    def copy(self) -> Self:
        return self.__class__(self.composition.copy())


if __name__ == '__main__':
    d = CompoundDict('C6H12O6+')
    c = CompoundDict('C6H14O7')
    f = d - c
    print(f.formula)
