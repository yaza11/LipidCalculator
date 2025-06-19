"""This module implements functions for parsing strings for (complex) formulae,
making use of a custom syntax and exposing the CompoundDict class, which allows
arithmetic operations on Compounds defined by their elemental counts."""
from __future__ import annotations

from typing import ItemsView, Iterable, Callable
from typing import Self

from rdkit.Chem import Mol, GetFormalCharge
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from LipidCalculator.isotopes.isotopes import isotope_mass
from LipidCalculator.isotopes.elements import elements

NEG_SIGN: str = chr(0x2796)
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


def ignore_square_brackets(s: str) -> str:
    """
    Ignore everything in square brackets (isotope type does not matter for element dict)
    """
    s_new = ''
    encountered_bracket_open = False
    for char in s:
        if char == '[':
            encountered_bracket_open = True
            continue
        elif char == ']':
            encountered_bracket_open = False
            continue

        if not encountered_bracket_open:
            s_new += char
    return s_new


def group_start_end(
        s, start_function: Callable[[str], bool], end_function: Callable[[str], bool], start_end_symbols: Iterable
):
    bracket_start, bracket_end = start_end_symbols
    s_new: str = ''
    encountered_start: bool = False
    for char in s:
        if start_function(char):
            s_new += bracket_start + char
            encountered_start = True
            continue
        # encountering end of element-count pair
        if encountered_start and (end_function(char)):
            s_new += char + bracket_end
            encountered_start = False
            continue
        # in all other cases just append symbol
        s_new += char
    return s_new


def is_part_of_numeric(x: str) -> bool:
    return x.isdigit() or (x == '.')


def numeric_ends(char: str) -> bool:
    """numeric or lowercase"""
    # islower only returns True if all symbols are lowercase letters
    return not (char.islower() or is_part_of_numeric(char))


def new_group(x: str) -> bool:
    return x in '({'


def end_group(x: str) -> bool:
    return x in '})'


def bracket_numbers(s: str) -> str:
    return group_start_end(s, is_part_of_numeric, lambda x: not is_part_of_numeric(x), '()')


def group_element_count(s: str) -> str:
    """
    identify start and end of elements and their counts
    """
    return group_start_end(s, start_function=lambda x: x.isupper(),
                           end_function=lambda x: end_group(x) or (x.isupper()) or (x in ''), start_end_symbols='{}')


def format_charge(s: str) -> str:
    """identify charge by +/- or ^{+/-...} (which is how the output will be formatted)"""
    return group_start_end(s, start_function=lambda x: x in '+-', end_function=..., start_end_symbols='{}')


def insert_operations(s: str) -> str:
    """add * before (, add + before { and replace NEG_SIGN by -"""
    s = s.replace(NEG_SIGN, '-')
    s_new = ''
    last_idx: int = len(s) - 1
    for idx, char in enumerate(s):
        if last_idx:
            continue
        if char == '}':
            # check that next index is another group or bracket
            if (s[idx + 1] in '({'):
                s_new += '+'
        elif char == ')':
            if (s[idx + 1] in '({'):
                s_new += '*'
    return s_new


def insert_elementary_cds(s: str) -> str:
    # turn {Xnum} into {"X": num} for evaluation
    pre_dict = 'CompoundDict('
    post_dict = ', skip_cleaning=True)'

    s_new = ''
    encountered_opening = False
    encountered_start_numeric = False
    for char in s:
        if char == '{':  # found start of
            encountered_opening = True
            encountered_start_numeric = False
            s += pre_dict + '{\'' + char
            continue
        # check for start of number
        if not encountered_start_numeric and is_numeric(char):
            ...


def better_parser(s: str) -> CompoundDict:
    s = ignore_square_brackets(s)
    s = bracket_numbers(s)
    s = group_element_count(s)
    s = format_charge(s)
    s = insert_operations(s)
    s = insert_elementary_cds(s)
    return eval(s)


def lexer(s: str) -> str:
    """
    Turn a sum formula into an algebraic expression by adding + and *
    operators and brackets.
    """
    assert set(s).issubset(VALID_CHARACTERS), \
        f'formula contains invalid characters: {set(s) - VALID_CHARACTERS}'

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
        elif (current_type == 'num') and (last_type not in ('num',)) and (not flag_ignore):
            s_new += '*'
        if (current_type == 'el_start') and (not flag_ignore):
            s_new += '{'

        last_type = current_type
        s_new += char
    if current_type in ('el_start', 'el', 'ignore_end'):
        s_new += '}'

    return s_new


def parse_formula(s: str) -> CompoundDict:
    def check_only_numeric_after_charge(chars: str):
        assert all([is_part_of_numeric(c) or (c == sym) for c in
                    chars]), f'charge contains non-numeric symbols, make sure provided string ends in number, you provided {chars}'

    if len(s) == 0:
        return CompoundDict({})
    # remove redundant brackets
    s = s.replace('()', '')
    if s.startswith('(') and s.endswith(')'):
        return parse_formula(s[1:-1])

    # handle charge first
    if (sym := '+') in s:
        s, charge = s.split('+', 1)
        check_only_numeric_after_charge(charge)
        charge = int(charge) if charge.isdigit() else len(charge) + 1
    elif (sym := '-') in s:
        s, charge = s.split('-', 1)
        check_only_numeric_after_charge(charge)
        charge = int(charge) if charge.isdigit() else len(charge) + 1
    else:
        charge = 0

    s = lexer(s)
    to_eval = s.replace('{', 'CompoundDict({"').replace('}', '": 1}, skip_cleaning=True)')
    cd = eval(
        to_eval
    )

    if charge != 0:
        cd += CompoundDict({sym: charge}, skip_cleaning=True)
    cd.clean()
    return cd


class CompoundDict:
    """Wrapper around dicts."""
    _elements_sorting = set('C H O N P S'.split())

    def __init__(self, compound_dict: dict[str, float | int] | str, skip_cleaning=False):
        if skip_cleaning:  # premature exit without checking keys and cleaning
            self._composition: dict[str, float | int] = compound_dict
            return

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

        self.clean()

    @classmethod
    def from_mol(cls, mol: Mol) -> Self:
        assert isinstance(mol, Mol), f'provided Molecule must be instance of {type(Mol)}, but got {type(mol)}'
        f_mol = CalcMolFormula(mol)
        charge = GetFormalCharge(mol)
        f_charge = ('+' if charge > 0 else '-') + str(abs(charge))
        cd = cls(f'{f_mol}{f_charge}')
        return cd

    @classmethod
    def from_elemental_dict(cls, ipt: dict[str, int | float]) -> Self:
        cd = cls(ipt, skip_cleaning=True)
        return cd

    @classmethod
    def from_diff(cls, ipt: str) -> Self:
        if NEG_SIGN in ipt:
            assert ipt.count(NEG_SIGN) <= 1, f'found more than one {NEG_SIGN}'
            pos, neg = ipt.split(NEG_SIGN)
            cd = cls(pos) - cls(neg)
        else:
            cd = cls(ipt)
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

    def clean(self) -> None:
        self._clean_charge()
        # new_dict = {}
        # for k, v in self.composition.items():
        #     if v != 0:
        #         new_dict[k] = v
        # self._composition = new_dict
        self._composition = {k: v for k, v in self.composition.items() if v != 0}

    @property
    def composition(self) -> dict[str, float | int]:
        return self._composition

    @property
    def formula(self) -> str:
        self.clean()

        out_pos = ''  # part of the formular with positive atom counts
        out_neg = ''  # part of the formula with negative atom counts
        # first, use common elements
        _els = set(self.composition.keys())
        if _has_pos_charge := ('+' in _els):
            _els.remove('+')
            _has_neg_charge = False
        elif _has_neg_charge := ('-' in _els):
            _els.remove('-')
        que = (self._elements_sorting & _els) | (_els - self._elements_sorting)
        for el in que:
            ct = self.composition[el]
            if (is_pos := (ct > 0)) and (ct == 1):
                out_pos += el
            elif is_pos:
                out_pos += el + str(ct)
            elif ct == -1:
                out_neg += el
            else:
                out_neg += el + str(-ct)

        # always put charge on positive part, if available
        if _has_pos_charge:
            c = str(round(self.composition['+'], 2))
            if c == '1':
                c = ''
            if len(out_pos) > 0:
                out_pos += '+' + c
            else:
                out_neg += '-' + c
        elif _has_neg_charge:
            c = str(round(self.composition['-'], 2))
            if c == '1':
                c = ''
            if len(out_pos) > 0:
                out_pos += '-' + c
            else:
                out_neg += '+' + c

        # use heavy minus sign to mark negative atom counts
        out = out_pos
        if len(out_neg) > 0:
            out += NEG_SIGN + out_neg

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
        return hash(self.formula)

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


def test_neg():
    s = f'C6{NEG_SIGN}O2'
    cd = CompoundDict(s)
    print(cd.composition)
    print(cd.formula)
    return cd


def test_performance():
    import time
    from tqdm import tqdm

    C_range = range(-10, 31)
    H_range = range(10, 81, 2)
    O_range = range(0, 21)
    N_range = range(11)
    charges = [2, 1, -1, -2]

    n_iter = len(C_range) * len(H_range) * len(O_range) * len(N_range) * len(charges)

    t0 = time.time()
    for c in tqdm(C_range, 'running test', total=len(C_range)):
        for h in H_range:
            for o in O_range:
                for n in N_range:
                    for charge in charges:
                        d = dict(C=c, H=h, O=o, N=n)
                        d['+'] = charge
                        cd = CompoundDict(d)
                        cd.formula
    t1 = time.time()
    print(f'took {t1 - t0:.3f} s for {n_iter} iterations')


def test_parsing():
    s = 'Mg((OH3)(AlC)2)4Pb[21]O23.33'
    print('inpt', s)
    s = ignore_square_brackets(s)
    print('ignoring []', s)
    s = bracket_numbers(s)
    print('() rd numbers', s)


if __name__ == '__main__':
    pass
    # print(lexer(NEG_SIGN + 'H2O'))
    # parse_formula('((C6H12O6)())')

    # d = CompoundDict('C6H12O6N+')
    # c = CompoundDict('C6H14O7-')
    # f = d - c
    #
    # # charge should be considered part of the molecule, so negating the formula also negates the charge
    # # so can safely parse the string by splitting at negative signs
    # # if this does not work, the input is malformatted
    #
    # print(f.formula)
    # f2 = CompoundDict(f.formula)
    # print(f2.formula)

    # test_neg()

    for i in range(5):
        test_performance()
