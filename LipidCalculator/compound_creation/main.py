"""Define compound objects for common geochemical compound groups."""
from typing import ItemsView, Iterable

import numpy as np

from LipidCalculator.consts import m_e

from .formula_parser import CompoundDict, remove_charge, get_charge_from_str


def check_formula_valid(formula: str):
    if (('+' in formula) and ('-' in formula)):
        return False
    elif formula.count('(') != formula.count(')'):
        return False
    # check for plus and minus anywhere else in string (can only be at the end)
    ...
    return True


class Compound:
    charge: int | None = None
    mass: float | None = None
    formula: str | None = None
    composition: CompoundDict | None = None
    abbreviation: str | None = None
    name: str | None = None
    smiles: str | None = None
    graph: np.ndarray | None = None

    def __init__(
            self,
            *,
            name: str | None = None,
            smiles: str | None = None,
            graph: np.ndarray | None = None,
            comp: dict | None = None,
            formula: str | None = None,
            mass: float | None = None,
            adduct: str | None = None,
            abbreviation: str | None = None
    ) -> None:
        """
        :param name: the name of the compound
        :param mass: the mass of the compound
        :param formula: the formula of the compound
        :param adduct: possible formula making up the adduct provided for the
            formula and/ or mass
        :param abbreviation: abbreviation to be used for this compound
        """
        # check for conflicting inputs
        has_formula: bool = formula is not None
        has_comp: bool = comp is not None
        has_mass: bool = mass is not None
        has_adduct: bool = adduct is not None

        # tests
        if has_mass:
            assert isinstance(mass, float | int), 'mass must be a float or int'
        if has_comp:
            if type(comp) is dict:
                comp = CompoundDict(comp)
            assert type(comp) is CompoundDict, 'composition must be a dict or CompoundDict'
        if has_formula:
            assert type(formula) is str, 'formula must be a str'
            assert check_formula_valid(formula), 'formula is not valid'
        if has_adduct:
            assert type(adduct) is str, 'adduct must be a str'
        if abbreviation is not None:
            assert type(abbreviation) is str, 'abbreviation must be a str'
        if name is not None:
            assert type(name) is str, 'name must be a str'
        if smiles is not None:
            assert type(smiles) is str, 'smiles must be a str'
        if graph is not None:
            assert isinstance(graph, np.ndarray), 'graph must be an array'

        assert has_mass or has_formula or has_comp, \
            'provide either a mass or formula or CompoundDict'
        assert (not has_comp) or (not has_adduct), \
            'if a composition is given, no adduct can be specified'

        if abbreviation is not None:
            self.abbreviation: str = abbreviation
        if name is not None:
            self.name: str = name

        # will assign 0 mass and empty CompoundDict if adduct is None
        adduct: CompoundDict = self._handle_adduct(adduct)

        if has_comp:
            formula: str = self._from_comp(comp, formula)
            has_formula: bool = True

        if has_formula:
            self._set_from_formula(formula, adduct, comp)

        if has_adduct and (not has_formula):
            self.mass -= adduct.mass

    @staticmethod
    def _handle_adduct(adduct: None | str) -> CompoundDict:
        """Calculate the mass of the provided adduct and return a CompoundDict instance"""
        if adduct is not None:
            assert type(adduct) is str, 'adduct must be a string'
            adduct: CompoundDict = CompoundDict(adduct)
            return adduct
        return CompoundDict({})

    @staticmethod
    def _verify_formula(comp: CompoundDict, formula: str) -> None:
        """Check if the provided formula matches the dict."""
        if comp != CompoundDict(formula):
            raise ValueError('provided dict does not match formula')

    def _from_comp(self, comp: CompoundDict, formula: str | None) -> str:
        """Set the composition of the compound based on composition or formula
        (if specified) and return the formula."""
        if formula is not None:
            self._verify_formula(comp, formula)
        self.composition: CompoundDict = comp

        # keep provided format, should the provided formula match
        return comp.formula if formula is None else formula

    def _set_from_formula(
            self,
            formula: str,
            adduct: CompoundDict,
            mass: float | None
    ) -> None:
        """Set the formula (without the adduct) and check the mass if provided."""
        formula: str = self._remove_adduct(formula, adduct)
        self.formula = formula

        if self.composition is None:
            self.composition = CompoundDict(self.formula)

        self.charge = get_charge_from_str(self.formula)

        self._set_mass_from_charge_and_comp()
        if mass is not None:
            assert np.abs(self.mass + adduct.mass - mass) < 1e-4, \
                ('provided mass does not match mass calculated from formula '
                 f'(calculated: {self.mass + adduct.mass}, expected: {mass})')

    @staticmethod
    def _remove_adduct(formula: str, adduct: CompoundDict) -> str:
        cd: CompoundDict = CompoundDict(remove_charge(formula))
        cd_new: CompoundDict = cd - adduct
        return cd_new.formula

    def _set_mass_from_charge_and_comp(self) -> None:
        # e.g. charge of +1 means one electron less compared to neutral
        m_charge: float = -m_e * self.charge
        self.mass: float = self.composition.mass + m_charge

    def items(self) -> ItemsView:
        return self.composition.items()

    def values(self) -> Iterable:
        return self.composition.values()

    def keys(self) -> Iterable:
        return self.composition.keys()


if __name__ == '__main__':
    pass

# class CombinedComponent():
#     def __init__(self, head: Component, body: Component, tail: Component,
#                  side_groups: list[Component] | None = None, name: str = None, abbreviation: str = None):
#
#         # side_groups: list of components
#         self.head = head
#         self.body = body
#         self.tail = tail
#         if name is None:
#             self.name = self.combine_names(self.head, self.body, self.tail)
#         else:
#             self.name = name
#         self.abbreviation = abbreviation
#         self.composition = self.combine_composition(self.head,
#                                                     self.body,
#                                                     self.tail,
#                                                     side_groups)
#         self.mass = self.combine_masses(self.head, self.body, self.tail)
#
#     def combine_names(self, head, body, tail):
#         # if possible use abbreviation, else fall back to name
#         part_names = [''] * 3
#         for i, part in enumerate([head, body, tail]):
#             if part.abbreviation is None:
#                 part_name = part.name
#             else:
#                 part_name = part.abbreviation
#             part_names[i] = part_name
#         return f'{part_names[0]}-{part_names[1]} {part_names[2]}'
#
#     def combine_composition(self, head, body, tail, side_groups):
#         '''
#         combine the count of elements for multiple components
#
#         Parameters
#         ----------
#         head : Component-Class
#             head group (usually has one bonding place).
#         body : Component-Class
#             body group (usually has one bonding place for head and
#                         two for chains).
#         tail : Component-Class
#             usually C-chain.
#         side_groups : list[Component-Class]
#             possiblity to add more functional groups to molecule.
#
#         Returns
#         -------
#         str
#             composition of combined component.
#
#         '''
#         dict_sum = {}
#         groups = [head, body, tail]
#         if side_groups is not None:
#             groups.extend(side_groups)
#         for comp in groups:
#             # split letters and numbers
#             list_entries = self.split_at_elements(comp.composition)
#             if (L := len(list_entries)) % 2 == 1:
#                 L += 1
#                 list_entries.append('1')
#             for i in range(L // 2):
#                 key = list_entries[2 * i]
#                 if key not in dict_sum:
#                     dict_sum[list_entries[2 * i]] = int(list_entries[2 * i + 1])
#                 else:
#                     dict_sum[list_entries[2 * i]] += int(list_entries[2 * i + 1])
#         return ''.join([key + str(value) for key, value in dict_sum.items()])
#
#     def combine_masses(self, head, body, tail):
#         return head.mass + body.mass + tail.mass
#
#     def split_at_elements(self, composition_str):
#         '''
#         returns a list of elements and counts for a composition str
#
#         Parameters
#         ----------
#         composition_str : str
#             composition of component.
#
#         Returns
#         -------
#         list_entries : list
#             list with element name followed by the number of that element for
#             all elements in the compound.
#
#         '''
#         list_entries = []
#         current_entry = ''
#         previous_str = ''
#         for s in composition_str:
#             # if s is an upper case letter, start new entry
#             if s.isupper():
#                 # append list if current is not empty
#                 if current_entry != '':
#                     list_entries.append(current_entry)
#                 # one element following another without number inbetween
#                 if previous_str.isalpha():
#                     list_entries.append('1')
#                 current_entry = s
#             elif s == '-':
#                 list_entries.append(current_entry)
#                 current_entry = s
#             # start new entry if previous str was letter
#             elif s.isdigit() and previous_str.isalpha():
#                 list_entries.append(current_entry)
#                 current_entry = s
#             else:
#                 current_entry += s
#
#             previous_str = s
#         # handle the last entry
#         if current_entry != '':
#             list_entries.append(current_entry)
#         if previous_str.isalpha():
#             list_entries.append('1')
#
#         return list_entries
#
#
# def C_chain(num_C, num_double_bounds=0):
#     '''
#
#
#     Parameters
#     ----------
#     num_C : int
#         number of C atoms in tail chains (two C-atoms are counted as belonging
#                                           to body part)
#     num_double_bounds : int
#         number of double bounds
#
#     Returns
#     -------
#     compound object of chain, notice that two C atoms less than one would
#     expect
#
#     '''
#     n_C = num_C - 2  # subtract 2 as they are part of body
#     n_H = n_C * 2 + 2 - 2 * num_double_bounds
#     return Component(
#         name=f'C{num_C}:{num_double_bounds}-Chain',
#         composition=f'C{n_C}H{n_H}',
#         abbreviation=f'C{num_C}:{num_double_bounds}'
#     )
#
#
# def menaquinone(n_isoprenyl, n_double_bounds=None):
#     if n_double_bounds is None:
#         n_double_bounds = n_isoprenyl
#     n_C = 11 + n_isoprenyl * 5
#     n_H = 7 + n_isoprenyl * 8 - 2 * (n_double_bounds - n_isoprenyl) + 1
#     return Component(
#         name=f'Menaquinone {n_isoprenyl}:{n_double_bounds}',
#         composition=f'C{n_C}H{n_H}O2',
#         abbreviation=f'MK{n_isoprenyl}:{n_double_bounds}'
#     )
#
#
# def ubiquinone(n_isoprenyl, n_double_bounds=None):
#     if n_double_bounds is None:
#         n_double_bounds = n_isoprenyl
#     n_C = 9 + n_isoprenyl * 5
#     n_H = 9 + n_isoprenyl * 8 - 2 * (n_double_bounds - n_isoprenyl) + 1
#     return Component(
#         name=f'Ubiquinone {n_isoprenyl}:{n_double_bounds}',
#         composition=f'C{n_C}H{n_H}O4',
#         abbreviation=f'UQ{n_isoprenyl}:{n_double_bounds}'
#     )
#
#
# def plastoquinone(n_isoprenyl, n_double_bounds=None):
#     if n_double_bounds is None:
#         n_double_bounds = n_isoprenyl
#     n_C = 8 + n_isoprenyl * 5
#     n_H = 7 + n_isoprenyl * 8 - 2 * (n_double_bounds - n_isoprenyl) + 1
#     return Component(
#         name=f'PLastoquinone {n_isoprenyl}:{n_double_bounds}',
#         composition=f'C{n_C}H{n_H}O2',
#         abbreviation=f'PQ{n_isoprenyl}:{n_double_bounds}'
#     )
#
#
# def chlorophyll(chl_type):
#     # ring E
#     ringE = Component(name='ringE', composition='C4H4O3')
#     if chl_type == 'a':
#         side_groups = [methyl, ethylenyl, methyl, ethyl, acrylate_phytyl]
#     elif chl_type == 'b':
#         side_groups = [methyl, ethylenyl, CHO, ethyl, acrylate_phytyl]
#     elif chl_type == 'c1':
#         side_groups = [methyl, ethylenyl, methyl, ethyl, acrylate,
#                        Component(composition='H-2')]
#     elif chl_type == 'c2':
#         side_groups = [methyl, ethylenyl, methyl, ethylenyl, acrylate,
#                        Component(composition='H-2')]
#     elif chl_type == 'd':
#         side_groups = [methyl, CHO, methyl, ethyl, acrylate_phytyl]
#     elif chl_type == 'f':
#         side_groups = [CHO, ethylenyl, methyl, ethyl, acrylate_phytyl]
#     # same for all of them
#     side_groups.extend([methyl, methyl])
#     # remove H atoms where we have functional groups instead
#     side_groups.append(Component(composition='H-7'))
#     # remove 2 H atoms that bind to Mg
#     side_groups.append(Component(composition='H-2'))
#     # remove 2 extra H atoms from somewhere
#     side_groups.append(Component(composition='H-2'))
#     return Combined_Component(
#         ringE,
#         chlorin,
#         central_Mg,
#         side_groups=side_groups,
#         name=f'Chlorophyll {chl_type}',
#         abbreviation=f'Chl {chl_type}'
#     )
#
#
# def bacterio_chlorophyll(chl_type):
#     # ring E
#     ringE = Component(name='ringE', composition='C2H2O')
#     if chl_type == 'a':
#         side_groups = [acetyl, methyl, ethyl, methyl, acetate, phytol,
#                        single_H, missing_double_bound]
#     elif chl_type == 'b':
#         side_groups = [acetyl, methyl, ethylenyl, methyl, acetate, phytol,
#                        single_H, missing_double_bound]
#     elif chl_type == 'c1':
#         side_groups = []
#     elif chl_type == 'c2':
#         side_groups = []
#     elif chl_type == 'd':
#         side_groups = []
#     elif chl_type == 'f':
#         side_groups = []
#     # same for all of them
#     side_groups.extend([methyl, methyl, Component(composition='C3H4O2')])
#     return Combined_Component(
#         ringE,
#         chlorin,
#         central_Mg,
#         side_groups=side_groups,
#         name=f'Bacteriochlorophyll {chl_type}',
#         abbreviation=f'BChl {chl_type}'
#     )
#
#
# def create_table_long():
#     min_chain_length = 28
#     max_chain_length = 45
#     max_double_bounds = 8
#     head_body_combinations = {
#         MG: [DAG, AEG],
#         DG: [DAG, AEG],
#         PC: [DAG, AEG, DEG],
#         BL: [DAG, AEG, OH_DAG, DEG],
#         SQ: [DAG, DEG],
#         OL: [empty_comp, OH],
#         TMOL: [empty_comp],
#         PG: [DAG, AEG, DEG],
#         oneG: [DEG, AEG, DAG, CER],
#         twoG: [DEG, AEG, DAG],
#         PME: [DAG],
#         PE: [DAG],
#         PI: [AEG],
#     }
#     component_names = []
#     component_compositions = []
#     # make all possible combinations
#     for head, bodies in head_body_combinations.items():
#         for body in bodies:
#             for c_length in range(min_chain_length, max_chain_length, 1):
#                 for n_double_bounds in range(max_double_bounds):
#                     tail = C_chain(c_length, n_double_bounds)
#                     comp = Combined_Component(head, body, tail)
#                     component_names.append(comp.name)
#                     component_compositions.append(comp.composition)
#
#     data_array = np.vstack([component_names, component_compositions]).T
#     df = pd.DataFrame(data=data_array,
#                       index=None,
#                       columns=['name', 'composition'])
#
#     return df
#
#
# def testing():
#     # OL_C40d2 = Component(name='Ornithine Lipid C38:2', composition='C43H80N2O5', adduct='Na+')
#
#     # tails
#     # two C atoms are countead as contributing to body group
#     C29 = C_chain(29)
#     C32 = C_chain(32)
#     C32d1 = C_chain(32, 1)
#     C30 = C_chain(30)
#     C34 = C_chain(34)
#     C34d4 = C_chain(34, 4)
#     C36 = C_chain(36)
#     C37 = C_chain(37)
#     C41d2 = C_chain(41, 2)
#
#     MG_DAG_C29 = Combined_Component(MGDG, empty_comp, C29)
#     MG_DAG_C29_2 = Combined_Component(MG, DAG, C29)  # this works as well
#     MG_DAG_C34d4 = Combined_Component(MGDG, empty_comp, C34d4)
#
#     DG_DAG_C34d4 = Combined_Component(DGDG, empty_comp, C34d4)
#
#     oneG_DAG_C32 = Combined_Component(oneG, DAG, C32)
#     twoG_DAG_C32 = Combined_Component(twoG, DAG, C32)
#     SQ_DAG_C34 = Combined_Component(SQ, DAG, C34)
#
#     PG_DAG_C34 = Combined_Component(PG, empty_comp, C34)
#     PE_DAG_C32 = Combined_Component(PE, empty_comp, C32)
#
#     oneG_AEG_C34 = Combined_Component(oneG, AEG, C34)
#     TMOL_C37 = Combined_Component(TMOL, C37, empty_comp)
#     # BL_DAG_C32 = Combined_Component(BL_DAG, empty_comp, C32)
#     BL_DAG_C32_ = Combined_Component(BL, DAG, C32)
#
#     PME_DAG_C30 = Combined_Component(PME, DAG, C30)
#     PC_DAG_C37 = Combined_Component(PC, DAG, C37)
#     PI_AEG_C41d2 = Combined_Component(PI, AEG, C41d2)
#
#     BL_DEG_C32d1 = Combined_Component(BL, DEG, C32d1)
#     oneG_CER_C36 = Combined_Component(oneG, CER, C36)
#
#     MK9d9 = menaquinone(9, 9)
#
#     BL_OH_DAG_C37d4 = Combined_Component(BL, OH_DAG, C_chain(37, 4))
#     OL_C30d1_OH = Combined_Component(OL, C_chain(30, 1), OH)
#     print(MG_DAG_C29.mass)
#     print(MG_DAG_C29_2.mass)
#     print(MG_DAG_C34d4.mass)
#     print(DG_DAG_C34d4.mass)
#     print(PG_DAG_C34.mass, 750.5478)
#     print(PE_DAG_C32.composition, 'C37H74N1O8P1')
#     print(twoG_DAG_C32.composition, 'C47H88O15')
#     print(oneG_AEG_C34.composition)
#     # print(OL_C37.composition, 'C45H88N2O5')
#     # print(BL_DAG_C32.composition, 'C42H81N1O7')
#     print(BL_DAG_C32_.composition, 'C42H81N1O7')
#     print(PME_DAG_C30.composition, 'C36H72N1O8P1')
#     print(PC_DAG_C37.composition, 'C45H90N1O8P1')
#     print(PI_AEG_C41d2.composition, 'C50H95O12P1')
#     print(BL_DEG_C32d1.composition, 'C42H83N1O5')
#     print(oneG_CER_C36.composition, 'C42H83N1O8')
#     print(MK9d9.composition, 'C56H80O2')
#     print(menaquinone(8, 8).composition, 'C51H72O2')
#     print(menaquinone(8, 7).composition, 'C51H74O2')
#     print(ubiquinone(8, 8).composition, 'C49H74O4')
#     print(ubiquinone(8, 7).composition, 'C49H76O4')
#     print(plastoquinone(9, 9).composition, 'C53H80O2')
#     print(plastoquinone(9, 8).composition, 'C53H82O2')
#     print(chlorophyll('a').composition, 'C55H72O5N4Mg')
#     print(chlorophyll('b').composition, 'C55H70O6N4Mg')
#     print(chlorophyll('c1').composition, 'C35H30O5N4Mg')
#     print(chlorophyll('c2').composition, 'C35H28O5N4Mg')
#     print(chlorophyll('d').composition, 'C54H70O6N4Mg')
#     print(chlorophyll('f').composition, 'C55H70O6N4Mg')
#     print(BL_OH_DAG_C37d4.mass, 790.619146 - 1.007)
#     print(OL_C30d1_OH.mass, 611.4994 - 1.007)
#     print(TMOL_C37.composition, 'C45H88N2O5')
#     print(Combined_Component(OL, C32, empty_comp).mass, 625.5514 - 1.007)
#
#
# empty_comp = Component(name='', composition='', abbreviation='')
# remove_H = Component(name='-H', composition='H-1', abbreviation='-H')
# remove_O = Component(name='-O', composition='O-1', abbreviation='-O')
# remove_OH = Component(name='-OH', composition='O-1H-1', abbreviation='-OH')
# remove_C4H6 = Component(name='-C4H6', composition='C-4H-6', abbreviation='-C4H6')
#
# # some functional groups
# methyl = Component(name='methyl', composition='CH3')
# ethyl = Component(name='ethyl', composition='C2H5')
# ethylenyl = Component(name='ethylenyl', composition='C2H3')
# ethoxide = Component(name='Ethoxide', composition='C2H5O')
# CHO = Component(name='Formyl', composition='CHO')
# OH = Component(name='OH', composition='O')
# acetyl = Component(name='Acetyl', composition='C2H3O')
# acetate = Component(name='Acetate', composition='C2H3O2')
# acrylate = Component(name='Acrylate', composition='C3H3O2')
# phytol = Component(name='Phytol', composition='C20H40O')
# farnesol = Component(name='Farnesol', composition='C15H26O')
# stearyl_alcohol = Component(name='Stearyl alcohol', composition='C18H38O')
# acrylate_phytyl = Combined_Component(acrylate, phytol, Component(name='x', composition=f'O-1'))
# chlorin = Component(name='Chlorin', composition='C20H16N4')
# central_Mg = Component(name='Mg', composition='Mg')
# single_H = Component(name='H', composition='H')
# missing_double_bound = Component(name='-double bound', composition='H2', abbreviation='-DB')
#
# # heads
# SQ = Component(name='Sulfoquinovosyl', composition='C6H11O7S', abbreviation='SQ')
# oneG = Component(name='Monoglycosyl', composition='C6H11O5', abbreviation='1G')
# twoG = Combined_Component(oneG, oneG, remove_H, abbreviation='2G')
# TMOL = Component(name='Trimethylornithine lipid', composition='C10H16O5N2', abbreviation='TMOL')
# OL = Component(name='Ornithine lipid', composition='C7H10O5N2', abbreviation='OL')
# BL = Component(name='Betaine Lipid', composition='C7H14O2N1', abbreviation='BL')
# MG = Component(name='Monogalactosyl', composition='C6H11O5', abbreviation='MG')
# DG = Combined_Component(MG, MG, empty_comp, abbreviation='DG')
# # BL_DAG = Component(name='Betaine lipid Diaclyglyceryl trimethylhomoserine', composition='C12H19O7N1', abbreviation='DGTS')
#
# # bodies
# DAG = Component(name='Diacylglycerol', composition='C5H5O5', abbreviation='DAG')
# AEG = Component(name='Acyl/ether-glycerol', composition='C5H7O4', abbreviation='AEG')
# DPG = Component(name='Diphosphatidyglycerol', composition='C3H8O7P2', abbreviation='DPG')  # didn't check this one yet
# DEG = Component(name='Diether-glycerol', composition='C5H9O3', abbreviation='DEG')
# CER = Component(name='Ceramide', composition='C6H8O3N1', abbreviation='CER')
# # 6 carbon atoms belong to one side chain, so if combined with lipid chain,
# # subtract C4 (so in total 6, because for the C-chain also two C's
# # are removed)
# CER = Combined_Component(CER, remove_C4H6, empty_comp, name='CER')
# OH_DAG = Combined_Component(OH, DAG, empty_comp, name='OH-DAG')
# # combined
# # defined with one less O as it belongs to MG as well as DAG
# MGDG = Combined_Component(MG, DAG, empty_comp, name='MGDG')
# DGDG = Combined_Component(DG, DAG, remove_H, name='DGDG')
# SQDG = Combined_Component(SQ, DAG, remove_H, name='SQDG')
#
# phosphati = Component(name='Phosphati', composition='O4P1', abbreviation='P')
# glycerol = Component(name='Glycerol', composition='C3H8O3')
# PG = Combined_Component(glycerol, phosphati, DAG)
# PG = Combined_Component(PG, remove_O, remove_O, name='PG')
#
# PE = Combined_Component(
#     Component(
#         name='x',
#         composition='C2H7O1N1'),
#     phosphati,
#     DAG
# )
# PE = Combined_Component(PE, remove_O, remove_O, name='PE')
#
# PME = Component(name='Phosphatidyl-(N) methylethanolamine', composition='C3H9O3P1N1', abbreviation='PME')
# PDME = Component(name='Phosphatidyl-(N, N) dimethylethanolamine', composition='C4H11O3P1N1', abbreviation='PME')  # didn't check this one
# PC = Component(name='Phosphatidylcholine', composition='C5H13O3N1P1', abbreviation='PC')
# PI = Component(name='Phosphatidylinositol', composition='C6H12O8P1', abbreviation='PI')
#
#
# def create_filled_df():
#     df = create_table_long()
#     # add missing columns
#     cols_df = [
#         'abbreviation', 'adduct_type', 'mass_with_adduct',
#         'M', 'M+', 'mz+', 'dm_M+', 'M+H+', 'mz+H+', 'dm_H+',
#         'M+Na+', 'mz+Na+', 'dm_Na+', 'M+K+,mz+K+', 'dm_K+',
#         'M+NH4+', 'mz+NH4+', 'dm_NH4+',
#         'headgroup_type', 'middle_group_type', 'chaingroup', 'metagroup', 'from'
#     ]
#     for col_name in cols_df:
#         df[col_name] = np.nan
#
#     df_filled = table_known_compounds.fill_table(df)
#     df_filled.to_excel(
#         directory_paths.known_compounds_table_long,
#         index=False
#     )
#     return df_filled
#
#
# def get_filled_df():
#     return functions.load_feature_table(directory_paths.known_compounds_table_long, file_type='xlsx')
#
#
# if __name__ == '__main__':
#     # df_filled = get_filled_df()
#     # df_reduced = table_known_compounds.create_df_knowns(df_filled, save_df=False)
#     # df_reduced.to_excel(
#     #     directory_paths.known_compounds_excel_long_xfile(),
#     #     index=False
#     # )
#     pass
