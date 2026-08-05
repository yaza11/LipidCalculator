"""
For a molecule with given mass, generate the tree corresponding to masses
from the isotopes and their adducts
"""
from itertools import product

import numpy as np
import pandas as pd
from IsoSpecPy import IsoDistribution, IsoTotalProb

from LipidCalculator import CompoundDict
from LipidCalculator.isotopes.iso_frac_predict import IsotopePattern
from LipidCalculator.rdkit.adduct.parser import get_mz_from_M_and_adduct, _get_adduct_composition

default_adducts = [
    '[M]+',
    '[M+H]+',
    '[M+NH4]+',
    '[M+Na]+',
    '[M+K]+',
    '[M+H-H2O]+',
    '[M+H+H]2+',  # double charged
    '[M+H+Na]2+',
    '[M+H+NH4]2+',
    '[M2]+',  # clusterd
    '[M2+H]+',
    '[M2+NH4]+',
    '[M2+Na]+',
]


def get_mzs_for_molecule(
        formula: str | CompoundDict,
        adduct_types: list[str],
        explained_intensity_isopattern: float,
        resolution: int = 40_000,
        adduct_abundances: list[int | float] = None
) -> IsotopePattern:
    if isinstance(formula, str):
        comp: CompoundDict = CompoundDict(formula)
    else:
        assert isinstance(formula, CompoundDict)
        comp: CompoundDict = formula
    if adduct_abundances is None:
        adduct_abundances = [1] * len(adduct_types)

    # level 1: molecule masses of isotopes
    pattern: IsoDistribution = IsoTotalProb(
        prob_to_cover=explained_intensity_isopattern, formula=comp.formula
    )
    masses_isotopes: list[float] = list(pattern.masses)
    ints_isotopes: list[float] = list(pattern.probs)

    # level 2: for each isotope, generate the m/z for the adduct types
    mzs: list[float] = []
    ints: list[float] = []
    for (iso_mass, iso_int), (adduct, adduct_weight) in product(
            zip(masses_isotopes, ints_isotopes),
            zip(adduct_types, adduct_abundances)
    ):
        mzs.append(get_mz_from_M_and_adduct(iso_mass, adduct))
        ints.append(iso_int * adduct_weight)

    # coarse-grain to resolving power m / dm
    isopattern = IsotopePattern(masses=mzs, intensities=ints)
    isopattern.bin_close_weighted(resolution=resolution)
    return isopattern


def get_mass_from_mz_and_adduct(mz: float, adduct: str) -> float:
    add_mass, add_c, molecule_multiplicity = _get_adduct_composition(adduct)
    return (mz * add_c - add_mass) / molecule_multiplicity


def get_possible_adduct_diffs(mz1, mz2, mz_precision, adducts) -> list[tuple[str, str]]:
    """Check which combination of adducts would fit the mass difference between the two m/z values"""
    match_precision = np.sqrt(2) * mz_precision
    # for all combinations of adducts, check whether mass difference matches the assigned adduct pair
    possible_adduct_changes = []
    for adduct1, adduct2 in product(adducts, adducts):
        if adduct1 == adduct2:  # this should be checked separately
            continue
        M1 = get_mass_from_mz_and_adduct(mz1, adduct1)
        M2 = get_mass_from_mz_and_adduct(mz2, adduct2)
        if abs(M1 - M2) < match_precision:
            possible_adduct_changes.append((adduct1, adduct2))
    return possible_adduct_changes


if __name__ == '__main__':
    # check against https://www.envipat.eawag.ch/index.php
    pattern = get_mzs_for_molecule(
        'C43H88O3',
        adduct_types=['[M2+H]+'],
        adduct_abundances=[1],
        explained_intensity_isopattern=0.999
    )

    pattern.plot()
