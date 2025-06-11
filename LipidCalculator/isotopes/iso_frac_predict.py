"""Given a formula, predict the isotope pattern"""
from typing import Iterable, OrderedDict, Self

import numpy as np
import pandas as pd
from IsoSpecPy import ParseFormula, IsoDistribution
from matplotlib import pyplot as plt
from scipy.optimize import least_squares, OptimizeResult, minimize, Bounds

from isotopes import isotope_properties as isotope_properties
import IsoSpecPy as isospec

# PDB standard
# TODO: which one to use? why are there multiple?
# f_VPDB_C13 = 0.01123720
f_VPDB_C13 = 0.0111802
f_VSMOV_O17 = 379.9e-6  # https://en.wikipedia.org/wiki/Vienna_Standard_Mean_Ocean_Water
f_VSMOV_H2 = 155.76e-6
ATOM2ISOS: dict[str, tuple[str, str]] = {
    'H': ('H[1]', 'H[2]'),
    'C': ('C[12]', 'C[13]'),
    'O': ('O[16]', 'O[18]')
}
DEFAULT_MASS_TOLERANCE: float = 5e-3


def C13C12to_delta13C(pC12, pC13):
    """Calculate delta value from C13 and C12 portions"""
    return (pC13 / pC12 / f_VPDB_C13 - 1) * 1000  # =delta13C in permil


def H2H1to_delta13C(pH1, pH2):
    """Calculate delta value """
    return (pH2 / pH1 / f_VSMOV_H2 - 1) * 1000


def O17O16to_delta13C(pO16, pO17):
    """Calculate delta value """
    return (pO17 / pO16 / f_VSMOV_O17 - 1) * 1000


def delta13C_to_f(delta13C):
    """Calculate fraction of C13 and C12 from delta value"""
    f = (delta13C / 1000 + 1) * f_VPDB_C13
    # since C13 + C12 = 1
    C13 = f / (1 + f)
    C12 = 1 - C13
    return C13, C12


def delta2H_to_f(delta2H):
    """Calculate fraction of C13 and C12 from delta value"""
    f = (delta2H / 1000 + 1) * f_VSMOV_H2
    # since 2H + H = 1
    H2 = f / (1 + f)
    H1 = 1 - H2
    return H2, H1


def delta17O_to_f(delta17O):
    """Calculate fraction of C13 and C12 from delta value"""
    f = (delta17O / 1000 + 1) * f_VSMOV_O17
    # since 2H + H = 1
    O17 = f / (1 + f)
    O16 = 1 - O17
    return O17, O16


class IsoTable:
    """Provides the probability and mass matrices/vectors/list of lists"""
    probabilities: list[float] | None = None

    def __init__(self, atoms_in_table: list[str]):
        self.atoms_in_table = atoms_in_table

        self.isotopes: list[str] = []
        for atom in self.atoms_in_table:
            self.isotopes.extend(ATOM2ISOS[atom])

        self.masses: list[float] = [
            isotope_properties.at[iso, 'Relative Atomic Mass']
            for iso in self.isotopes
        ]

    def set_natural_probabilities(self):
        self.probabilities: list[float] = [
            isotope_properties.at[iso, 'Isotopic Composition']
            for iso in self.isotopes
        ]

    def set_custom_probabilities(self, mapper: dict[str, float]) -> None:
        """
        :param mapper: keys: isotope names, values: probabilities. Not provided
            values will be filled with the natural abundances
        :return: None
        """
        self.probabilities: list[float] = [
            mapper.get(iso, isotope_properties.at[iso, 'Isotopic Composition'])
            for iso in self.isotopes
        ]

    def set_probabilities_from_ratios(self, ratios: dict[str, float]) -> None:
        """
        :param ratios: dictionary where the keys are the element abbreviations
            and the values the ratios (rare over common). Missing elements will
            be filled with the natural abundances.
        :return: None
        """
        self.probabilities = []
        for a in self.atoms_in_table:
            if a in ratios:
                r = ratios[a]
                self.probabilities.extend([
                    1 / (1 + r),  # e.g. 1H
                    r / (1 + r)  # e.g. 2H
                ])
            else:
                isos = ATOM2ISOS[a]
                self.probabilities.extend(
                    [isotope_properties.at[iso, 'Isotopic Composition']
                     for iso in isos]
                )

    def set_probabilites_from_delta(
            self,
            d13C: float | None = None,
            d2H: float | None = None,
            d17O: float | None = None
    ) -> None:
        """Only supports d13C right now. Will use natural abundances for other elements"""
        mapper: dict[str, float] = {}
        if d13C is not None:
            C13, C12 = delta13C_to_f(d13C)
            mapper |= {'C[13]': C13, 'C[12]': C12}
        if d2H is not None:
            H2, H1 = delta2H_to_f(d2H)
            mapper |= {'H[2]': H2, 'H[1]': H1}
        if d17O is not None:
            O17, O16 = delta17O_to_f(d17O)
            mapper |= {'O[17]': O17, 'O[16]': O16}
        self.set_custom_probabilities(mapper)

    @property
    def ratios(self) -> list[float]:
        n_iso = len(self.probabilities) // 2
        fs = []
        for i in range(n_iso):
            fs.append(self.probabilities[i * 2 + 1] / self.probabilities[i * 2])
        return fs

    @property
    def ps_isospec(self) -> list[list[float]]:
        # iso expects list of lists
        ps: list[list[float]] = []
        for idx, a in enumerate(self.atoms_in_table):
            probs_el = [self.probabilities[idx * 2], self.probabilities[idx * 2 + 1]]
            ps.append(probs_el)
        return ps

    @property
    def ms_isospec(self) -> list[list[float]]:
        ms: list[list[float]] = []
        for idx, a in enumerate(self.atoms_in_table):
            ms_el = [self.masses[idx * 2], self.masses[idx * 2 + 1]]
            ms.append(ms_el)
        return ms

    def as_deltas(self):
        el_to_f = {
            'C': C13C12to_delta13C,
            'H': H2H1to_delta13C,
            'O': O17O16to_delta13C
        }

        out = {}
        for el, row in zip(self.atoms_in_table, self.ps_isospec):
            out[el] = el_to_f[el](*row)
        return out

    def renormalize(self):
        """Make sure probabilites for each atom add up to 1"""
        for idx, a in enumerate(self.atoms_in_table):
            n = self.probabilities[idx * 2] + self.probabilities[idx * 2 + 1]
            self.probabilities[idx * 2] /= n
            self.probabilities[idx * 2 + 1] /= n

    def __repr__(self) -> str:
        return pd.DataFrame(data=self.ps_isospec, index=self.atoms_in_table).to_string()


class IsotopePattern:
    def __init__(self, masses, intensities):
        self.masses: list[float] = list(masses)
        self.intensities: list[float] = list(intensities)
        assert len(self.masses) == len(self.intensities), \
            'number of entries for masses and intensities do not match'
        self.n_peaks: int = len(self.masses)

    def bin_into_targets(
            self,
            iso_masses_measured: Iterable[float],
            mass_tol: float = DEFAULT_MASS_TOLERANCE
    ) -> None:
        """
        IsoSpecPy uses fine-structure spectrum, we can only measure with finite
        resolution, so bin the predicted masses into those of the measured once
        with a given tolerance
        """
        iso_masses_measured = np.array(iso_masses_measured)

        # cannot use n_peaks since we may have fewer peaks than target masses
        binned_intensities_predicted: list[float] = [0] * iso_masses_measured.shape[0]
        for m, i in zip(self.masses, self.intensities):
            dists = np.abs(iso_masses_measured - m)
            idx_dist = np.argmin(dists)
            if dists[idx_dist] < mass_tol:
                binned_intensities_predicted[idx_dist] += i
            else:
                print(
                    f'unmatched peak: {m=:.4f}, {i=:.2f}, distance: {dists[idx_dist]:.4f}, bins: {np.round(iso_masses_measured, 4)}')

        self.intensities = binned_intensities_predicted
        self.masses = list(iso_masses_measured)
        self.n_peaks = len(self.masses)

    def bin_close(self, mass_tol: float = DEFAULT_MASS_TOLERANCE):
        bins: list[float] = list(
            map(lambda m: round(m / mass_tol) * mass_tol, self.masses)
        )
        self.bin_into_targets(bins, mass_tol=mass_tol)

    def plot(self, ax: plt.Axes | None = None, shift: int = 0, **kwargs) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()

        masses_plot = list(map(lambda m: m + shift * DEFAULT_MASS_TOLERANCE, self.masses))
        ax.stem(masses_plot, self.intensities, markerfmt=kwargs.pop('markerfmt', ''), **kwargs)
        ax.set_xlabel('mass in Da')
        ax.set_ylabel('Fraction')
        return ax

    @classmethod
    def from_iso_table(
            cls, formula, iso_table: IsoTable, mass_tolerance: float = DEFAULT_MASS_TOLERANCE
    ) -> Self:
        ...

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({'mass': self.masses, 'intensity': self.intensities})

    def __repr__(self) -> str:
        return self.as_dataframe().to_string()


class IsoPatternFit:
    """
    Fit the isotope pattern of a molecule with a given formula for possibly
    non-natural isotope abundances.
    """

    def __init__(
            self,
            formula: str,
            atoms_fit: list[str],
            measured_isotope_pattern: IsotopePattern | None = None,
            iso_table: IsoTable | None = None
    ) -> None:
        self.formula = formula

        options_fit_elements: list[str] = 'H C O'.split()
        assert all([f in options_fit_elements for f in atoms_fit]), \
            f'can only fit isotopes for elements {options_fit_elements}'
        self.atoms_fit: list[str] = atoms_fit

        # decompose formula into part that is used to fit spectrum and remainder
        self.atom_counts, self.remainder = self._get_atom_counts_fitted_elements()

        if iso_table is None:
            iso_table = IsoTable(atoms_in_table=atoms_fit)
            iso_table.set_natural_probabilities()
        else:
            assert hasattr(iso_table, 'probabilities'), \
                'if isotable is provided, the probabilites must be set'
        self.iso_table: IsoTable = iso_table

        if measured_isotope_pattern is None:
            self.update_predicted_pattern()
            self.iso_pattern.bin_close()
        else:
            self.iso_pattern: IsotopePattern = measured_isotope_pattern

        self._target_masses: list[float] = self.iso_pattern.masses.copy()
        self._target_intensities: list[float] = self.iso_pattern.intensities.copy()

    def _get_atom_counts_fitted_elements(self) -> tuple[list[int], OrderedDict]:
        """Return the atom counts of elements to be fitted and remaining formula"""
        parsed: OrderedDict = ParseFormula(self.formula)
        atom_counts: list[int] = [parsed[el] for el in self.atoms_fit]
        for remove_element in self.atoms_fit:
            parsed.pop(remove_element)
        return atom_counts, parsed

    def update_predicted_pattern(self) -> None:
        # TODO: isotope pattern is slightly off ... why?
        predicted_pattern: IsoDistribution = isospec.IsoThreshold(
            formula=self.remainder,
            threshold=.05,
            atomCounts=self.atom_counts,
            isotopeProbabilities=self.iso_table.ps_isospec,
            isotopeMasses=self.iso_table.ms_isospec
        )
        self.iso_pattern = IsotopePattern(predicted_pattern.masses, predicted_pattern.probs)

    def fit_spectrum(
            self,
            mass_tol=DEFAULT_MASS_TOLERANCE,
            scalar_min: bool = False
    ) -> OptimizeResult:
        """iteratively tweak probabilities to match spectrum using IsoSpecPy"""

        def _error(pred_intensities: np.ndarray[float]) -> np.ndarray[float] | float:
            diff_vec = (pred_intensities - self._target_intensities) ** 2
            return diff_vec.sum() if scalar_min else diff_vec

        def pred(fs: np.ndarray[float]) -> np.ndarray[float]:
            # convert fs to ps
            ratios_mapper = dict(zip(self.atoms_fit, fs))
            self.iso_table.set_probabilities_from_ratios(ratios_mapper)
            self.update_predicted_pattern()
            # bin into provided masses
            self.iso_pattern.bin_into_targets(self._target_masses, mass_tol=mass_tol)
            pred_intensities_binned = np.array(self.iso_pattern.intensities)
            return _error(pred_intensities_binned)

        fs0 = np.array(self.iso_table.ratios)

        eps = 1e-21  # p of 0 is not allowed
        if scalar_min:
            bounds = [(eps, 10) for _ in fs0]
            return minimize(pred, x0=fs0, bounds=bounds)
        else:
            return least_squares(pred, x0=fs0, bounds=(eps, 10))

    def as_deltas(self) -> dict:
        el_to_f = {
            'C': C13C12to_delta13C,
            'H': H2H1to_delta13C,
            'O': O17O16to_delta13C
        }

        out = {}

        for i, el in enumerate(self.atoms_fit):
            ps = self.iso_table.ps_isospec[i]
            out[el] = el_to_f[el](*ps)
        return out


def direct_predict_d13C(formula, iso_pattern: IsotopePattern, d2H: float = -200) -> float:
    """Assume H O N occur with their natural abundances"""
    parsed = ParseFormula(formula)
    nH = parsed.get('H', 0)
    nC = parsed.get('C', 0)
    nO = parsed.get('O', 0)
    nN = parsed.get('N', 0)

    M0 = iso_pattern.intensities[0]
    M1 = iso_pattern.intensities[1]
    f_H = f_VSMOV_H2 * (d2H / 1000 + 1)
    # note: in equation in manuscript we are not dividing by 1 + f_H but this
    # changes the H contribution only by a factor of .9999 (assuming d2H = -200)
    contribution_H = f_H * nH  # / (1 + f_H)
    contribution_O = isotope_properties.loc['O[17]', 'Isotopic Composition'] * nO
    contribution_N = isotope_properties.loc['N[15]', 'Isotopic Composition'] * nN

    # M1_corr = M1 - contribution_H - contribution_O - contribution_N
    # rC13 = M1_corr / M0 / nC
    rC13 = (M1 / M0 - contribution_H - contribution_O - contribution_N) / nC
    return (rC13 / f_VPDB_C13 - 1) * 1000


def direct_predict_d13C_exact(formula, iso_pattern: IsotopePattern, d2H: float = -200) -> float:
    parsed = ParseFormula(formula)
    nH = parsed.get('H', 0)
    nC = parsed.get('C', 0)
    nO = parsed.get('O', 0)
    nN = parsed.get('N', 0)

    if nC == 0:
        raise ValueError('cannot calculate d13C for molecule without carbon')

    M0 = iso_pattern.intensities[0]
    M1 = iso_pattern.intensities[1]

    if nH > 0:
        f_H = f_VSMOV_H2 * (d2H / 1000 + 1) * (nH - 1)
    else:
        f_H = 0
    if nO > 0:
        f_O = (nO - 1) * isotope_properties.loc['O[17]', 'Isotopic Composition'] / isotope_properties.loc[
            'O[16]', 'Isotopic Composition']
    else:
        f_O = 0
    if nN > 0:
        f_N = (nN - 1) * isotope_properties.loc['N[15]', 'Isotopic Composition'] / isotope_properties.loc[
            'N[14]', 'Isotopic Composition']
    else:
        f_N = 0

    fC = (M1 / M0 - f_H - f_O - f_N) / (nC - 1)
    return (fC / f_VPDB_C13 - 1) * 1000


def test_archaeol():
    formula = 'C43H88O3'

    _pH1 = isotope_properties.loc['H[1]', 'Isotopic Composition']
    _pH2 = isotope_properties.loc['H[2]', 'Isotopic Composition']
    _pC12 = isotope_properties.loc['C[12]', 'Isotopic Composition']
    _pC13 = isotope_properties.loc['C[13]', 'Isotopic Composition']
    _pN14 = isotope_properties.loc['N[14]', 'Isotopic Composition']
    _pN15 = isotope_properties.loc['N[15]', 'Isotopic Composition']
    _pO16 = isotope_properties.loc['O[16]', 'Isotopic Composition']
    _pO17 = isotope_properties.loc['O[17]', 'Isotopic Composition']
    # _pO18 = isotope_properties.loc['O[18]', 'Isotopic Composition']
    # P has only one stable isotope
    # _pS32 = isotope_properties.loc['S[32]', 'Isotopic Composition']
    # _pS33 = isotope_properties.loc['S[33]', 'Isotopic Composition']
    # _pS34 = isotope_properties.loc['S[34]', 'Isotopic Composition']
    # _pS36 = isotope_properties.loc['S[36]', 'Isotopic Composition']

    parsed = ParseFormula(formula)
    nH = parsed.get('H', 0)
    nC = parsed.get('C', 0)
    nO = parsed.get('O', 0)
    nN = parsed.get('N', 0)

    use_one_correc = False
    I0_th = _pC12 ** nC * _pO16 ** nO * _pH1 ** nH * _pN14 ** nN
    I1_C = _pC12 ** (nC - 1) * _pC13 * _pO16 ** nO * _pH1 ** nH * _pN14 ** nN * (nC - use_one_correc)
    I1_C2 = _pC13 / _pC12 * nC * I0_th
    I1_H = _pC12 ** nC * _pO16 ** nO * _pH1 ** (nH - 1) * _pH2 * _pN14 ** nN * (nH - use_one_correc)
    I1_H2 = _pH2 / _pH1 * nH * I0_th
    I1_O = _pC12 ** nC * _pO16 ** (nO - 1) * _pO17 * _pH1 ** nH * _pN14 ** nN * (nO - use_one_correc)
    I1_O2 = _pO17 / _pO16 * nO * I0_th
    I1_N = _pC12 ** nC * _pO16 ** nO * _pH1 ** nH * _pN14 ** (nN - 1) * _pN15 * (nN - use_one_correc)
    I1_N2 = _pN15 / _pN14 * nN * I0_th

    I1_th = I1_C + I1_H + I1_O + I1_N
    I1_th2 = (_pC13 / _pC12 * nC + _pO17 / _pO16 * nO + _pH2 / _pH1 * nH + _pN15 / _pN14 * nN) * I0_th

    print((_pH2 / _pH1 / f_VSMOV_H2 - 1) * 1000)
    print((_pC13 / _pC12 / f_VPDB_C13 - 1) * 1000)

    rC = 1 / nC * (I1_th / I0_th - _pH2 / _pH1 * nH - _pO17 / _pO16 * nO - _pN15 / _pN14 * nN)
    print((rC / f_VPDB_C13 - 1) * 1000)

    # print(I1_C2 - I1_C)
    # print(I1_H2 - I1_H)
    # print(I1_O2 - I1_O)
    # print(I1_N2 - I1_N)

    # predict the pattern
    iso_table = IsoTable(atoms_in_table='C H O'.split())
    iso_table.set_natural_probabilities()
    iso_fit = IsoPatternFit(formula=formula, iso_table=iso_table, atoms_fit='C H O'.split())
    iso_pattern = iso_fit.iso_pattern

    # predicted_pattern = isospec.IsoThreshold(
    #     formula=formula,
    #     threshold=.05
    # )

    print(iso_pattern)
    print(I0_th)
    print(I1_th)
    # print(I1_th2)


if __name__ == '__main__':
    f_archaeol = 'C43H88O3'

    # generate artificial spectrum
    actual_deltaC13 = -100
    actual_deltaH2 = -200

    fit_elements = ['H', 'C']

    iso_table = IsoTable(atoms_in_table=fit_elements)
    iso_table.set_probabilites_from_delta(d13C=actual_deltaC13, d2H=actual_deltaH2)

    print(iso_table.as_deltas())

    # natural abundance
    iso_fit = IsoPatternFit(
        formula=f_archaeol, atoms_fit=fit_elements)
    # actual (synthetic) pattern
    iso_fit2 = IsoPatternFit(
        formula=f_archaeol, atoms_fit=fit_elements, iso_table=iso_table)
    # fit to actual pattern
    # iso_fit3 = IsoPatternFit(
    #     formula=f_archaeol,
    #     atoms_fit=fit_elements,
    #     measured_isotope_pattern=iso_fit2.iso_pattern)
    # iso_fit3.fit_spectrum()

    iso_fit4 = IsoPatternFit(
        formula=f_archaeol,
        atoms_fit=fit_elements,
        measured_isotope_pattern=iso_fit2.iso_pattern)
    iso_fit4.fit_spectrum(scalar_min=True)

    ax = iso_fit.iso_pattern.plot()
    iso_fit2.iso_pattern.plot(ax=ax, linefmt='orange', shift=1)
    # iso_fit3.iso_pattern.plot(ax=ax, linefmt='green', shift=2)
    iso_fit4.iso_pattern.plot(ax=ax, linefmt='red', shift=2)
    ax.legend(['natural', 'fractionated', 'predicted', 'predicted scalar'])
    plt.show()

    fitted_deltas = iso_fit4.as_deltas()

    print('fitted:', fitted_deltas)

    print(direct_predict_d13C(f_archaeol, iso_fit.iso_pattern, d2H=-261.6))
    print(direct_predict_d13C_exact(f_archaeol, iso_fit.iso_pattern, d2H=-261.6))

    print(direct_predict_d13C(f_archaeol, iso_fit2.iso_pattern))
    print(direct_predict_d13C_exact(f_archaeol, iso_fit2.iso_pattern))

    # test_archaeol()
