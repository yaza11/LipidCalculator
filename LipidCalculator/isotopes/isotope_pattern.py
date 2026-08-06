"""Given a formula, predict the isotope pattern"""
from itertools import product
from typing import Iterable, OrderedDict, Self, Literal

import numpy as np
import pandas as pd
from IsoSpecPy import ParseFormula, IsoDistribution, IsoTotalProb
from matplotlib import pyplot as plt
from msIO import PeakList
from scipy.optimize import least_squares, OptimizeResult, minimize

from LipidCalculator import CompoundDict
from LipidCalculator.isotopes.standards_and_deltas import f_VPDB_C13, f_VSMOV_H2, ATOM2ISOS, DEFAULT_MASS_TOLERANCE, \
    C13C12to_delta13C, H2H1to_delta13C, O17O16to_delta17O, S34S32to_delta34S, delta13C_to_f, delta2H_to_f, \
    delta17O_to_f, delta34S_to_f
from LipidCalculator.rdkit.adduct.parser import _get_adduct_composition, Adduct
from LipidCalculator.isotopes.isotopes import isotope_properties as isotope_properties

import IsoSpecPy as isospec
import logging

from LipidCalculator.util.spectrum import recursively_merge_peaks_to_resolution

logger = logging.getLogger(__name__)

iso_abundance_input_type = Literal['probabilities', 'ratio', 'delta', 'natural', 'p', 'r', 'd', 'n']


class IsoTable:
    """Provides the probability and mass matrices/vectors/list of lists"""
    probabilities: list[float] | None = None
    atoms_in_table: list[str] = None
    isotope_names: list[str] = None
    masses: list[float] = None
    _dataframe: pd.DataFrame = None

    def __init__(
            self,
            *quantities: tuple[str, iso_abundance_input_type, float | tuple[float, float] | None]
    ):
        """
        Relative abundances of isotopes must be provided as a tuple of element name, input type and value

        Notice that no check for uniquenes of values is performed, therefore when providing e.g. dict(ratio_S=.98, d34S=0),
        the later value will override previous ones
        """
        iso_to_prob: dict[str, float] = {}
        atoms_in_table = set()
        for element, input_type, value in quantities:
            assert element in ATOM2ISOS, f'{element=} not supported'
            atoms_in_table.add(element)
            iso_names = ATOM2ISOS[element]
            # parse key to determine input type (prob, ratio or delta)
            if input_type in ('probabilities', 'p'):  # probability provided directly
                assert len(value) == 2
            elif input_type in ('ratio', 'r'):
                assert value > 0
                value = self._probs_from_ratio(value)
            elif input_type in ('delta', 'd'):
                value = self._probs_from_delta(element, value)[::-1]
            elif input_type in ('natural', 'n'):
                isos = ATOM2ISOS[element]
                value = [isotope_properties.at[iso, 'Isotopic Composition'] for iso in isos]
            else:
                raise TypeError(f'Invalid input type {input_type}')

            for iso_name, v in zip(iso_names, value):
                assert 0 <= v <= 1
                iso_to_prob[iso_name] = v

        self.atoms_in_table = list(atoms_in_table)

        self.isotope_names: list[str] = []
        for atom in self.atoms_in_table:
            self.isotope_names.extend(ATOM2ISOS[atom])

        self.masses: list[float] = [
            isotope_properties.at[iso, 'Relative Atomic Mass']
            for iso in self.isotope_names
        ]
        self.probabilities: list[float] = [iso_to_prob[iso_name] for iso_name in self.isotope_names]
        self._renormalize()

    @staticmethod
    def _probs_from_ratio(r):
        # probability of denum and num in ratio (e.g., S[32] and S[34])
        return 1 / (1 + r), r / (1 + r)

    @staticmethod
    def _probs_from_delta(element_name, delta_value):
        el_to_func = {
            'C': delta13C_to_f,
            'H': delta2H_to_f,
            'O': delta17O_to_f,
            'S': delta34S_to_f
        }
        assert element_name in el_to_func, f'delta values for element {element_name} not supported'
        return el_to_func[element_name](delta_value)

    def _as_deltas(self):
        """Convert probabilities to delta values"""
        el_to_f = {
            'C': C13C12to_delta13C,
            'H': H2H1to_delta13C,
            'O': O17O16to_delta17O,
            'S': S34S32to_delta34S
        }

        out = {}
        for el, row in zip(self.atoms_in_table, self.ps_isospec):
            out[el] = el_to_f[el](*row)
        return out

    def _renormalize(self):
        """Make sure probabilites for each atom add up to 1"""
        for idx, a in enumerate(self.atoms_in_table):
            n = self.probabilities[idx * 2] + self.probabilities[idx * 2 + 1]
            self.probabilities[idx * 2] /= n
            self.probabilities[idx * 2 + 1] /= n

    @property
    def dataframe(self):
        if self._dataframe is None:
            self._dataframe = pd.DataFrame(
                data=self.ps_isospec, index=self.atoms_in_table, columns=['p0', 'p1']
            )
            self._dataframe.loc[:, 'ratio'] = self._dataframe.p1 / self._dataframe.p0
            deltas = self._as_deltas()
            self._dataframe.loc[:, 'delta'] = np.nan
            for el, delta in deltas.items():
                self._dataframe.loc[el, 'delta'] = delta
            self._dataframe.loc[:, 'name0'] = self.isotope_names[0::2]
            self._dataframe.loc[:, 'name1'] = self.isotope_names[1::2]
            self._dataframe.loc[:, 'm0'] = self.masses[0::2]
            self._dataframe.loc[:, 'm1'] = self.masses[1::2]
        return self._dataframe

    @property
    def ratios(self) -> list[float]:
        """Calculate probability ratios for elements"""
        n_iso = len(self.probabilities) // 2
        fs = []
        for i in range(n_iso):
            fs.append(self.probabilities[i * 2 + 1] / self.probabilities[i * 2])
        return fs

    @property
    def ps_isospec(self) -> list[list[float]]:
        """Format probabilities to be provided to IsoSpecPy"""
        # iso expects list of lists
        ps: list[list[float]] = []
        for idx, a in enumerate(self.atoms_in_table):
            probs_el = [self.probabilities[idx * 2], self.probabilities[idx * 2 + 1]]
            ps.append(probs_el)
        return ps

    @property
    def ms_isospec(self) -> list[list[float]]:
        """Format masses to be provided to IsoSpecPy"""
        ms: list[list[float]] = []
        for idx, a in enumerate(self.atoms_in_table):
            ms_el = [self.masses[idx * 2], self.masses[idx * 2 + 1]]
            ms.append(ms_el)
        return ms

    def __repr__(self) -> str:
        return self.dataframe.to_string()


class IsotopePattern:
    masses: list[float] = None
    intensities: list[float] = None
    n_peaks: int = None

    def __init__(self, masses, intensities, normalize=False):
        self.masses: list[float] = list(masses)
        if normalize:
            i_max = max(intensities)
            intensities = [i / i_max for i in intensities]
        self.intensities: list[float] = list(intensities)
        assert len(self.masses) == len(self.intensities), \
            'number of entries for masses and intensities do not match'
        self.n_peaks: int = len(self.masses)

    @classmethod
    def from_formula(
            cls,
            formula: str,
            adduct: str = None,
            mass_accuracy: float = None,
            mass_resolution: int = None,
            merge_method: Literal['weighted_average', 'none'] = 'none'
    ) -> Self:
        formula: CompoundDict = CompoundDict(formula)
        if adduct is not None:
            # multiply formula by multiplicity and add adduct elements
            adduct: Adduct = Adduct(adduct)
            formula: CompoundDict = formula * adduct.multiplicity + adduct.composition

        ms1 = IsoTotalProb(formula=formula.formula, prob_to_cover=.9999)
        # shift/scale masses according to adduct
        masses = list(ms1.masses)
        if adduct is not None:
            mzs = [adduct.isopattern_molecule_mass_to_mz(mass) for mass in masses]
        else:
            mzs = masses

        new = cls(mzs, ms1.probs)
        if merge_method == 'none':
            return new
        elif merge_method == 'weighted_average':
            new.bin_close_weighted(mass_tol=mass_accuracy, resolution=mass_resolution)
        else:
            raise ValueError(f'unknown merge method: {merge_method}')
        return new

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
                logger_func = logger.warning if i > .05 else logger.debug
                logger_func(
                    f'unmatched peak: {m=:.4f}, {i=:.2f}, '
                    f'distance: {dists[idx_dist]:.4f}, '
                    f'bins: {np.round(iso_masses_measured, 4)}'
                )

        self.intensities = binned_intensities_predicted
        self.masses = list(iso_masses_measured)
        self.n_peaks = len(self.masses)

    def bin_close(self, mass_tol: float = DEFAULT_MASS_TOLERANCE):
        bins: list[float] = list(
            map(lambda m: round(m / mass_tol) * mass_tol, self.masses)
        )
        self.bin_into_targets(bins, mass_tol=mass_tol)

    def bin_close_weighted(self, *, mass_tol: float = None, resolution: int = None):
        """merge close mz values until the smallest difference is above the mass tolerance"""
        self.masses, self.intensities = recursively_merge_peaks_to_resolution(
            self.masses, self.intensities, mass_tol, resolution
        )
        self.n_peaks = len(self.masses)

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

    def combine_with(
            self,
            other: Self,
            merge_method: Literal['none', 'weighted_average'] = 'none',
            mass_resolution=None
    ) -> Self:
        masses = self.masses + other.masses
        intensities = self.intensities + other.intensities
        new = self.__class__(masses, intensities)
        if merge_method == 'weighted_average':
            new.bin_close_weighted(resolution=mass_resolution)
        return new


class IsoPatternFit:
    """
    Fit the isotope pattern of a molecule with a given formula for possibly
    non-natural isotope abundances.
    """
    formula: str = None
    atoms_fit: list[str] = None  # for which patterns to tweak the isotope abundances in order to improve the fit
    atom_counts: list[int] = None  # corresponding number of atoms according to the formula
    atoms_fixed: OrderedDict = None  # dict with elements and counts that are not manipulated
    iso_table: IsoTable = None
    iso_pattern: IsotopePattern = None  # measured isotope pattern that is to be achieved by adjusting the isotope abundances

    def __init__(
            self,
            formula: str,
            atoms_fit: list[str],
            measured_isotope_pattern: IsotopePattern = None,
            iso_table: IsoTable = None
    ) -> None:
        self.formula = formula

        _options_fit_elements: list[str] = 'H C O S'.split()
        assert all([f in _options_fit_elements for f in atoms_fit]), \
            f'can only fit isotopes for elements {_options_fit_elements}'
        self.atoms_fit: list[str] = atoms_fit

        # decompose formula into part that is used to fit spectrum and remainder
        self.atom_counts, self.atoms_fixed = self._get_atom_counts_fitted_elements()

        if iso_table is None:
            iso_table = IsoTable(
                *[(element, 'natural', None) for element in self.atoms_fit]
            )
        else:
            assert hasattr(iso_table, 'probabilities'), \
                'if isotable is provided, the probabilites must be set'
        self.iso_table: IsoTable = iso_table

        if measured_isotope_pattern is None:
            self.update_predicted_pattern()
            self.iso_pattern.bin_close()
        else:
            self.iso_pattern: IsotopePattern = measured_isotope_pattern

        self._target_masses: np.ndarray[float] = np.array(self.iso_pattern.masses)
        self._target_intensities: np.ndarray[float] = np.array(self.iso_pattern.intensities)

    def _get_atom_counts_fitted_elements(self) -> tuple[list[int], OrderedDict]:
        """Return the atom counts of elements to be fitted and remaining formula"""
        parsed: OrderedDict = ParseFormula(self.formula)
        atom_counts: list[int] = [parsed[el] for el in self.atoms_fit]
        for remove_element in self.atoms_fit:
            parsed.pop(remove_element)
        return atom_counts, parsed

    def update_predicted_pattern(self) -> None:
        # predicted_pattern: IsoDistribution = isospec.IsoThreshold(
        iso_probabilities = [self.iso_table.dataframe.loc[el, ['p0', 'p1']].to_list() for el in self.atoms_fit]
        iso_masses = [self.iso_table.dataframe.loc[el, ['m0', 'm1']].to_list() for el in self.atoms_fit]
        predicted_pattern: IsoDistribution = isospec.IsoTotalProb(
            formula=CompoundDict(self.atoms_fixed).formula,
            prob_to_cover=.999,
            atomCounts=self.atom_counts,
            isotopeProbabilities=iso_probabilities,
            isotopeMasses=iso_masses
        )
        self.iso_pattern = IsotopePattern(predicted_pattern.masses, predicted_pattern.probs)

    def fit_spectrum(
            self,
            mass_tol=DEFAULT_MASS_TOLERANCE,
            scalar_min: bool = False
    ) -> OptimizeResult:
        """iteratively tweak probabilities to match spectrum using IsoSpecPy"""

        def _error(pred_intensities: np.ndarray[float]) -> np.ndarray[float] | float:
            # scale up probabilities for better numerical stability
            diff_vec = (pred_intensities * 1000 - self._target_intensities * 1000) ** 2
            return diff_vec.sum() if scalar_min else diff_vec

        def pred(deltas: np.ndarray[float]) -> np.ndarray[float]:
            # convert fs to ps
            self.iso_table = IsoTable(
                *[(element, 'delta', delta) for element, delta in zip(self.atoms_fit, deltas)]
            )
            self.update_predicted_pattern()
            # bin into provided masses
            self.iso_pattern.bin_into_targets(self._target_masses, mass_tol=mass_tol)
            pred_intensities_binned = np.array(self.iso_pattern.intensities)
            return _error(pred_intensities_binned)

        deltas0 = np.array(self.iso_table.dataframe.delta)
        # print('initial deltas:', self.iso_table.dataframe.delta.to_dict())

        eps = 1e-21  # p of 0 is not allowed
        if scalar_min:
            bounds = [(-500, 500) for _ in deltas0]
            return minimize(pred, x0=deltas0, bounds=bounds)
        else:
            return least_squares(pred, x0=deltas0, bounds=(-500, 500), ftol=3e-16, gtol=3e-16, xtol=3e-16)

    def as_deltas(self) -> dict:
        el_to_f = {
            'C': C13C12to_delta13C,
            'H': H2H1to_delta13C,
            'O': O17O16to_delta17O,
            'S': S34S32to_delta34S
        }

        out = {}

        for i, el in enumerate(self.atoms_fit):
            ps = self.iso_table.ps_isospec[i]
            out[el] = el_to_f[el](*ps)
        return out


def predict_deltas_for_ms1(
        ms1_pattern: PeakList,
        formula: str,
        adduct_type,
        atoms_to_fit: list[str],
        plts=False
) -> dict[str, float]:
    # we need to account for the fact that the MS1 spectrum contains IONS
    #  generally, we assume that peaks in the MS1 spectrum are the [M+adduct]+ masses, so it is easiest to convert the
    #  m/z values to M using the adduct type
    ...
    mzs = ms1_pattern.mzs
    # convert to M using adduct_type
    adduct_mass, adduct_charge, *_ = _get_adduct_composition(adduct_type, format_template='metaboscape')
    Ms = [mz * adduct_charge - adduct_mass for mz in mzs]
    iso_pattern = IsotopePattern(
        masses=Ms,
        intensities=ms1_pattern.intensities
    )
    model = IsoPatternFit(formula=formula, atoms_fit=atoms_to_fit, measured_isotope_pattern=iso_pattern)
    model.fit_spectrum(scalar_min=False)
    if plts:
        ax = model.iso_pattern.plot(linefmt='blue')
        iso_pattern.plot(ax=ax, shift=1, linefmt='orange')
        # plt.legend(['measured', 'fitted'])
        plt.show()

    return model.iso_table.dataframe.delta.to_dict()


def direct_predict_d13C(formula, iso_pattern: IsotopePattern | tuple, d2H: float = -200) -> float:
    """Assume H O N occur with their natural abundances"""
    parsed = ParseFormula(formula)
    nH = parsed.get('H', 0)
    nC = parsed.get('C', 0)
    nO = parsed.get('O', 0)
    nN = parsed.get('N', 0)

    if isinstance(iso_pattern, IsotopePattern):
        I0, I1 = iso_pattern.intensities[:1]
    else:
        assert len(iso_pattern) == 2
        I0, I1 = iso_pattern
    f_H = f_VSMOV_H2 * (d2H / 1000 + 1)
    # note: in equation in manuscript we are not dividing by 1 + f_H but this
    # changes the H contribution only by a factor of .9999 (assuming d2H = -200)
    contribution_H = f_H * nH  # / (1 + f_H)
    contribution_O = isotope_properties.loc['O[17]', 'Isotopic Composition'] * nO
    contribution_N = isotope_properties.loc['N[15]', 'Isotopic Composition'] * nN

    # M1_corr = M1 - contribution_H - contribution_O - contribution_N
    # rC13 = M1_corr / M0 / nC
    rC13 = (I1 / I0 - contribution_H - contribution_O - contribution_N) / nC
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
    iso_table._set_natural_probabilities()
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


def test_multi_delta_fit():
    formula = 'C30H58O5S'

    # generate artificial spectrum
    actual_deltaC13 = -100
    actual_deltaH2 = -200
    actual_deltaS34 = 50

    fit_elements = ['C', 'S', 'H']

    iso_table = IsoTable(
        ('H', 'delta', actual_deltaH2),
        ('C', 'delta', actual_deltaC13),
        ('S', 'delta', actual_deltaS34)
    )

    # natural abundance
    unbiased_pattern = IsoPatternFit(
        formula=formula, atoms_fit=fit_elements)
    # actual (synthetic) pattern
    measured_pattern = IsoPatternFit(
        formula=formula, atoms_fit=fit_elements, iso_table=iso_table)

    fitted_pattern = IsoPatternFit(
        formula=formula,
        atoms_fit=fit_elements,
        measured_isotope_pattern=measured_pattern.iso_pattern)
    deltas_fit = fitted_pattern.fit_spectrum(scalar_min=False)

    print('unbiased:', unbiased_pattern.iso_table.dataframe.delta.to_dict())
    print('measured:', measured_pattern.iso_table.dataframe.delta.to_dict())
    print('fitted:', fitted_pattern.iso_table.dataframe.delta.to_dict())

    ax = unbiased_pattern.iso_pattern.plot()
    measured_pattern.iso_pattern.plot(ax=ax, linefmt='orange', shift=1)
    # iso_fit3.iso_pattern.plot(ax=ax, linefmt='green', shift=2)
    fitted_pattern.iso_pattern.plot(ax=ax, linefmt='red', shift=2)
    ax.legend(['unbiased', 'measured', 'predicted', 'predicted scalar'])
    plt.show()


if __name__ == '__main__':
    pass

    formula = 'C43H88O3'
    adduct = '[M+H2]2+'

    iso_pattern = IsotopePattern.from_formula(formula, adduct, mass_resolution=40_000)
    iso_pattern.plot()
    # ms1_measured = PeakList(
    #     mzs=[690.57093, 691.57305, 692.57612],
    #     intensities=[137821, 64095, 28280]
    # )
    #
    # res = predict_deltas_for_ms1(ms1_measured, formula='C39H79NO6S', adduct_type='[M+H]+', atoms_to_fit='CS',
    #                              plts=True)
    #
    # print(res)
