import logging

import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import csc_matrix

from LipidCalculator import Adduct, CompoundDict
from LipidCalculator.formulae_generation.constants_for_checks import EL_TO_DBE, VALENCE_ELECTRONS, SENIOR_VALENCES

# TODO: encode DB, LEWIS and SENIOR check
ELEMENTS: list[str] = ["C", "H", "N", "O", "P", "S"]
ATOMIC_MASSES: list[float] = [CompoundDict(el).mass for el in ELEMENTS]

logger = logging.getLogger(__name__)


def get_biggest_possible_bounds(
        M_obs: float,
        element_masses: list[float]
) -> tuple[list[int], list[int]]:
    return [0] * len(element_masses), [int(np.ceil(M_obs / el_m)) for el_m in element_masses]


def get_ks_solution(
        M_obs: float,
        element_masses: list[float],
        element_bounds_lower: list[int],
        element_bounds_upper: list[int],
        mass_tolerance_ppm=None,
        mass_tolerance_Da=None,
        min_ratios: dict[tuple[int, int], float] = None,
        max_ratios: dict[tuple[int, int], float] = None,
        other_solutions: list[list[int]] = None,
        add_multi_el_constraint_for: list[int] = None,
        elements: list[str] = None,
        dbe_range: tuple[float, float] = None,
        check_lewis: bool = True,
        check_senior: bool = True,
):
    if other_solutions is None:
        other_solutions = []
    if min_ratios is None:
        min_ratios = {}
    if max_ratios is None:
        max_ratios = {}
    if mass_tolerance_Da is None and mass_tolerance_ppm is None:
        mass_tolerance = float('inf')
    elif mass_tolerance_Da is not None and mass_tolerance_ppm is not None:
        # take stricter one
        mass_tolerance = min(mass_tolerance_Da, mass_tolerance_ppm * 1e-6 * M_obs)
    elif mass_tolerance_Da is not None:
        mass_tolerance = mass_tolerance_Da
    elif mass_tolerance_ppm is not None:
        mass_tolerance = mass_tolerance_ppm * 1e-6 * M_obs
    else:
        raise RuntimeError('internal error')

    k: int = len(other_solutions)

    n_elements: int = len(element_masses)

    if k > 0:
        if any([len(sol) != n_elements for sol in other_solutions]):
            raise ValueError('previous solutions must have the same number of elements')

    n_binary_variables_other_solutions = 2 * n_elements * k
    n_multi_el = (
        len(add_multi_el_constraint_for)
        if add_multi_el_constraint_for is not None
        else 0
    )
    n_variables: int = 1 + n_elements + n_binary_variables_other_solutions + check_lewis + check_senior + n_multi_el

    rows_in_A: list[list[float | int]] = []
    lower_bounds: list[float | int] = []
    upper_bounds: list[float | int] = []

    # objective: minimize distance between formula mass and M_obs
    # abs can not be encoded as linear constraint, so need auxiliary variable t
    row = [0] * n_variables
    row[0] = -1
    for idx, el_mass in enumerate(element_masses):
        row[1 + idx] = el_mass / mass_tolerance
    rows_in_A.append(row)
    lower_bounds.append(float('-inf'))
    upper_bounds.append(M_obs / mass_tolerance)

    row = [0] * n_variables
    row[0] = -1
    for idx, el_mass in enumerate(element_masses):
        row[1 + idx] = -el_mass / mass_tolerance
    rows_in_A.append(row)
    lower_bounds.append(float('-inf'))
    upper_bounds.append(-M_obs / mass_tolerance)

    if dbe_range is not None:
        assert elements is not None
        row = [0] * n_variables
        for el in elements:
            idx = elements.index(el)
            row[1 + idx] = EL_TO_DBE[el]
        rows_in_A.append(row)
        # DBE = 1 + sum(dbe_i * x_i)
        lower_bounds.append(dbe_range[0] - 1)
        upper_bounds.append(dbe_range[1] - 1)

    if check_lewis:
        assert elements is not None
        # sum ai xi - 2k = 0
        row = [0] * n_variables
        for el in elements:
            idx = elements.index(el)
            row[1 + idx] = VALENCE_ELECTRONS[el]
        row[1 + n_elements + n_binary_variables_other_solutions] = -2
        rows_in_A.append(row)
        lower_bounds.append(0)
        upper_bounds.append(0)

    if check_senior:
        # first condition: senior valence must be mod 2
        assert elements is not None
        # sum ai xi - 2k = 0
        row = [0] * n_variables
        for el in elements:
            idx = elements.index(el)
            row[1 + idx] = SENIOR_VALENCES[el]
        row[1 + n_elements + n_binary_variables_other_solutions + check_lewis] = -2
        rows_in_A.append(row)
        lower_bounds.append(0)
        upper_bounds.append(0)
        # second condition: highest valence atom must find enough partners
        v_max = max([SENIOR_VALENCES[el] for el in elements])
        row = [0] * n_variables
        for el in elements:
            idx = elements.index(el)
            row[1 + idx] = SENIOR_VALENCES[el]
        rows_in_A.append(row)
        lower_bounds.append(2 * v_max)
        upper_bounds.append(float("inf"))
        # third condition: connecting n atoms requires at least n - 1 bonds
        # V >= 2 * (n - 1) <==> sum (v_i - 2) x_i >= -2
        row = [0] * n_variables
        for idx, el in enumerate(elements):
            row[1 + idx] = SENIOR_VALENCES[el] - 2
        rows_in_A.append(row)
        lower_bounds.append(-2)
        upper_bounds.append(float('inf'))

    if add_multi_el_constraint_for is not None:
        # variables are after lewis and senior check auxilaries
        z_start = (
                1 + n_elements
                + n_binary_variables_other_solutions
                + check_lewis + check_senior
        )

        # Link each selected element to its binary variable:
        # x_i <= U_i * z_i
        for j, idx in enumerate(add_multi_el_constraint_for):
            row = [0] * n_variables
            row[1 + idx] = 1
            row[z_start + j] = -element_bounds_upper[idx]

            rows_in_A.append(row)
            lower_bounds.append(float("-inf"))
            upper_bounds.append(0)

        # At most one selected element may be nonzero:
        # sum(z_i) <= 1
        row = [0] * n_variables
        for j in range(n_multi_el):
            row[z_start + j] = 1

        rows_in_A.append(row)
        lower_bounds.append(float("-inf"))
        upper_bounds.append(1)

    # implement ratio constraints
    for (idx_el_nom, idx_el_denom), min_ratio in min_ratios.items():
        # min el_nom/el_denom = r ==> 0 <= num_el_nom - num_el_denom * r
        row = [0] * n_variables
        row[1 + idx_el_nom] = 1
        row[1 + idx_el_denom] = -min_ratio
        rows_in_A.append(row)
        lower_bounds.append(0)
        upper_bounds.append(float('inf'))
    for (idx_el_nom, idx_el_denom), max_ratio in max_ratios.items():
        # min el_nom/el_denom = r ==> 0 <= num_el_nom - num_el_denom * r
        row = [0] * n_variables
        row[1 + idx_el_nom] = 1
        row[1 + idx_el_denom] = -max_ratio
        rows_in_A.append(row)
        lower_bounds.append(float('-inf'))
        upper_bounds.append(0)

    # exclude existing solutions
    for idx_sol, sol in enumerate(other_solutions):
        # idx_sol determines which variables we are setting
        for el_idx, n_el_sol in enumerate(sol):
            # z+
            Li = element_bounds_lower[el_idx]
            coeffs_row = [0] * n_variables
            coeffs_row[1 + el_idx] = 1
            coeffs_row[1 + n_elements + 2 * n_elements * idx_sol + el_idx * 2] = -(n_el_sol + 1 - Li)

            rows_in_A.append(coeffs_row)
            lower_bounds.append(Li)
            upper_bounds.append(float('inf'))

            # z-
            Ui = element_bounds_upper[el_idx]
            coeffs_row = [0] * n_variables
            coeffs_row[1 + el_idx] = 1
            coeffs_row[1 + n_elements + 2 * n_elements * idx_sol + el_idx * 2 + 1] = Ui - n_el_sol + 1
            rows_in_A.append(coeffs_row)
            lower_bounds.append(float('-inf'))
            upper_bounds.append(Ui)
        # require that at least one of the zs for this solution is non-zero
        row = [0] * n_variables
        idcs = slice((1 + n_elements + 2 * n_elements * idx_sol), (1 + n_elements + 2 * n_elements * (idx_sol + 1)))
        row[idcs] = [1] * 2 * n_elements
        rows_in_A.append(row)
        lower_bounds.append(1)
        upper_bounds.append(float('inf'))

    lb = [0.] + element_bounds_lower + [0] * 2 * n_elements * k
    ub = [1] + element_bounds_upper + [1] * 2 * n_elements * k
    integrality = [1] * n_variables
    integrality[0] = 0  # t is continuous

    if check_lewis:  # auxillary variable added for mod calculation
        lb.append(sum([VALENCE_ELECTRONS[el] * element_bounds_lower[idx] for idx, el in enumerate(elements)]))
        ub.append(sum([VALENCE_ELECTRONS[el] * element_bounds_upper[idx] for idx, el in enumerate(elements)]))
    if check_senior:
        lb.append(sum([SENIOR_VALENCES[el] * element_bounds_lower[idx] for idx, el in enumerate(elements)]))
        ub.append(sum([SENIOR_VALENCES[el] * element_bounds_upper[idx] for idx, el in enumerate(elements)]))
    if add_multi_el_constraint_for is not None:
        lb.extend([0] * n_multi_el)
        ub.extend([1] * n_multi_el)

    bounds = Bounds(lb=lb, ub=ub)

    c = [0] * n_variables
    # objective is to minimize t
    c[0] = 1

    constraints = LinearConstraint(
        csc_matrix(rows_in_A),
        np.asarray(lower_bounds, dtype=float),
        np.asarray(upper_bounds, dtype=float),

    )

    res = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options=dict(disp=False)
    )
    if res.status == 4:
        # HiGHS presolve occasionally fails on this badly scaled problem (depends on the
        # exact tolerance value), solving without presolve is more robust
        logger.debug(f'milp failed ({res.message}), retrying without presolve')
        res = milp(
            c=c,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options=dict(disp=False, presolve=False)
        )
    return res


def milp_result_to_cd(coeffs: np.ndarray[float], element_names: list[str]):
    return CompoundDict({el: round(coeff) for el, coeff in zip(element_names, coeffs)})


def encode_ratio_constraints(
        min_ratios: dict[tuple[str, str], float],
        max_ratios: dict[tuple[str, str], float],
        elements: list[str]
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    min_ratios_encode = {}
    for (el_nom, el_denom), min_ratio in min_ratios.items():
        idx_el_nom = elements.index(el_nom)
        idx_el_denom = elements.index(el_denom)
        min_ratios_encode[(idx_el_nom, idx_el_denom)] = min_ratio

    max_ratios_encoded = {}
    for (el_nom, el_denom), max_ratio in max_ratios.items():
        idx_el_nom = elements.index(el_nom)
        idx_el_denom = elements.index(el_denom)
        max_ratios_encoded[(idx_el_nom, idx_el_denom)] = max_ratio
    return min_ratios_encode, max_ratios_encoded


def get_min_max_ratios_extended() -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    # lower limits of 0 do not restrict the solution at all since this is the same as 0 <= num_el_nom
    min_ratios = {
        ('H', 'C'): .429
    }
    max_ratios = {
        ('H', 'C'): 3.5,
        ('O', 'C'): 4.0,
    }
    return min_ratios, max_ratios


def get_common_element_bounds(
        M_obs: float,
        elements: list[str]
) -> tuple[list[int], list[int]]:
    lower, upper = get_biggest_possible_bounds(M_obs, [CompoundDict(el).mass for el in elements])
    ranges = {
        'N': (0, 7),
        'P': (0, 4),
        'S': (0, 2),
    }
    for el, (l, u) in ranges.items():
        if el not in elements:
            continue
        # attempt to restrict the ranges
        el_idx = elements.index(el)
        if lower[el_idx] < l:
            lower[el_idx] = l
        if upper[el_idx] > u:
            upper[el_idx] = u
    return lower, upper


def find_k_top_solutions(
        M_obs: float,
        element_masses: list[float],
        element_bounds_lower: list[int],
        element_bounds_upper: list[int],
        mass_tolerance_ppm: float = None,
        mass_tolerance_Da: float = None,
        min_ratios: dict[tuple[int, int], float] = None,
        max_ratios: dict[tuple[int, int], float] = None,
        add_multi_el_constraint_for: list[int] = None,
        elements: list[str] = None,
        dbe_range: tuple[float, float] = None,
        k: int = 3,
        **kwargs
):
    sols: list[list[int]] = []
    reses = []
    for sol_idx in range(k):
        res = get_ks_solution(
            M_obs,
            element_masses,
            element_bounds_lower,
            element_bounds_upper,
            mass_tolerance_ppm=mass_tolerance_ppm,
            mass_tolerance_Da=mass_tolerance_Da,
            min_ratios=min_ratios,
            max_ratios=max_ratios,
            add_multi_el_constraint_for=add_multi_el_constraint_for,
            elements=elements,
            dbe_range=dbe_range,
            other_solutions=sols,
            **kwargs
        )

        if (res.status == 2) or (not res.success):
            logger.info(f"Could only find {len(sols)} formulas: {res.message}")
            return sols, reses
        elif res.status != 0:
            logger.warning(f"There is a problem with milp after {len(sols)} solutions: {res.message}")
            return sols, reses
        sols.append([round(v) for v in res.x[1:len(element_masses) + 1]])
        reses.append(res)
    return sols, reses


def get_formulas_default_constraints(
        M_obs: float,
        mass_tolerance_ppm: float = None,
        mass_tolerance_Da: float = None,
        elements: list[str] = None,
        n_formulas_max=10,
        dbe_range: tuple[float, float] = (0, 36),
        exclusive_elements: str | list[str] = 'NPS',
        **kwargs
):
    """exclusive_elements: at most one of those elements may be present in a formula"""
    if elements is None:
        elements = ELEMENTS.copy()
    atomic_masses = [CompoundDict(el).mass for el in elements]

    lower, upper = get_common_element_bounds(M_obs, elements)
    min_ratios, max_ratios = get_min_max_ratios_extended()
    # only keep ratios for which both elements are in the element list
    min_ratios = {k: v for k, v in min_ratios.items() if set(k) <= set(elements)}
    max_ratios = {k: v for k, v in max_ratios.items() if set(k) <= set(elements)}
    min_ratios, max_ratios = encode_ratio_constraints(min_ratios, max_ratios, elements)

    sols, res = find_k_top_solutions(
        M_obs,
        atomic_masses,
        lower,
        upper,
        min_ratios=min_ratios,
        max_ratios=max_ratios,
        mass_tolerance_Da=mass_tolerance_Da,
        mass_tolerance_ppm=mass_tolerance_ppm,
        k=n_formulas_max,
        add_multi_el_constraint_for=[elements.index(el) for el in exclusive_elements if el in elements] or None,
        elements=elements,
        dbe_range=dbe_range,
        **kwargs
    )
    compounds = [milp_result_to_cd(sol, elements) for sol in sols]
    errs_ppm = [(cd.mass - M_obs) / cd.mass * 1e6 for cd in compounds]

    return pd.DataFrame(
        index=range(len(compounds)),
        data=dict(
            formula=[c.formula for c in compounds],
            DBE=[c.DBE for c in compounds],
            Lewis=[c.is_lewis for c in compounds],
            Senior=[c.is_senior for c in compounds],
            err_ppm=np.around(errs_ppm, 2)
        )
    )


if __name__ == '__main__':
    err_ppm = .05

    M_obs = CompoundDict('C86H172O6').mass
    mz = Adduct('[M+H]+').mass_to_mz(M_obs)
    # M_obs = CompoundDict('C6H12O6').mass
    # M_obs += err_ppm * M_obs / 1e6
    # M_obs = Adduct('[M+H]+').mz_to_mass(551.25)

    df = get_formulas_default_constraints(
        M_obs,
        n_formulas_max=20,
        mass_tolerance_ppm=10,
        check_senior=False,
        check_lewis=False
    )

    cd = CompoundDict('C85H166N7O')
