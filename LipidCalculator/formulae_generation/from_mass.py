"""Module for predicting formulas from mass for given element constraints.
Implemenentation not finished, consider using chemcalc through API.
"""
from typing import Self

import numpy as np

from LipidCalculator.compound_creation.formula_parser import CompoundDict

ELECTRON = 0.000549  # difference in m/z caused by electron (if z=1)
DEFAULT_ELEMENTS: list[str] = ['C', 'H', 'O', 'N', 'S', 'P', 'Na', 'K']


class Constraint:
    def __init__(self):
        self._constraints = []
        self._i = 0

    def add_constrain_abundance(
            self, element: str, limits: tuple[int | None, int | None]
    ) -> Self:
        if limits[0]:  # not None or 0
            self._constraints.append((element, '>=', limits[0]))
        if limits[1] is not None:
            self._constraints.append((element, '<=', limits[1]))
        return self

    def add_constrain_ratio(
            self, element: str, limits: tuple[float | None, float | None]
    ) -> Self:
        assert limits[0] <= 1, 'relative abundance must be between 0 and 1'
        assert limits[0] >= 0, 'relative abundance must be between 0 and 1'

        if limits[0]:  # not None or 0
            self._constraints.append((element, '>=', limits[0]))
        if limits[1] is not None:
            self._constraints.append((element, '<=', limits[1]))
        return self

    def add_constrain_relative_ratio(
            self,
            element_nominator,
            element_denominator,
            limits: tuple[float | None, float | None]
    ) -> Self:
        if limits[0] is not None:
            self._constraints.append(
                (element_nominator, '>=', limits[0], element_denominator)
            )
        if limits[1] is not None:
            self._constraints.append(
                (element_nominator, '<=', limits[1], element_denominator)
            )

        return self

    @property
    def _n_constraints(self) -> int:
        return len(self._constraints)

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self._constraints) - 1:
            raise StopIteration
        else:
            self._i += 1
            return self.constraints[self._i - 1]

    @property
    def constraints(self) -> list[tuple]:
        return self._constraints

    def to_str(self, mass: float) -> str:
        """Use absolute and relative abundance constrains to get possible value range for elements"""
        out = {}
        for c in self.constraints:
            # TODO
            ...


# C:N:P:S between 52:5:1:1 and 108:8:1:1
default_constrains = Constraint()
default_constrains.add_constrain_ratio('C', (0.8, 1.))
default_constrains.add_constrain_ratio('N', (0., .1))
default_constrains.add_constrain_abundance('P', (0, 2))
default_constrains.add_constrain_abundance('S', (0, 2))


def get_unconstrained_candidates(
        mz: float,
        tolerance: float,
        elements: list[str],
        current_formula: CompoundDict | None = None
) -> list[CompoundDict] | None:
    # do a tree search: recursively subtract elements
    if current_formula is None:
        # initialize empty dict
        current_formula = CompoundDict(dict())
    results = []

    # update current mass
    mz_current: float = current_formula.mass

    if abs(mz - mz_current) < tolerance:
        # reached target mass, return
        return [current_formula]
    elif mz_current > mz + tolerance:
        return []
    else:
        elements_new = elements.copy()
        for el in elements:
            res = get_unconstrained_candidates(
                mz,
                tolerance,
                elements_new,
                current_formula.copy() + CompoundDict({el: 1}),
            )
            elements_new.remove(el)
            results.extend(res)

    return results


def preprocess_params(
        mz,
        tolerance: float,
        elements: list[str],
        current_formula=None
):
    # sort elements by weight (heaviest first)
    elements = np.sort([CompoundDict({el: 1}).mass for el in elements])[::-1]
    elements_whitelist: np.ndarray[bool] = np.ones_like(elements, dtype=bool)
    current_formula = np.zeros_like(elements)

    return mz, tolerance, elements, current_formula, elements_whitelist


def get_unconstrained_candidates_fast(
        mz,
        tolerance: float,
        elements: np.ndarray[float],
        current_formula: np.ndarray[float],
        elements_whitelist: np.ndarray[bool]
) -> list[np.ndarray[int]]:
    # do a tree search: recursively add elements
    results = []

    # update current mass
    mz_current: float = (current_formula * elements).sum()

    if abs(mz - mz_current) < tolerance:
        # reached target mass, return
        return [current_formula.copy()]
    elif mz_current > mz + tolerance:
        return []
    else:

        for i, el in enumerate(elements):
            if not elements_whitelist[i]:
                continue
            current_formula[i] += 1
            res = get_unconstrained_candidates_fast(
                mz,
                tolerance,
                elements,
                current_formula,
                elements_whitelist.copy()
            )
            # undo
            current_formula[i] -= 1

            elements_whitelist[i] = False
            results.extend(res)

    return results


def check_constraint(formula: CompoundDict, constraint: tuple) -> bool:
    el1, operation, lim, *el2 = constraint

    if len(el2):
        lim = formula.composition[el2[0]] * lim

    if operation == ">=":
        return el1 >= lim
    elif operation == "<=":
        return el1 <= lim
    else:
        raise NotImplementedError(f'{operation} not implemented!')


def check_constraints(formula: CompoundDict, constraints: Constraint) -> bool:
    for constraint in constraints:
        if not check_constraint(formula, constraint):
            return False
    return True


def restrict_el_counts(mz, elements):
    pass


def get_candidates(
        mz: float,
        ionization: int,
        tolerance: float,
        tolerance_unit: str = 'mDa',
        constraints: Constraint | None = None,
        elements: str | list[str] = 'default',
) -> list[CompoundDict]:
    assert tolerance_unit in ['mDa', 'ppm', 'Da']

    # concert tolerance to Da
    if tolerance_unit == 'ppm':
        tolerance: float = mz * tolerance
    elif tolerance_unit == 'mDa':
        tolerance: float = tolerance * 1e-3
    elif tolerance_unit == 'Da':
        pass
    else:
        raise NotImplementedError('internal error')

    if elements == 'default':
        elements: list[str] = DEFAULT_ELEMENTS

    # correct for missing / additional electrons
    mz += -ionization * ELECTRON

    # first, generate a list of all possible combinations
    # then, apply constrains
    params_fast = preprocess_params(mz=mz, tolerance=tolerance, elements=elements)
    res = get_unconstrained_candidates_fast(*params_fast)
    candidates = [CompoundDict(dict(zip(elements, r))) for r in res]
    # candidates: list[CompoundDict] = get_unconstrained_candidates(
    #     mz, tolerance, elements
    # )

    if constraints is None:
        return candidates

    # apply constrains
    candidates_constrained = []
    for candidate in candidates:
        if check_constraints(candidate, constraints):
            candidates_constrained.append(candidate)

    return candidates_constrained


class PredictFormulas:
    def __init__(self, mass: float, delta_mass, elements: list[str], restrictions: str | None = None) -> None:
        self.mass: float = mass
        self.delta_mass: float = delta_mass
        el_masses = []
        for el in elements:
            el_d = CompoundDict({el: 1})
            el_masses.append(el_d.mass)

        o = np.argsort(el_masses)
        self.element_masses: np.ndarray[float] = np.array(el_masses)[o]
        self.element_names: np.ndarray[str] = np.array(elements)[o]

        if restrictions is None:
            restrictions = 'default_organic'

        if restrictions == 'default_organic':
            # TODO
            ...
            # restrictions: dict[str, tuple[int, int]] = {'P': 3, ''}

    def predict(self):
        ...


if __name__ == '__main__':
    import time
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    # res = get_unconstrained_candidates(30.010565, 1e-3, ['C', 'H', 'O'], )
    # res = get_unconstrained_candidates(36, 1e-3, ['C', 'H', 'O'], )
    # res = get_unconstrained_candidates(30.010565, 1e-3, elements=DEFAULT_ELEMENTS)
    # res = get_candidates(30.010565, tolerance=1e-3, ionization=0, elements=DEFAULT_ELEMENTS)
    # C83H168N0O5
    # res = get_candidates(1245.289175, tolerance=3e-3, ionization=0, elements='C H N O'.split())
    # mz = 180.063390

    mz = 551.749
    tolerance = 6e-3  # mDa
    mzs = [92.04369999999994]
    res_slow = get_unconstrained_candidates(mzs[0], tolerance=tolerance, elements='C H N O P S'.split())
    for cd in res_slow:
        print(cd.formula, cd.mass)
    elements = 'C H N O P S'.split()

    # plt.figure()
    # nres = []
    # for mz in tqdm(mzs):
    #     args = preprocess_params(mz, tolerance=tolerance, elements=elements)
    #     # t0 = time.time()
    #     res = get_unconstrained_candidates_fast(*args)
    #     # t1 = time.time()
    #     # res_fast = [CompoundDict(dict(zip(elements, r))) for r in res]
    #     # ms = [r.mass for r in res_fast]
    #     nres.append(len(res))
    #     # print(f'finding {len(res)} took {(t1 - t0) * 1e3:.0f} ms')
    #     plt.scatter(mz, nres[-1])
    # plt.show()
    # pass
