"""This script generates all possible formulas for a wide mass range for wide element ratios and ranges"""
import math
import os
from array import array
from functools import partial
from multiprocessing import Pool

import numpy as np

from LipidCalculator import CompoundDict


def get_vec() -> list[int]:
    return [0] * N_ELEMENTS


def get_mass(vec: list[int]) -> float:
    return sum(n * MASSES[idx] for idx, n in enumerate(vec))


def check_mass(m: float) -> bool:
    return MASS_RANGE[0] <= m <= MASS_RANGE[1]


def get_dbe2(vec: list[int]) -> int:
    """Twice the double bond equivalent (kept as int to avoid floats): 2 + sum n_i * (v_i - 2)."""
    return 2 + sum(n * (VALENCES[idx] - 2) for idx, n in enumerate(vec))


def check_sums(vec: list[int]) -> bool:
    for els, el_range in SUM_RANGES.items():
        s: int = sum(vec[ELEMENT_TO_INDEX[el]] for el in els)
        if s < el_range[0] or s > el_range[1]:
            return False
    return True


def check_dbe(vec: list[int]) -> bool:
    dbe2: int = get_dbe2(vec)
    if not ALLOW_HALF_INTEGER_DBE and dbe2 % 2:
        return False
    return 2 * DBE_RANGE[0] <= dbe2 <= 2 * DBE_RANGE[1]


def check_ranges(vec: list[int]) -> bool:
    for el, el_range in RANGES.items():
        idx: int = ELEMENT_TO_INDEX[el]
        if vec[idx] < el_range[0] or vec[idx] > el_range[1]:
            return False
    for (el_nom, el_denom), ratio in RATIOS.items():
        idx_nom: int = ELEMENT_TO_INDEX[el_nom]
        idx_denom: int = ELEMENT_TO_INDEX[el_denom]
        r = vec[idx_nom] / vec[idx_denom]
        if r < ratio[0] or r > ratio[1]:
            return False
    return check_sums(vec) and check_dbe(vec)


MASS_RANGE = (100, 2000)

RANGES = dict(N=(0, 35), P=(0, 6), S=(0, 6), C=(3, 130))
RATIOS = {
    ('H', 'C'): (0, 6.),
    ('O', 'C'): (0, 4.)
}
# limits for the summed counts of several elements
SUM_RANGES = {
    ('N', 'P', 'S'): (0, 35),
    ('N', 'P'): (0, 35),
    ('S', 'P'): (0, 8),
}
DBE_RANGE = (0, 40)
# half-integer DBEs correspond to radicals, which are excluded for neutral molecules
ALLOW_HALF_INTEGER_DBE = False

ELEMENTS: list[str] = 'C H O N P S'.split()
N_ELEMENTS: int = len(ELEMENTS)
MASSES: list[float] = [CompoundDict(el).mass for el in ELEMENTS]
ELEMENT_TO_INDEX: dict[str, int] = dict(zip(ELEMENTS, range(N_ELEMENTS)))
# valences used for the DBE, set P to 5 to treat it as pentavalent
VALENCES: list[int] = [dict(C=4, H=1, O=2, N=3, P=3, S=2)[el] for el in ELEMENTS]

IDX_C: int = ELEMENT_TO_INDEX['C']
IDX_H: int = ELEMENT_TO_INDEX['H']


def get_upper_bound(el: str, n_c: int, mass_left: float) -> int:
    """Largest count of el that fits the remaining mass, the range and the ratio to C."""
    idx: int = ELEMENT_TO_INDEX[el]
    upper: int = math.floor(mass_left / MASSES[idx])
    if el in RANGES:
        upper = min(upper, RANGES[el][1])
    for (el_nom, el_denom), ratio in RATIOS.items():
        if el_nom == el and el_denom == 'C':
            upper = min(upper, math.floor(ratio[1] * n_c))
    return upper


def get_h_range(vec: list[int], mass: float) -> range:
    """Hydrogen counts compatible with the mass range, H/C ratio and DBE, given all other elements in vec."""
    n_c: int = vec[IDX_C]
    lower: int = max(0, math.ceil((MASS_RANGE[0] - mass) / MASSES[IDX_H]))
    upper: int = get_upper_bound('H', n_c, MASS_RANGE[1] - mass)
    if ('H', 'C') in RATIOS:
        lower = max(lower, math.ceil(RATIOS[('H', 'C')][0] * n_c))
    # every H lowers 2 * DBE by one
    dbe2_without_h: int = get_dbe2(vec) - vec[IDX_H] * (VALENCES[IDX_H] - 2)
    lower = max(lower, dbe2_without_h - 2 * DBE_RANGE[1])
    upper = min(upper, dbe2_without_h - 2 * DBE_RANGE[0])
    if not ALLOW_HALF_INTEGER_DBE:
        # H needs the same parity as 2 * DBE without H for an integer DBE
        lower += (lower - dbe2_without_h) % 2
        return range(lower, upper + 1, 2)
    return range(lower, upper + 1)


def get_candidates_for_c(n_c: int) -> np.ndarray:
    """Generate all candidates with a fixed number of carbons as (n_candidates, N_ELEMENTS) array."""
    # heaviest elements first so the mass constraint prunes as early as possible,
    # H is solved last directly from the remaining constraints
    others: list[str] = sorted(
        (el for el in ELEMENTS if el not in ('C', 'H')),
        key=lambda el: MASSES[ELEMENT_TO_INDEX[el]],
        reverse=True
    )
    # flat buffer of unsigned shorts, much more compact than a list of lists
    candidates: array = array('H')
    vec: list[int] = get_vec()
    vec[IDX_C] = n_c

    def recurse(depth: int, mass: float):
        if depth == len(others):
            if not check_sums(vec):
                return
            for n_h in get_h_range(vec, mass):
                vec[IDX_H] = n_h
                if check_mass(mass + n_h * MASSES[IDX_H]) and check_ranges(vec):
                    candidates.extend(vec)
            vec[IDX_H] = 0
            return
        el: str = others[depth]
        idx: int = ELEMENT_TO_INDEX[el]
        for n in range(get_upper_bound(el, n_c, MASS_RANGE[1] - mass) + 1):
            vec[idx] = n
            recurse(depth + 1, mass + n * MASSES[idx])
        vec[idx] = 0

    recurse(0, n_c * MASSES[IDX_C])
    return np.frombuffer(candidates, dtype=np.uint16).reshape(-1, N_ELEMENTS)


def save_candidates_for_c(n_c: int, path_folder_save: str) -> tuple[int, int]:
    """Generate the candidates for n_c carbons, save them and return (n_c, number of candidates)."""
    candidates: np.ndarray = get_candidates_for_c(n_c)
    np.save(os.path.join(path_folder_save, f'C{n_c}.npy'), candidates)
    return n_c, candidates.shape[0]


def save_all_candidates(path_folder_save: str, n_processes: int | None = None) -> int:
    """
    Generate all candidates in parallel and save them to path_folder_save.

    Each carbon count is written to its own file C{n}.npy of shape (n_candidates, N_ELEMENTS),
    so the full set never has to fit into memory. Columns are ordered like ELEMENTS.
    Returns the total number of candidates.
    """
    # at least one C is needed for the ratios to be defined
    min_c: int = max(1, RANGES['C'][0]) if 'C' in RANGES else 1
    max_c: int = math.floor(MASS_RANGE[1] / MASSES[IDX_C])
    if 'C' in RANGES:
        max_c = min(max_c, RANGES['C'][1])
    # largest carbon counts are the most expensive, so hand them out first
    c_counts: list[int] = list(range(max_c, min_c - 1, -1))

    os.makedirs(path_folder_save, exist_ok=True)

    counts: dict[int, int] = {}
    with Pool(n_processes) as pool:
        worker = partial(save_candidates_for_c, path_folder_save=path_folder_save)
        for n_c, n_candidates in pool.imap_unordered(worker, c_counts):
            counts[n_c] = n_candidates
            print(f'C{n_c}: {n_candidates} candidates ({len(counts)}/{len(c_counts)})')
    return sum(counts.values())


if __name__ == '__main__':
    path_folder_save = r"C:\Users\Yannick Zander\Downloads\formulas"

    n_compounds: int = save_all_candidates(path_folder_save)
    print(f'saved {n_compounds} candidates to {path_folder_save}')
