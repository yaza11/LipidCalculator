"""Test generation of possible formulae for a given mass and tolerance."""
import time

from LipidCalculator.compound_creation.formula_parser import CompoundDict
from from_mass import get_unconstrained_candidates_fast, preprocess_params

elements = 'O N C H'.split()

args = preprocess_params(mz=180.063390, tolerance=3e-3, elements=elements)
# args = preprocess_params(mz=500, tolerance=3e-3, elements=elements)
t0 = time.time()
res = get_unconstrained_candidates_fast(*args)
t1 = time.time()

res_fast = [CompoundDict(dict(zip(elements, r))) for r in res]
ms = [r.mass for r in res_fast]

print(t1 - t0)
