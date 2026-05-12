import numpy as np
import itertools

from matplotlib import pyplot as plt
from tqdm import tqdm

from LipidCalculator.isotopes.iso_frac_predict import IsoTable, IsoPatternFit

n_c = 51
n_s = 21
n_h = 5

delta_h_values = [-200]
delta_c_values = np.linspace(-100, 100, n_c)
delta_s_values = np.linspace(-50, 50, n_s)

formulas = ['C33H40O2S2', 'C53H80O2S2', 'C39H79NO6S']
fit_elements = 'CSH'

# result tables
result_images = {}
for f in formulas:
    for el in fit_elements:
        result_images[(el, f)] = np.full(shape=(n_c, n_s), fill_value=np.nan, dtype=float)

for formula, i, j in tqdm(
        itertools.product(formulas, range(n_c), range(n_s)),
        total=n_c * n_s * len(formulas),
):
    try:
        deltas_true = delta_c_values[i], delta_s_values[j], delta_h_values[0]
        true_pattern = dict(zip(fit_elements, deltas_true))
        iso_table = IsoTable(
            *[(el, 'delta', true_pattern[el]) for el in fit_elements],
        )
        pattern = IsoPatternFit(formula=formula, atoms_fit=fit_elements, iso_table=iso_table)
        model = IsoPatternFit(formula=formula, atoms_fit=fit_elements, measured_isotope_pattern=pattern.iso_pattern)
        fit_result = model.fit_spectrum(scalar_min=False)
        deltas_fit = model.iso_table.dataframe.delta.to_dict()
        # print(true_pattern)
        # print(deltas_fit)

        for el in fit_elements:
            result_images[(el, formula)][i, j] = true_pattern[el] - deltas_fit[el]
    except Exception as err:
        print(f'encountered exception for {formula=} and {deltas_true=}: {err}')

fig, axs = plt.subplots(nrows=len(fit_elements), ncols=len(formulas), figsize=(15, 15), sharex=True, sharey=True,
                        constrained_layout=True)
for i, el in enumerate(fit_elements):
    for j, f in enumerate(formulas):
        axs[i, j].imshow(result_images[(el, f)], vmin=-.001, vmax=.001)
