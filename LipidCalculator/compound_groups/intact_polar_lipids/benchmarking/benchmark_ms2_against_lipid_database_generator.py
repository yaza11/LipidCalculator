"""This module benchmarks MS2 spectra generated with LipidDatabase_Generator against ipl_automatic_bonds"""
import numpy as np
from matplotlib import pyplot as plt
from msIO import MSPReader
import os

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from LipidCalculator.compound_groups.intact_polar_lipids.benchmarking.benchmark_util import pieces_from_ldg_name
from LipidCalculator.compound_groups.intact_polar_lipids.frag_from_alpha_cleavage import predict_ms2, \
    plot_ms2_prediction, predict_losses
from LipidCalculator.compound_groups.intact_polar_lipids.generate_ipl import ipl_automatic_bonds

path_folder = r'\\hlabstorage.dmz.marum.de\scratch\Yannick\compounds\julius\fragments'
# file = r'1G-DAG_pos.msp'
files = os.listdir(path_folder)

# %%
plt.close('all')
file = 'PC-DAG_pos.msp'
# file = np.random.choice(files)

rdr = MSPReader(os.path.join(path_folder, file), splitter_peaks_list=' ')

# compare predicted MS2 spectra for a few examples
idx = np.random.choice(rdr.df_features.index[rdr.df_features.ms_level == 2])
ms2 = rdr.peak_list[idx]

name = rdr.df_features.at[idx, 'name']
mol = ipl_automatic_bonds(pieces_from_ldg_name(name, use_second=True), plts=False)
print(ExactMolWt(mol))

smiles = Chem.MolToSmiles(mol)

frags = predict_ms2(smiles=smiles, max_recursion_depth=1)
fig, axs = plot_ms2_prediction(mol, fragments=frags)

ints = [-i / 1000 for i in ms2.intensities]
axs[1].stem(ms2.mzs, ints, linefmt='g', markerfmt='')
# add annotations
for mz, i, txt in zip(ms2.mzs, ints, ms2.annotations):
    axs[1].text(mz, .8 * i, txt, rotation=45)

fig.suptitle(name)
plt.show()
