"""This module benchmarks MS2 spectra generated with LipidDatabase_Generator against ipl_automatic_bonds"""
from matplotlib import pyplot as plt
from msIO import MSPReader
import os

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from LipidCalculator.compound_groups.intact_polar_lipids.benchmark_util import pieces_from_ldg_name
from LipidCalculator.compound_groups.intact_polar_lipids.frag_from_alpha_cleavage import predict_ms2, \
    plot_ms2_prediction, predict_losses
from LipidCalculator.compound_groups.intact_polar_lipids.generate_ipl import ipl_automatic_bonds

path_folder = r'\\hlabstorage.dmz.marum.de\scratch\Yannick\compounds'
file = r'1G-DAG_pos.msp'

rdr = MSPReader(os.path.join(path_folder, file), splitter_peaks_list=' ')

# compare predicted MS2 spectra for a few examples
idx = rdr.df_features.index[rdr.df_features.ms_level == 2][0]
ms2 = rdr.peak_lists[idx]

name = rdr.df_features.at[idx, 'name']
mol = ipl_automatic_bonds(pieces_from_ldg_name(name), plts=False)
print(ExactMolWt(mol))

smiles = Chem.MolToSmiles(mol)

frags = predict_ms2(smiles, max_recursion_depth=0)
fig, axs = plot_ms2_prediction(mol, fragments=frags)

axs[1].stem(ms2.mzs, [i / 1000 for i in ms2.intensities], linefmt='g', markerfmt='')

# let's also plot the neutral losses here in case we get wrong which fragment gets the electron
losses = predict_losses(smiles, max_recursion_depth=0)
axs[1].stem(list(losses.keys()), [1] * len(losses), linefmt='k--', markerfmt='')

plt.show()
