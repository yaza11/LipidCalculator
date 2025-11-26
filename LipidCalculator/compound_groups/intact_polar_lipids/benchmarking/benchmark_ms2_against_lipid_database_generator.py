"""This module benchmarks MS2 spectra generated with LipidDatabase_Generator against ipl_automatic_bonds"""
import numpy as np
from matplotlib import pyplot as plt
from msIO import MSPReader, PeakList
import os

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from LipidCalculator.compound_groups.intact_polar_lipids.benchmarking.benchmark_util import pieces_from_ldg_name, \
    get_fragment_from_lipiddatabase_generator
from LipidCalculator.cleaving.generate_fragments import FragmentTree
from LipidCalculator.compound_groups.intact_polar_lipids.benchmarking.paths import path_folder_fragments
from LipidCalculator.compound_groups.intact_polar_lipids.generate_ipl import ipl_automatic_bonds


def plot_predicted_against_measured(name, adduct_type, ms_spec: PeakList, **kwargs):
    mol = ipl_automatic_bonds(name.split())
    tree = FragmentTree(mol, adduct_type=adduct_type, max_recursion_depth=kwargs.get('max_recursion_depth', 1))
    # initiate fragmentation
    tree.get_all_fragments()

    fig = tree.plot_ms2(add_struct_plots=False)
    axs = fig.get_axes()

    ints = [-i / 1000 for i in ms_spec.intensities]
    axs[1].stem(ms_spec.mzs, ints, linefmt='g', markerfmt='')
    # add annotations
    for mz, i, txt in zip(ms_spec.mzs, ints, ms_spec.annotations):
        axs[1].text(mz, .8 * i, txt, rotation=45)

    fig.suptitle(name)
    plt.show()


# %%
# plt.close('all')
name = 'AR'

mol = ipl_automatic_bonds(name.split(), plts=False, idx_plt=False, split_chain=True)
tree = FragmentTree(mol, adduct_type='[M+NH4]+', max_recursion_depth=1)
# initiate fragmentation
tree.get_all_fragments()

fig = tree.plot_ms2(add_struct_plots=False)
axs = fig.get_axes()

try:
    frag_julius: PeakList = get_fragment_from_lipiddatabase_generator(name)

    ints = [-i / 1000 for i in frag_julius.intensities]
    axs[1].stem(frag_julius.mzs, ints, linefmt='g', markerfmt='')
    # add annotations
    for mz, i, txt in zip(frag_julius.mzs, ints, frag_julius.annotations):
        axs[1].text(mz, .8 * i, txt, rotation=45)
except Exception as f:
    print(f)

fig.suptitle(name)
plt.show()
