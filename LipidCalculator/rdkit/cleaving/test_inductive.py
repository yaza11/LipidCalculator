from LipidCalculator.rdkit.cleaving.inductive_cleavage import get_inductively_cleaved
from LipidCalculator.compound_groups.intact_polar_lipids.generate_ipl import ipl_automatic_bonds
from LipidCalculator.rdkit.plotting import plt_indices_bond

mol = ipl_automatic_bonds('DAG C32:0'.split(), plts=False, idx_plt=False, split_chain=True)

plt_indices_bond([mol])

get_inductively_cleaved(mol, other_atom_idx=1, hetero_atom_idx=0, plts=True)
