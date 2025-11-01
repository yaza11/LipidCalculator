"""This module benchmarks formulas generated with LipidDatabase_Generator against ipl_automatic_bonds"""
import os
import pandas as pd
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm

from LipidCalculator import CompoundDict
from LipidCalculator.compound_groups.intact_polar_lipids.benchmark_util import pieces_from_ldg_name
from LipidCalculator.compound_groups.intact_polar_lipids.generate_ipl import ipl_automatic_bonds

path_folder = r'\\hlabstorage.dmz.marum.de\scratch\Yannick\compounds'

tables = [f for f in os.listdir(path_folder) if f.endswith('.csv')]
# filter out tables we know won't work
tables = [t for t in tables if 'cmpd' not in t]

results = []

for t in tqdm(tables):
    df = pd.read_csv(os.path.join(path_folder, t), sep='\t')
    for idx, row in df.iterrows():
        try:
            pieces = pieces_from_ldg_name(row.Name)
            mol = ipl_automatic_bonds(pieces)
            f = rdMolDescriptors.CalcMolFormula(mol)
            is_same = CompoundDict(f) == CompoundDict(row.Formula)
            results.append(
                dict(f_ldg=row.Formula, f_lc=f, name_in=row.Name, name_parsed=' '.join(pieces), is_same=is_same))
        except Exception as e:
            results.append(dict(f_ldg=row.Formula, name_in=row.Name, err=e))

results = pd.DataFrame.from_records(results)
print(
    f'out of the {results.err.isna().sum()} compounds build without errors'
    f' {results.loc[results.err.isna(), 'is_same'].mean():.2%} match'
)

# TODO: errors for BL-DEG and similar since both BL and DEG are considered core pieces --> need to define how they are connected
# TODO: errors for PEth --> unknown piece
# TODO: DMK, MK, MMK, MP, MTK, UQ
r = results.loc[~results.err.isna(), :]
