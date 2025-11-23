"""
building look-up table for target molecules from Rossel paper
https://www.sciencedirect.com/science/article/pii/S0146638008000715?via%3Dihub#app1

Only using compounds with >= 10 %
Only using 2G-GDGTs before the first >> in ring distribution (so 3 and 2)

"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol, rdMolDescriptors
from rdkit.Chem.Descriptors import ExactMolWt

from LipidCalculator.compound_groups.intact_polar_lipids.generate_ipl import ipl_automatic_bonds, \
    remove_placeholder_atoms, get_struct
from LipidCalculator.rdkit.util import mol_from_str

ANME1 = 'ANME-1'
ANME2 = 'ANME-2'
ANME3 = 'ANME-3'
BNME1 = 'bacterial ANME-1'
BNME2 = 'bacterial ANME-2'
BNME3 = 'bacterial ANME-3'

ANME12 = ANME1 + ' & ' + ANME2
BNME12 = BNME1 + ' & ' + BNME2

# use only compounds with >= 10 % as cut-off
organism_to_targets: dict[str, list[str]] = {
    ANME1: ['2G GDGT3', '2G GDGT2'],
    BNME1: ['PE DEG C30:0', 'PE DAG C33:2', 'PE AEG C33:2', 'PE DEG C31:1', 'PE DAG C35:2', 'PE AEG C35:2',
            'PE DAG C32:2', 'PE AEG C32:2'],
    ANME12: ['PG AR', '2G GDGT3', '2G GDGT2', 'P AR'],
    BNME12: ['PE DAG C31:2', 'PE AEG C31:2', 'PE DEG C32:1', 'PE AEG C31:1', 'PE DAG C31:1'],
    ANME2: ['PG OH-AR', 'PE OH-AR', 'PI OH-AR', 'PS OH-AR', 'PG AR', '2G AR'],
    BNME2: ['PE DAG C32:2', 'PE AEG C32:2', 'PG DAG C34:2', 'PG AEG C34:2'],
    ANME3: ['PG OH-AR', 'PS OH-AR'],
    BNME3: ['PDME DAG C32:2', 'PDME AEG C32:2', 'PE DAG C32:2', 'PE AEG C32:2', 'PDME DAG C34:2', 'PDME AEG C34:2']
}

targets_to_organisms = {}
for org, targets in organism_to_targets.items():
    orgs = set(org.split(' & '))
    for target in targets:
        if target in targets_to_organisms:
            targets_to_organisms[target] |= orgs
        else:
            targets_to_organisms[target] = orgs

targets_to_mols: dict[str, Mol] = {}
targets_to_heads: dict[str, Mol] = {}
targets_to_cores: dict[str, Mol] = {}
for org, targets in organism_to_targets.items():
    for target in targets:
        names = target.split()
        mol = ipl_automatic_bonds(names, plts=False)
        targets_to_mols[target] = mol

        head = mol_from_str(get_struct(names[0]))
        head = remove_placeholder_atoms(head)
        targets_to_heads[target] = head

        core = ipl_automatic_bonds(names[1:], plts=False)
        targets_to_cores[target] = core

targets_to_masses: dict[str, float] = {t: ExactMolWt(mol) for t, mol in targets_to_mols.items()}

e = 5.485799090441e-4
adduct_masses: dict[str, float] = {
    'M+': -e,
    '[M + H]+': 1.007825 - e,
    '[M + NH4]+': 18.034374 - e,
    '[M + K]+': 38.963708 - e,
    '[M + Na]+': 22.989770 - e,
}

# build table with columns
# target name | target SMILES | target formula | target mass | target mass (ion) | ion type | organisms

# TODO: workaround for now, head and core are not being split correctly (O should be part of headgroup)
mO = 15.994915

entries: list[dict] = []
for target, mol in targets_to_mols.items():
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mass = ExactMolWt(mol)
    head_mass = ExactMolWt(targets_to_heads[target]) + mO
    core_mass = ExactMolWt(targets_to_cores[target]) - mO - 1.007825
    for ion_type, ion_mass in adduct_masses.items():
        # add head and core masses for MS2 identification through characteristic head group loss
        entry = dict(
            name=target,
            smiles=Chem.MolToSmiles(mol),
            formula=formula,
            M=round(mass, 4),
            mz=round(mass + ion_mass, 4),
            ion_type=ion_type,
            head_M=head_mass,
            core_M=core_mass,
            organisms=targets_to_organisms[target]  # or fill based on your metadata
        )
        entries.append(entry)

df = pd.DataFrame(entries)
df.to_pickle(
    r'C:\Users\Yannick Zander\Nextcloud2\Promotion\T-GUAYMAS\from_lipid_calculator\rossel_target_compounds.pickle'
)

# modify cfm-id input
with open(r"C:\Users\Yannick Zander\cfmid\in.txt", 'w') as f:
    for target, mol in targets_to_mols.items():
        f.write(f'{target.replace(' ', '-')} {Chem.MolToSmiles(mol)}\n')

if __name__ == '__main__':
    pass
