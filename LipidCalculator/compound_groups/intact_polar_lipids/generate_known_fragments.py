from typing import Literal

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.Descriptors import ExactMolWt
from tqdm import tqdm

from LipidCalculator.compound_creation.formula_parser import CompoundDict
from LipidCalculator.compound_groups.intact_polar_lipids.frag_from_alpha_cleavage import predict_ms2, \
    plot_ms2_prediction, \
    get_neutral_equivalents_for_predicted_ms2, predict_losses
from LipidCalculator.compound_groups.intact_polar_lipids.pieces_from_json import ABBREVIATION_TO_GROUP
from LipidCalculator.compound_groups.intact_polar_lipids.generate_ipl import ipl_automatic_bonds, get_mol_from_abbr, \
    CHAIN_EMPTY_PLACEHOLDER
from LipidCalculator.compound_groups.common_fragments import MASS_PROTON

PIECE_TYPES: list[str] = ['head', 'core', 'chain1', 'chain2']
CHAIN_EMPTY_PLACEHOLDER_TYPE = 'chain_empty'
PieceTypes: type = Literal[*PIECE_TYPES]
NEGATIVE_FORMULA = chr(0x2796)


def remove_protons(mol: Mol):
    """set radicals and formals to 0"""
    for a in mol.GetAtoms():
        a.SetNumRadicalElectrons(0)
        a.SetFormalCharge(0)


def compare_fragment_to_substructures(
        fragment: Mol,
        pieces: dict[str, Mol]
) -> pd.DataFrame:
    """For a given fragments and its building blocks, attempt to match the
    fragment to a building block."""

    def _ct_atoms(mol_: Mol) -> int:
        """Count number of non-H-atoms in molecule"""
        return len(Chem.RemoveHs(mol_).GetAtoms())

    def _check_smart_in_mol(mol_: Mol, pattern: Mol) -> bool:
        """Check whether smart is substructure of the mol (equivalent to 'is pattern in mol?')"""
        return len(mol_.GetSubstructMatches(pattern)) > 0

    def _check_sub_both_ways(a: Mol, b: Mol) -> bool:
        """check whether a is substructure of b or vice versa"""
        b_in_a: bool = _check_smart_in_mol(a, b)
        a_in_b: bool = _check_smart_in_mol(b, a)
        return a_in_b or b_in_a

    def get_atoms_not_in_substruct(substructure: Mol | int) -> float | int:
        """Calculate how many more/less non-H-atoms the fragment has than the given substructure."""
        if (substructure is None) or (not _check_sub_both_ways(substructure, fragment)):
            return np.nan
        return n_atoms_fragment - _ct_atoms(substructure)

    # neutralize fragment for better substructure match

    n_atoms_fragment: int = _ct_atoms(fragment)

    # preliminary check for protonated atom by mass
    mz: float = ExactMolWt(fragment)
    cd_fragment = CompoundDict.from_mol(fragment)

    # calculate "scores" for pieces and combinations of pieces
    combinations = [
        ['head'], ['core'], ['chain1'], ['chain2'],
        ['head', 'core'], ['core', 'chain1', ], ['core', CHAIN_EMPTY_PLACEHOLDER_TYPE, 'chain2'],
        ['head', 'core', 'chain1', CHAIN_EMPTY_PLACEHOLDER_TYPE],
        ['head', 'core', CHAIN_EMPTY_PLACEHOLDER_TYPE, 'chain2'], ['core', 'chain1', 'chain2'],
        ['head', 'core', 'chain1', 'chain2']
    ]
    # filter combinations
    possible_combinations = []
    for comb in combinations:
        if all([piece in pieces for piece in comb]):
            possible_combinations.append(comb)

    # for each combination, check how far the fragment is away in terms of mass and substructure
    substruct_names = []
    substruct_masses = []
    fragment_in_substructs = []
    substruct_in_fragments = []
    mass_diffs = []
    formula_diffs = []
    ann = []
    dn_atoms = []
    for comb in possible_combinations:
        if len(comb) == 1:  # just a single piece, don't need to form bonds
            piece_label: str = pieces[comb[0]]
            informative_substructure: Mol = get_mol_from_abbr(piece_label)
        else:  # multiple pieces
            names: list[str] = []
            # use C0:0 for placeholders to make sure chains are in the right
            # positions (for substructure match)
            for piece_type in comb:
                if piece_type == CHAIN_EMPTY_PLACEHOLDER_TYPE:
                    names.append(CHAIN_EMPTY_PLACEHOLDER)
                else:
                    names.append(pieces[piece_type])
            informative_substructure: Mol = ipl_automatic_bonds(
                names, split_chain=False, sort_chains=False)
            piece_label = '-'.join(names)
        mass: float = ExactMolWt(informative_substructure)
        cd_substruct = CompoundDict.from_mol(informative_substructure)

        # calculate relative properties
        mass_diff: float = mz - mass
        is_fragment_in_substruct: bool = _check_smart_in_mol(informative_substructure, fragment)
        is_substruct_in_fragment: bool = _check_smart_in_mol(fragment, informative_substructure)
        cd_diff = cd_fragment - cd_substruct
        formula_diff = cd_diff.formula
        fragment_label = f'{piece_label} ({formula_diff})'
        n_atoms_diff = get_atoms_not_in_substruct(informative_substructure)

        substruct_names.append(piece_label)
        substruct_masses.append(mass)
        mass_diffs.append(mass_diff)
        fragment_in_substructs.append(is_fragment_in_substruct)
        substruct_in_fragments.append(is_substruct_in_fragment)
        dn_atoms.append(n_atoms_diff)
        formula_diffs.append(formula_diff)
        ann.append(fragment_label)

    df = pd.DataFrame(dict(substruct_name=substruct_names,
                           substruct_mass=substruct_masses,
                           dm=mass_diffs,
                           fragment_in_substruct=fragment_in_substructs,
                           substruct_in_fragment=substruct_in_fragments,
                           dn_atom=dn_atoms,
                           formula_diff=formula_diffs,
                           fragment_name=ann))

    def score_formula_diff(formula):
        if is_neg := (NEGATIVE_FORMULA in formula):
            formula = formula.replace(NEGATIVE_FORMULA, '')

        weights = {
            'H': 1,
            'C': 100,
        }
        cd = CompoundDict(formula)
        if is_neg:
            cd = cd * (-1)

        score = 0
        for atm, ct in cd.items():
            if atm in ('+', '-'):
                continue
            weight = weights.get(atm, 20)
            score += abs(ct * weight)
        return score

    # closest match is the onewith the smallest change in the formula of differences
    df.loc[:, 'score'] = df.formula_diff.apply(score_formula_diff)

    return df


def find_biggest_substructure(fragment: Mol, pieces: dict[PieceTypes, Mol]) -> str:
    df = compare_fragment_to_substructures(fragment, pieces)
    # scores = df.dm.abs()
    return df.fragment_name.iloc[np.argmin(df.score)].replace(NEGATIVE_FORMULA, '-')


ADDUCTS = ['[M+H]+', '[M+H+H]2+', '[M+H+H2]3+', '[M+NH4]+', '[M+K]+', '[M+Na]+']

# does not work for multipliy charged ions
ADDUCT_MINUS_PROTONATED = {
    k: CompoundDict(k.split('[M+')[1].split(']')[0] + c).mass - MASS_PROTON
    for k in ADDUCTS
    if (c := k.split(']')[1]) in ('+', '-')
}


# %% test molecule
def annotate_fragments(
        pieces: dict[PieceTypes, Mol], mol: Mol
) -> dict[float, str | None]:
    assert "core" in pieces

    predicted_ms2: dict[float, Mol] = predict_ms2(Chem.MolToSmiles(mol))
    # neutral_equivalents = get_neutral_equivalents_for_predicted_ms2(Chem.MolToSmiles(mol))
    # attempt to annotate fragments
    annotations: dict[float, str | None] = {}
    M_H = ExactMolWt(mol) + MASS_PROTON
    for mz, frag in predicted_ms2.items():
        # TODO: looks like its not working that well, do some more tests and PG is missing
        annotations[mz] = find_biggest_substructure(
            frag,
            pieces
        )

    return annotations


# %% test for specific compound
def test_for_single_mol():
    pieces = dict(chain1='C10:0', chain2='C22:0', head='PC', core='DAG')
    mol = ipl_automatic_bonds(pieces.values(), split_chain=False, plts=False, idx_plt=False, sort_chains=False)

    smiles = Chem.MolToSmiles(mol)
    predicted = predict_ms2(smiles)
    # neutrals = get_neutral_equivalents_for_predicted_ms2(smiles)
    annotations = annotate_fragments(pieces, mol)
    mz_to_df = {}
    for mz, frag in predicted.items():
        # TODO: pick one with smallest mz dev for now
        df = compare_fragment_to_substructures(frag, pieces)
        mz_to_df[mz] = df
    plot_ms2_prediction(mol, predicted, annotations=annotations)

    # ch3_frag
    mz = min(predicted.keys())
    fragment = frag = predicted[mz]

    label = find_biggest_substructure(fragment, pieces)

    print(Chem.MolToSmiles(fragment))
    remove_protons(fragment)
    print(Chem.MolToSmiles(fragment))


# %% predict all
def add_pred(pieces):
    mol = ipl_automatic_bonds(pieces.values(), split_chain=False, sort_chains=False)
    _name = ' '.join(pieces.values())

    mols[_name] = mol

    smiles = Chem.MolToSmiles(mol)
    pred_ms2[_name] = predict_ms2(smiles)
    pred_losses[_name] = predict_losses(smiles)


chain1s = [f'C{i}:0' for i in range(6, 31, 1)] + [f'C{i}:1' for i in range(6, 31, 1)]
# chain2 = 'C5:2'
chain2s = [f'C{i}:0' for i in range(6, 31, 1)] + [f'C{i}:1' for i in range(6, 31, 1)]
# mol_chain1: Mol = get_mol_from_abbr(chain1)
# mol_chain2: Mol = get_mol_from_abbr(chain2)
# combine heads, cores, chains in all possible combinations to get characteristic fragments
HEADS = [k for k, v in ABBREVIATION_TO_GROUP.items() if v == 'head groups']
CORES = [k for k, v in ABBREVIATION_TO_GROUP.items() if v == 'core lipids']

mols: dict[str, Mol] = {}
pred_ms2: dict[str, dict[float, str | None]] = {}
pred_losses: dict[str, dict[float, str | None]] = {}
for head in tqdm(HEADS, total=len(HEADS)):
    mol_head: Mol = get_mol_from_abbr(head)
    for core in CORES:
        if ('GDGT' in core) or ('AR' in core):  # no chains
            pieces = dict(head=head, core=core)
            add_pred(pieces)
        elif core in ('DGTS', 'BL', 'OL'):
            for chain1 in chain1s:
                pieces = dict(core=core, chain1=chain1)
                add_pred(pieces)
        else:
            for chain1 in chain1s:
                for chain2 in chain2s:
                    pieces = dict(head=head, core=core, chain1=chain1, chain2=chain2)
                    add_pred(pieces)

# %% find fragments, losses that are in all representatives of each head-core combination
characteristic_frags: dict[float, list[str]] = {}
characteristic_losses: dict[float, list[str]] = {}
comps = set(pred_ms2.keys()) | set(pred_losses.keys())

for comp in comps:
    frags = pred_ms2[comp]
    losses = pred_losses[comp]
    for mz in frags.keys():
        if mz not in characteristic_frags:
            characteristic_frags[mz] = [comp]
        else:
            characteristic_frags[mz].append(comp)
    for mz in losses.keys():
        if mz not in characteristic_losses:
            characteristic_losses[mz] = [comp]
        else:
            characteristic_losses[mz].append(comp)

# save to csv for lookup
with open('fragment_db.csv', 'w') as f:
    for frag_mass, comps in characteristic_frags.items():
        f.write('\t'.join([f'{frag_mass:.4f}'] + comps) + '\n')

with open('loss_db.csv', 'w') as f:
    for loss_mass, comps in characteristic_losses.items():
        f.write('\t'.join([f'{loss_mass:.4f}'] + comps) + '\n')

# combine compounds if
# - chains are all the same
# - heads are all the same

# mz_epsilon = 1e-6  # tolerance for comparing mz float values
# for head in HEADS:
#     for core in CORES:
#         group = ' '.join([head, core])
#         comps_in_group = [n for n in comps if n.startswith(group)]
#         if len(comps_in_group) == 0:
#             continue
#         shared_frags = set(pred_ms2[comps_in_group[0]].keys())
#         shared_losses = set(pred_losses[comps_in_group[0]].keys())
#         for comp in comps_in_group:
#             shared_frags &= set(pred_ms2[comp].keys())
#             shared_losses &= set(pred_losses[comp].keys())
#
#         characteristic_frags[group] = shared_frags
#         characteristic_losses[group] = shared_losses
