import warnings
from typing import Iterable, Literal

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol, EXPLICIT
from rdkit.Chem.Descriptors import ExactMolWt

from LipidCalculator.cleaving.generate_fragments import ChargeType, PERIODIC_TABLE, Fragment
from LipidCalculator.cleaving.inductive_cleavage import get_inductively_cleaved
from LipidCalculator.compound_groups.common_fragments import MASS_PROTON
from LipidCalculator.rdkit.plotting import mplt_mol

CLEAVAGE_POS: list[str] = ['CO', 'CN', 'CP', 'CS']
CleavagePos: type = Literal[*CLEAVAGE_POS]


def _convert_to_modes_dict(fragments: Iterable[Mol]) -> dict[ChargeType, list[Mol]]:
    """Split an iterable of fragments into neutral, positive and negative ones"""
    out: dict[ChargeType, list[Mol]] = {'neutral': [], 'positive': [], 'negative': []}
    for frag in fragments:
        Chem.SanitizeMol(frag)
        c = Chem.GetFormalCharge(frag)
        if c == 0:
            out['neutral'].append(frag)
        elif c > 0:
            out['positive'].append(frag)
        else:
            out['negative'].append(frag)
    return out  # tuple of Mol objects


def split_bond(mol, atom_idx1, atom_idx2, is_inductive=None) -> tuple[Mol, Mol]:
    """Cleave a molecule at the bond between the specified atoms. Return the generated fragments."""
    rw_mol = Chem.RWMol(mol)
    atom1 = rw_mol.GetAtomWithIdx(atom_idx1)
    atom2 = rw_mol.GetAtomWithIdx(atom_idx2)
    rw_mol.RemoveBond(atom1.GetIdx(), atom2.GetIdx())
    Chem.SanitizeMol(rw_mol)
    frags: tuple[Mol] = Chem.GetMolFrags(rw_mol, asMols=True, sanitizeFrags=True)
    return frags


def reduce_charge(mol):
    """Reduce formal charge of the one with the highest by one"""
    raise NotImplementedError
    rw_mol = Chem.RWMol(mol)  # make mutable
    idx_max: int = -1
    score = -float('inf')
    for i, atom in enumerate(rw_mol.GetAtoms()):
        allowed_valences = list(PERIODIC_TABLE.GetValenceList(atom.GetAtomicNum()))
        if (atom.GetTotalValence()) in allowed_valences:  # valid candidate
            # reducing the formal charge increases the valence, so ideally the atom we are changing has a lower than default valence
            _score = PERIODIC_TABLE.GetDefaultValence(atom.GetAtomicNum()) - atom.GetTotalValence()
            if _score > score:
                score = _score
                idx_max = i
    if idx_max == -1:
        mplt_mol(rw_mol)
    assert idx_max >= 0

    atom_to_modify = rw_mol.GetAtomWithIdx(idx_max)
    formal_charge = atom_to_modify.GetFormalCharge()
    print(
        f'changing formal charge of atom with index {idx_max} of type {atom_to_modify.GetAtomicNum()} to {formal_charge - 1} (new valence is {atom_to_modify.GetValence(which=EXPLICIT)})'
    )
    print('before:', atom_to_modify.GetFormalCharge(), atom_to_modify.GetValence(which=EXPLICIT),
          atom_to_modify.GetTotalValence(),
          atom_to_modify.GetNumImplicitHs(), atom_to_modify.GetIsAromatic())
    atom_to_modify.SetFormalCharge(formal_charge - 1)
    atom_to_modify.UpdatePropertyCache()

    # check this worked
    _atom = rw_mol.GetAtomWithIdx(idx_max)
    print('after:', _atom.GetFormalCharge(), _atom.GetValence(which=EXPLICIT), _atom.GetTotalValence(),
          _atom.GetNumImplicitHs())

    Chem.SanitizeMol(rw_mol)  # recheck valences, update
    return rw_mol.GetMol()


def _find_cleavage_bonds(mol: Mol) -> dict[CleavagePos, list[tuple[int, int]]]:
    """
    For a given molecule, identify bonds that are prone to inductive cleavages.
    This applies to Carbon atoms bonded to O, N, S(?), P
    """
    matches = {}
    for s_pattern in CLEAVAGE_POS:
        pattern: Mol = Chem.MolFromSmarts(s_pattern)
        matches[s_pattern] = list(mol.GetSubstructMatches(pattern))

    return matches


def _get_fragments(
        parent_ion: Fragment,
        keep_neutral=False,
        max_recursion_depth=3,
        _current_recursion_depth=0
) -> list[Fragment]:
    """Split mol at cleavage pos. C atom gets positive charge.

    :returns
    a list of tuples in which the first entry is the positive and the second
    the neutral fragment.
    """

    def _process_fragments(fragments: list[Mol]) -> list[Fragment]:
        processed = [
            Fragment(mol=frag, recursion_depth=_current_recursion_depth)
            for frag in fragments
        ]
        return processed

    mol = parent_ion.mol
    inchkeys = set()
    fragments = []

    cleavage_pos = _find_cleavage_bonds(mol)
    for k, bonds in cleavage_pos.items():
        for bond in bonds:
            if keep_neutral:
                # fragments_neut.extend(split_bond(mol, *bond))
                _frags: list[Mol] = split_bond(mol, *bond)
                _frags: list[Fragment] = _process_fragments(_frags)
            else:
                for is_inductive in [True, False]:
                    _frags: list[Mol] = get_inductively_cleaved(
                        mol, *bond, is_inductive=is_inductive
                    )
                    _frags: list[Fragment] = _process_fragments(_frags)
                    # only append fragments when the inchkey does not exist yet
                    for _frag in _frags:
                        if _frag.inchkey in inchkeys:
                            continue
                        fragments.append(_frag)
                        inchkeys.add(_frag.inchkey)
                        # enter recursive call if fragment is positive and max
                        # recursion depth is not reached yet
    return fragments


def _predict_ms2(
        *,
        mol: Mol = None,
        smiles: str = None,
        adduct: str = None,
        **kwargs
) -> list[Fragment]:
    """Generate the fragment spectrum for a smiles. Assumes molecule is protonated."""
    # TODO: use FragmentTree instead
    if mol is None:
        assert smiles is not None
        mol: Mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    out = get_fragments(mol, **kwargs)

    # TODO: implement for different adduct types
    if adduct is None:
        adduct_mass = MASS_PROTON

    parent_ion = add_adduct_to_heteroatom(mol)
    parent_fragment = Fragment(parent_ion, recursion_depth=0)
    fragments = [parent_fragment]
    for ik, frag, charge_type, i, rd in zip(*out):
        if charge_type != 'positive':
            continue
        fragments.append(Fragment(mol=frag, recursion_depth=rd))
    return fragments


def _predict_losses(smiles: str, **kwargs) -> dict[float, Mol]:
    mol: Mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    frags = get_fragments(mol, **kwargs)

    desc = [add_adduct_to_heteroatom(mol)]
    mzs = [ExactMolWt(desc[0])]
    for t in frags:
        frag = t[1]
        if frag is None:
            continue
        mz = ExactMolWt(frag)
        mzs.append(mz)
        desc.append(frag)

    return dict(zip(mzs, desc))


def get_neutral_equivalents_for_predicted_ms2(smiles):
    """Instead of the protonated fragments, this returns the neutral molecules
    for better substructure match"""
    mol: Mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    cleavage_pos = _find_cleavage_bonds(mol)
    predicted = predict_ms2(smiles)
    neutral_frags_conv: list[tuple[Mol, Mol]] = get_fragments(
        mol, cleavage_pos, keep_neutral=True
    )
    # deconvolute
    neutral_frags: list[Mol] = []
    for neuts in neutral_frags_conv:
        neutral_frags.extend(neuts)

    neutral_equivalents = {}
    for mz, mol in predicted.items():
        dists = np.array([np.abs((mz - MASS_PROTON) - ExactMolWt(neut))
                          for neut in neutral_frags])
        closest_idx = np.argmin(dists)
        print(dists, closest_idx)
        dist = dists[closest_idx]
        if dist > 1e-3:
            warnings.warn(f'found large deviation for {smiles=} and {mz=}')
        neutral_equivalents[mz] = neutral_frags[closest_idx]
    return neutral_equivalents
