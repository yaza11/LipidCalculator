"""Use alpha/induced cleavages (breakage of carbon-carbon bonds next to
heteroatoms) to predict fragments."""
import warnings
from typing import Literal, Iterable

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from rdkit import Chem
from rdkit.Chem import Mol, rdMolDescriptors, EXPLICIT, inchi
from rdkit.Chem.Descriptors import ExactMolWt

from LipidCalculator import CompoundDict
from LipidCalculator.adduct.rdkit_add_adduct import get_mol_with_adduct
from LipidCalculator.cleaving.inductive_cleavage import get_inductively_cleaved
from LipidCalculator.compound_groups.intact_polar_lipids.generate_ipl import ipl_automatic_bonds
from LipidCalculator.rdkit.plotting import mol_to_img, mplt_mol, plt_indices_bond
from LipidCalculator.compound_groups.common_fragments import MASS_PROTON

CLEAVAGE_POS: list[str] = ['CO', 'CN', 'CP', 'CS']
CleavagePos: type = Literal[*CLEAVAGE_POS]
ChargeType: type = Literal['neutral', 'positive', 'negative']

PERIODIC_TABLE = Chem.GetPeriodicTable()


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


# def split_bond(mol, atom_idx1, atom_idx2, is_inductive=None) -> tuple[Mol, Mol]:
#     """Cleave a molecule at the bond between the specified atoms. Return the generated fragments."""
#     rw_mol = Chem.RWMol(mol)
#     atom1 = rw_mol.GetAtomWithIdx(atom_idx1)
#     atom2 = rw_mol.GetAtomWithIdx(atom_idx2)
#     rw_mol.RemoveBond(atom1.GetIdx(), atom2.GetIdx())
#     Chem.SanitizeMol(rw_mol)
#     frags: tuple[Mol] = Chem.GetMolFrags(rw_mol, asMols=True, sanitizeFrags=True)
#     return frags


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


def plot_ms2_prediction(
        mol,
        fragments: list['Fragment'],
        annotations: dict[float, str] = None,
        res_pixels_parent=2000
):
    """Draw the original molecule, the generated fragments and the fragment pattern."""
    if annotations is None:
        annotations = {k: None for k in fragments}
    else:
        # TODO: check annotations match fragments
        pass

    img = mol_to_img(mol, res_pixels=res_pixels_parent)
    # fig, axs = plt.subplots(nrows=2, dpi=res_pixels)
    fig, axs = plt.subplots(nrows=2)
    axs[0].imshow(img)
    axs[0].axis('off')

    # plot fragments with different colors depending on recursion depth
    mzs = np.array([f.mz for f in fragments])
    ints = np.array([f.probability for f in fragments])
    rds = np.array([f.recursion_depth for f in fragments])
    for c, rd in enumerate(np.unique(rds)):
        mask = rds == rd
        axs[1].stem(
            mzs[mask],
            ints[mask],
            markerfmt='',
            basefmt='k',
            linefmt=f'C{c}'
        )

    # add annotations
    tr = axs[1].get_xaxis_transform()
    for i, (frag, ann) in enumerate(zip(fragments, annotations.values())):
        mz = frag.mz
        p = (mz, np.random.random() * .8 + .1)
        axs[1].plot(*p, transform=tr, marker='o', c='r')
        inset_ax = inset_axes(axs[1], width=1.5, height=1.5, loc='center',
                              bbox_to_anchor=p,
                              bbox_transform=tr,
                              borderpad=0)
        lbl_mz = f'{mz:.4f}'
        txt = f'{frag.formula}\n({lbl_mz})' if ann is None else ann + '\n' + f'({lbl_mz})'
        axs[1].text(mz, 1, txt, rotation=45)
        frag_img = mol_to_img(mol=frag.mol, res_pixels=int(res_pixels_parent / 10))
        inset_ax.imshow(frag_img)
        inset_ax.axis('off')

    axs[1].set_xlabel('fragment m/z in Da')
    return fig, axs


class Fragment:
    _inchkey: str = None
    mol: Mol = None
    _ion: Mol = None
    recursion_depth: int = None
    _mass: float = None
    _charge: float = None
    _charge_type: ChargeType = None
    _mz: float = None
    _formula: str = None
    probability: float = None

    _cleavage_type: tuple[CleavagePos, bool] = None
    _child_fragments: list['Fragment'] = None
    _max_recursion_depth: int = None

    def set_ion(self, adduct_type: str):
        self._ion = get_mol_with_adduct(self.mol, adduct_type, return_mode='first')

    @classmethod
    def from_uncharged(cls, mol, adduct_type: str = '[M+H]+'):
        frag = Fragment(mol=mol)
        frag.set_ion(adduct_type)
        return frag

    @property
    def ion(self):
        """Apart from the root, the ion is the molecule itself ...
        it is assumed that the adduct is lost during fragmentation"""
        return self._ion if self._ion is not None else self.mol

    def __init__(self, mol: Mol, recursion_depth: int = 0, probability: float = 1):
        self.mol = mol
        self.recursion_depth = recursion_depth
        self.probability = probability

    @property
    def inchkey(self):
        if self._inchkey is None:
            self._inchkey = inchi.MolToInchiKey(mol=self.mol)
        return self._inchkey

    @property
    def mass(self):
        if self._mass is None:
            self._mass = ExactMolWt(self.mol)
        return self._mass

    @property
    def charge(self):
        if self._charge is None:
            self._charge = Chem.GetFormalCharge(self.mol)
        return self._charge

    @property
    def mz(self):
        if self._mz is None:
            self._mz = self.mass / self.charge if self.charge != 0 else np.nan
        return self._mz

    @property
    def formula(self):
        if self._formula is None:
            self._formula = rdMolDescriptors.CalcMolFormula(self.mol)
        return self._formula

    def _set_child_fragments(
            self, max_recursion_depth: int, only_inductive: bool
    ):
        self._max_recursion_depth = max_recursion_depth

        # keep track of fragments
        _inchkeys = set()
        self._child_fragments: list['Fragment'] = []

        is_cleavage_type_inductive: list[bool] = [True]
        if not only_inductive:
            is_cleavage_type_inductive.append(False)

        print(only_inductive, is_cleavage_type_inductive)

        # maximum recursion depth reached
        if self.recursion_depth > max_recursion_depth:
            return

        # only add hydrogen for first cleavage
        is_first_order: bool = self.recursion_depth == 0
        # split_func = split_bond_add_H if is_first_order else split_bond
        split_func = get_inductively_cleaved

        cleavage_pos: dict[CleavagePos, list[tuple[int, int]]] = (
            _find_cleavage_bonds(self.mol)
        )
        for bond_type, positions in cleavage_pos.items():
            for position in positions:
                # one time C gets charge, other time heteroatom
                for is_inductive in is_cleavage_type_inductive:
                    # is_inductive not used in this case
                    parts: tuple[Mol, Mol] = split_func(
                        mol=self.mol,
                        c_atom_idx=position[0],
                        hetero_atom_idx=position[1],
                        is_inductive=is_inductive
                    )
                    parts_fragments: list[Fragment] = []
                    for part in parts:
                        frag: Fragment = Fragment(
                            mol=part, recursion_depth=self.recursion_depth + 1
                        )
                        # this either enters the recursion or returns immediately
                        if frag.inchkey in _inchkeys:
                            continue
                        _inchkeys.add(frag.inchkey)
                        if frag.charge > 0:  # fragment positive fragments further
                            frag._set_child_fragments(
                                max_recursion_depth=max_recursion_depth,
                                only_inductive=only_inductive
                            )
                        self._child_fragments.append(frag)

    def get_child_fragments(
            self, max_recursion_depth: int = 1, only_inductive: bool = True
    ):
        if (self._child_fragments is None) or (self._max_recursion_depth != max_recursion_depth):
            self._set_child_fragments(max_recursion_depth, only_inductive)
        return self._child_fragments

    def get_all_fragments(self, **kwargs) -> list['Fragment']:
        frags: list[Fragment] = []
        visited: set[str] = set()

        # Ensure at least one layer of children is computed
        stack = list(self.get_child_fragments(**kwargs))

        while stack:
            frag = stack.pop()
            ik = frag.inchkey
            if ik in visited:
                continue
            visited.add(ik)
            frags.append(frag)

            # Only traverse already generated children (do not force generation here)
            if frag._child_fragments is not None:
                stack.extend(frag._child_fragments)

        return frags


class FragmentTree:
    root: Fragment = None
    max_recursion_depth: int = None
    only_inductive: bool = None

    @classmethod
    def from_mol(cls, mol: Mol, **kwargs):
        root = Fragment.from_uncharged(mol=mol)
        new = cls(root, **kwargs)
        return new

    @classmethod
    def from_ipl_name(cls, ipl_name: str, **kwargs):
        mol = ipl_automatic_bonds(ipl_name.split())
        new = cls.from_mol(mol=mol, **kwargs)
        return new

    def __init__(
            self,
            root: Fragment,
            max_recursion_depth=1,
            only_inductive: bool = True
    ):
        self.root = root
        self.max_recursion_depth = max_recursion_depth
        self.only_inductive = only_inductive

    def get_all_positive_fragments(self) -> list[Fragment]:
        fragments: list[Fragment] = self.root.get_all_fragments(
            max_recursion_depth=self.max_recursion_depth,
            only_inductive=self.only_inductive
        )
        return [f for f in fragments if f.charge > 0]

    def plot_ms2(self):
        frags = self.get_all_positive_fragments()
        # deduplicate with formulas
        frags_plot = []
        cds_plot: set[CompoundDict] = set()
        for frag in frags:
            f = CompoundDict(frag.formula)
            if f not in cds_plot:
                cds_plot.add(f)
                frags_plot.append(frag)
        plot_ms2_prediction(self.root.ion, frags_plot)


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


def test_inductive_cleavage_simple():
    smiles = "CCOCC"  # Contains O, N (heteroatoms)
    mol: Mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    plt_indices_bond([mol])
    frags = get_inductively_cleaved(mol, 1, 2, plts=True)


if __name__ == "__main__":
    name = 'AR'
    # max_recursion_depth = 1
    # mol = ipl_automatic_bonds(name.split(), plts=False, idx_plt=False, split_chain=True)

    # print(Chem.MolToSmiles(mol))
    # M = ExactMolWt(mol)
    # f = rdMolDescriptors.CalcMolFormula(mol)

    tree = FragmentTree.from_ipl_name(name, max_recursion_depth=0, only_inductive=False)
    tree.plot_ms2()

    # plt_indices_bond([mol])
    # cleavage_pos = find_cleavage_bonds(mol)
    # split_bond_add_H(mol, 4, 1, plts=True, is_inductive=False)
    # split_bond_add_H(mol, 2, 3, plts=True)
    # frags = get_fragments(mol, max_recursion_depth=max_recursion_depth)

    # ms = predict_ms2(mol=mol, max_recursion_depth=max_recursion_depth)

    # mol = Chem.MolFromSmiles(smiles)

    # split_bond_add_H(mol, 19, 20, plts=True)

    # frags = get_fragments(mol, max_recursion_depth=0)
    # pos = [t[0] for t in frags]
    # neut = [t[1] for t in frags]
    #
    # c = [Chem.GetFormalCharge(frag) for frag in pos]
    # n = [Chem.GetFormalCharge(frag) for frag in neut]
    #
    # m_pos = [ExactMolWt(frag) for frag in pos]
    # m_neut = [ExactMolWt(frag) for frag in neut]

    # mplt_mol(neut[0])

    # fig, axs = plot_ms2_prediction(mol, ms)
    # fig.suptitle(f'MS2 prediction for {name}\n({f}, {M=:.4f})')
    # plt.show()
