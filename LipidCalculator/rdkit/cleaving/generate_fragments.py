"""Use alpha/induced cleavages (breakage of carbon-carbon bonds next to
heteroatoms) to predict fragments."""
from typing import Literal, Iterable, Callable

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from rdkit import Chem
from rdkit.Chem import Mol, rdMolDescriptors, inchi, GetFormalCharge, RWMol, Atom, ValenceType, AddHs
from rdkit.Chem.Descriptors import ExactMolWt

from LipidCalculator import CompoundDict
from LipidCalculator.adduct.rdkit_add_adduct import get_mol_with_adduct, steal_charge_from_adduct, \
    find_adduct_positions, add_formal_charge_for_atom, find_available_bond_locations_for_adduct
from LipidCalculator.cleaving.alpha_cleavage import find_alpha_cleavage_positions, get_alpha_cleaved
from LipidCalculator.cleaving.inductive_cleavage import get_inductively_cleaved, find_inductive_cleavage_positions
from LipidCalculator.cleaving.sigma_cleavage import find_sigma_cleavage_positions, get_sigma_cleaved
from LipidCalculator.rdkit.plotting import mol_to_img, plt_indices_bond
import logging

from LipidCalculator.rdkit.util import get_num_hs, remove_one_h_for_atom

logger = logging.getLogger(__name__)

DEBUG_SPLITTING = True
DEBUG_IONS = False
DEBUG_CHARGE_MOVEMENT = False

CLEAVAGE_TYPES = ['SIGMA', 'INDUCTIVE', 'ALPHA']
CleavageType: type = Literal[*CLEAVAGE_TYPES]

ChargeType: type = Literal['neutral', 'positive', 'negative']

PERIODIC_TABLE = Chem.GetPeriodicTable()

CLEAVAGE_TYPE_TO_FINDER: dict[CleavageType, Callable[[Mol], list[tuple[int, int]]]] = {
    'SIGMA': find_sigma_cleavage_positions,
    'INDUCTIVE': find_inductive_cleavage_positions,
    'ALPHA': find_alpha_cleavage_positions
}


def _find_cleavage_positions(
        mol: Mol, cleavage_types: list[CleavageType]
) -> dict[CleavageType, tuple[int, int]]:
    out = {ct: CLEAVAGE_TYPE_TO_FINDER[ct](mol) for ct in cleavage_types}
    return out


def _fragments_sort_by_charge(fragments: Iterable['Fragment']) -> dict[ChargeType, list['Fragment']]:
    """Split an iterable of fragments into neutral, positive and negative ones"""
    out: dict[ChargeType, list['Fragment']] = {'neutral': [], 'positive': [], 'negative': []}
    for frag in fragments:
        if (c := frag.charge) == 0:
            out['neutral'].append(frag)
        elif c > 0:
            out['positive'].append(frag)
        else:
            out['negative'].append(frag)
    return out


def get_predicted_pattern_from_fragments(frags: Iterable['Fragment']) -> list['Fragment']:
    """Bin Fragments with the same formula"""
    frags_binned: list[Fragment] = []
    formulas_plot: list[CompoundDict] = []
    for frag in frags:
        if (cd := CompoundDict(frag.formula)) not in formulas_plot:
            frags_binned.append(frag.copy())
            formulas_plot.append(cd)
        else:
            idx = formulas_plot.index(cd)
            frag_plot = frags_binned[idx]
            frag_plot.probability += frag.probability
            frag_plot.recursion_depth = min(
                frag_plot.recursion_depth, frag.recursion_depth
            )
    # rescale
    _max = max([f.probability for f in frags_binned])
    for f in frags_binned:
        f.probability /= _max

    return frags_binned


def plot_ms2_prediction(
        fragments: list['Fragment'],
        as_negative: bool = False,
        res_pixels_child=1000,
        add_struct_plots: bool = True,
        ax: plt.Axes = None
) -> plt.Axes:
    """Draw the original molecule, the generated fragments and the fragment pattern."""
    if ax is None:
        _, ax = plt.subplots()

    if len(fragments) == 0:
        print('No fragments to plot')
        return ax

    is_neutral: bool = fragments[0].charge == 0
    assert all([(f.charge == 0) is is_neutral for f in fragments]), \
        'all fragments must either be neutral or positive'

    def x_getter(frag: Fragment) -> float:
        if is_neutral:
            return frag.mass
        return frag.mz

    # combine fragments with the same formula
    # add probabilities, use lower recursion
    frags_plot = get_predicted_pattern_from_fragments(fragments)

    ct_to_color = {
        None: 'blue',
        'INDUCTIVE': 'red',
        'ALPHA': 'green',
        'SIGMA': 'yellow',
    }
    linestyles = ['solid', 'dashed', 'dotted']

    tr = ax.get_xaxis_transform()
    for i, frag in enumerate(frags_plot):
        x = x_getter(frag)
        ymin = 0
        ymax = frag.probability
        if as_negative:
            ymin, ymax = -ymax, ymin
        ax.vlines(x, ymin=ymin, ymax=ymax, color=ct_to_color.get(frag.cleavage_type, 'k'),
                  linestyles=linestyles[frag.recursion_depth % len(linestyles)])

        # add annotations
        # TODO: use absolute position and respect negative
        p = (x, np.random.random() * .8 + .1)
        lbl_mz = f'{x:.4f}'
        if (ct := frag.cleavage_type) is None:
            cleave_type_ann = ''
        elif ct == 'INDUCTIVE':
            cleave_type_ann = 'I'
        elif ct == 'ALPHA':
            cleave_type_ann = r'$\alpha$'
        elif ct == 'SIGMA':
            cleave_type_ann = r'$\sigma$'
        else:
            raise NotImplementedError
        txt = f'{frag.formula}\n({cleave_type_ann}, {lbl_mz})'
        ax.text(x, ymin if as_negative else ymax, txt, rotation=45)
        # only add structure for more than 5 atoms
        cd = CompoundDict(frag.formula)
        num_non_h = sum([v for k, v in cd.composition.items() if k not in ('H', '+', '-')])
        if (not add_struct_plots) or (num_non_h <= 3):
            continue
        ax.plot(*p, transform=tr, marker='o', c='r')
        inset_ax = inset_axes(ax, width=4, height=4, loc='center',
                              bbox_to_anchor=p,
                              bbox_transform=tr,
                              borderpad=0)
        frag_img = mol_to_img(mol=frag.mol, res_pixels=res_pixels_child)
        inset_ax.imshow(frag_img)
        inset_ax.axis('off')

    ax.set_xlabel(f'fragment {'mass' if is_neutral else 'm/z'} in Da')
    return ax


def _find_movable_charges(mol: Mol, charge_type: ChargeType, plts=False) -> list[int]:
    """Find atoms whose formal charge can be removed"""

    def _is_charge_right(atm: Atom) -> bool:
        if charge_type == 'positive':
            return atm.GetFormalCharge() == 1
        elif charge_type == 'negative':
            return atm.GetFormalCharge() == -1
        else:
            raise ValueError

    def _is_charge_movable(atm: Atom) -> bool:
        """This is only the case if there is an h atom or a radical"""
        # TODO: is this really required
        can_compensate_valence = (atm.GetNumRadicalElectrons() > 0) or (
                get_num_hs(atm) > 0)
        return can_compensate_valence

    candidates: list[int] = []
    for idx, atom in enumerate(mol.GetAtoms()):
        if _is_charge_right(atom) and _is_charge_movable(atom):
            candidates.append(idx)
    if plts:
        plt_indices_bond(mol)
        print(f'identified the following positions as movable: {candidates}')
    return candidates


def _move_charge(mol: Mol, idx_start, idx_end, as_h_plus: bool, plts=False):
    rw_mol = RWMol(Mol(mol))

    atom_with_charge = rw_mol.GetAtomWithIdx(idx_start)
    atom_getting_charge = rw_mol.GetAtomWithIdx(idx_end)

    # remove charge
    c = atom_with_charge.GetFormalCharge()
    atom_with_charge.SetFormalCharge(0)
    # add charge
    atom_getting_charge.SetFormalCharge(c)

    new_mol = add_formal_charge_for_atom(mol=rw_mol, atom_idx=idx_end, add_H=as_h_plus)
    # need to remove H after since this will change the indices
    remove_one_h_for_atom(new_mol, idx_start)

    if plts:
        fig, ax = plt.subplots()
        plt_indices_bond(mol, ax=ax)
        ax.set_title(f'attempting to move charge from {idx_start} to {idx_end}')
        plt.show()
    print(f'final radicals on charged: {new_mol.GetAtomWithIdx(idx_start).GetNumRadicalElectrons()}')
    print(f'final Hs on charged: {new_mol.GetAtomWithIdx(idx_start).GetNumExplicitHs()}')
    if failed:
        raise ValueError()
    return new_mol


def _figure_wait_unit_pressed(fig):
    plt.show(block=False)
    print("Press any key or click in the plot window to continue (don't close the figure!)...")
    plt.waitforbuttonpress()  # waits indefinitely for a key press or mouse click
    plt.close(fig)


class Fragment:
    _inchkey: str = None
    mol: Mol = None
    recursion_depth: int = None
    _mass: float = None
    _charge: float = None
    _charge_type: ChargeType = None
    allow_charge_relocation: bool = None

    _mz: float = None
    _formula: str = None
    probability: float = None

    cleavage_type: CleavageType = None
    cleavage_pos_in_parent: tuple[int, int] = None
    _child_fragments: list['Fragment'] = None
    _max_recursion_depth: int = None

    def __init__(
            self,
            mol: Mol,
            recursion_depth: int = 0,
            probability: float = 1,
            cleavage_type: CleavageType = None,
            cleavage_pos_in_parent: tuple[int, int] = None,
            allow_charge_relocation: bool = True,
            charge_movement_as_h_plus: bool = False,
    ):
        assert allow_charge_relocation is not None
        self.mol = mol
        self.recursion_depth = recursion_depth
        self.probability = probability
        self.cleavage_type = cleavage_type
        self.cleavage_pos_in_parent = cleavage_pos_in_parent
        self.allow_charge_relocation = allow_charge_relocation
        self.charge_movement_as_h_plus = charge_movement_as_h_plus

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

    def copy(self):
        return Fragment(
            mol=Mol(self.mol),
            recursion_depth=self.recursion_depth,
            probability=self.probability,
            cleavage_type=self.cleavage_type,
            cleavage_pos_in_parent=self.cleavage_pos_in_parent,
            allow_charge_relocation=self.allow_charge_relocation,
        )

    def _set_child_fragments(
            self, cleavage_types: list[CleavageType], max_recursion_depth: int
    ):
        self._max_recursion_depth = max_recursion_depth

        # keep track of fragments
        _inchkeys = set()
        self._child_fragments: list['Fragment'] = []

        # maximum recursion depth reached
        if self.recursion_depth >= max_recursion_depth:
            return

        cleavage_pos: dict[CleavageType, list[tuple[int, int]]] = (
            _find_cleavage_positions(self.mol, cleavage_types)
        )

        for cleavage_type, positions in cleavage_pos.items():
            if cleavage_type == 'SIGMA':
                split_func = get_sigma_cleaved
            elif cleavage_type == 'ALPHA':
                split_func = get_alpha_cleaved
            elif cleavage_type == 'INDUCTIVE':
                split_func = get_inductively_cleaved
            else:
                raise ValueError(f'Unknown cleavage type: {cleavage_type}')
            for position in positions:
                logger.info(f'splitting at {position} with {cleavage_type} for {self}')
                try:
                    parts: tuple[Mol, Mol] = split_func(self.mol, *position)
                    if DEBUG_SPLITTING:
                        fig, axs = plt.subplots(nrows=2)
                        plt_indices_bond(self.mol, ax=axs[0])
                        plt_indices_bond(parts, ax=axs[1])
                        fig.suptitle(
                            f'Splitting of molecule at {position} with {cleavage_type}'
                        )
                        _figure_wait_unit_pressed(fig)

                except Exception as e:
                    print(e)
                    continue
                for part in parts:
                    frag: Fragment = Fragment(
                        mol=part,
                        recursion_depth=self.recursion_depth + 1,
                        cleavage_type=cleavage_type,
                        cleavage_pos_in_parent=position,
                        probability=self.probability * .8,
                        allow_charge_relocation=self.allow_charge_relocation
                    )
                    # this either enters the recursion or returns immediately
                    if frag.inchkey in _inchkeys:
                        continue
                    _inchkeys.add(frag.inchkey)
                    self._child_fragments.append(frag)
                    if frag.charge == 0:
                        continue
                    frag._set_child_fragments(
                        max_recursion_depth=max_recursion_depth,
                        cleavage_types=cleavage_types
                    )
                    # for now only positive is supported
                    assert self.allow_charge_relocation is not None
                    if not self.allow_charge_relocation:
                        logger.info('skipping charge relocation')
                        continue
                    else:
                        logger.info('not skipping charge relocation')
                    movable_charges = _find_movable_charges(
                        frag.mol, charge_type='positive'
                    )
                    logger.info(f'found {len(movable_charges)} possible other charge position(s)')
                    assert len(movable_charges) <= 1, \
                        'not expecting and cannot handle multiple movable charges'
                    if len(movable_charges) == 0:
                        continue
                    idx_charge = movable_charges[0]

                    idcs_destination = find_available_bond_locations_for_adduct(frag.mol)
                    for idx_destination in idcs_destination:
                        if DEBUG_CHARGE_MOVEMENT:
                            fig, axs = plt.subplots(nrows=2)
                            fig.suptitle(
                                f'Charge moved from {idx_charge} to {idx_destination} (as H+={self.charge_movement_as_h_plus})')
                            plt_indices_bond(frag.mol, ax=axs[0])
                        mol_with_moved_charge = _move_charge(
                            frag.mol,  # need to create copy
                            idx_charge,
                            idx_destination,
                            as_h_plus=self.charge_movement_as_h_plus
                        )
                        if DEBUG_CHARGE_MOVEMENT:
                            plt_indices_bond(mol_with_moved_charge, ax=axs[1])
                            _figure_wait_unit_pressed(fig)

                        frag_charged: Fragment = Fragment(
                            mol=mol_with_moved_charge,
                            recursion_depth=self.recursion_depth + 1,
                            cleavage_type=cleavage_type,
                            probability=self.probability * .8 ** 2,
                            allow_charge_relocation=self.allow_charge_relocation
                        )
                        frag_charged._set_child_fragments(
                            max_recursion_depth=max_recursion_depth,
                            cleavage_types=cleavage_types
                        )
                        # need to link modified fragments to this fragment,
                        # even though the composition is almost the same
                        self._child_fragments.append(frag_charged)

    def get_child_fragments(
            self, cleavage_types: list[CleavageType], max_recursion_depth: int = 1,
    ):
        if (self._child_fragments is None) or (self._max_recursion_depth != max_recursion_depth):
            logger.info(f'setting child fragments to {max_recursion_depth}')
            self._set_child_fragments(cleavage_types, max_recursion_depth)
        return self._child_fragments

    def get_all_fragments(self, **kwargs) -> list['Fragment']:
        logger.info(f'getting all child fragments for fragment {self}')
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
        logger.info(f'got {len(frags)} child fragments for fragment {self}')
        return frags


class FragmentTree:
    root: Fragment = None
    max_recursion_depth: int = None
    only_inductive: bool = None
    _ions: list[Fragment] = None
    _ion_location_and_types: list[tuple[int, str, bool]] = None

    allow_charge_relocation: bool = None
    cleavage_types: list[CleavageType] = None

    def __init__(
            self,
            mol: Mol,
            adduct_type: str,
            max_recursion_depth: int = 1,
            cleavage_types: list[CleavageType] = None,
            allow_charge_relocation: bool = True
    ):
        assert allow_charge_relocation is not None
        self.mol = AddHs(mol)  # we are always treating Hs explicitly, only removed for plotting
        self.allow_charge_relocation = allow_charge_relocation
        if GetFormalCharge(mol) == 0:
            logger.info('creating fragment tree from charged molecule')
            self._set_ions(adduct_type)
        else:
            logger.info('creating fragment tree from non-charged molecule')
            self._ions = [
                Fragment(
                    mol=mol,
                    allow_charge_relocation=allow_charge_relocation
                )
            ]
            logger.info(f'created {len(self._ions)} fragments')

        self.max_recursion_depth = max_recursion_depth

        if cleavage_types is None:
            cleavage_types = CLEAVAGE_TYPES.copy()
        else:
            assert all([ct in CLEAVAGE_TYPES for ct in cleavage_types])
        self.cleavage_types: list[CleavageType] = cleavage_types

    def _set_ions(self, adduct_type: str):
        """
        Add adducts to all neutral heteroatoms and simulate stripping away the
        adduct, attempting to one time
        keep and another time remove the H (only if adduct contains H)
        """
        _keep_hs = [False]
        if 'H' in adduct_type:
            _keep_hs.append(True)

        _adduct_positions: list[int] = find_adduct_positions(self.mol)
        self._ions: list[Fragment] = []
        self._ion_location_and_types: list[tuple[int, str, bool]] = []

        for _adduct_pos in _adduct_positions:
            _mol_with_adduct = get_mol_with_adduct(
                self.mol, add=adduct_type, return_mode='index', idx=_adduct_pos
            )
            for _keep_h in _keep_hs:
                if adduct_type == 'M+':  # nothing to steal from
                    _mol_with_h_plus = _mol_with_adduct
                else:
                    _mol_with_h_plus = steal_charge_from_adduct(
                        _mol_with_adduct, keep_h=_keep_h, plts=False
                    )
                self._ions.append(
                    Fragment(
                        mol=_mol_with_h_plus,
                        allow_charge_relocation=self.allow_charge_relocation,
                        charge_movement_as_h_plus=_keep_h
                    )
                )
                self._ion_location_and_types.append(
                    (_adduct_pos, adduct_type, _keep_h)
                )

        if DEBUG_IONS:
            self.plot_ions()

    def plot_ions(self):
        if self._ion_location_and_types is None:
            positions = ['unknown']
            types = ['unknown']
            keep_hs = ['unknown']
        else:
            positions = [t[0] for t in self._ion_location_and_types]
            types = [t[1] for t in self._ion_location_and_types]
            keep_hs = [t[2] for t in self._ion_location_and_types]
        for _ion, _pos, _type, _keep_h in zip(self._ions, positions, types, keep_hs):
            fig, ax = plt.subplots()
            plt_indices_bond(_ion.mol, ax=ax)
            fig.suptitle(
                f'Ion with {_type} adduct at idx {_pos} (H was {'' if _keep_h else 'not '}kept, m/z={_ion.mz:.4f}, f={_ion.formula})')
            if DEBUG_IONS:
                _figure_wait_unit_pressed(fig)

    def get_all_fragments(self) -> list[Fragment]:
        logger.info('getting all fragments')
        fragments: list[Fragment] = []
        for parent_ion in self._ions:
            fragments.append(parent_ion)
            fragments.extend(
                parent_ion.get_all_fragments(
                    max_recursion_depth=self.max_recursion_depth,
                    cleavage_types=self.cleavage_types
                )
            )
        logger.info(f'got {len(fragments)} fragments')
        return fragments

    def plot_ms2(self, fig=None, res_pixels_parent=5000, **kwargs):
        if fig is None:
            fig, axs = plt.subplots(nrows=3)
        else:
            axs = fig.get_axes()
            assert len(axs) >= 3

        img = mol_to_img(self.mol, res_pixels=res_pixels_parent)
        axs[0].imshow(img)
        axs[0].axis('off')

        frags = self.get_all_fragments()
        frags_sorted: dict[ChargeType, list[Fragment]] = (
            _fragments_sort_by_charge(frags))

        plot_ms2_prediction(fragments=frags_sorted['positive'], ax=axs[1], **kwargs)
        axs[1].set_title('Positive fragments')
        plot_ms2_prediction(fragments=frags_sorted['neutral'], ax=axs[2], **kwargs)
        axs[2].set_title('Neutral fragments')

        # equal scaling for axes
        xmin1, xmax1 = axs[1].get_xlim()
        xmin2, xmax2 = axs[2].get_xlim()
        xmin = min(xmin1, xmin2)
        xmax = max(xmax1, xmax2)
        span = xmax - xmin
        for i in [1, 2]:
            axs[i].set_xlim((xmin - .05 * span, xmax + .05 * span))

        return fig


def predict_ms2(
        mol: Mol, adduct_type: str, **kwargs
) -> list[Fragment]:
    """Predict characteristic fragments for a given molecule with specified adduct type. """
    tree = FragmentTree(mol, adduct_type=adduct_type, **kwargs)
    frags = tree.get_all_fragments()
    return _fragments_sort_by_charge(frags)['positive']


def predict_losses(
        mol: Mol, adduct_type: str, **kwargs
):
    tree = FragmentTree(mol, adduct_type=adduct_type, **kwargs)
    frags = tree.get_all_fragments()
    return _fragments_sort_by_charge(frags)['neutral']


def test_inductive_cleavage_simple():
    smiles = "CCOCC"  # Contains O, N (heteroatoms)
    mol: Mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    plt_indices_bond([mol])
    frags = get_inductively_cleaved(mol, 1, 2, plts=True)


def testing_get_comp(adduct_pos: Literal['head', 'chain'], plts=False):
    from LipidCalculator.compound_groups.intact_polar_lipids.generate_ipl import ipl_automatic_bonds

    if adduct_pos == 'head':
        adduct_idx = 10
    elif adduct_pos == 'chain':
        adduct_idx = 14
    else:
        raise ValueError('adduct_pos must be "head" or "chain"')

    name = 'PC DAG C15:0 C15:0'
    mol = ipl_automatic_bonds(name.split(), plts=False, idx_plt=False, split_chain=False)

    mol_with_adduct = get_mol_with_adduct(mol, add='[M+NH4]+', return_mode='index', idx=adduct_idx)
    if plts:
        plt_indices_bond(mol_with_adduct)

    return mol_with_adduct


def test_inductive_cleavage_head():
    mol_with_adduct = testing_get_comp(adduct_pos='head', plts=False)
    mol_with_h_plus = steal_charge_from_adduct(mol_with_adduct, keep_h=True, plts=False)
    plt_indices_bond(mol_with_h_plus)
    pos = _find_cleavage_positions(mol_with_h_plus)
    assert (11, 10) in pos['INDUCTIVE']

    cleaved = get_inductively_cleaved(mol_with_h_plus, 11, 10)
    plt_indices_bond(cleaved)

    # cleave at OP
    cleaved = get_inductively_cleaved(mol_with_h_plus, 7, 10)
    plt_indices_bond(cleaved)


def test_inductive_cleavage_chain():
    mol_with_adduct = testing_get_comp(adduct_pos='chain', plts=False)
    mol_with_h_plus = steal_charge_from_adduct(mol_with_adduct, keep_h=True, plts=False)
    plt_indices_bond(mol_with_h_plus)

    pos = _find_cleavage_positions(mol_with_h_plus)
    assert (15, 14) in pos['INDUCTIVE']

    cleaved = get_inductively_cleaved(mol_with_h_plus, 15, 14)
    plt_indices_bond(cleaved)


def test_sanitize_remove_hs():
    # sanitize does not make H atoms that are neighbors implicit
    name = 'PI AR'
    mol = ipl_automatic_bonds(name.split(), plts=False, idx_plt=False, split_chain=False)

    mol_ = AddHs(mol)
    plt_indices_bond(mol_, remove_hs=False)

    mol__ = RWMol(Mol(mol_))
    mol___ = mol__.GetMol()
    Chem.SanitizeMol(mol___)

    plt_indices_bond(mol___, remove_hs=False)


if __name__ == "__main__":
    # TODO: beta-H rearrangement: need O[H+]CC, H jumps from further C to O, bond between OC is broken, double bond between CC
    pass
    # test_inductive_cleavage_head()
    # test_inductive_cleavage_chain()

    from LipidCalculator.compound_groups.intact_polar_lipids.generate_ipl import ipl_automatic_bonds

    name = 'PI AR'
    mol = ipl_automatic_bonds(name.split(), plts=False, idx_plt=False, split_chain=False)

    # plt.show()
    # plt.pause(1)
    #
    # adduct_pos = int(input('Adduct position: '))
    #
    # mol_with_adduct = get_mol_with_adduct(mol, add='[M+NH4]+', return_mode='index', idx=adduct_pos)
    # mol_with_h_plus = steal_charge_from_adduct(mol_with_adduct, keep_h=True, plts=False)
    # plt_indices_bond(mol_with_h_plus)
    #
    # print(ExactMolWt(mol_with_h_plus))
    # print(rdMolDescriptors.CalcMolFormula(mol_with_h_plus))

    # pos = _find_cleavage_positions(mol_with_h_plus)
    #
    # for p in pos['SIGMA']:
    #     # frags = get_inductively_cleaved(mol_with_h_plus, *p)
    #     # frags = get_alpha_cleaved(mol_with_h_plus, *p)
    #     frags = get_sigma_cleaved(mol_with_h_plus, *p)
    #     plt_indices_bond(frags)

    # frag = Fragment(mol=mol_with_h_plus)
    # frag.get_child_fragments(only_inductive=False)

    # print(Chem.MolToSmiles(mol))
    # M = ExactMolWt(mol_with_adduct)
    # f = rdMolDescriptors.CalcMolFormula(mol)

    logger.setLevel(logging.INFO)

    tree = FragmentTree(
        mol,
        adduct_type='[M+H]+',
        max_recursion_depth=2,
        allow_charge_relocation=True,
        cleavage_types=None,
    )
    # tree.plot_ions()
    fig = tree.plot_ms2(add_struct_plots=False, res_pixels_child=500)

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
