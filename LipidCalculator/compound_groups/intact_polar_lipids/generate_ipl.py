"""
CONVENTIONS for IPL naming
==========================
- IPLs are build of a head, core/ backbone, and tails
- an IPL does not have to have all of those parts
- an IPL can have at most
    - one head
    - one core
    - two tails
- tails are carbon chains
    - they may have double bonds, hydrogen groups, branches, rings
    - right now we only support double bonds
    - if only one tail is specified, the chain is split into equal parts
- as of right now, heads can be
    - OH
    - 1G
    - 2G
    - 3G
    - PC
    - PE
    - PDME
    - PME
    - PG
    - PS
    - PE
    - PI
    - DPG
    - MMPE
    - SQ
- bonds with the core are formed at Fr atoms
- cores can be
    - DGTS / BL
    - OL
    - DAG
    - DEG
    - AEG
    - CER
    - GDGTs/ OH-GDGTs (0-5)
    - AR, OH-AR, 2OH-AR
- AR and GDGTs can not be connected to chains
- chains are connected to the core at Rb and Cs atoms
- C atoms in the core contribute to the chain lengths and double bonds

"""
import re
import warnings
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import rdkit
import logging
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from rdkit.Chem.rdchem import Mol, Atom, EditableMol
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import IPythonConsole

from LipidCalculator.compound_groups.intact_polar_lipids.pieces_from_json import PLACEHOLDER_ELEMENTS, \
    get_smiles, CORE_IMPLICIT_CHAIN, ABBREVIATION_TO_GROUP, ABBREVIATIONS, IPL_PIECES, CORE_GROUPS, \
    CHAINS_IN_ISOPRENE_UNITS, CHAIN_PLACEHOLDERS, implicit_chain
from LipidCalculator.compound_groups.intact_polar_lipids.isoprenoid_chains import modify_isoprenoid_chains, \
    clear_chain_marks, add_tetraether_double_bonds
from LipidCalculator.compound_groups.intact_polar_lipids.isoprenoid_blocks import is_block_core, block_core_smiles
from LipidCalculator.compound_groups.intact_polar_lipids.archlips_cores import is_archlips_core, archlips_core_smiles

from LipidCalculator.rdkit.plotting import mplt_mol, plt_indices_bond
from LipidCalculator.rdkit.util import add_indices, mol_from_str, get_combined

logger = logging.getLogger(__name__)

IPythonConsole.drawOptions.addAtomIndices = True
IPythonConsole.molSize = 500, 500
IPythonConsole.ipython_useSVG = True

CHAIN_EMPTY_PLACEHOLDER = 'C0:0'
# chains are given as e.g. C16:0, C18:1;(2OH) (hydroxyl group at C2), C24:0;O (one OH, first free position from C2)
CHAIN_RE = re.compile(r'C(\d+):(\d+)((?:;[^;]+)*)')
# atom property marking the attachment point of an empty chain (C0:0), value: carbons of the chain in the piece
EMPTY_CHAIN_PROP = 'empty_chain'
EMPTY_CHAIN_STRICT_PROP = 'empty_chain_strict'


def ceil(a, b):
    return -(a // -b)


def _combine_molecules_at_positions(
        mols_or_combo: Iterable[Mol] | Mol,
        bond_positions: Iterable[tuple[int | str, int | str]],
        bond_orders: Iterable[int] = None
):
    """
    Combine two molecules at given atoms. Can either provide the index
    (from plt_indices_bond) or placeholder atoms (if they are unique)

    following https://asteeves.github.io/blog/2015/01/14/editing-in-rdkit/
    """

    def is_bond_atom(combo: Mol, mol_pos: int | str) -> bool:
        """If pos refers to placeholder, mol_is_bond_atom must be False, otherwise True"""
        atom_type: str = mol_pos if isinstance(mol_pos, str) else combo.GetAtomWithIdx(mol_pos).GetSymbol()
        # if the atom type is one of the placeholders, the molecule position cannot refer to the bond atom
        return not (atom_type in PLACEHOLDER_ELEMENTS)

    def get_bond_atom_for_placeholder(combo: Mol, idx: int) -> Atom:
        """Get the atom that is bonded to the placeholder"""
        placeholder: Atom = combo.GetAtomWithIdx(idx)
        neighbours: list[Atom] = placeholder.GetNeighbors()

        assert len(
            neighbours) == 1, \
            f'placeholder has more than one bond, please specify atom at which to bond by index'
        return neighbours[0]

    def get_placeholder_for_bond_atom(combo: Mol, idx: int) -> Atom | None:
        """Find the placeholder atom for an atom specified by its index"""
        neighbours = combo.GetAtomWithIdx(idx).GetNeighbors()

        placeholders: list[Atom] = []
        for n in neighbours:
            if n.GetSymbol() in PLACEHOLDER_ELEMENTS:
                placeholders.append(n)

        n_placeholders: int = len(placeholders)
        assert n_placeholders <= 1, \
            f'expected one or no placeholder item for atom with index {n.GetIdx()}'
        return placeholders[0] if n_placeholders > 0 else None

    def get_bond_atom(combo: Mol, mol_pos: int | str, idx_is_bond_atom: bool) -> tuple[int, int | None]:
        """Determine indices of bond and placeholder atoms from inputs. Placeholder index may be missing."""
        if isinstance(mol_pos, str):  # placeholder provided by symbol
            assert mol_pos in PLACEHOLDER_ELEMENTS, f'atom {mol_pos} is not a placeholder'
            assert not idx_is_bond_atom, \
                'if atom type is given, can only determine bond atom from a placeholder'
            placeholders: list[Atom] = []
            for atom in combo.GetAtoms():
                if atom.GetSymbol() == mol_pos:
                    placeholders.append(atom)
            assert len(placeholders) == 1, \
                f'expected to find exactly 1 {mol_pos} atom, but found {len(placeholders)}'
            placeholder_idx: int = placeholders[0].GetIdx()
            bond_atom_idx: int = get_bond_atom_for_placeholder(combo, placeholder_idx).GetIdx()
        elif idx_is_bond_atom:  # bond atom given, find placeholder
            bond_atom_idx: int = mol_pos
            placeholder: Atom | None = get_placeholder_for_bond_atom(combo, bond_atom_idx)
            placeholder_idx = placeholder.GetIdx() if isinstance(placeholder, Atom) else None
        else:  # idx of placeholder given, find bond atom
            placeholder_idx: int = mol_pos
            bond_atom_idx: int = get_bond_atom_for_placeholder(combo, placeholder_idx).GetIdx()
        return bond_atom_idx, placeholder_idx

    def combine_at_indices(
            em: EditableMol,
            mol1_idx: int,
            mol2_idx: int,
            mol1_placeholder: int | None,
            mol2_placeholder: int | None
    ) -> None:
        """Add a bond by replacing two placeholder molecules, so
        R1-Fr Rb-R2 --> R1-R2
        """
        if mol1_placeholder is not None:
            placeholders_to_remove.append(mol1_placeholder)
        if mol2_placeholder is not None:
            placeholders_to_remove.append(mol2_placeholder)
        em.AddBond(mol1_idx, mol2_idx, order=rdkit.Chem.rdchem.BondType.SINGLE)

    # Combine molecules
    if not isinstance(mols_or_combo, Mol):
        _combo = get_combined(mols_or_combo)
    else:
        _combo = mols_or_combo

    n_bonds: int = len(bond_positions)

    if bond_orders is None:
        bond_orders = [1] * n_bonds
    else:
        assert len(bond_orders) == n_bonds, 'need one bond order for each bond'

    em = Chem.EditableMol(_combo)
    # form bonds between pairs of provided indices/ atoms
    placeholders_to_remove: list[int] = []
    for bond, bond_order in zip(bond_positions, bond_orders):
        mol1_position, mol2_position = bond
        # attempt to infer whether bond atom or placeholder has been provided
        _mol1_idx_is_bond_atom = is_bond_atom(_combo, mol1_position)
        _mol2_idx_is_bond_atom = is_bond_atom(_combo, mol2_position)

        try:
            _mol1_idx, _mol1_placeholder = get_bond_atom(_combo, mol1_position, _mol1_idx_is_bond_atom)
            _mol2_idx, _mol2_placeholder = get_bond_atom(_combo, mol2_position, _mol2_idx_is_bond_atom)
            # Remove placeholders and form bond
            combine_at_indices(em, _mol1_idx, _mol2_idx, _mol1_placeholder, _mol2_placeholder)
        except AssertionError as _e:
            # supress warning if
            warnings.warn(f'bond between placeholders {mol1_position, mol2_position} could not be formed: {_e}')

    for ph_idx in sorted(placeholders_to_remove, reverse=True):
        em.RemoveAtom(ph_idx)

    _combo = em.GetMol()
    # _combo = Chem.RemoveHs(_combo)

    return _combo


def _get_chain_smiles(chain_length: int, double_bonds: int, hydroxy: Iterable[int] = ()) -> str:
    """
    Generate the smiles for a chain of a certain length with a certain number of double bonds. Spaces double bonds
    equally along the chain. Cs is used as a placeholder. hydroxy are the indices of chain atoms (starting at 0
    for the atom bonded to the placeholder) with an OH group."""
    assert chain_length >= double_bonds, \
        f"Cannot fit {double_bonds} in chain of length {chain_length}"
    positions_double_bonds = np.linspace(0, chain_length, double_bonds + 2)[1:-1]
    idcs_double_bonds = np.round(positions_double_bonds).astype(int) - 1
    hydroxy = set(hydroxy)
    assert all(0 <= i < chain_length for i in hydroxy), f'hydroxyl groups {hydroxy} outside of chain'

    chain = '[Cs]'

    for i in range(chain_length):
        chain += 'C'
        if i in hydroxy:
            chain += '(O)'
        if i in idcs_double_bonds:
            chain += '='

    return chain


def _parse_chain(chain: str) -> tuple[int, int, str]:
    """E.g., 'C18:1;(2OH)' --> (18, 1, ';(2OH)')"""
    match = CHAIN_RE.fullmatch(chain)
    assert match is not None, f'{chain} is not a valid chain (e.g. C18:1, C16:0;(2OH), C24:0;O)'
    return int(match.group(1)), int(match.group(2)), match.group(3)


def _hydroxy_indices(modifications: str, chain_length: int, double_bonds: int, n_core_carbons: int) -> list[int]:
    """Chain atom indices of the hydroxyl groups, the numbering of the modifications includes the carbon atoms of
    the core (C2 of an acyl chain is the first atom of the chain if the core contains the carbonyl carbon)"""
    positions: list[int] = []
    n_unspecified = 0
    for part in modifications.split(';')[1:]:
        if m := re.fullmatch(r'\((\d+)OH\)', part):
            positions.append(int(m.group(1)))
        elif m := re.fullmatch(r'(\d*)O', part):
            n_unspecified += int(m.group(1) or 1)
        else:
            raise ValueError(f'unknown chain modification {part}')
    indices = [pos - 1 - n_core_carbons for pos in positions]
    # unspecified hydroxyl groups at the first free carbon atoms (from C2), avoiding double bonds
    db = set(np.round(np.linspace(0, chain_length, double_bonds + 2)[1:-1]).astype(int) - 1)
    candidates = [i for i in range(max(1 - n_core_carbons, 0), chain_length)
                  if i not in indices and i not in db and i - 1 not in db]
    assert len(candidates) >= n_unspecified, f'chain too short for {n_unspecified} hydroxyl groups'
    return indices + candidates[:n_unspecified]


def _remove_empty_chain_atoms(mol: Mol) -> Mol:
    """For empty chains (C0:0, e.g. lyso lipids, or attachment points without chain), also remove the chain carbon
    contained in the piece (e.g. the carbonyl carbon of an acyl chain), so that it keeps an OH (or NH2) instead of a
    formyl group"""
    marked = [a for a in mol.GetAtoms() if a.HasProp(EMPTY_CHAIN_PROP)]
    if not marked:
        return mol
    rw = Chem.RWMol(mol)
    to_remove: set[int] = set()
    for placeholder in marked:
        n_carbons = placeholder.GetIntProp(EMPTY_CHAIN_PROP)
        if n_carbons == 0:
            continue
        if n_carbons > 1:
            # explicitly requested empty chains must be removable, otherwise only the placeholder is removed
            assert not placeholder.GetBoolProp(EMPTY_CHAIN_STRICT_PROP), \
                f'the chain at {placeholder.GetSymbol()} cannot be empty, the piece contains {n_carbons} of its carbons'
            continue
        carbon = placeholder.GetNeighbors()[0]
        to_remove.add(carbon.GetIdx())
        for bond in carbon.GetBonds():
            other = bond.GetOtherAtom(carbon)
            if other.GetSymbol() == 'O' and bond.GetBondType() == Chem.BondType.DOUBLE:
                to_remove.add(other.GetIdx())
            elif other.GetIdx() != placeholder.GetIdx():
                # gets an H instead of the chain
                rw.GetAtomWithIdx(other.GetIdx()).SetNoImplicit(False)
    for idx in sorted(to_remove, reverse=True):
        rw.RemoveAtom(idx)
    return rw.GetMol()


def remove_placeholder_atoms(
        mol: Mol
) -> Mol:
    """Remove atoms that are being used as placeholders."""
    edmol: Chem.EditableMol = Chem.EditableMol(mol)
    atoms_to_remove = [atom.GetIdx()
                       for atom in mol.GetAtoms()
                       if atom.GetSymbol() in PLACEHOLDER_ELEMENTS]

    # Remove in reverse order to avoid index shift
    for idx in sorted(atoms_to_remove, reverse=True):
        edmol.RemoveAtom(idx)

    clean: Mol = edmol.GetMol()
    clean = Chem.RemoveHs(clean)

    return clean


def connect_ipl_pieces(
        head: str,
        core: str,
        chain1: tuple[int, int] | str | None = None,
        chain2: tuple[int, int] | str | None = None,
        bond_positions: Iterable[tuple[str | int, str | int]] = None,
        plts: bool = False
):
    """Piece together an intact polar lipid from the provided head, core chains and bond positions. Not intended to be called directly."""
    if isinstance(chain1, str):
        chain1 = _parse_chain_str(chain1)
    if isinstance(chain2, str):
        chain2 = _parse_chain_str(chain2)

    if chain1 is not None:
        chain1: str = _get_chain_smiles(*chain1)
    if chain2 is not None:
        chain2: str = _get_chain_smiles(*chain2)

    # turn smiles into molecules
    # print(head, core, chain1, chain2)

    mols: list[Mol] = [mol_from_str(mol) for mol in [head, core, chain1, chain2] if mol is not None]

    # print(mols)

    if bond_positions is None:  # manual input
        bond_positions = get_bonds_interactively(mols)
    # combine
    combined = _combine_molecules_at_positions(
        mols, bond_positions
    )
    clean = remove_placeholder_atoms(combined)
    # clean = combined

    if plts:
        mplt_mol(clean)
        plt.show()

        # fig, axs = plt.subplots()
        # mol = add_numbers(clean.__copy__())
        #
        # # match_head = mol.GetSubstructMatch(head)
        # # match_core = mol.GetSubstructMatch(remove_placeholders(remove_placeholders(core)))
        # # match_chain = mol.GetSubstructMatch(add_chain(Chem.MolFromSmiles('[Cs]'), chain1)[0])
        # # if chain2 is not None:
        # #     match_chain2 = add_chain(Chem.MolFromSmiles('[Cs]'), chain2)
        # #     match_chain = match_chain + match_chain2
        #
        # # add highlights
        # highlight_colors = {}
        # # Assign colors to each part
        # colors = {
        #     'head': (1, 0, 0),  # Red
        #     'core': (0, 1, 0),  # Green
        #     'tail': (0, 0, 1)  # Blue
        # }
        #
        # # for atom_idx in match_head:
        # #     highlight_colors[atom_idx] = colors['head']
        # # # Highlight core atoms
        # # for atom_idx in match_core:
        # #     highlight_colors[atom_idx] = colors['core']
        # # Highlight tail atoms
        # # for atom_idx in match_chain:
        # #     highlight_colors[atom_idx] = colors['tail']
        #
        # drawer = rdMolDraw2D.MolDraw2DCairo(2000, 2000)
        # # drawer.DrawMolecule(
        # #     mol,
        # #     highlightAtoms=sum([match_head, match_core, match_chain], ()),
        # #     highlightAtomColors=highlight_colors
        # # )
        #
        # # drawer.DrawMolecule(
        # #     mol,
        # #     highlightAtoms=match_head
        # #     highlightAtomColors=highlight_colors
        # # )
        #
        # # highlight_atoms = list(sum([match_head, match_core, match_chain], ()))
        # # highlight_atoms = list(sum([match_head, match_core], ()))
        #
        # # drawer.FinishDrawing()
        # rdMolDraw2D.PrepareAndDrawMolecule(
        #     drawer=drawer,
        #     mol=mol,
        #     # highlightAtoms=highlight_atoms,
        #     highlightAtomColors=highlight_colors
        # )
        #
        # drawer.FinishDrawing()
        # img_data = drawer.GetDrawingText()
        # img = Image.open(BytesIO(img_data))
        #
        # # img = np.array(img)
        #
        # # img = Draw.MolToImage(
        # #     mol,
        # #     size=(1000, 1000),
        # #     highlightAtoms=sum([match_head, match_core, match_chain], ()),
        # #     atomColors=highlight_colors,
        # #     bla=1
        # # )
        #
        # axs.imshow(img, interpolation='none')
        # axs.set_axis_off()
        # plt.show()

    return clean


def _parse_chain_str(chain: str | tuple[int, int]) -> tuple[int, int]:
    """E.g., 'C30:2' --> (30, 2), modifications of the chain are ignored"""
    if not isinstance(chain, str):
        assert len(chain) == 2
        return chain
    l, db, _ = _parse_chain(chain)
    return l, db


def get_struct(inpt: str | tuple[int, int]) -> str | None:
    """Input is either chain (str or tuple) or name of an IPL piece"""
    if not isinstance(inpt, str):
        assert len(inpt) == 2
        return _get_chain_smiles(*inpt)
    if CHAIN_RE.fullmatch(inpt):
        l, db, modifications = _parse_chain(inpt)
        return _get_chain_smiles(l, db, _hydroxy_indices(modifications, l, db, 0))
    elif is_block_core(inpt):
        # isoprenoid core build from isoprene units, e.g. GDGT(40:0_40:0(2cp)) or AR(20:0_25:0)
        return block_core_smiles(inpt)
    elif inpt not in ABBREVIATIONS and is_archlips_core(inpt):
        # tetraether core of the ArchLips library, e.g. GDGT-2a
        return archlips_core_smiles(inpt)
    elif inpt not in ABBREVIATIONS:
        warnings.warn(f'structure with name {inpt} not found, returning None')
        return None
    return get_smiles(inpt)


def get_mol_from_abbr(inpt: str, keep_placeholders: bool = False) -> Mol:
    """Get a piece by abbreviation (removes placeholders if placeholders is
    set to False, which is the default)"""
    mol = mol_from_str(get_struct(inpt))
    if not keep_placeholders:
        mol = remove_placeholder_atoms(mol)
    return mol


def get_bonds_interactively(mols: Iterable[Mol]) -> list[tuple[int, int]]:
    """Draw a molecule with atom indices and ask for bonds between indices"""
    plt_indices_bond(mols)
    bond_positions = []
    while True:
        inpt = input('Specify indices with gap inbetween: ')
        if inpt == '':
            break
        if inpt.count(' ') != 1:
            print('indices must be separated by space')
            continue
        p1, p2 = inpt.split(' ')
        if not p1.isnumeric() or not p2.isnumeric():
            print('indices must be whole numbers')
            continue
        bond_positions.append((int(p1), int(p2)))
        print('current bonds:', bond_positions)
    return bond_positions


def correct_chain_length_for_core(
        core: str,
        chain: tuple[int, int],
        core_placeholder: str
) -> tuple[int, int]:
    """the chain length includes C atoms in the backbone, so AEG + C10 results
    in different chain length than BL + C10"""
    key = (core, core_placeholder)
    if key not in CORE_IMPLICIT_CHAIN:
        warnings.warn(f'found no rule for {core=} and {core_placeholder} for chain correction')
        return 0, 0

    correction: tuple[int, int] = CORE_IMPLICIT_CHAIN[key]

    lc: int = chain[0] - correction[0]
    dbc: int = chain[1] - correction[1]
    assert lc >= 0, f'chain has length smaller than 0 after accounting for C atoms in core!'
    assert dbc >= 0, f'chain has less than 0 double bonds after accounting for C atoms in core!'
    return lc, dbc


def _assert_no_peroxide(mol: Mol, names: list[str]) -> None:
    """The core provides the O at its attachment points, head groups bond with their own atom (e.g. P or the anomeric
    C), so O-O bonds point to a piece that does not follow this convention"""
    if mol.HasSubstructMatch(Chem.MolFromSmarts('[O]-[O]')):
        raise ValueError(f'combining {names} gives an O-O bond, head groups must not contain the O they are bonded '
                         f'through (the core provides it)')


def ipl_automatic_bonds(
        names: Iterable[str],
        split_chain: bool = True,
        sort_chains: bool = True,
        plts=False,
        idx_plt: bool = False,
        core_chains: str | None = None,
        core_double_bonds: int | None = None
) -> Mol:
    """
    Attempt to form bonds between pieces automatically by idenitfying piece types and placeholders.

    the core provides the O at its attachment points (Fr), head groups bond to it with their own atom (e.g. P of
    phospholipids or the anomeric C of sugars). Chains are attached to the chain placeholders (Cs, Rb) of the core,
    then of the head groups, the carbons of a chain that are contained in a piece (connected to its placeholder
    through C-C bonds, e.g. the carbonyl carbon of acyl chains) count to the chain length.

    cores with two attachment points (e.g. GDGTs) can take two head groups, the first one is bonded to the
    first attachment point. core_chains sets the isoprenoid chains of the core, e.g. '20:1_25:0' for archaeols,
    core_double_bonds the number of double bonds in the biphytanyl chains of tetraethers (see isoprenoid_chains.py).
    """

    def _check_piece_types(names_) -> tuple[list[str], list[bool], list[bool], int, str]:
        is_chain_: list[bool] = [CHAIN_RE.fullmatch(name) is not None for name in names_]
        grps_: list[str] = [ABBREVIATION_TO_GROUP.get(name,
                                                      'chain' if _is_chain
                                                      else 'core lipids' if is_block_core(name) or is_archlips_core(name)
                                                      else None) for name, _is_chain in zip(names_, is_chain_)]
        assert sum(grp in CORE_GROUPS for grp in grps_) == 1, \
            f'automatic bonding only works with exactly one core unit but got {dict(zip(names_, grps_))}'

        is_core_: list[bool] = [grp in CORE_GROUPS for grp in grps_]
        core_: str = names[is_core_.index(True)]  # always need core, so we can assign it here

        assert None not in grps_, \
            (f'cannot use name(s) {[names_[idx] for idx, grp in enumerate(grps_) if grp is None]} '
             f'because the are not associated with a group')
        assert grps_.count('head groups') + grps_.count('functional groups') <= 2, \
            'automatic bonding only works with at most two functional or head group units'

        return grps_, is_chain_, is_core_, sum(is_chain_), core_

    def _split_chains(chain_: str) -> tuple[str, str]:
        logging.info(f'splitting chain: {chain_}')
        l, db, modifications = _parse_chain(chain_)
        assert not modifications, f'cannot split chain with modifications {chain_}'
        l1, l2 = l // 2, ceil(l, 2)
        db1, db2 = db // 2, ceil(db, 2)

        assert l1 + l2 == l
        assert db1 + db2 == db
        chain1_ = f'C{l1}:{db1}'
        chain2_ = f'C{l2}:{db2}'
        logging.info(f'new chains: {chain1_}, {chain2_}')
        return chain1_, chain2_

    # ensure list and make copy
    names: list[str] = list(names)

    grps, is_chain, is_core, n_chains, core = _check_piece_types(names)

    # pieces (core and head groups) in the order of names, chains are added later
    piece_names: list[str] = [name for name, _is_chain in zip(names, is_chain) if not _is_chain]
    piece_groups: list[str] = [grp for grp, _is_chain in zip(grps, is_chain) if not _is_chain]
    mols: list[Mol] = [mol_from_str(get_struct(name)) for name in piece_names]
    core_piece: int = piece_groups.index(next(g for g in piece_groups if g in CORE_GROUPS))
    head_pieces: list[int] = [i for i, g in enumerate(piece_groups) if g in ('head groups', 'functional groups')]

    # attachment points of chains: core first, then head groups, in the order of the placeholders in the SMILES
    targets: list[tuple[int, int, str]] = [
        (i, atom.GetIdx(), atom.GetSymbol())
        for i in [core_piece] + head_pieces
        for atom in mols[i].GetAtoms() if atom.GetSymbol() in CHAIN_PLACEHOLDERS
    ]

    if split_chain and (n_chains == 1) and (len(targets) >= 2):  # splitting chain
        idx_chain = grps.index('chain')
        chain: str = names[idx_chain]
        chain1, chain2 = _split_chains(chain)
        names.remove(chain)
        names.extend([chain1, chain2])
        # call again with split chains
        return ipl_automatic_bonds(names, split_chain=False, sort_chains=sort_chains, plts=plts,
                                   core_chains=core_chains, core_double_bonds=core_double_bonds)

    chains: list[tuple[int, int, str]] = [_parse_chain(name) for name, _is_chain in zip(names, is_chain) if _is_chain]
    assert len(chains) <= len(targets), \
        f'{"-".join(piece_names)} can take at most {len(targets)} chain(s), got {len(chains)}'
    # some pieces contain parts of the chain (e.g. the carbonyl carbon), the chain lengths include those
    corrections: list[tuple[int, int]] = [implicit_chain(mols[i], idx) for i, idx, _ in targets]
    # chains with modifications are given at their positions and are not sorted
    if sort_chains and (n_chains == 2) and (targets[0][0] == targets[1][0] == core_piece) \
            and not any(modifications for _, _, modifications in chains):
        # longest chain first, combined with the bigger correction
        if chains[1][0] > chains[0][0]:
            chains.reverse()
        if corrections[1][0] > corrections[0][0]:
            targets[:2], corrections[:2] = targets[1::-1], corrections[1::-1]
    logging.info(f'chains {chains} at {targets} with corrections {corrections}')

    def _global(piece: int, idx: int) -> int:
        return sum(m.GetNumAtoms() for m in mols[:piece]) + idx

    # head groups take the free attachment points of the core in order
    bond_pairs: list[tuple[int, int]] = []
    # atom map numbers on the placeholders set the order (cores build from isoprene units), otherwise the SMILES order
    core_attachments: list[int] = [a.GetIdx() for a in sorted(
        (a for a in mols[core_piece].GetAtoms() if a.GetSymbol() == 'Fr'), key=lambda a: a.GetAtomMapNum())]
    for i in head_pieces:
        assert core_attachments, f'core {core} has no free attachment point left for {piece_names[i]}'
        head_attachment = next(a.GetIdx() for a in mols[i].GetAtoms() if a.GetSymbol() == 'Fr')
        bond_pairs.append((_global(i, head_attachment), _global(core_piece, core_attachments.pop(0))))

    # chains
    for k, ((i, idx, _), (n_core_c, n_core_db)) in enumerate(zip(targets, corrections)):
        if k >= len(chains) or chains[k][:2] == (0, 0):
            # no chain: carbons of the chain in the piece are removed later
            atom = mols[i].GetAtomWithIdx(idx)
            atom.SetIntProp(EMPTY_CHAIN_PROP, n_core_c)
            atom.SetBoolProp(EMPTY_CHAIN_STRICT_PROP, k < len(chains))
            continue
        l, db, modifications = chains[k]
        lc, dbc = l - n_core_c, db - n_core_db
        assert lc >= 0, f'chain C{l}:{db} is shorter than the {n_core_c} carbon atom(s) included in {piece_names[i]}'
        assert dbc >= 0, f'chain C{l}:{db} has less double bonds than included in {piece_names[i]}'
        mols.append(mol_from_str(_get_chain_smiles(lc, dbc, _hydroxy_indices(modifications, lc, dbc, n_core_c))))
        chain_placeholder = next(a.GetIdx() for a in mols[-1].GetAtoms() if a.GetSymbol() == 'Cs')
        bond_pairs.append((_global(len(mols) - 1, chain_placeholder), _global(i, idx)))

    if idx_plt:
        _mol = get_combined(mols)
        _mol = add_indices(_mol)
        mplt_mol(_mol)
        plt.show()

    # supress warnings for placeholders
    mol = _combine_molecules_at_positions(mols, bond_positions=bond_pairs)
    mol = _remove_empty_chain_atoms(mol)
    mol = remove_placeholder_atoms(mol)
    _assert_no_peroxide(mol, names)

    if core_chains:
        mol = modify_isoprenoid_chains(mol, core_chains, in_units=core in CHAINS_IN_ISOPRENE_UNITS)
    if core_double_bonds:
        mol = add_tetraether_double_bonds(mol, core_double_bonds)
    # atom map numbers only mark the isoprenoid chains in the pieces
    mol = clear_chain_marks(mol)

    formula = rdMolDescriptors.CalcMolFormula(mol)
    logger.info(f'successfully build molecule with formula {formula}')

    if plts:
        mplt_mol(mol)
        plt.show()

    Chem.SanitizeMol(mol)
    return mol


def interactively_build_molecule(plts: bool = False, always_manual=False) -> Mol:
    """Requires console input"""

    def parse_input(inpt: str):
        """get smiles or inchis"""
        blocks: list[str] = inpt.split(',')
        processed: list[str] = []
        for block in blocks:
            processed.append(get_struct(block))
        return processed

    print('enter your building blocks separated by ","')
    print(f'options: {ABBREVIATIONS}')
    print('to specify a chain, specify the number of C atoms and double bonds like this: "C30:5"')
    structs: str = input("blocks:")
    if structs == '':
        return Mol()

    if not always_manual:
        try:
            return ipl_automatic_bonds(structs.split(','), plts=plts)
        except Exception as e:
            print('automatic structure pairing failed:', e)
            print('enter bonds manually')

    structs: list[str] = parse_input(structs)

    # logging.info(f'fetched structures {structs}')

    mols = [mol_from_str(mol) for mol in structs if (mol is not None) and (mol != '')]
    # print(mols)

    bond_positions = get_bonds_interactively(mols)

    mol = _combine_molecules_at_positions(mols, bond_positions)
    mol = remove_placeholder_atoms(mol)

    formula = rdMolDescriptors.CalcMolFormula(mol)
    logger.info(f'successfully build molecule with formula {formula}')

    if plts:
        mplt_mol(mol)
        plt.show()

    return mol


def test_ipl():
    head_core2pos = {
        ('PC', 'DAG'): (7, 12)
    }

    pc: str = IPL_PIECES['head groups'][0]
    dag: str = IPL_PIECES['core lipids'][0]

    smiles_pc = pc['SMILES']
    smiles_dag = dag['SMILES']

    # plt_indices_bond(head=smiles_pc, core=smiles_dag)

    mol = mol_from_str(smiles_dag)

    # plt_indices_bond([mol])

    clean = PC_DAG_C24d12 = connect_ipl_pieces(
        head=smiles_pc,
        core=smiles_dag,
        chain1=(24, 12),
        plts=True
    )


def test_ipl_automatic():
    ipl_automatic_bonds('1G AR'.split(), plts=True)


if __name__ == '__main__':
    while False:
        try:
            mol = build_molecule(plts=True)
            print(Chem.MolToSmiles(mol))
            print(rdMolDescriptors.CalcMolFormula(mol))
            print(ExactMolWt(mol))
        except:
            pass
    # ipl_automatic_bonds('2G,DAG,C18:0,C20:4'.split(','), plts=True, idx_plt=True)
    # mol = ipl_automatic_bonds('PE,DEG,C10:0,C1:0'.split(','), plts=True, idx_plt=True, split_chain=False)
    # mol = ipl_automatic_bonds('PE,AEG,C0:0,C10:0'.split(','), plts=True, idx_plt=True,
    #                           split_chain=False, sort_chains=False)
    # mol = ipl_automatic_bonds('SQ DAG C22:0 C6:0'.split(), plts=True, idx_plt=False)
    mol = ipl_automatic_bonds('PI DEG C34:2'.split(), plts=False, idx_plt=False, split_chain=True)

    # mplt_mol(mol)

    # from LipidCalculator.compound_groups.intact_polar_lipids.frag_from_alpha_cleavage import add_proton_to_heteroatom
    #
    # add_proton_to_heteroatom(mol)
    #
    print(Chem.MolToSmiles(mol))
    print(ExactMolWt(mol))
    print(rdMolDescriptors.CalcMolFormula(mol))
