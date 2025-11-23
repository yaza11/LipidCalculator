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
import warnings
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import rdkit
import logging
from rdkit import Chem

from rdkit.Chem.rdchem import Mol, Atom, EditableMol
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import IPythonConsole

from LipidCalculator.compound_groups.intact_polar_lipids.pieces_from_json import PLACEHOLDER_ELEMENTS, \
    get_smiles, CORE_IMPLICIT_CHAIN, ABBREVIATION_TO_GROUP, ABBREVIATIONS, IPL_PIECES

from LipidCalculator.rdkit.plotting import mplt_mol, plt_indices_bond
from LipidCalculator.rdkit.util import add_indices, mol_from_str, get_combined

logger = logging.getLogger(__name__)

IPythonConsole.drawOptions.addAtomIndices = True
IPythonConsole.molSize = 500, 500
IPythonConsole.ipython_useSVG = True

CHAIN_EMPTY_PLACEHOLDER = 'C0:0'


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


def _get_chain_smiles(chain_length: int, double_bonds: int) -> str:
    """
    Generate the smiles for a chain of a certain length with a certain number of double bonds. Spaces double bonds
    equally along the chain. Cs is used as a placeholder"""
    assert chain_length >= double_bonds, \
        f"Cannot fit {double_bonds} in chain of length {chain_length}"
    positions_double_bonds = np.linspace(0, chain_length, double_bonds + 2)[1:-1]
    idcs_double_bonds = np.round(positions_double_bonds).astype(int) - 1

    chain = '[Cs]'

    for i in range(chain_length):
        chain += 'C'
        if i in idcs_double_bonds:
            chain += '='

    return chain


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
    """E.g., 'C30:2' --> (30, 2)"""
    if not isinstance(chain, str):
        assert len(chain) == 2
        return chain

    hain = chain[1:]
    assert len(hain.split(':')) == 2
    l, db = hain.split(':')
    return int(l), int(db)


def get_struct(inpt: str | tuple[int, int]) -> str | None:
    """Input is either chain (str or tuple) or name of an IPL piece"""
    if not isinstance(inpt, str):
        assert len(inpt) == 2
        return _get_chain_smiles(*inpt)
    if inpt.startswith('C') and ':' in inpt:
        l, db = _parse_chain_str(inpt)
        return _get_chain_smiles(l, db)
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


def replace_OOP_with_OP(mol):
    rw_mol = Chem.RWMol(mol)

    # Find O–O–P
    match = rw_mol.GetSubstructMatch(Chem.MolFromSmarts("[O]-[O]-[P]"))
    if not match:
        return mol  # no change

    o1_idx, o2_idx, p_idx = match  # first O, middle O, P

    # Delete the middle oxygen
    rw_mol.AddBond(o1_idx, p_idx, Chem.BondType.SINGLE)

    rw_mol.RemoveAtom(o2_idx)

    Chem.SanitizeMol(rw_mol)
    return rw_mol.GetMol()


def ipl_automatic_bonds(
        names: Iterable[str],
        split_chain: bool = True,
        sort_chains: bool = True,
        remove_double_oxygen: bool = True,
        plts=False,
        idx_plt: bool = False
) -> Mol:
    """
    Attempt to form bonds between pieces automatically by idenitfying piece types and placeholders.

    for head groups that contain a phospahte group and are connected to a glycerol backbone, it is convention to
    consider molecules that connect the OH group of the glycerol directly to P instead of having a O-O-P fragment.
    """

    def _check_piece_types(names_) -> tuple[list[str], list[bool], list[bool], int, str]:
        is_chain_: list[bool] = [(name.startswith('C') and ':' in name) for name in names_]
        grps_: list[str] = [ABBREVIATION_TO_GROUP.get(name,
                                                      'chain' if _is_chain
                                                      else None) for name, _is_chain in zip(names_, is_chain_)]
        assert grps_.count('core lipids') == 1, \
            f'automatic bonding only works with exactly one core unit but got {dict(zip(names_, grps_))}'

        is_core_: list[bool] = [grp == 'core lipids' for grp in grps_]
        # assert sum(is_core_) == 1, f'need exactly one core but none of {names_} is'
        core_: str = names[is_core_.index(True)]  # always need core, so we can assign it here

        assert None not in grps_, \
            (f'cannot use name(s) {[names_[idx] for idx, grp in enumerate(grps_) if grp is None]} '
             f'because the are not associated with a group')

        assert (n_chains_ := sum(is_chain_)) <= 2, \
            'automatic bonding only works with at most two chains'
        assert grps_.count('head groups') + grps_.count('functional groups') <= 1, \
            'automatic bonding only works with at most one functional or head group unit'

        return grps_, is_chain_, is_core_, n_chains_, core_

    def _split_chains(chain_: str) -> tuple[str, str]:
        logging.info('splitting chain:', chain_)
        l, db = _parse_chain_str(chain_)
        l1, l2 = l // 2, ceil(l, 2)
        db1, db2 = db // 2, ceil(db, 2)

        assert l1 + l2 == l
        assert db1 + db2 == db
        chain1_ = f'C{l1}:{db1}'
        chain2_ = f'C{l2}:{db2}'
        logging.info('new chains:', chain1_, chain2_)
        return chain1_, chain2_

    def _clean_chains(chains_: list[tuple[int, int]], sort_chains: bool) -> list[str]:
        if len(chains_) == 0:
            return []
        """Could be 0, 1 or 2 chains"""
        # make sure longest chain comes first
        if sort_chains and (n_chains > 1) and (chains_[1][0] > chains_[0][0]):
            chains_.reverse()
        _corrections: list[tuple[int, int]] = [
            CORE_IMPLICIT_CHAIN[(core, 'Rb')], CORE_IMPLICIT_CHAIN[(core, 'Cs')]
        ]
        logging.info(f'correcting {chains_=} for {core=} with {_corrections=}')
        if (n_chains > 1) and (_corrections[1][0] > _corrections[0][0]):
            _corrections.reverse()
        # combine bigger correction with bigger chain
        # if placeholder is used "(0,0)", stay at 0,0
        chains_new_: list[str] = [
            f'C{_chain[0] - _correction[0]}:{_chain[1] - _correction[1]}'
            if _chain != (0, 0) else CHAIN_EMPTY_PLACEHOLDER
            for _chain, _correction in zip(chains_, _corrections)
        ]

        logging.info(f'chains after correction: {chains_new_}')
        return chains_new_

    def _find_bonds_placeholders(piece_groups: list[str], pieces: list[Mol], is_empty_chain_placeholder: list[bool]):
        # need to know indices of placeholders
        # indices of atoms in subsequent molecules are shifted by number of atoms
        # in preceding molecules
        # match Cs, Rb to Cs of chains and Fr to Fr
        # connections = [{'Cs', 'Cs'}, {'Cs', 'Rb'}, {'Fr', 'Fr'}]

        # gather placeholder indicies
        placeholder_indices: list[int] = []
        placeholder_symbols: list[str] = []
        placeholder_mol: list[int] = []
        placeholder_group: list[str] = []
        atom_count: int = 0
        for mol_idx, (grp, piece) in enumerate(zip(piece_groups, pieces)):
            for atom in piece.GetAtoms():
                if (s := atom.GetSymbol()) in PLACEHOLDER_ELEMENTS:
                    placeholder_indices.append(atom_count)
                    placeholder_symbols.append(s)
                    placeholder_mol.append(mol_idx)
                    placeholder_group.append(grp)
                atom_count += 1

        # pair non-core groups to core
        placeholder_indices: np.ndarray[int] = np.array(placeholder_indices)
        placeholder_symbols: np.ndarray[str] = np.array(placeholder_symbols)
        placeholder_mol: np.ndarray[int] = np.array(placeholder_mol)
        placeholder_group: np.ndarray[str] = np.array(placeholder_group)
        placeholder_taken: np.ndarray[bool] = np.zeros_like(placeholder_indices, dtype=bool)

        # print(placeholder_indices)
        # print(placeholder_symbols)
        # print(placeholder_mol)
        # print(placeholder_group)

        bond_pairs_: list[tuple[int, int]] = []
        # print(grps)
        for mol_idx, (group, is_empty) in enumerate(zip(grps, is_empty_chain_placeholder)):
            if group == 'core lipids':
                continue
            elif group in ('head groups', 'functional groups'):
                mask_fr_head = (placeholder_mol == mol_idx) & (placeholder_symbols == 'Fr')
                idx_fr_head: int = int(placeholder_indices[mask_fr_head][0])

                mask_fr_core = (placeholder_group == 'core lipids') & (placeholder_symbols == 'Fr')
                idx_fr_core: int = int(placeholder_indices[mask_fr_core][0])

                bond_pairs_.append((idx_fr_head, idx_fr_core))
            elif group == 'chain':
                mask_chain = (placeholder_mol == mol_idx) & (placeholder_symbols == 'Cs')
                idx_chain: int = int(placeholder_indices[mask_chain][0])

                mask_core = (
                        (placeholder_group == 'core lipids')
                        & (~placeholder_taken)
                        & ((placeholder_symbols == 'Cs') | (placeholder_symbols == 'Rb'))
                )
                idx_core = int(placeholder_indices[mask_core][0])
                placeholder_taken[np.argwhere(mask_core)[0, 0]] = True

                if not is_empty:
                    bond_pairs_.append((idx_chain, idx_core))
            else:
                raise ValueError(f'group {group} not recognized')

            logging.info(f'determined bonds between {bond_pairs_}')
        return bond_pairs_

    # ensure list and make copy
    names: list[str] = list(names)

    assert len(names) <= 4, \
        'can automatically bond 1 core, at most two chains, at most one functional or head group'

    grps, is_chain, is_core, n_chains, core = _check_piece_types(names)
    # print(dict(zip(names, grps)), dict(zip(names, is_chain)))

    if split_chain and (n_chains == 1):  # splitting chain
        idx_chain = grps.index('chain')
        chain: str = names[idx_chain]
        chain1, chain2 = _split_chains(chain)
        names.remove(chain)
        names.extend([chain1, chain2])
        # call again with split chains
        return ipl_automatic_bonds(names, split_chain=False, plts=plts)

    # core chain correction: some core pieces contain parts of the chain
    chains_parsed: list[tuple[int, int]] = [
        _parse_chain_str(name)
        for name, _is_chain in zip(names, is_chain)
        if _is_chain
    ]
    chains: list[str] = _clean_chains(chains_parsed, sort_chains=sort_chains)

    # check that number of chains is compatible with core
    if any(is_chain):
        if core in ('DGTS', 'BL', 'OL'):
            assert sum(is_chain) <= 1, f'can have only 1 chain for core {core}, got {names}'
        else:
            assert sum(is_chain) <= 2, \
                'can only do chain length correction for two chains, split first'

    # insert into names
    names_new = []
    for name, _is_chain in zip(names, is_chain):
        if _is_chain:
            names_new.append(chains.pop(0))
        else:
            names_new.append(name)
    names = names_new
    is_chain_placeholder: list[bool] = [name == CHAIN_EMPTY_PLACEHOLDER for name in names]

    mols: list[Mol] = [mol_from_str(get_struct(name)) for name in names]

    if idx_plt:
        _mol = get_combined(mols)
        _mol = add_indices(_mol)
        mplt_mol(_mol)
        plt.show()

    bond_pairs = _find_bonds_placeholders(grps, mols, is_chain_placeholder)

    # supress warnings for placeholders
    mol = _combine_molecules_at_positions(mols, bond_positions=bond_pairs)
    mol = remove_placeholder_atoms(mol)
    if remove_double_oxygen:
        mol = replace_OOP_with_OP(mol)

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
    mol = ipl_automatic_bonds('1G DAG C33:1'.split(), plts=False, idx_plt=False, split_chain=True)

    mplt_mol(mol)

    # from LipidCalculator.compound_groups.intact_polar_lipids.frag_from_alpha_cleavage import add_proton_to_heteroatom
    #
    # add_proton_to_heteroatom(mol)
    #
    # print(Chem.MolToSmiles(mol))
    # print(ExactMolWt(mol))
    # print(rdMolDescriptors.CalcMolFormula(mol))
