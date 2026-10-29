"""GUI for constructing intact polar lipids (IPLs) from head groups, cores and chains."""
import re
import warnings
import tkinter as tk
from functools import lru_cache
from itertools import product
from tkinter import ttk, filedialog, messagebox

import pandas as pd
from PIL import ImageTk
from rdkit import Chem
from rdkit.Chem import Mol, rdMolDescriptors
from rdkit.Chem.Descriptors import ExactMolWt

from LipidCalculator import Adduct
from LipidCalculator.compound_groups.intact_polar_lipids.generate_ipl import ipl_automatic_bonds, get_struct, \
    remove_placeholder_atoms
from LipidCalculator.compound_groups.intact_polar_lipids.isoprenoid_chains import describe_isoprenoid_chains, \
    has_biphytanyl_chains, n_hydroxy_positions, n_methyl_positions
from LipidCalculator.compound_groups.intact_polar_lipids.isoprenoid_blocks import (
    CORE_TYPES, DEFAULT_CARBONS, ISOPRENE_CARBONS, ChainBlock, format_block_core, is_block_core
)
from LipidCalculator.compound_groups.intact_polar_lipids.pieces_from_json import (
    DATA_FRAME_IPL_PIECES, CORE_IMPLICIT_CHAIN, CORE_GROUPS, CHAINS_IN_ISOPRENE_UNITS, ABBREVIATIONS
)
from LipidCalculator.rdkit.depiction import DepictionOptions, draw as draw_lipid
from LipidCalculator.rdkit.util import mol_from_str
from LipidCalculator.compound_groups.intact_polar_lipids.archlips_cores import (
    CORE_NAME, TETRAETHER_FAMILIES, archlips_cores, is_archlips_core
)
from LipidCalculator.nomenclature import Lipid, standard_name
from LipidCalculator.nomenclature.archlips import TETRAETHER_PIECE
from LipidCalculator.nomenclature.blocks import ipl_name
from LipidCalculator.gui.common import (
    PAD, ADDUCTS, ResultTable, SearchCombobox, SendTargets, add_labeled, hill_formula, run_in_thread, send_buttons,
    show_error
)

NO_HEAD: str = '(none)'
# structures that can be drawn in the single IPL tab
VIEWS: list[str] = ['Molecule', 'Head group', 'Head group 2', 'Core']
SEP: str = ' – '
DEFAULT_BATCH_ADDUCTS: list[str] = ['[M+H]+', '[M+NH4]+', '[M+Na]+']


def _piece_labels(groups: list[str]) -> list[str]:
    df = DATA_FRAME_IPL_PIECES[DATA_FRAME_IPL_PIECES.group.isin(groups)]
    return [abbr if pd.isna(name) or name == abbr else f'{abbr}{SEP}{name}'
            for abbr, name in zip(df.abbreviation, df.name)]


HEAD_LABELS: list[str] = [NO_HEAD] + _piece_labels(['head groups', 'functional groups'])
CORE_PIECES: list[str] = DATA_FRAME_IPL_PIECES[DATA_FRAME_IPL_PIECES.group == 'core lipids'].abbreviation.tolist()


def _core_labels(groups: list[str] = CORE_GROUPS) -> list[str]:
    """Cores (and quinones, carotenoids), tetraethers by family (the number of rings and isomer are chosen
    separately)"""
    out = []
    for label in _piece_labels(groups):
        tetraether = tetraether_of(abbreviation(label))
        if tetraether is None:
            out.append(label)
        elif (family_label := f'{tetraether[0]}{SEP}{CORE_TYPES[tetraether[0]].description}') not in out:
            out.append(family_label)
    return out


def abbreviation(label: str) -> str | None:
    abbr = label.split(SEP)[0]
    return None if abbr == NO_HEAD else abbr


def takes_chains(piece: str) -> bool:
    """Whether chains can be attached to the core or head group"""
    return any((piece, placeholder) in CORE_IMPLICIT_CHAIN for placeholder in ('Cs', 'Rb'))


@lru_cache(maxsize=None)
def isoprenoid_chains(core: str) -> str:
    """Isoprenoid chains of a core piece, e.g. '20:0_20:0', empty if it has none"""
    return describe_isoprenoid_chains(mol_from_str(get_struct(core)), in_units=core in CHAINS_IN_ISOPRENE_UNITS)


@lru_cache(maxsize=None)
def takes_core_double_bonds(core: str) -> bool:
    """Whether the core has biphytanyl chains (tetraethers) that can be unsaturated"""
    return has_biphytanyl_chains(mol_from_str(get_struct(core)))


@lru_cache(maxsize=None)
def max_core_hydroxy(core: str) -> int:
    """Number of hydroxyl groups the rule can add to the core (C3 of its ether-bound isoprenoid chains)"""
    return n_hydroxy_positions(mol_from_str(get_struct(core)))


@lru_cache(maxsize=None)
def max_core_methyls(core: str) -> int:
    """Number of methyl groups the rule can add to the core (C13 of its biphytanyl chains)"""
    return n_methyl_positions(mol_from_str(get_struct(core)))


def _rings(rings: str) -> str:
    """Number of rings, crenarchaeols (4 cyclopentane and 1 cyclohexane rings) have 5"""
    return '5' if rings.lower() == 'cren' else rings


@lru_cache(maxsize=None)
def tetraether_isomers() -> dict[tuple[str, str], list[str]]:
    """(family, rings) --> unmodified ArchLips cores (the regioisomers) and core pieces with other structures (e.g.
    GDGT5' with a parallel glycerol), e.g. ('GDGT', '5') --> [GDGT-5a, GDGT-5b, GDGT-Cren, GDGT5, GDGT5']"""
    out: dict[tuple[str, str], list[str]] = {}
    for name in archlips_cores():
        m = CORE_NAME.fullmatch(name)
        if m and not (m['uns'] or m['homo'] or m['oh']):
            out.setdefault((m['family'], _rings(m['rings'])), []).append(name)
    out = {key: sorted(names) for key, names in out.items()}
    for piece in CORE_PIECES:
        if (key := tetraether_of(piece)) is None:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mol = ipl_automatic_bonds([piece])
        Chem.RemoveStereochemistry(mol)
        if Chem.MolToSmiles(mol) not in {archlips_cores()[name] for name in out.get(key, []) if is_archlips_core(name)}:
            out.setdefault(key, []).append(piece)
    return out


def tetraether_of(core: str) -> tuple[str, str] | None:
    """(family, rings) of an unmodified tetraether core, e.g. GDGT2 --> (GDGT, 2), GDGT-Cren --> (GDGT, 5)"""
    if is_archlips_core(core):
        m = CORE_NAME.fullmatch(core)
        if m['uns'] or m['homo'] or m['oh']:
            return None
        return m['family'], _rings(m['rings'])
    if m := TETRAETHER_PIECE.fullmatch(core):
        return m['family'], _rings(m['rings'])
    return None


def tetraether_piece(family: str, rings: str) -> str | None:
    """Core piece of package.json for a tetraether family and number of rings"""
    for piece in (f'{family}{rings}', f'{family}-Cren' if rings == '5' else None):
        if piece in ABBREVIATIONS:
            return piece
    return None


def tetraether_rings(family: str) -> list[str]:
    """Numbers of rings of a tetraether family with an ArchLips core or a core piece"""
    rings = {r for f, r in tetraether_isomers() if f == family}
    rings |= {r for r in map(str, range(9)) if tetraether_piece(family, r)}
    return sorted(rings, key=int)


CORE_LABELS: list[str] = _core_labels()
BATCH_CORE_LABELS: list[str] = _core_labels(['core lipids'])


def parse_chains(chains: str) -> list[str]:
    """'16:0_18:1' or '18:1/16:0;(2OH)' --> ['C16:0', 'C18:1'] or ['C18:1', 'C16:0;(2OH)'], empty --> []"""
    out = []
    for chain in re.split(r'[_/]', chains.strip()) if chains.strip() else []:
        match = re.fullmatch(r'C?(\d+):(\d+)((?:;(?:\(\d+OH\)|\d*O))*)', chain.strip())
        if match is None:
            raise ValueError(f'chain "{chain}" must have the form 34:1 (carbon atoms:double bonds), optionally '
                             f'with hydroxyl groups like 16:0;(2OH) or 24:0;O')
        out.append(f'C{match.group(1)}:{match.group(2)}{match.group(3)}')
    return out


def build_ipl(
        head: str | None,
        core: str,
        chains: list[str],
        split_chain: bool = True,
        sort_chains: bool = True,
        head2: str | None = None,
        core_chains: str | None = None,
        core_double_bonds: int = 0,
        core_hydroxy: int = 0,
        core_methyls: int = 0
) -> Mol:
    """Build an IPL, chains like C16:0 or C16:0;(2OH), chain lengths include the C atoms of the core (lipid
    shorthand), chains are attached to the core first, then to head groups with chains. head2 is bonded to the
    second attachment point of the core (e.g. other glycerol of GDGTs), core_chains sets the isoprenoid chains
    of the core (e.g. '20:1_25:0'), core_double_bonds the double bonds in the biphytanyl chains of tetraethers,
    core_hydroxy the hydroxyl groups at C3 of the isoprenoid chains, core_methyls the additional methyl groups at C13
    of the biphytanyl chains. The core can also be build from isoprene units, e.g. GDGT(40:0_40:0(2cp)) or
    AR(20:0_25:0)."""
    if is_block_core(core):
        if chains or core_chains or core_double_bonds:
            raise ValueError('cores build from isoprene units take no chains, set them in the core instead')
    elif core not in ABBREVIATIONS and not is_archlips_core(core):
        raise ValueError(f'unknown core {core}')
    for head_ in (head, head2):
        if head_ and head_ not in ABBREVIATIONS:
            raise ValueError(f'unknown head group {head_}')
    if chains and not any(takes_chains(p) for p in (head, head2, core) if p):
        raise ValueError(f'{core} cannot be connected to chains')
    if core_chains and not isoprenoid_chains(core):
        raise ValueError(f'core {core} has no isoprenoid chains')
    if core_double_bonds and not takes_core_double_bonds(core):
        raise ValueError(f'core {core} has no biphytanyl chains')
    names = [h for h in (head, head2) if h] + [core] + list(chains)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return ipl_automatic_bonds(
            names,
            split_chain=split_chain and len(chains) == 1,
            sort_chains=sort_chains,
            core_chains=core_chains or None,
            core_double_bonds=core_double_bonds or None,
            core_hydroxy=core_hydroxy or None,
            core_methyls=core_methyls or None
        )


def display_name(head: str | None, core: str, chains: list[str], head2: str | None = None,
                 core_chains: str | None = None, core_double_bonds: int = 0, core_hydroxy: int = 0,
                 core_methyls: int = 0) -> str:
    """Standardized name (ArchLips for archaeal lipids, LIPID MAPS shorthand for others), the LipidCalculator name of
    the building blocks if there is none"""
    modifications = dict(core_chains=core_chains, core_double_bonds=core_double_bonds, core_hydroxy=core_hydroxy,
                         core_methyls=core_methyls)
    if head or not head2:
        lipid = Lipid(tuple(h for h in (head, head2) if h), core, tuple(chains), **modifications)
        try:
            return standard_name(lipid)
        except Exception:
            pass
    return ipl_name(head, core, chains, head2=head2, **modifications)


class _StructureFrame(ttk.Frame):
    """Shows a built structure with its properties and adduct m/z values (subclasses build the inputs)."""

    def __init__(self, master, **kwargs):
        super().__init__(master, padding=PAD, **kwargs)
        self.name_var = tk.StringVar(self, value='')
        self.formula_var = tk.StringVar(self, value='')
        self.mass_var = tk.StringVar(self, value='')
        self.smiles_var = tk.StringVar(self, value='')
        self.mol: Mol | None = None
        self._img = None
        self._photo = None
        # errors of the automatic build (e.g. while typing)
        self.status_var = tk.StringVar(self, value='')
        self._pending_build: str | None = None
        # drawing options (see LipidCalculator/rdkit/depiction.py)
        self.straight_var = tk.BooleanVar(self, value=True)
        self.hide_heads_var = tk.BooleanVar(self, value=False)
        self.shorten_var = tk.BooleanVar(self, value=False)
        self.shorten_n_var = tk.IntVar(self, value=4)
        self.fixed_scale_var = tk.BooleanVar(self, value=False)
        for var in (self.straight_var, self.hide_heads_var, self.shorten_var, self.shorten_n_var,
                    self.fixed_scale_var):
            var.trace_add('write', lambda *_: self._draw())

    def build(self) -> None:
        raise NotImplementedError

    def _init_auto_build(self, variables: list[tk.Variable]) -> None:
        """Rebuild the molecule shortly after any of the variables changes"""
        for var in variables:
            var.trace_add('write', lambda *_: self._schedule_build())
        self._schedule_build(delay=0)

    def _schedule_build(self, delay: int = 300) -> None:
        """Build after delay ms without further changes"""
        if self._pending_build is not None:
            self.after_cancel(self._pending_build)
        self._pending_build = self.after(delay, self._auto_build)

    def _cancel_build(self) -> None:
        if self._pending_build is not None:
            self.after_cancel(self._pending_build)
            self._pending_build = None

    def _auto_build(self) -> None:
        self._pending_build = None
        self.build()

    def _build_status(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, textvariable=self.status_var, foreground='#b00020', justify='left',
                  wraplength=360).pack(anchor='w', pady=(0, PAD))

    def _build_properties(self, left: ttk.Frame) -> None:
        props = ttk.LabelFrame(left, text='Properties', padding=PAD)
        props.pack(fill='x')
        for row, (label, var) in enumerate([('Name', self.name_var), ('Formula', self.formula_var),
                                            ('Monoisotopic mass', self.mass_var), ('SMILES', self.smiles_var)]):
            add_labeled(props, row, label, ttk.Entry(props, textvariable=var, state='readonly', width=34))
        btns = ttk.Frame(props)
        btns.grid(row=4, column=0, columnspan=2, sticky='w', pady=PAD)
        ttk.Button(btns, text='Copy SMILES', command=self._copy_smiles).pack(side='left')
        ttk.Button(btns, text='Save image…', command=self._save_image).pack(side='left', padx=PAD)
        ttk.Button(btns, text='Save MOL…', command=self._save_mol).pack(side='left')

    def _build_output(self, send_targets: SendTargets | None) -> None:
        right = ttk.PanedWindow(self, orient='vertical')
        right.grid(row=0, column=1, sticky='nsew')
        plot = ttk.Frame(right)
        toolbar = ttk.Frame(plot)
        toolbar.pack(fill='x', pady=(0, PAD))
        for text, var in (('Straight chains', self.straight_var), ('Hide head groups', self.hide_heads_var),
                          ('Shorten chains to', self.shorten_var)):
            ttk.Checkbutton(toolbar, text=text, variable=var).pack(side='left', padx=(0, PAD))
        ttk.Spinbox(toolbar, from_=1, to=30, textvariable=self.shorten_n_var, width=3).pack(side='left')
        ttk.Label(toolbar, text='C').pack(side='left', padx=(2, PAD * 2))
        ttk.Checkbutton(toolbar, text='Fixed scale', variable=self.fixed_scale_var).pack(side='left')
        self.canvas = tk.Canvas(plot, background='white', highlightthickness=0, height=320)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda _: self._show_image())
        right.add(plot, weight=3)
        self.table = ResultTable(right, ['Adduct', 'm/z'], height=6)
        right.add(self.table, weight=1)
        send_buttons(self.table.toolbar, send_targets, self._payload).pack(side='right')

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

    def depiction_options(self, pieces: bool = False) -> DepictionOptions:
        """Drawing options of the toolbar, building blocks have no head groups to hide"""
        try:
            n_keep = int(self.shorten_n_var.get()) if self.shorten_var.get() else 0
        except (tk.TclError, ValueError):
            n_keep = 0
        return DepictionOptions(straight_chains=self.straight_var.get(),
                                hide_heads=self.hide_heads_var.get() and not pieces,
                                shorten_chains=max(n_keep, 0), fixed_scale=self.fixed_scale_var.get())

    def _render(self, mol: Mol, pieces: bool = False):
        return draw_lipid(mol, self.depiction_options(pieces), size=(1200, 675))

    def _draw(self) -> None:
        """Draw the molecule with the options of the toolbar"""
        if self.mol is None:
            return
        self._img = self._render(self.mol)
        self._show_image()

    def show_mol(self, mol: Mol, name: str) -> None:
        self.mol = mol
        mass = ExactMolWt(mol)
        self.name_var.set(name)
        self.formula_var.set(rdMolDescriptors.CalcMolFormula(mol))
        self.mass_var.set(f'{mass:.5f}')
        self.smiles_var.set(Chem.MolToSmiles(mol))
        self.table.set_rows([(a, f'{Adduct(a).mass_to_mz(mass):.5f}') for a in ADDUCTS])
        self._draw()

    def _show_image(self) -> None:
        if self._img is None:
            return
        w, h = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        img = self._img.copy()
        img.thumbnail((w, h))
        self._photo = ImageTk.PhotoImage(img, master=self.canvas)
        self.canvas.delete('all')
        self.canvas.create_image(w // 2, h // 2, image=self._photo, anchor='center')

    def _payload(self) -> dict | None:
        if self.mol is None:
            show_error(self, 'build an IPL first')
            return None
        row = self.table.selected_row()
        # formula from rdkit may contain charges (e.g. PC is zwitterionic), so remove them
        formula = hill_formula(re.sub(r'[+-]\d*$', '', self.formula_var.get()))
        return dict(formula=formula) if row is None else dict(formula=formula, adduct=row['Adduct'])

    def _copy_smiles(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self.smiles_var.get())

    def _save_image(self) -> None:
        if self._img is None:
            return
        path = filedialog.asksaveasfilename(parent=self.winfo_toplevel(), defaultextension='.png',
                                            initialfile=self.name_var.get().replace('/', '_').replace(':', '-'),
                                            filetypes=[('PNG', '*.png')])
        if path:
            self._img.save(path)

    def _save_mol(self) -> None:
        if self.mol is None:
            return
        path = filedialog.asksaveasfilename(parent=self.winfo_toplevel(), defaultextension='.mol',
                                            initialfile=self.name_var.get().replace('/', '_').replace(':', '-'),
                                            filetypes=[('MOL file', '*.mol')])
        if path:
            Chem.MolToMolFile(self.mol, path)


class SingleIPLFrame(_StructureFrame):
    """Build one IPL and show its structure, formula and adduct m/z values."""

    def __init__(self, master, send_targets: SendTargets = None, **kwargs):
        super().__init__(master, **kwargs)
        self.head_var = tk.StringVar(self, value=next(l for l in HEAD_LABELS if l.startswith('PC')))
        self.core_var = tk.StringVar(self, value=next(l for l in CORE_LABELS if l.startswith('DAG')))
        self.chains_var = tk.StringVar(self, value='34:1')
        self.head2_var = tk.StringVar(self, value=NO_HEAD)
        self.core_chains_var = tk.StringVar(self, value='')
        self.core_chains_hint_var = tk.StringVar(self, value='')
        self.core_db_var = tk.IntVar(self, value=0)
        self.core_oh_var = tk.IntVar(self, value=0)
        self.core_me_var = tk.IntVar(self, value=0)
        # number of rings and regioisomer of tetraether cores
        self.rings_var = tk.StringVar(self, value='')
        self.isomer_var = tk.StringVar(self, value='')
        self._tetraether_family: str | None = None
        # drawn structure: the molecule or one of its building blocks
        self.view_var = tk.StringVar(self, value=VIEWS[0])
        self.placeholders_var = tk.BooleanVar(self, value=True)
        self.split_var = tk.BooleanVar(self, value=True)
        self.sort_var = tk.BooleanVar(self, value=True)
        # core whose fields are shown (a new core resets them)
        self._state_core: str | None = None
        self._build(send_targets)
        self._init_auto_build([self.head_var, self.head2_var, self.core_var, self.chains_var, self.core_chains_var,
                               self.core_db_var, self.core_oh_var, self.core_me_var, self.isomer_var, self.split_var,
                               self.sort_var])

    def _build(self, send_targets: SendTargets | None) -> None:
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky='nsw', padx=(0, PAD * 2))

        inputs = ttk.LabelFrame(left, text='Building blocks', padding=PAD)
        inputs.pack(fill='x')
        # typing shows the matching entries
        for row, (label, var) in enumerate([('Head group', self.head_var), ('Head group 2', self.head2_var)]):
            add_labeled(inputs, row, label, SearchCombobox(inputs, textvariable=var, values=HEAD_LABELS, width=34))
        self.core_cb = add_labeled(inputs, 2, 'Core', SearchCombobox(inputs, textvariable=self.core_var,
                                                                     values=CORE_LABELS, width=34))
        self.core_chains_entry = add_labeled(inputs, 3, 'Isoprenoid chains',
                                             ttk.Entry(inputs, textvariable=self.core_chains_var))
        ttk.Label(inputs, textvariable=self.core_chains_hint_var, foreground='gray', justify='left',
                  wraplength=320).grid(row=4, column=0, columnspan=2, sticky='w', padx=PAD)
        # modifications of the isoprenoid chains of the core, placed by rules (see isoprenoid_chains.py)
        modifications = ttk.Frame(inputs)
        modifications.grid(row=5, column=0, columnspan=2, sticky='w')
        self.core_db_entry = add_labeled(modifications, 0, 'Double bonds',
                                         ttk.Spinbox(modifications, from_=0, to=8, textvariable=self.core_db_var,
                                                     width=4))
        self.core_oh_entry = add_labeled(modifications, 0, 'OH', column=2,
                                         widget=ttk.Spinbox(modifications, from_=0, to=4, textvariable=self.core_oh_var,
                                                            width=4))
        self.core_me_entry = add_labeled(modifications, 0, 'Me', column=4,
                                         widget=ttk.Spinbox(modifications, from_=0, to=4, textvariable=self.core_me_var,
                                                            width=4))
        self.rings_cb = add_labeled(modifications, 1, 'Rings',
                                    ttk.Combobox(modifications, textvariable=self.rings_var, state='readonly', width=5))
        self.isomer_cb = add_labeled(modifications, 1, 'Isomer', column=2,
                                     widget=ttk.Combobox(modifications, textvariable=self.isomer_var, state='readonly',
                                                         width=14))
        self.isomer_cb.grid_configure(columnspan=3, sticky='w')
        self.rings_cb.bind('<<ComboboxSelected>>', lambda _: self._update_isomers())
        self.chains_entry = add_labeled(inputs, 6, 'Chains (C:DB)', ttk.Entry(inputs, textvariable=self.chains_var))
        ttk.Label(
            inputs, foreground='gray', justify='left', wraplength=320,
            text='Chains separated by _ or /, e.g. 16:0_18:1 or 18:1/16:0;(2OH) (hydroxyl groups: ;(2OH), ;O). '
                 'Chain lengths include the C atoms of the core. A single total composition (e.g. 34:1) is split '
                 'between two chains. Chains go to the core first, then to head groups with chains (e.g. Acyl for '
                 'TG). Head group 2 is bonded to the second attachment point of the core (other glycerol of '
                 'tetraethers, 3-OH of ceramides). Isoprenoid cores take double bonds (Δ2, Δ6, ... of the '
                 'biphytanyl chains), hydroxyl groups (C3, sn-2 chains first) and additional methyl groups (C13 of '
                 'the biphytanyl chains), e.g. OH-GDGT-0 or Homo-Me-GDGT-0. Tetraethers can be switched between '
                 'numbers of rings and the regioisomers of the ArchLips library (e.g. GDGT-2a, GDGT-2b).'
        ).grid(row=7, column=0, columnspan=2, sticky='w', padx=PAD)

        opts = ttk.LabelFrame(left, text='Options', padding=PAD)
        opts.pack(fill='x', pady=PAD)
        # options of the chains, disabled together with the chains field
        self.chain_options: list[ttk.Checkbutton] = [
            ttk.Checkbutton(opts, text='Split single chain into two', variable=self.split_var),
            ttk.Checkbutton(opts, text='Longest chain first', variable=self.sort_var),
        ]
        for check in self.chain_options:
            check.pack(anchor='w')
        draw = ttk.Frame(opts)
        draw.pack(anchor='w', pady=(PAD, 0))
        ttk.Label(draw, text='Draw').pack(side='left')
        view_cb = ttk.Combobox(draw, textvariable=self.view_var, values=VIEWS, state='readonly', width=13)
        view_cb.pack(side='left', padx=PAD)
        view_cb.bind('<<ComboboxSelected>>', lambda _: self._draw())
        ttk.Checkbutton(draw, text='Placeholders', variable=self.placeholders_var,
                        command=self._draw).pack(side='left')
        self._build_status(left)
        self._build_properties(left)
        self._build_output(send_targets)

    def _auto_build(self) -> None:
        self._pending_build = None
        # a new core resets its fields (number of rings, modifications, ...), which schedules another build
        self._update_chain_state(keep_values=abbreviation(self.core_var.get()) == self._state_core)
        self._cancel_build()
        self.build()

    def _update_chain_state(self, keep_values: bool = False) -> None:
        core = abbreviation(self.core_var.get())
        pieces = [core, abbreviation(self.head_var.get()), abbreviation(self.head2_var.get())]
        takes = ['!disabled'] if any(p and takes_chains(p) for p in pieces) else ['disabled']
        for widget in (self.chains_entry, *self.chain_options):
            widget.state(takes)
        if keep_values:
            return
        self._state_core = core
        # tetraethers are selected by family (or a piece of a family), the number of rings and the isomer
        if core in TETRAETHER_FAMILIES:
            tetraether = (core, '0' if '0' in tetraether_rings(core) else tetraether_rings(core)[0])
        else:
            tetraether = tetraether_of(core) if core in ABBREVIATIONS else None
        self._tetraether_family = tetraether[0] if tetraether else None
        self.rings_cb.configure(values=tetraether_rings(tetraether[0]) if tetraether else [])
        self.rings_var.set(tetraether[1] if tetraether else '')
        self.rings_cb.state(['!disabled'] if tetraether else ['disabled'])
        self._update_isomers()
        core = self._core()
        known = core in ABBREVIATIONS or is_archlips_core(core)
        self.core_db_entry.state(['!disabled'] if known and takes_core_double_bonds(core) else ['disabled'])
        self.core_db_var.set(0)
        for entry, var, maximum in ((self.core_oh_entry, self.core_oh_var, max_core_hydroxy),
                                    (self.core_me_entry, self.core_me_var, max_core_methyls)):
            n = maximum(core) if known else 0
            entry.configure(to=max(n, 1))
            entry.state(['!disabled'] if n else ['disabled'])
            var.set(0)
        chains = isoprenoid_chains(core) if known else ''
        self.core_chains_var.set(chains)
        self.core_chains_entry.state(['!disabled'] if chains else ['disabled'])
        if not chains:
            self.core_chains_hint_var.set('')
        elif core in CHAINS_IN_ISOPRENE_UNITS:
            self.core_chains_hint_var.set('Side chain as isoprene units:double bonds, e.g. 7:7')
        else:
            self.core_chains_hint_var.set('Per chain C atoms:double bonds (+ (OH)/(Me) to identify modified '
                                          'chains), e.g. 20:1_25:0. Double bonds start at Δ2, 0:0 removes a '
                                          'chain.')

    def _draw(self) -> None:
        """Draw the molecule or the selected building block (with or without its placeholders: Fr for the bonds to
        head groups, Cs and Rb for chains)"""
        view = self.view_var.get()
        if view == VIEWS[0]:
            if self.mol is None:
                return
            mol = self.mol
        else:
            piece = {'Head group': abbreviation(self.head_var.get()), 'Head group 2': abbreviation(self.head2_var.get()),
                     'Core': self._core()}[view]
            if not piece:
                self.status_var.set(f'No {view.lower()} selected')
                return
            mol = mol_from_str(get_struct(piece))
            if not self.placeholders_var.get():
                mol = remove_placeholder_atoms(mol)
        self._img = self._render(mol, pieces=view != VIEWS[0])
        self._show_image()

    def _update_isomers(self) -> None:
        """Regioisomers of the selected tetraether family and number of rings, the first one is selected"""
        isomers = tetraether_isomers().get((self._tetraether_family, self.rings_var.get()), []) \
            if self._tetraether_family else []
        self.isomer_cb.configure(values=isomers)
        self.isomer_var.set(isomers[0] if isomers else '')
        self.isomer_cb.state(['!disabled'] if isomers else ['disabled'])

    def _core(self) -> str:
        """Core to build: the selected isomer of tetraethers, otherwise the selected core piece"""
        core = abbreviation(self.core_var.get())
        if self._tetraether_family is None:
            return core
        if self.isomer_var.get():
            return self.isomer_var.get()
        return tetraether_piece(self._tetraether_family, self.rings_var.get()) or core

    def _read_chains(self, core: str) -> list[str]:
        if self.chains_entry.instate(['disabled']):
            return []
        return parse_chains(self.chains_var.get())

    def build(self) -> None:
        head = abbreviation(self.head_var.get())
        head2 = abbreviation(self.head2_var.get())
        core = self._core()
        core_chains = self.core_chains_var.get().strip() or None
        try:
            modifications = {key: int(var.get()) if entry.instate(['!disabled']) else 0
                             for key, var, entry in (('core_double_bonds', self.core_db_var, self.core_db_entry),
                                                     ('core_hydroxy', self.core_oh_var, self.core_oh_entry),
                                                     ('core_methyls', self.core_me_var, self.core_me_entry))}
            chains = self._read_chains(core)
            mol = build_ipl(head, core, chains, split_chain=self.split_var.get(), sort_chains=self.sort_var.get(),
                            head2=head2, core_chains=core_chains, **modifications)
        except Exception as e:
            self.status_var.set(f'Could not build: {e}')
            return
        self.status_var.set('')

        # only name the isoprenoid chains if they differ from the core
        if core_chains == isoprenoid_chains(core):
            core_chains = None
        self.show_mol(mol, display_name(head, core, chains, head2=head2, core_chains=core_chains, **modifications))


BLOCK_CORE_TYPE_LABELS: list[str] = [f'{core_type}{SEP}{t.description}' for core_type, t in CORE_TYPES.items()]
MAX_BLOCK_CHAINS: int = max(len(t.chains) for t in CORE_TYPES.values())
# label, attribute of ChainBlock, default, maximum
BLOCK_CHAIN_FIELDS: list[tuple[str, str, int, int]] = [
    ('C atoms', 'n_carbons', 40, 80),
    ('Cyclopentane rings', 'n_cyclopentanes', 0, 8),
    ('Cyclohexane rings', 'n_cyclohexanes', 0, 4),
    ('Hydroxyl groups', 'n_hydroxy', 0, 8),
    ('Double bonds', 'n_double_bonds', 0, 16),
]


class IsoprenoidBlocksFrame(_StructureFrame):
    """Build isoprenoid cores (tetraethers, diethers, hydrocarbons) with head groups from isoprene units."""

    def __init__(self, master, send_targets: SendTargets = None, **kwargs):
        super().__init__(master, **kwargs)
        self.type_var = tk.StringVar(self, value=BLOCK_CORE_TYPE_LABELS[0])
        self.head_var = tk.StringVar(self, value=NO_HEAD)
        self.head2_var = tk.StringVar(self, value=NO_HEAD)
        self.layout_hint_var = tk.StringVar(self, value='')
        # one variable per chain and field
        self.chain_vars: list[dict[str, tk.IntVar]] = [
            {attr: tk.IntVar(self, value=default) for _, attr, default, _ in BLOCK_CHAIN_FIELDS}
            for _ in range(MAX_BLOCK_CHAINS)
        ]
        self._build(send_targets)
        self._update_chain_state()
        self._init_auto_build([self.type_var, self.head_var, self.head2_var,
                               *(var for chain_vars in self.chain_vars for var in chain_vars.values())])

    def _build(self, send_targets: SendTargets | None) -> None:
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky='nsw', padx=(0, PAD * 2))

        inputs = ttk.LabelFrame(left, text='Building blocks', padding=PAD)
        inputs.pack(fill='x')
        type_cb = add_labeled(inputs, 0, 'Type', ttk.Combobox(inputs, textvariable=self.type_var, state='readonly',
                                                              values=BLOCK_CORE_TYPE_LABELS, width=34))
        type_cb.bind('<<ComboboxSelected>>', lambda _: self._update_chain_state(set_defaults=True))
        self.head_cbs: list[SearchCombobox] = []
        for row, (label, var) in enumerate([('Head group 1', self.head_var), ('Head group 2', self.head2_var)], 1):
            self.head_cbs.append(add_labeled(inputs, row, label, SearchCombobox(inputs, textvariable=var,
                                                                                values=HEAD_LABELS, width=34)))

        chains = ttk.LabelFrame(left, text='Chains', padding=PAD)
        chains.pack(fill='x', pady=PAD)
        ttk.Label(chains, textvariable=self.layout_hint_var, justify='left', wraplength=320).grid(
            row=0, column=0, columnspan=MAX_BLOCK_CHAINS + 1, sticky='w', padx=PAD)
        for col in range(MAX_BLOCK_CHAINS):
            ttk.Label(chains, text=f'Chain {col + 1}').grid(row=1, column=col + 1, padx=PAD)
        # spinboxes per chain
        self.chain_spinboxes: list[list[ttk.Spinbox]] = [[] for _ in range(MAX_BLOCK_CHAINS)]
        for row, (label, attr, _, maximum) in enumerate(BLOCK_CHAIN_FIELDS, 2):
            ttk.Label(chains, text=label).grid(row=row, column=0, sticky='w', padx=PAD, pady=PAD)
            for k, chain_vars in enumerate(self.chain_vars):
                spinbox = ttk.Spinbox(chains, from_=0, to=maximum, textvariable=chain_vars[attr], width=5)
                spinbox.grid(row=row, column=k + 1, padx=PAD, pady=PAD)
                self.chain_spinboxes[k].append(spinbox)
        self.carbon_spinboxes: list[ttk.Spinbox] = [spinboxes[0] for spinboxes in self.chain_spinboxes]
        ttk.Label(
            chains, foreground='gray', justify='left', wraplength=320,
            text='5 C atoms per isoprene unit (only regular alkanes can be shortened, e.g. pristane HT(19:0)). '
                 'Positions are counted from C1 of each half (bonded to the ether O). Cyclopentane rings: C7 of the '
                 'first, C7 of the second half, then C3, C11. Cyclohexane rings: C11-C15 (crenarchaeol). Hydroxyl '
                 'groups: C3, then C7. Double bonds: Δ2, Δ6, ... alternating between the halves. Head group 1 bonds '
                 'to the first polyol, head group 2 to the second one (or to the free OH of the chains).'
        ).grid(row=len(BLOCK_CHAIN_FIELDS) + 2, column=0, columnspan=MAX_BLOCK_CHAINS + 1, sticky='w', padx=PAD)

        self._build_status(left)
        self._build_properties(left)
        self._build_output(send_targets)

    def _core_type(self) -> str:
        return abbreviation(self.type_var.get())

    def _update_chain_state(self, set_defaults: bool = False) -> None:
        core_type = self._core_type()
        layout = CORE_TYPES[core_type]
        n_chains = len(layout.chains)
        for k, spinboxes in enumerate(self.chain_spinboxes):
            for spinbox in spinboxes:
                spinbox.state(['!disabled'] if k < n_chains else ['disabled'])
        # only regular alkanes can be shortened by single C atoms
        for spinbox in self.carbon_spinboxes:
            spinbox.configure(increment=1 if core_type == 'HT' else ISOPRENE_CARBONS)
        if set_defaults:
            for chain_vars, chain_layout in zip(self.chain_vars, layout.chains):
                chain_vars['n_carbons'].set(DEFAULT_CARBONS[chain_layout.kind])
        # hydrocarbons take no head groups, cores with OH groups or polyols take up to two
        n_attachments = len(layout.polyols) + sum(a == 'OH' for c in layout.chains for a in c.anchors)
        for k, head_cb in enumerate(self.head_cbs):
            head_cb.state(['!disabled'] if k < n_attachments else ['disabled'])
        self.layout_hint_var.set(', '.join(f'{k + 1}: {c.description}' for k, c in enumerate(layout.chains)))

    def core_spec(self) -> str:
        """e.g. GDGT(40:0_40:0(2cp))"""
        core_type = self._core_type()
        try:
            chains = [ChainBlock(**{attr: int(var.get()) for attr, var in chain_vars.items()})
                      for chain_vars in self.chain_vars[:len(CORE_TYPES[core_type].chains)]]
        except (tk.TclError, ValueError):
            raise ValueError('the chain properties must be whole numbers')
        return format_block_core(core_type, chains)

    def build(self) -> None:
        heads = [abbreviation(var.get()) if cb.instate(['!disabled']) else None
                 for var, cb in zip((self.head_var, self.head2_var), self.head_cbs)]
        try:
            core = self.core_spec()
            mol = build_ipl(heads[0], core, [], head2=heads[1])
        except Exception as e:
            self.status_var.set(f'Could not build: {e}')
            return
        self.status_var.set('')
        self.show_mol(mol, display_name(heads[0], core, [], head2=heads[1]))


class BatchIPLFrame(ttk.Frame):
    """Generate tables of IPLs for ranges of chain compositions (e.g. target lists)."""

    def __init__(self, master, **kwargs):
        super().__init__(master, padding=PAD, **kwargs)
        self.c_min_var = tk.StringVar(self, value='30')
        self.c_max_var = tk.StringVar(self, value='38')
        self.db_min_var = tk.StringVar(self, value='0')
        self.db_max_var = tk.StringVar(self, value='2')
        self.even_var = tk.BooleanVar(self, value=True)
        # series of isoprenoid cores: (min, max) per property
        self.iso_ranges: dict[str, tuple[tk.StringVar, tk.StringVar]] = {
            key: (tk.StringVar(self, value=lo), tk.StringVar(self, value=hi))
            for key, lo, hi in (('rings', '0', '8'), ('core_double_bonds', '0', '0'), ('core_hydroxy', '0', '0'),
                                ('core_methyls', '0', '0'), ('archaeol_db', '0', '0'))
        }
        self.all_isomers_var = tk.BooleanVar(self, value=False)
        self.archaeol_carbons_var = tk.StringVar(self, value='20, 25')
        self.status_var = tk.StringVar(self, value='Select head groups, cores and adducts, then press "Generate".')
        self._build()

    @staticmethod
    def _listbox(parent, values: list[str], selected: list[str], height: int = 10) -> tk.Listbox:
        frame = ttk.Frame(parent)
        lb = tk.Listbox(frame, selectmode='multiple', exportselection=False, height=height, width=30)
        sb = ttk.Scrollbar(frame, orient='vertical', command=lb.yview)
        lb.configure(yscrollcommand=sb.set)
        for idx, v in enumerate(values):
            lb.insert('end', v)
            if v.split(SEP)[0] in selected:
                lb.selection_set(idx)
        lb.pack(side='left', fill='both', expand=True)
        sb.pack(side='left', fill='y')
        frame.pack(fill='both', expand=True)
        return lb

    def _build(self) -> None:
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky='nsw', padx=(0, PAD * 2))
        lists = ttk.Frame(left)
        lists.pack(fill='both', expand=True)
        for col, (title, attr, values, selected) in enumerate([
            ('Head groups', 'heads_lb', HEAD_LABELS, ['PC', 'PE', 'PG']),
            ('Cores', 'cores_lb', BATCH_CORE_LABELS, ['DAG']),
            ('Adducts', 'adducts_lb', ADDUCTS, DEFAULT_BATCH_ADDUCTS),
        ]):
            lf = ttk.LabelFrame(lists, text=title, padding=PAD)
            lf.grid(row=0, column=col, sticky='nsew', padx=(0, PAD))
            setattr(self, attr, self._listbox(lf, values, selected))
        lists.rowconfigure(0, weight=1)

        ranges = ttk.LabelFrame(left, text='Total chain composition', padding=PAD)
        ranges.pack(fill='x', pady=PAD)
        for row, (label, v_min, v_max) in enumerate([('Carbon atoms', self.c_min_var, self.c_max_var),
                                                     ('Double bonds', self.db_min_var, self.db_max_var)]):
            frame = ttk.Frame(ranges)
            ttk.Entry(frame, textvariable=v_min, width=6).pack(side='left')
            ttk.Label(frame, text='to').pack(side='left', padx=PAD)
            ttk.Entry(frame, textvariable=v_max, width=6).pack(side='left')
            add_labeled(ranges, row, label, frame)
        ttk.Checkbutton(ranges, text='Even carbon numbers only', variable=self.even_var).grid(
            row=2, column=0, columnspan=2, sticky='w', padx=PAD)

        iso = ttk.LabelFrame(left, text='Isoprenoid cores (tetraethers, archaeol)', padding=PAD)
        iso.pack(fill='x', pady=(0, PAD))
        for row, (label, key) in enumerate([('Rings (crenarchaeol: 5)', 'rings'),
                                            ('Biphytanyl double bonds', 'core_double_bonds'),
                                            ('Hydroxyl groups', 'core_hydroxy'), ('Extra methyl groups', 'core_methyls'),
                                            ('Archaeol double bonds per chain', 'archaeol_db')]):
            frame = ttk.Frame(iso)
            ttk.Entry(frame, textvariable=self.iso_ranges[key][0], width=6).pack(side='left')
            ttk.Label(frame, text='to').pack(side='left', padx=PAD)
            ttk.Entry(frame, textvariable=self.iso_ranges[key][1], width=6).pack(side='left')
            add_labeled(iso, row, label, frame)
        add_labeled(iso, 5, 'Archaeol chain C atoms', ttk.Entry(iso, textvariable=self.archaeol_carbons_var, width=14))
        ttk.Checkbutton(iso, text='All regioisomers (otherwise the first per number of rings)',
                        variable=self.all_isomers_var).grid(row=6, column=0, columnspan=2, sticky='w', padx=PAD)

        self.run_button = ttk.Button(left, text='Generate', command=self.run)
        self.run_button.pack(fill='x')

        self.table = ResultTable(self, ['Name'])
        self.table.grid(row=0, column=1, sticky='nsew')
        ttk.Label(self, textvariable=self.status_var, foreground='gray').grid(
            row=1, column=0, columnspan=2, sticky='w', pady=(PAD, 0))
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

    @staticmethod
    def _selected(lb: tk.Listbox) -> list[str]:
        return [lb.get(i) for i in lb.curselection()]

    def _read_inputs(self) -> dict:
        heads = [abbreviation(h) for h in self._selected(self.heads_lb)]
        cores = [abbreviation(c) for c in self._selected(self.cores_lb)]
        adducts = self._selected(self.adducts_lb)
        if not heads or not cores:
            raise ValueError('select at least one head group (or "(none)") and one core')
        try:
            c_range = range(int(self.c_min_var.get()), int(self.c_max_var.get()) + 1)
            db_range = range(int(self.db_min_var.get()), int(self.db_max_var.get()) + 1)
        except ValueError:
            raise ValueError('chain ranges must be whole numbers')
        chains = [f'C{c}:{db}' for c, db in product(c_range, db_range)
                  if (not self.even_var.get() or c % 2 == 0) and db <= c // 2]
        try:
            iso = {key: range(int(lo.get()), int(hi.get()) + 1) for key, (lo, hi) in self.iso_ranges.items()}
            archaeol_carbons = [int(c) for c in re.split(r'[,;\s]+', self.archaeol_carbons_var.get().strip()) if c]
        except ValueError:
            raise ValueError('the ranges of the isoprenoid cores must be whole numbers')
        return dict(heads=heads, cores=cores, adducts=adducts, chains=chains, iso=iso,
                    archaeol_carbons=archaeol_carbons, all_isomers=self.all_isomers_var.get())

    @staticmethod
    def _combinations(head: str | None, core: str, params: dict) -> list[dict]:
        """Keyword arguments of build_ipl for one head group and core: chain compositions for cores with chains,
        series of rings, isomers, double bonds, hydroxyl and methyl groups for tetraethers, chains for archaeols
        (combinations the rules cannot place are left out)"""
        iso = params['iso']
        if takes_chains(core):
            return [dict(head=head, core=core, chains=[chain]) for chain in params['chains']]
        if core in TETRAETHER_FAMILIES:
            cores = []
            for rings in tetraether_rings(core):
                if int(rings) not in iso['rings']:
                    continue
                isomers = tetraether_isomers().get((core, rings)) or [tetraether_piece(core, rings)]
                cores += [i for i in (isomers if params['all_isomers'] else isomers[:1]) if i]
            out = []
            for c in cores:
                n_hydroxy, n_methyls = max_core_hydroxy(c), max_core_methyls(c)
                out += [dict(head=head, core=c, chains=[], core_double_bonds=db, core_hydroxy=oh, core_methyls=me)
                        for db, oh, me in product(iso['core_double_bonds'], iso['core_hydroxy'], iso['core_methyls'])
                        if oh <= n_hydroxy and me <= n_methyls]
            return out
        if core == 'AR':
            # pairs of chains, each pair once (ArchLips does not distinguish the sn positions), hydroxyl groups at
            # C3 as in ArchLips names (e.g. 20:1(OH)_20:0), on either chain if they differ
            chains = sorted({f'{c}:{db}' for c in params['archaeol_carbons'] for db in iso['archaeol_db']
                             if c % 5 == 0 and db <= c // 5}, key=lambda s: tuple(map(int, s.split(':'))))
            out = []
            for k, a in enumerate(chains):
                for b in chains[k:]:
                    for oh in iso['core_hydroxy']:
                        variants = {0: [(a, b)], 1: [(f'{a}(OH)', b)] + ([(f'{b}(OH)', a)] if a != b else []),
                                    2: [(f'{a}(OH)', f'{b}(OH)')]}.get(oh, [])
                        out += [dict(head=head, core=core, chains=[], core_chains=f'{c1}_{c2}') for c1, c2 in variants]
            return out
        return [dict(head=head, core=core, chains=[])]

    @staticmethod
    def _modifications(combo: dict) -> str:
        """e.g. '2 DB, 1 OH'"""
        return ', '.join(f'{combo[key]} {label}' for key, label in (('core_double_bonds', 'DB'), ('core_hydroxy', 'OH'),
                                                                    ('core_methyls', 'Me')) if combo.get(key))

    def run(self) -> None:
        try:
            params = self._read_inputs()
        except Exception as e:
            show_error(self, e)
            return
        combos = [combo for head, core in product(params['heads'], params['cores'])
                  for combo in self._combinations(head, core, params)]
        if len(combos) > 20_000 and not messagebox.askyesno(
                'Large batch', f'This generates {len(combos)} structures and may take a while. Continue?',
                parent=self.winfo_toplevel()):
            return
        adducts = [Adduct(a) for a in params['adducts']]

        def compute():
            rows, failed = [], []
            for combo in combos:
                head, core, chains = combo['head'], combo['core'], combo['chains']
                modifications = {key: value for key, value in combo.items() if key not in ('head', 'core', 'chains')}
                name = display_name(head, core, chains, **modifications)
                try:
                    mol = build_ipl(head, core, chains, **modifications)
                except Exception:
                    failed.append(name)
                    continue
                mass = ExactMolWt(mol)
                rows.append(
                    [name, head or '', core,
                     combo.get('core_chains') or '/'.join(chain[1:] for chain in chains),
                     self._modifications(combo), rdMolDescriptors.CalcMolFormula(mol), f'{mass:.5f}']
                    + [f'{a.mass_to_mz(mass):.5f}' for a in adducts]
                    + [Chem.MolToSmiles(mol)]
                )
            return rows, failed

        def done(result):
            rows, failed = result
            self.run_button.state(['!disabled'])
            columns = ['Name', 'Head', 'Core', 'Chains', 'Modifications', 'Formula', 'Mass'] + params['adducts'] \
                + ['SMILES']
            self.table.set_columns(columns, widths={'Name': 150, 'Head': 55, 'Core': 70, 'Chains': 80,
                                                    'Modifications': 90, 'Formula': 130, 'SMILES': 300})
            self.table.set_rows(rows)
            msg = f'Generated {len(rows)} IPLs.'
            if failed:
                msg += f' {len(failed)} combination(s) could not be built, e.g. {", ".join(failed[:3])}.'
            self.status_var.set(msg)

        def failed_cb(e):
            self.run_button.state(['!disabled'])
            show_error(self, e)

        self.run_button.state(['disabled'])
        self.status_var.set(f'Building {len(combos)} structures…')
        run_in_thread(self, compute, done, failed_cb)


class IPLBuilderFrame(ttk.Frame):
    """Combines single IPL construction and batch generation"""

    def __init__(self, master, send_targets: SendTargets = None, **kwargs):
        super().__init__(master, padding=PAD, **kwargs)
        nb = ttk.Notebook(self)
        nb.pack(fill='both', expand=True)
        self.single = SingleIPLFrame(nb, send_targets=send_targets)
        self.blocks = IsoprenoidBlocksFrame(nb, send_targets=send_targets)
        self.batch = BatchIPLFrame(nb)
        nb.add(self.single, text='Single IPL')
        nb.add(self.blocks, text='Isoprenoid blocks')
        nb.add(self.batch, text='Batch / target list')


def main():
    from LipidCalculator.gui.common import run_standalone
    run_standalone(IPLBuilderFrame, 'IPL builder', geometry='1150x750')


if __name__ == '__main__':
    main()
