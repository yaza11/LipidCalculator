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
from LipidCalculator.compound_groups.intact_polar_lipids.generate_ipl import ipl_automatic_bonds, get_struct
from LipidCalculator.compound_groups.intact_polar_lipids.isoprenoid_chains import describe_isoprenoid_chains, \
    has_biphytanyl_chains
from LipidCalculator.compound_groups.intact_polar_lipids.pieces_from_json import (
    DATA_FRAME_IPL_PIECES, CORE_IMPLICIT_CHAIN, CORE_GROUPS, CHAINS_IN_ISOPRENE_UNITS, ABBREVIATIONS
)
from LipidCalculator.rdkit.plotting import mol_to_img
from LipidCalculator.rdkit.util import mol_from_str
from LipidCalculator.gui.common import (
    PAD, ADDUCTS, ResultTable, SendTargets, add_labeled, hill_formula, run_in_thread, send_buttons, show_error
)

NO_HEAD: str = '(none)'
SEP: str = ' – '
# cores that only take a single chain (see ipl_automatic_bonds)
SINGLE_CHAIN_CORES: tuple[str, ...] = ('OL',)
DEFAULT_BATCH_ADDUCTS: list[str] = ['[M+H]+', '[M+NH4]+', '[M+Na]+']


def _piece_labels(groups: list[str]) -> list[str]:
    df = DATA_FRAME_IPL_PIECES[DATA_FRAME_IPL_PIECES.group.isin(groups)]
    return [abbr if pd.isna(name) else f'{abbr}{SEP}{name}' for abbr, name in zip(df.abbreviation, df.name)]


HEAD_LABELS: list[str] = [NO_HEAD] + _piece_labels(['head groups', 'functional groups'])
# cores, quinones and carotenoids
CORE_LABELS: list[str] = _piece_labels(CORE_GROUPS)
BATCH_CORE_LABELS: list[str] = _piece_labels(['core lipids'])


def abbreviation(label: str) -> str | None:
    abbr = label.split(SEP)[0]
    return None if abbr == NO_HEAD else abbr


def takes_chains(core: str) -> bool:
    return (core, 'Cs') in CORE_IMPLICIT_CHAIN


@lru_cache(maxsize=None)
def isoprenoid_chains(core: str) -> str:
    """Isoprenoid chains of a core piece, e.g. '20:0_20:0', empty if it has none"""
    return describe_isoprenoid_chains(mol_from_str(get_struct(core)), in_units=core in CHAINS_IN_ISOPRENE_UNITS)


@lru_cache(maxsize=None)
def takes_core_double_bonds(core: str) -> bool:
    """Whether the core has biphytanyl chains (tetraethers) that can be unsaturated"""
    return has_biphytanyl_chains(mol_from_str(get_struct(core)))


def parse_chain(chain: str) -> tuple[int, int] | None:
    """'34:1' or 'C34:1' --> (34, 1), empty --> None"""
    chain = chain.strip()
    if chain == '':
        return None
    match = re.fullmatch(r'C?(\d+):(\d+)', chain)
    if match is None:
        raise ValueError(f'chain "{chain}" must have the form 34:1 (carbon atoms:double bonds)')
    return int(match.group(1)), int(match.group(2))


def build_ipl(
        head: str | None,
        core: str,
        chains: list[tuple[int, int]],
        split_chain: bool = True,
        sort_chains: bool = True,
        remove_double_oxygen: bool = True,
        head2: str | None = None,
        core_chains: str | None = None,
        core_double_bonds: int = 0
) -> Mol:
    """Build an IPL, chain lengths include the C atoms of the core (lipid shorthand). head2 is bonded to the
    second attachment point of the core (e.g. other glycerol of GDGTs), core_chains sets the isoprenoid chains
    of the core (e.g. '20:1_25:0'), core_double_bonds the double bonds in the biphytanyl chains of tetraethers."""
    if core not in ABBREVIATIONS:
        raise ValueError(f'unknown core {core}')
    if chains and not takes_chains(core):
        raise ValueError(f'core {core} cannot be connected to chains')
    if core in SINGLE_CHAIN_CORES and len(chains) > 1:
        raise ValueError(f'core {core} only takes one chain')
    if core_chains and not isoprenoid_chains(core):
        raise ValueError(f'core {core} has no isoprenoid chains')
    if core_double_bonds and not takes_core_double_bonds(core):
        raise ValueError(f'core {core} has no biphytanyl chains')
    names = [h for h in (head, head2) if h] + [core] + [f'C{n_c}:{n_db}' for n_c, n_db in chains]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return ipl_automatic_bonds(
            names,
            split_chain=split_chain and len(chains) == 1 and core not in SINGLE_CHAIN_CORES,
            sort_chains=sort_chains,
            remove_double_oxygen=remove_double_oxygen,
            core_chains=core_chains or None,
            core_double_bonds=core_double_bonds or None
        )


UNSATURATION_PREFIXES: list[str] = ['', 'Mono', 'Di', 'Tri', 'Tetra', 'Penta', 'Hexa', 'Hepta', 'Octa']


def ipl_name(head: str | None, core: str, chains: list[tuple[int, int]], head2: str | None = None,
             core_chains: str | None = None, core_double_bonds: int = 0) -> str:
    """e.g. PE-AR(20:0_20:1), 2G-(PG-)Di-uns-GDGT0 or PC-DAG 34:1"""
    name = core if not core_chains else f'{core}({core_chains})'
    if core_double_bonds:
        prefix = UNSATURATION_PREFIXES[core_double_bonds] if core_double_bonds < len(UNSATURATION_PREFIXES) \
            else str(core_double_bonds)
        name = f'{prefix}-uns-{name}'
    if head2:
        name = f'({head2}-){name}'
    if head:
        name = f'{head}-{name}'
    if chains:
        name += ' ' + '/'.join(f'{n_c}:{n_db}' for n_c, n_db in chains)
    return name


class SingleIPLFrame(ttk.Frame):
    """Build one IPL and show its structure, formula and adduct m/z values."""

    def __init__(self, master, send_targets: SendTargets = None, **kwargs):
        super().__init__(master, padding=PAD, **kwargs)
        self.head_var = tk.StringVar(self, value=next(l for l in HEAD_LABELS if l.startswith('PC')))
        self.core_var = tk.StringVar(self, value=next(l for l in CORE_LABELS if l.startswith('DAG')))
        self.chain1_var = tk.StringVar(self, value='34:1')
        self.chain2_var = tk.StringVar(self, value='')
        self.head2_var = tk.StringVar(self, value=NO_HEAD)
        self.core_chains_var = tk.StringVar(self, value='')
        self.core_chains_hint_var = tk.StringVar(self, value='')
        self.core_db_var = tk.IntVar(self, value=0)
        self.split_var = tk.BooleanVar(self, value=True)
        self.sort_var = tk.BooleanVar(self, value=True)
        self.oop_var = tk.BooleanVar(self, value=True)
        self.name_var = tk.StringVar(self, value='')
        self.formula_var = tk.StringVar(self, value='')
        self.mass_var = tk.StringVar(self, value='')
        self.smiles_var = tk.StringVar(self, value='')
        self.mol: Mol | None = None
        self._img = None
        self._photo = None
        self._build(send_targets)
        self._update_chain_state()
        self.after_idle(self.build)

    def _build(self, send_targets: SendTargets | None) -> None:
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky='nsw', padx=(0, PAD * 2))

        inputs = ttk.LabelFrame(left, text='Building blocks', padding=PAD)
        inputs.pack(fill='x')
        add_labeled(inputs, 0, 'Head group', ttk.Combobox(inputs, textvariable=self.head_var, values=HEAD_LABELS,
                                                          state='readonly', width=34))
        add_labeled(inputs, 1, 'Head group 2', ttk.Combobox(inputs, textvariable=self.head2_var, values=HEAD_LABELS,
                                                            state='readonly', width=34))
        # editable, typing filters the long list of cores
        self.core_cb = add_labeled(inputs, 2, 'Core', ttk.Combobox(inputs, textvariable=self.core_var,
                                                                   values=CORE_LABELS, width=34))
        self.core_cb.bind('<<ComboboxSelected>>', lambda _: self._update_chain_state())
        self.core_cb.bind('<KeyRelease>', self._filter_cores)
        self.core_cb.bind('<Return>', lambda _: (self._update_chain_state(), self.build()))
        self.core_chains_entry = add_labeled(inputs, 3, 'Isoprenoid chains',
                                             ttk.Entry(inputs, textvariable=self.core_chains_var))
        ttk.Label(inputs, textvariable=self.core_chains_hint_var, foreground='gray', justify='left',
                  wraplength=320).grid(row=4, column=0, columnspan=2, sticky='w', padx=PAD)
        self.core_db_entry = add_labeled(inputs, 5, 'Biphytanyl double bonds',
                                         ttk.Spinbox(inputs, from_=0, to=8, textvariable=self.core_db_var, width=5))
        self.chain1_entry = add_labeled(inputs, 6, 'Chain 1 (C:DB)', ttk.Entry(inputs, textvariable=self.chain1_var))
        self.chain2_entry = add_labeled(inputs, 7, 'Chain 2 (C:DB)', ttk.Entry(inputs, textvariable=self.chain2_var))
        for entry in (self.chain1_entry, self.chain2_entry, self.core_chains_entry, self.core_db_entry):
            entry.bind('<Return>', lambda _: self.build())
        ttk.Label(
            inputs, foreground='gray', justify='left', wraplength=320,
            text='Chain lengths include the C atoms of the core. Leave chain 2 empty to give the total '
                 'composition (e.g. 34:1), which is split equally between both chains. Head group 2 is '
                 'bonded to the second attachment point of the core (other glycerol of tetraethers).'
        ).grid(row=8, column=0, columnspan=2, sticky='w', padx=PAD)

        opts = ttk.LabelFrame(left, text='Options', padding=PAD)
        opts.pack(fill='x', pady=PAD)
        ttk.Checkbutton(opts, text='Split single chain into two', variable=self.split_var).pack(anchor='w')
        ttk.Checkbutton(opts, text='Longest chain first', variable=self.sort_var).pack(anchor='w')
        ttk.Checkbutton(opts, text='Glycerol O bonds directly to P (O-O-P → O-P)', variable=self.oop_var).pack(
            anchor='w')
        ttk.Button(left, text='Build', command=self.build).pack(fill='x', pady=PAD)

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

        right = ttk.PanedWindow(self, orient='vertical')
        right.grid(row=0, column=1, sticky='nsew')
        self.canvas = tk.Canvas(right, background='white', highlightthickness=0, height=320)
        self.canvas.bind('<Configure>', lambda _: self._show_image())
        right.add(self.canvas, weight=3)
        self.table = ResultTable(right, ['Adduct', 'm/z'], height=6)
        right.add(self.table, weight=1)
        send_buttons(self.table.toolbar, send_targets, self._payload).pack(side='right')

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

    def _filter_cores(self, event) -> None:
        if event.keysym in ('Up', 'Down', 'Return', 'Escape', 'Tab'):
            return
        text = self.core_var.get().lower()
        self.core_cb['values'] = [label for label in CORE_LABELS if text in label.lower()] or CORE_LABELS

    def _update_chain_state(self) -> None:
        core = abbreviation(self.core_var.get())
        state = ['!disabled'] if takes_chains(core) else ['disabled']
        self.chain1_entry.state(state)
        self.chain2_entry.state(state)
        tetraether = core in ABBREVIATIONS and takes_core_double_bonds(core)
        self.core_db_entry.state(['!disabled'] if tetraether else ['disabled'])
        self.core_db_var.set(0)
        chains = isoprenoid_chains(core) if core in ABBREVIATIONS else ''
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

    def _read_chains(self, core: str) -> list[tuple[int, int]]:
        if not takes_chains(core):
            return []
        return [c for c in (parse_chain(self.chain1_var.get()), parse_chain(self.chain2_var.get())) if c is not None]

    def build(self) -> None:
        head = abbreviation(self.head_var.get())
        head2 = abbreviation(self.head2_var.get())
        core = abbreviation(self.core_var.get())
        core_chains = self.core_chains_var.get().strip() or None
        try:
            core_db = int(self.core_db_var.get()) if self.core_db_entry.instate(['!disabled']) else 0
            chains = self._read_chains(core)
            mol = build_ipl(head, core, chains, split_chain=self.split_var.get(), sort_chains=self.sort_var.get(),
                            remove_double_oxygen=self.oop_var.get(), head2=head2, core_chains=core_chains,
                            core_double_bonds=core_db)
        except Exception as e:
            show_error(self, e, title='Could not build IPL')
            return

        self.mol = mol
        mass = ExactMolWt(mol)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        # only name the isoprenoid chains if they differ from the core
        if core_chains == isoprenoid_chains(core):
            core_chains = None
        self.name_var.set(ipl_name(head, core, chains, head2=head2, core_chains=core_chains,
                                   core_double_bonds=core_db))
        self.formula_var.set(formula)
        self.mass_var.set(f'{mass:.5f}')
        self.smiles_var.set(Chem.MolToSmiles(mol))
        self.table.set_rows([(a, f'{Adduct(a).mass_to_mz(mass):.5f}') for a in ADDUCTS])
        self._img = mol_to_img(mol, res_pixels=1200)
        self._show_image()

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


class BatchIPLFrame(ttk.Frame):
    """Generate tables of IPLs for ranges of chain compositions (e.g. target lists)."""

    def __init__(self, master, **kwargs):
        super().__init__(master, padding=PAD, **kwargs)
        self.c_min_var = tk.StringVar(self, value='30')
        self.c_max_var = tk.StringVar(self, value='38')
        self.db_min_var = tk.StringVar(self, value='0')
        self.db_max_var = tk.StringVar(self, value='2')
        self.even_var = tk.BooleanVar(self, value=True)
        self.oop_var = tk.BooleanVar(self, value=True)
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
        ttk.Checkbutton(ranges, text='Glycerol O bonds directly to P (O-O-P → O-P)', variable=self.oop_var).grid(
            row=3, column=0, columnspan=2, sticky='w', padx=PAD)

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
        chains = [(c, db) for c, db in product(c_range, db_range)
                  if (not self.even_var.get() or c % 2 == 0) and db <= c // 2]
        return dict(heads=heads, cores=cores, adducts=adducts, chains=chains, oop=self.oop_var.get())

    def run(self) -> None:
        try:
            params = self._read_inputs()
        except Exception as e:
            show_error(self, e)
            return
        combos = [
            (head, core, [chain] if takes_chains(core) else [])
            for head, core in product(params['heads'], params['cores'])
            for chain in (params['chains'] if takes_chains(core) else [None])
        ]
        if len(combos) > 20_000 and not messagebox.askyesno(
                'Large batch', f'This generates {len(combos)} structures and may take a while. Continue?',
                parent=self.winfo_toplevel()):
            return
        adducts = [Adduct(a) for a in params['adducts']]

        def compute():
            rows, failed = [], []
            for head, core, chains in combos:
                try:
                    mol = build_ipl(head, core, chains, remove_double_oxygen=params['oop'])
                except Exception:
                    failed.append(ipl_name(head, core, chains))
                    continue
                mass = ExactMolWt(mol)
                rows.append(
                    [ipl_name(head, core, chains), head or '', core,
                     '/'.join(f'{c}:{db}' for c, db in chains),
                     rdMolDescriptors.CalcMolFormula(mol), f'{mass:.5f}']
                    + [f'{a.mass_to_mz(mass):.5f}' for a in adducts]
                    + [Chem.MolToSmiles(mol)]
                )
            return rows, failed

        def done(result):
            rows, failed = result
            self.run_button.state(['!disabled'])
            columns = ['Name', 'Head', 'Core', 'Chains', 'Formula', 'Mass'] + params['adducts'] + ['SMILES']
            self.table.set_columns(columns, widths={'Name': 150, 'Head': 55, 'Core': 60, 'Chains': 60,
                                                    'Formula': 130, 'SMILES': 300})
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
        self.batch = BatchIPLFrame(nb)
        nb.add(self.single, text='Single IPL')
        nb.add(self.batch, text='Batch / target list')


def main():
    from LipidCalculator.gui.common import run_standalone
    run_standalone(IPLBuilderFrame, 'IPL builder', geometry='1150x750')


if __name__ == '__main__':
    main()
