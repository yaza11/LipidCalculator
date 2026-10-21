"""Shared widgets and helpers for the LipidCalculator tkinter GUIs."""
import csv
import threading
import traceback
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Callable, Any, Iterable

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from LipidCalculator import Adduct, CompoundDict

logger = logging.getLogger(__name__)

# entry in adduct selections that denotes a neutral molecule
NEUTRAL: str = 'neutral (M)'

ADDUCTS_POS: list[str] = [
    '[M+H]+', '[M+NH4]+', '[M+Na]+', '[M+K]+', '[M+H-H2O]+', '[M]+',
    '[M+2H]2+', '[M+H+NH4]2+', '[M+H+Na]2+', '[2M+H]+', '[2M+Na]+',
]
ADDUCTS_NEG: list[str] = [
    '[M-H]-', '[M+Cl]-', '[M+HCOO]-', '[M+CH3COO]-', '[M-H2O-H]-', '[M-2H]2-',
]
ADDUCTS: list[str] = ADDUCTS_POS + ADDUCTS_NEG

PAD: int = 4

# payload passed between modules, keys: 'formula', 'adduct', 'mz'
SendTargets = dict[str, Callable[[dict], None]]


def parse_adduct(adduct: str) -> Adduct | None:
    """Return None for neutral molecules"""
    adduct = adduct.strip()
    if adduct in ('', NEUTRAL, 'M', '[M]'):
        return None
    return Adduct(adduct)


def parse_formula(formula: str) -> CompoundDict:
    formula = formula.strip().replace(' ', '')
    if formula == '':
        raise ValueError('please provide a formula')
    cd = CompoundDict(formula)
    if any(v < 0 for v in cd.composition.values()):
        raise ValueError(f'formula {formula} contains negative element counts')
    return cd


def ion_composition(formula: str | CompoundDict, adduct: Adduct | None) -> CompoundDict:
    """Composition of the ion (charge removed), e.g. C2H6O + [M+H]+ --> C2H7O"""
    cd = CompoundDict(formula) if isinstance(formula, str) else formula.copy()
    if adduct is None:
        return cd
    cd = cd * adduct.multiplicity + adduct.composition
    for sign in '+-':
        cd.composition.pop(sign, None)
    if any(v < 0 for v in cd.composition.values()):
        raise ValueError(f'adduct {adduct} cannot be formed from {hill_formula(formula)}')
    return cd


def hill_formula(cd: CompoundDict | str) -> str:
    """Format composition in Hill notation (C, H, then alphabetical)"""
    if isinstance(cd, str):
        cd = CompoundDict(cd)
    comp = {el: n for el, n in cd.composition.items() if el not in ('+', '-') and n != 0}
    if 'C' in comp:
        els = sorted(comp, key=lambda el: (el != 'C', el != 'H', el))
    else:
        els = sorted(comp)
    out = ''
    for el in els:
        n = comp[el]
        n = int(n) if float(n).is_integer() else n
        out += el + (str(n) if n != 1 else '')
    return out


def run_in_thread(
        widget: tk.Misc,
        func: Callable[[], Any],
        on_success: Callable[[Any], None],
        on_error: Callable[[Exception], None] = None,
        poll_ms: int = 50
) -> None:
    """Run func in a worker thread and call the callbacks from the tk main loop
    (tkinter is not thread-safe, so the callbacks must not run in the worker)."""
    result: dict = {}

    def target():
        try:
            result['value'] = func()
        except Exception as e:
            logger.debug(traceback.format_exc())
            result['error'] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()

    def poll():
        if thread.is_alive():
            widget.after(poll_ms, poll)
        elif 'error' in result:
            (on_error or (lambda e: show_error(widget, e)))(result['error'])
        else:
            on_success(result['value'])

    widget.after(poll_ms, poll)


def show_error(parent: tk.Misc, exc: Exception | str, title: str = 'Error') -> None:
    msg = exc if isinstance(exc, str) else f'{type(exc).__name__}: {exc}'
    messagebox.showerror(title, msg, parent=parent.winfo_toplevel())


def add_labeled(
        parent: tk.Misc, row: int, label: str, widget: tk.Widget, unit: str = None, column: int = 0
) -> tk.Widget:
    """Grid a label, widget and optional unit in a row"""
    ttk.Label(parent, text=label).grid(row=row, column=column, sticky='w', padx=PAD, pady=PAD)
    widget.grid(row=row, column=column + 1, sticky='ew', padx=PAD, pady=PAD)
    if unit is not None:
        ttk.Label(parent, text=unit).grid(row=row, column=column + 2, sticky='w', padx=(0, PAD))
    return widget


def adduct_combobox(parent: tk.Misc, var: tk.StringVar, include_neutral: bool = True, width: int = 16):
    values = ([NEUTRAL] if include_neutral else []) + ADDUCTS
    # editable so that custom adducts like [M+Li]+ can be typed in
    return ttk.Combobox(parent, textvariable=var, values=values, width=width)


def send_buttons(parent: tk.Misc, send_targets: SendTargets | None, get_payload: Callable[[], dict | None]):
    """Add one button per send target, returns the frame containing them"""
    frame = ttk.Frame(parent)
    for label, target in (send_targets or {}).items():
        def cmd(target=target):
            payload = get_payload()
            if payload is not None:
                target(payload)

        ttk.Button(frame, text=label, command=cmd).pack(side='left', padx=(0, PAD))
    return frame


class ResultTable(ttk.Frame):
    """Treeview with scrollbars, click-to-sort headers, copy and CSV export."""

    def __init__(self, master, columns: Iterable[str], height: int = 12, toolbar: bool = True, **kwargs):
        super().__init__(master, **kwargs)
        self.tree = ttk.Treeview(self, show='headings', height=height, selectmode='extended')
        ysb = ttk.Scrollbar(self, orient='vertical', command=self.tree.yview)
        xsb = ttk.Scrollbar(self, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self.tree.grid(row=0, column=0, sticky='nsew')
        ysb.grid(row=0, column=1, sticky='ns')
        xsb.grid(row=1, column=0, sticky='ew')
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        if toolbar:
            self.toolbar = ttk.Frame(self)
            self.toolbar.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(PAD, 0))
            ttk.Button(self.toolbar, text='Copy', command=self.copy_to_clipboard).pack(side='left')
            ttk.Button(self.toolbar, text='Export CSV…', command=self.export_csv).pack(side='left', padx=PAD)

        self.tree.bind('<Control-c>', lambda _: self.copy_to_clipboard())
        self.columns: list[str] = []
        self.set_columns(columns)

    def set_columns(self, columns: Iterable[str], widths: dict[str, int] = None) -> None:
        self.columns = list(columns)
        widths = widths or {}
        self.tree.delete(*self.tree.get_children())
        self.tree.configure(columns=self.columns)
        for col in self.columns:
            self.tree.heading(col, text=col, command=lambda c=col: self.sort_by(c))
            self.tree.column(col, width=widths.get(col, 100), anchor='w', stretch=True)

    def set_rows(self, rows: Iterable[Iterable]) -> None:
        self.tree.delete(*self.tree.get_children())
        for row in rows:
            self.tree.insert('', 'end', values=[('' if v is None else v) for v in row])

    def rows(self, selected_only: bool = False) -> list[tuple]:
        items = self.tree.selection() if selected_only else self.tree.get_children()
        return [tuple(self.tree.item(i, 'values')) for i in items]

    def selected_row(self) -> dict | None:
        """First selected row as dict column -> value"""
        sel = self.tree.selection()
        if not sel:
            return None
        return dict(zip(self.columns, self.tree.item(sel[0], 'values')))

    def sort_by(self, col: str, descending: bool = False) -> None:
        def key(v: str):
            try:
                return 0, float(v)
            except ValueError:
                return 1, v

        items = [(self.tree.set(i, col), i) for i in self.tree.get_children('')]
        items.sort(key=lambda t: key(t[0]), reverse=descending)
        for idx, (_, i) in enumerate(items):
            self.tree.move(i, '', idx)
        # next click sorts the other way round
        self.tree.heading(col, command=lambda: self.sort_by(col, not descending))

    def copy_to_clipboard(self) -> None:
        rows = self.rows(selected_only=True) or self.rows()
        text = '\n'.join(['\t'.join(self.columns)] + ['\t'.join(map(str, r)) for r in rows])
        self.clipboard_clear()
        self.clipboard_append(text)

    def export_csv(self) -> None:
        if not self.tree.get_children():
            show_error(self, 'nothing to export', title='Export')
            return
        path = filedialog.asksaveasfilename(
            parent=self.winfo_toplevel(), defaultextension='.csv',
            filetypes=[('CSV', '*.csv'), ('All files', '*.*')]
        )
        if not path:
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.columns)
            writer.writerows(self.rows())


class PlotCanvas(ttk.Frame):
    """Embedded matplotlib figure with navigation toolbar"""

    def __init__(self, master, figsize: tuple[float, float] = (6, 4), **kwargs):
        super().__init__(master, **kwargs)
        self.figure = Figure(figsize=figsize, dpi=100, layout='constrained')
        self.ax = self.figure.add_subplot()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side='bottom', fill='x')
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

    def clear(self) -> None:
        self.ax.clear()

    def draw(self) -> None:
        self.canvas.draw_idle()


def run_standalone(frame_cls: type, title: str, geometry: str = '1100x750', **kwargs) -> None:
    """Show a single module in its own window"""
    root = tk.Tk()
    root.title(title)
    root.geometry(geometry)
    frame_cls(root, **kwargs).pack(fill='both', expand=True)
    root.mainloop()
