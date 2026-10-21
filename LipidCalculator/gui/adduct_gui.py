"""GUI for converting between formulas, neutral masses and adduct m/z values."""
import tkinter as tk
from tkinter import ttk

from LipidCalculator import Adduct
from LipidCalculator.gui.common import (
    PAD, ADDUCTS, ResultTable, SendTargets, add_labeled, hill_formula, ion_composition, parse_formula,
    send_buttons, show_error
)


class AdductCalculatorFrame(ttk.Frame):
    """Formula -> m/z of common adducts, and m/z -> neutral mass for each adduct."""

    def __init__(self, master, send_targets_formula: SendTargets = None, send_targets_mz: SendTargets = None,
                 **kwargs):
        super().__init__(master, padding=PAD * 2, **kwargs)
        self.formula_var = tk.StringVar(self, value='C42H82NO8P')
        self.mz_var = tk.StringVar(self, value='760.5851')
        self.extra_var = tk.StringVar(self, value='')
        self.info_var = tk.StringVar(self, value='')
        self._build(send_targets_formula, send_targets_mz)

    def _build(self, send_targets_formula: SendTargets | None, send_targets_mz: SendTargets | None) -> None:
        settings = ttk.Frame(self)
        settings.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, PAD))
        ttk.Label(settings, text='Additional adducts (comma separated):').pack(side='left')
        ttk.Entry(settings, textvariable=self.extra_var, width=40).pack(side='left', padx=PAD)
        ttk.Label(settings, text='e.g. [M+Li]+, [M+H-2H2O]+', foreground='gray').pack(side='left')

        # formula --> m/z
        left = ttk.LabelFrame(self, text='Formula → m/z', padding=PAD)
        left.grid(row=1, column=0, sticky='nsew', padx=(0, PAD))
        row = ttk.Frame(left)
        row.pack(fill='x')
        entry = add_labeled(row, 0, 'Formula', ttk.Entry(row, textvariable=self.formula_var, width=24))
        entry.bind('<Return>', lambda _: self.calc_formula())
        ttk.Button(row, text='Calculate', command=self.calc_formula).grid(row=0, column=2, padx=PAD)
        ttk.Label(left, textvariable=self.info_var, justify='left').pack(anchor='w', pady=PAD)
        self.formula_table = ResultTable(left, ['Adduct', 'Ion formula', 'Charge', 'm/z'])
        self.formula_table.pack(fill='both', expand=True)
        send_buttons(self.formula_table.toolbar, send_targets_formula, self._formula_payload).pack(side='right')

        # m/z --> neutral mass
        right = ttk.LabelFrame(self, text='m/z → neutral mass', padding=PAD)
        right.grid(row=1, column=1, sticky='nsew', padx=(PAD, 0))
        row = ttk.Frame(right)
        row.pack(fill='x')
        entry = add_labeled(row, 0, 'm/z', ttk.Entry(row, textvariable=self.mz_var, width=14))
        entry.bind('<Return>', lambda _: self.calc_mz())
        ttk.Button(row, text='Calculate', command=self.calc_mz).grid(row=0, column=2, padx=PAD)
        ttk.Label(right, text='Neutral mass M assuming each adduct type', foreground='gray').pack(
            anchor='w', pady=PAD)
        self.mz_table = ResultTable(right, ['Adduct', 'Neutral mass'])
        self.mz_table.pack(fill='both', expand=True)
        send_buttons(self.mz_table.toolbar, send_targets_mz, self._mz_payload).pack(side='right')

        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(1, weight=1)

    def _adducts(self) -> list[str]:
        extra = [a.strip() for a in self.extra_var.get().split(',') if a.strip()]
        return ADDUCTS + [a for a in extra if a not in ADDUCTS]

    def set_input(self, payload: dict) -> None:
        if payload.get('formula'):
            self.formula_var.set(payload['formula'])
            self.calc_formula()
        if payload.get('mz') is not None:
            self.mz_var.set(f"{payload['mz']:.6f}")
            self.calc_mz()

    def calc_formula(self) -> None:
        try:
            cd = parse_formula(self.formula_var.get())
            M = cd.mass
        except Exception as e:
            show_error(self, e)
            return
        info = f'{hill_formula(cd)}   monoisotopic mass: {M:.6f} Da'
        try:
            info += f'   DBE: {cd.DBE:g}   Lewis: {cd.is_lewis}   Senior: {cd.is_senior}'
        except (KeyError, ValueError):
            # rules are not defined for all elements
            pass
        self.info_var.set(info)

        rows, skipped = [], []
        for adduct_str in self._adducts():
            try:
                adduct = Adduct(adduct_str)
                ion = ion_composition(cd, adduct)
                rows.append((adduct_str, hill_formula(ion), adduct.charge, f'{adduct.mass_to_mz(M):.5f}'))
            except Exception:
                skipped.append(adduct_str)
        self.formula_table.set_rows(rows)
        if skipped:
            self.info_var.set(info + f'\nnot possible / invalid: {", ".join(skipped)}')

    def calc_mz(self) -> None:
        try:
            mz = float(self.mz_var.get())
        except ValueError:
            show_error(self, 'm/z must be a number')
            return
        rows, invalid = [], []
        for adduct_str in self._adducts():
            try:
                M = Adduct(adduct_str).mz_to_mass(mz)
            except Exception:
                invalid.append(adduct_str)
                continue
            if M > 0:
                rows.append((adduct_str, f'{M:.5f}'))
        self.mz_table.set_rows(rows)
        if invalid:
            show_error(self, f'could not parse adduct(s): {", ".join(invalid)}')

    def _formula_payload(self) -> dict | None:
        row = self.formula_table.selected_row()
        try:
            formula = hill_formula(parse_formula(self.formula_var.get()))
        except Exception as e:
            show_error(self, e)
            return None
        # without selection the receiving module keeps its adduct
        return dict(formula=formula) if row is None else dict(formula=formula, adduct=row['Adduct'])

    def _mz_payload(self) -> dict | None:
        row = self.mz_table.selected_row()
        if row is None:
            show_error(self, 'select an adduct first', title='No selection')
            return None
        return dict(mz=float(self.mz_var.get()), adduct=row['Adduct'])


def main():
    from LipidCalculator.gui.common import run_standalone
    run_standalone(AdductCalculatorFrame, 'Mass & adduct calculator', geometry='1000x600')


if __name__ == '__main__':
    main()
