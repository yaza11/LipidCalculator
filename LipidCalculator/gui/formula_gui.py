"""GUI for generating candidate sum formulas for an observed m/z."""
import tkinter as tk
from tkinter import ttk

from LipidCalculator import CompoundDict
from LipidCalculator.formulae_generation.from_mass_milp import get_formulas_default_constraints
from LipidCalculator.gui.common import (
    PAD, ResultTable, SendTargets, add_labeled, adduct_combobox, hill_formula, parse_adduct,
    run_in_thread, send_buttons, show_error
)

# elements offered in the GUI and whether they are selected by default
ELEMENT_OPTIONS: dict[str, bool] = {
    'C': True, 'H': True, 'N': True, 'O': True, 'P': True, 'S': True,
    'F': False, 'Cl': False, 'Br': False, 'I': False, 'Si': False,
}

COLUMNS: list[str] = ['#', 'Formula', 'Neutral mass', 'm/z (theo.)', 'Error (ppm)', 'Error (mDa)', 'DBE', 'Lewis',
                      'Senior']


class FormulaGeneratorFrame(ttk.Frame):
    """Find the sum formulas closest to an observed m/z (MILP based search)."""

    def __init__(self, master, send_targets: SendTargets = None, **kwargs):
        super().__init__(master, padding=PAD * 2, **kwargs)
        self.mz_var = tk.StringVar(self, value='760.5851')
        self.adduct_var = tk.StringVar(self, value='[M+H]+')
        self.tol_var = tk.StringVar(self, value='5')
        self.tol_unit_var = tk.StringVar(self, value='ppm')
        self.n_var = tk.IntVar(self, value=10)
        self.dbe_min_var = tk.StringVar(self, value='0')
        self.dbe_max_var = tk.StringVar(self, value='36')
        self.lewis_var = tk.BooleanVar(self, value=True)
        self.senior_var = tk.BooleanVar(self, value=True)
        self.exclusive_var = tk.BooleanVar(self, value=False)
        self.element_vars = {el: tk.BooleanVar(self, value=v) for el, v in ELEMENT_OPTIONS.items()}
        self.status_var = tk.StringVar(self, value='Enter an m/z value and press "Find formulas".')
        # adduct used for the last search, needed when sending results
        self._last_adduct: str | None = None

        self._build(send_targets)

    def _build(self, send_targets: SendTargets | None) -> None:
        inputs = ttk.LabelFrame(self, text='Input', padding=PAD)
        inputs.grid(row=0, column=0, sticky='nsw', padx=(0, PAD * 2))

        mz_entry = add_labeled(inputs, 0, 'm/z (or neutral mass)', ttk.Entry(inputs, textvariable=self.mz_var, width=14))
        mz_entry.bind('<Return>', lambda _: self.run())
        add_labeled(inputs, 1, 'Adduct', adduct_combobox(inputs, self.adduct_var))

        tol_frame = ttk.Frame(inputs)
        ttk.Entry(tol_frame, textvariable=self.tol_var, width=7).pack(side='left')
        ttk.Combobox(tol_frame, textvariable=self.tol_unit_var, values=['ppm', 'mDa'], width=5,
                     state='readonly').pack(side='left', padx=(PAD, 0))
        add_labeled(inputs, 2, 'Tolerance', tol_frame)
        add_labeled(inputs, 3, 'Max. candidates', ttk.Spinbox(inputs, from_=1, to=200, textvariable=self.n_var, width=7))

        dbe_frame = ttk.Frame(inputs)
        ttk.Entry(dbe_frame, textvariable=self.dbe_min_var, width=5).pack(side='left')
        ttk.Label(dbe_frame, text='to').pack(side='left', padx=PAD)
        ttk.Entry(dbe_frame, textvariable=self.dbe_max_var, width=5).pack(side='left')
        add_labeled(inputs, 4, 'DBE range', dbe_frame)

        el_frame = ttk.LabelFrame(inputs, text='Elements', padding=PAD)
        el_frame.grid(row=5, column=0, columnspan=3, sticky='ew', pady=PAD)
        for idx, (el, var) in enumerate(self.element_vars.items()):
            ttk.Checkbutton(el_frame, text=el, variable=var).grid(row=idx // 6, column=idx % 6, sticky='w')

        rules = ttk.LabelFrame(inputs, text='Rules', padding=PAD)
        rules.grid(row=6, column=0, columnspan=3, sticky='ew', pady=PAD)
        ttk.Checkbutton(rules, text='Lewis rule (even electron count)', variable=self.lewis_var).pack(anchor='w')
        ttk.Checkbutton(rules, text='Senior rules (valences)', variable=self.senior_var).pack(anchor='w')
        ttk.Checkbutton(rules, text='At most one of N, P, S', variable=self.exclusive_var).pack(anchor='w')
        ttk.Label(
            rules, foreground='gray', justify='left',
            text='Always applied: H/C 0.43–3.5, O/C ≤ 4,\nN ≤ 7, P ≤ 4, S ≤ 2'
        ).pack(anchor='w', pady=(PAD, 0))

        self.run_button = ttk.Button(inputs, text='Find formulas', command=self.run)
        self.run_button.grid(row=7, column=0, columnspan=3, sticky='ew', pady=PAD)

        results = ttk.LabelFrame(self, text='Candidates', padding=PAD)
        results.grid(row=0, column=1, sticky='nsew')
        self.table = ResultTable(results, COLUMNS, height=18)
        self.table.set_columns(COLUMNS, widths={'#': 35, 'Formula': 140, 'Lewis': 55, 'Senior': 55, 'DBE': 55})
        self.table.pack(fill='both', expand=True)
        send_buttons(self.table.toolbar, send_targets, self._selected_payload).pack(side='right')
        ttk.Label(self, textvariable=self.status_var, foreground='gray').grid(
            row=1, column=0, columnspan=2, sticky='w', pady=(PAD, 0))

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

    def set_input(self, payload: dict) -> None:
        """Fill in m/z and adduct (e.g. from another module) and start the search"""
        if payload.get('mz') is not None:
            self.mz_var.set(f"{payload['mz']:.6f}")
        if payload.get('adduct'):
            self.adduct_var.set(payload['adduct'])
        self.run()

    def _selected_payload(self) -> dict | None:
        row = self.table.selected_row()
        if row is None:
            show_error(self, 'select a candidate first', title='No selection')
            return None
        return dict(formula=row['Formula'], adduct=self._last_adduct)

    def _read_inputs(self) -> dict:
        try:
            mz = float(self.mz_var.get())
            tol = float(self.tol_var.get())
            dbe_range = float(self.dbe_min_var.get()), float(self.dbe_max_var.get())
            n_max = int(self.n_var.get())
        except (ValueError, tk.TclError):
            raise ValueError('m/z, tolerance, DBE range and max. candidates must be numbers')
        if mz <= 0 or tol <= 0:
            raise ValueError('m/z and tolerance must be positive')
        elements = [el for el, var in self.element_vars.items() if var.get()]
        if len(elements) == 0:
            raise ValueError('select at least one element')
        adduct_str = self.adduct_var.get().strip()
        adduct = parse_adduct(adduct_str)

        # tolerance is given for the m/z, the solver works with neutral masses
        tol_mz_da = tol * 1e-6 * mz if self.tol_unit_var.get() == 'ppm' else tol * 1e-3
        if adduct is None:
            M_obs, tol_M_da = mz, tol_mz_da
        else:
            M_obs = adduct.mz_to_mass(mz)
            tol_M_da = tol_mz_da * abs(adduct.charge) / adduct.multiplicity
        if M_obs <= 0:
            raise ValueError(f'neutral mass for {adduct_str} would be {M_obs:.4f}')
        return dict(
            mz=mz, adduct=adduct, adduct_str=adduct_str if adduct is not None else None,
            M_obs=M_obs, tol_M_da=tol_M_da, elements=elements, n_max=n_max, dbe_range=dbe_range,
            check_lewis=self.lewis_var.get(), check_senior=self.senior_var.get(),
            exclusive_elements='NPS' if self.exclusive_var.get() else '',
        )

    def run(self) -> None:
        try:
            params = self._read_inputs()
        except Exception as e:
            show_error(self, e)
            return

        def compute():
            df = get_formulas_default_constraints(
                params['M_obs'],
                mass_tolerance_Da=params['tol_M_da'],
                elements=params['elements'],
                n_formulas_max=params['n_max'],
                dbe_range=params['dbe_range'],
                exclusive_elements=params['exclusive_elements'],
                check_lewis=params['check_lewis'],
                check_senior=params['check_senior'],
            )
            rows = []
            for idx, formula in enumerate(df.formula, start=1):
                cd = CompoundDict(formula)
                M = cd.mass
                mz_theo = M if params['adduct'] is None else params['adduct'].mass_to_mz(M)
                err = params['mz'] - mz_theo
                rows.append((
                    idx, hill_formula(cd), f'{M:.5f}', f'{mz_theo:.5f}', f'{err / mz_theo * 1e6:.2f}',
                    f'{err * 1e3:.3f}', f'{cd.DBE:g}', cd.is_lewis, cd.is_senior
                ))
            return rows

        def done(rows):
            self.run_button.state(['!disabled'])
            self._last_adduct = params['adduct_str']
            self.table.set_rows(rows)
            self.status_var.set(
                f"Found {len(rows)} candidate(s) for neutral mass {params['M_obs']:.5f} "
                f"(± {params['tol_M_da'] * 1e3:.3f} mDa)."
            )

        def failed(e):
            self.run_button.state(['!disabled'])
            self.status_var.set('Search failed.')
            show_error(self, e)

        self.run_button.state(['disabled'])
        self.status_var.set('Searching…')
        run_in_thread(self, compute, done, failed)


def main():
    from LipidCalculator.gui.common import run_standalone
    run_standalone(FormulaGeneratorFrame, 'Formula from m/z')


if __name__ == '__main__':
    main()
