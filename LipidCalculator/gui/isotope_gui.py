"""GUI for predicting isotope patterns and comparing them with measured ones."""
import re
import tkinter as tk
from tkinter import ttk

import numpy as np

from LipidCalculator.isotopes.isotope_pattern import IsotopePattern, direct_predict_d13C
from LipidCalculator.gui.common import (
    PAD, NEUTRAL, PlotCanvas, ResultTable, add_labeled, adduct_combobox, hill_formula,
    ion_composition, parse_adduct, parse_formula, run_in_thread, show_error
)

COLUMNS: list[str] = ['Peak', 'm/z', 'Rel. intensity (%)', 'Fraction']

# mass difference between 13C and 12C
DELTA_C13: float = 1.003355


def parse_peak_list(text: str) -> tuple[list[float], list[float]]:
    """Parse lines of 'm/z intensity' separated by whitespace, comma, semicolon or tab"""
    mzs, intensities = [], []
    for line_number, line in enumerate(text.strip().splitlines(), start=1):
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        parts = [p for p in re.split(r'[\s,;]+', line) if p]
        if len(parts) != 2:
            raise ValueError(f'line {line_number}: expected "m/z intensity", got "{line}"')
        mzs.append(float(parts[0]))
        intensities.append(float(parts[1]))
    return mzs, intensities


def bin_nominal(mzs, intensities, mz_mono: float, charge: int, n_bins: int = 2) -> list[float]:
    """Sum intensities around the nominal isotope positions M+0, M+1, ..."""
    mzs, intensities = np.asarray(mzs), np.asarray(intensities)
    out = []
    for k in range(n_bins):
        center = mz_mono + k * DELTA_C13 / charge
        mask = np.abs(mzs - center) < .3 / charge
        out.append(float(intensities[mask].sum()))
    return out


class IsotopePatternFrame(ttk.Frame):
    """Predict the isotope pattern of a formula/adduct and overlay a measured spectrum."""

    def __init__(self, master, **kwargs):
        super().__init__(master, padding=PAD * 2, **kwargs)
        self.formula_var = tk.StringVar(self, value='C43H88O3')
        self.adduct_var = tk.StringVar(self, value='[M+H]+')
        self.resolution_var = tk.StringVar(self, value='40000')
        self.min_rel_var = tk.StringVar(self, value='0.01')
        self.d2H_var = tk.StringVar(self, value='-200')
        self.status_var = tk.StringVar(self, value='Enter a formula and press "Predict".')
        self.d13C_var = tk.StringVar(self, value='')
        self._last: dict | None = None

        self._build()

    def _build(self) -> None:
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky='nsw', padx=(0, PAD * 2))

        inputs = ttk.LabelFrame(left, text='Input', padding=PAD)
        inputs.pack(fill='x')
        formula_entry = add_labeled(inputs, 0, 'Formula', ttk.Entry(inputs, textvariable=self.formula_var, width=18))
        formula_entry.bind('<Return>', lambda _: self.run())
        add_labeled(inputs, 1, 'Adduct', adduct_combobox(inputs, self.adduct_var))
        add_labeled(inputs, 2, 'Resolution', ttk.Entry(inputs, textvariable=self.resolution_var, width=10),
                    unit='(empty: fine structure)')
        add_labeled(inputs, 3, 'Hide peaks below', ttk.Entry(inputs, textvariable=self.min_rel_var, width=10),
                    unit='%')
        self.run_button = ttk.Button(inputs, text='Predict', command=self.run)
        self.run_button.grid(row=4, column=0, columnspan=3, sticky='ew', pady=PAD)

        measured = ttk.LabelFrame(left, text='Measured pattern (optional)', padding=PAD)
        measured.pack(fill='x', pady=(PAD * 2, 0))
        ttk.Label(measured, text='One peak per line: m/z intensity', foreground='gray').pack(anchor='w')
        self.measured_text = tk.Text(measured, width=30, height=7, font=('Consolas', 9))
        self.measured_text.pack(fill='x', pady=PAD)
        d2H_frame = ttk.Frame(measured)
        d2H_frame.pack(fill='x')
        ttk.Label(d2H_frame, text='Assumed δ²H (‰)').pack(side='left')
        ttk.Entry(d2H_frame, textvariable=self.d2H_var, width=8).pack(side='left', padx=PAD)
        btns = ttk.Frame(measured)
        btns.pack(fill='x', pady=PAD)
        ttk.Button(btns, text='Compare', command=self.run).pack(side='left')
        ttk.Button(btns, text='Clear', command=self._clear_measured).pack(side='left', padx=PAD)
        ttk.Label(measured, textvariable=self.d13C_var, justify='left', wraplength=260).pack(anchor='w')

        right = ttk.PanedWindow(self, orient='vertical')
        right.grid(row=0, column=1, sticky='nsew')
        self.plot = PlotCanvas(right, figsize=(6, 3.5))
        right.add(self.plot, weight=3)
        self.table = ResultTable(right, COLUMNS, height=6)
        right.add(self.table, weight=1)

        ttk.Label(self, textvariable=self.status_var, foreground='gray').grid(
            row=1, column=0, columnspan=2, sticky='w', pady=(PAD, 0))
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

    def _clear_measured(self) -> None:
        self.measured_text.delete('1.0', 'end')
        self.d13C_var.set('')
        if self._last is not None:
            self._last['measured'] = None
            self._draw(self._last)

    def set_input(self, payload: dict) -> None:
        if payload.get('formula'):
            self.formula_var.set(payload['formula'])
        if 'adduct' in payload:
            self.adduct_var.set(payload['adduct'] or NEUTRAL)
        self.run()

    def _read_inputs(self) -> dict:
        cd = parse_formula(self.formula_var.get())
        adduct_str = self.adduct_var.get().strip()
        adduct = parse_adduct(adduct_str)
        res_str = self.resolution_var.get().strip()
        resolution = int(float(res_str)) if res_str not in ('', '0') else None
        min_rel = float(self.min_rel_var.get() or 0)
        measured_text = self.measured_text.get('1.0', 'end').strip()
        measured = parse_peak_list(measured_text) if measured_text else None
        d2H = float(self.d2H_var.get())
        return dict(cd=cd, adduct=adduct, adduct_str=adduct_str, resolution=resolution, min_rel=min_rel, measured=measured, d2H=d2H)

    def run(self) -> None:
        try:
            params = self._read_inputs()
        except Exception as e:
            show_error(self, e)
            return

        def compute():
            adduct = params['adduct']
            # validates that the adduct can be formed
            ion = ion_composition(params['cd'], adduct)
            pattern = IsotopePattern.from_formula(
                hill_formula(params['cd']),
                adduct=params['adduct_str'] if adduct is not None else None,
                mass_resolution=params['resolution'],
                merge_method='weighted_average' if params['resolution'] else 'none',
            )
            order = np.argsort(pattern.masses)
            mzs = np.asarray(pattern.masses)[order]
            fractions = np.asarray(pattern.intensities)[order]
            return dict(params, ion=ion, mzs=mzs, fractions=fractions)

        def done(result):
            self.run_button.state(['!disabled'])
            self._last = result
            self._fill_table(result)
            self._estimate_d13C(result)
            self._draw(result)
            adduct = result['adduct']
            self.status_var.set(
                f"Ion {hill_formula(result['ion'])}"
                f"{'' if adduct is None else f' ({adduct})'}: {len(result['mzs'])} peaks, "
                f"monoisotopic m/z {result['mzs'][0]:.5f}"
            )

        def failed(e):
            self.run_button.state(['!disabled'])
            show_error(self, e)

        self.run_button.state(['disabled'])
        self.status_var.set('Calculating…')
        run_in_thread(self, compute, done, failed)

    @staticmethod
    def _charge(result: dict) -> int:
        return 1 if result['adduct'] is None else abs(result['adduct'].charge)

    def _visible(self, result: dict) -> tuple[np.ndarray, np.ndarray]:
        rel = result['fractions'] / result['fractions'].max() * 100
        mask = rel >= result['min_rel']
        return result['mzs'][mask], rel[mask]

    def _fill_table(self, result: dict) -> None:
        mzs, rel = self._visible(result)
        mz_mono = result['mzs'][0]
        z = self._charge(result)
        fractions = result['fractions'][np.isin(result['mzs'], mzs)]
        rows = [
            (f'M+{round((mz - mz_mono) * z)}', f'{mz:.5f}', f'{r:.3f}', f'{f:.5f}')
            for mz, r, f in zip(mzs, rel, fractions)
        ]
        self.table.set_rows(rows)

    def _estimate_d13C(self, result: dict) -> None:
        if result['measured'] is None:
            self.d13C_var.set('')
            return
        z = self._charge(result)
        mz_mono = result['mzs'][0]
        I0, I1 = bin_nominal(*result['measured'], mz_mono=mz_mono, charge=z)
        P0, P1 = bin_nominal(result['mzs'], result['fractions'], mz_mono=mz_mono, charge=z)
        if I0 <= 0:
            self.d13C_var.set(f'No measured peak near the monoisotopic m/z {mz_mono:.4f}.')
            return
        text = f'M+1/M measured: {I1 / I0:.4f}\nM+1/M predicted (natural): {P1 / P0:.4f}'
        if 'C' in result['ion'].composition and I1 > 0:
            d13C = direct_predict_d13C(hill_formula(result['ion']), (I0, I1), d2H=result['d2H'])
            text += f'\nEstimated δ¹³C: {d13C:.1f} ‰ (VPDB)'
        self.d13C_var.set(text)

    def _draw(self, result: dict) -> None:
        ax = self.plot.ax
        self.plot.clear()
        mzs, rel = self._visible(result)
        ax.vlines(mzs, 0, rel, color='tab:blue', label='predicted')
        # label the most intense peaks
        for mz, r in zip(mzs, rel):
            if r >= 5:
                ax.annotate(f'{mz:.4f}', (mz, r), textcoords='offset points', xytext=(0, 2),
                            ha='center', fontsize=7)

        if result['measured'] is not None:
            m_mzs, m_int = map(np.asarray, result['measured'])
            m_rel = m_int / m_int.max() * 100
            # mirror plot
            ax.vlines(m_mzs, 0, -m_rel, color='tab:orange', label='measured')
            ax.axhline(0, color='k', linewidth=.5)
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.set_ylim(0, 110)

        span = max(mzs.max() - mzs.min(), 1)
        ax.set_xlim(mzs.min() - .1 * span - .5, mzs.max() + .1 * span + .5)
        ax.set_xlabel('m/z' if result['adduct'] is not None else 'mass (Da)')
        ax.set_ylabel('relative intensity (%)')
        ax.set_title(hill_formula(result['ion']) + ('' if result['adduct'] is None else f"  {result['adduct']}"),
                     fontsize=10)
        self.plot.draw()


def main():
    from LipidCalculator.gui.common import run_standalone
    run_standalone(IsotopePatternFrame, 'Isotope pattern')


if __name__ == '__main__':
    main()
