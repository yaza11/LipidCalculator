"""Main window combining all LipidCalculator GUI modules in one notebook."""
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable

from LipidCalculator.gui.adduct_gui import AdductCalculatorFrame
from LipidCalculator.gui.formula_gui import FormulaGeneratorFrame
from LipidCalculator.gui.ipl_gui import IPLBuilderFrame
from LipidCalculator.gui.isotope_gui import IsotopePatternFrame


class LipidCalculatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('LipidCalculator')
        self.geometry('1250x800')
        self.minsize(900, 600)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=4, pady=4)

        self.isotopes = IsotopePatternFrame(self.notebook)
        self.formulas = FormulaGeneratorFrame(self.notebook, send_targets={
            'Isotope pattern →': self._sender('isotopes'),
            'Adducts →': self._sender('adducts'),
        })
        self.adducts = AdductCalculatorFrame(
            self.notebook,
            send_targets_formula={'Isotope pattern →': self._sender('isotopes')},
            send_targets_mz={'Find formulas →': self._sender('formulas')},
        )
        self.ipls = IPLBuilderFrame(self.notebook, send_targets={
            'Isotope pattern →': self._sender('isotopes'),
            'Adducts →': self._sender('adducts'),
        })

        self.notebook.add(self.formulas, text='Formula from m/z')
        self.notebook.add(self.isotopes, text='Isotope pattern')
        self.notebook.add(self.adducts, text='Mass & adducts')
        self.notebook.add(self.ipls, text='IPL builder')

        self._build_menu()

    def _sender(self, attr: str) -> Callable[[dict], None]:
        """Switch to the tab stored in attribute attr and hand over the payload.
        The frame is looked up when sending, so it may be created later."""

        def send(payload: dict) -> None:
            frame = getattr(self, attr)
            self.notebook.select(frame)
            frame.set_input(payload)

        return send

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label='Quit', command=self.destroy)
        menubar.add_cascade(label='File', menu=file_menu)
        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label='About', command=self._about)
        menubar.add_cascade(label='Help', menu=help_menu)
        self.config(menu=menubar)

    def _about(self) -> None:
        messagebox.showinfo('About', 'LipidCalculator\nToolbox for dealing with intact polar lipids (IPLs).\n\n'
                                     'Modules: formula generation from m/z, isotope pattern prediction, '
                                     'mass & adduct calculation, IPL construction.', parent=self)


def main():
    LipidCalculatorApp().mainloop()


if __name__ == '__main__':
    main()
