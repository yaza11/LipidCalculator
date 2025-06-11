import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from IsoSpecPy import IsoTotalProb

from LipidCalculator.compound_creation import Compound
from LipidCalculator.compound_creation.formula_parser import CompoundDict
from LipidCalculator.isotopes.isotopes import most_common_isotopes_to_mass_number


class IsotopeSeries:
    def __init__(
            self,
            compound: Compound,
            total_abundance_percent: int | float = 99.9
    ) -> None:
        self.compound: Compound = compound
        # self._set_all_isotopes(min_abundance_percent=min_abundance_percent)
        res = IsoTotalProb(.999, formula=self.compound.formula)
        self.masses = np.array(list(res.masses))
        self.probabilities = np.array(list(res.probs))

    def plt_isotopes(self, **kwargs) -> None:
        def format_isotope_eq(formula: str) -> str:
            cd = CompoundDict(formula)
            d = cd.composition

            s = ''
            start = r'$'
            end = r'$'

            for k, v in d.items():
                el, *num = k.split('[')

                s += start
                # mass number high in front
                if len(num) == 1:
                    num = num[0]
                    num = num[:-1]
                    if most_common_isotopes_to_mass_number[el] != int(num):
                        print(f'{k}: {most_common_isotopes_to_mass_number[el]}', num)
                        s += r'^{' + num + r'}'
                s += r'\mathrm{' + el + r'}'
                if v > 1:
                    s += r'_{' + str(v) + '}'

                s += end
            return s

        xs: pd.Series = self.masses
        ys: pd.Series = self.probabilities * 100
        # labels: list[str] = [format_isotope_eq(l) for l in df.index]

        plt.figure()
        plt.stem(xs, ys, markerfmt='', basefmt='')
        # for x, y, l in zip(xs, ys, labels):
        #     plt.text(x, y, l, ha='left', va='bottom', rotation=60)
        plt.xlim(xs.min() - 2, xs.max() + 2)
        plt.ylim(0, ys.max() * 1.2)
        plt.ylabel('Abundance in %')
        plt.xlabel('Mass in Da')
        plt.title(f'Isotopes of {format_isotope_eq(self.compound.formula)}')
        plt.show()
