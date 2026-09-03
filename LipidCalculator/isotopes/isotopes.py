import pandas as pd
import os

from LipidCalculator.consts import m_e

IN_FILE: str = 'isotopes.txt'

# column names in csv
c_num: str = 'Atomic Number'
c_el: str = 'Element'
c_mnum: str = 'Mass number'
c_mass: str = 'Relative Atomic Mass'
c_comp: str = 'Isotopic Composition'
c_weight: str = 'Standard Atomic Weight'
c_notes: str = 'Notes'


def create_df():
    """Parse txt file to create more structured csv file"""
    lines = []
    # previous_entries = None
    with open(IN_FILE, 'r') as f:
        for it, line in enumerate(f):
            if it <= 2:
                continue
            entries = line.split('\t')
            entries = [entry.strip(' \n') for entry in entries]
            # if n_entries := len(entries) < 7:
            #     entries.extend(previous_entries[6 - n_entries:])
            lines.append(entries)
            # previous_entries = entries

    df = pd.DataFrame(lines[1:-1])

    df.columns = lines[0]

    df_mean = df.iloc[:, :5].copy()
    df_mean[c_mass] = df_mean.apply(
        lambda row: row[c_mass].split('(', 1)[0].replace(' ', '')
        if len(row[c_mass]) > 0 else '0',
        axis=1
    )

    df_mean[c_comp] = df_mean.apply(
        lambda row: row[c_comp].split('(', 1)[0].replace(' ', '')
        if len(row[c_comp]) > 0 else '0',
        axis=1
    )

    df_mean.index = df_mean.apply(
        lambda row: f"{row['Element']}[{row['Mass number']}]",
        axis=1
    )

    df_mean = df_mean.astype({
        c_num: 'int',
        c_el: 'str',
        c_mnum: 'int',
        c_mass: 'float',
        c_comp: 'float'
    })
    df_mean.to_csv('Isotopes.csv')
    return df_mean


# df = create_df()

isotope_properties: pd.DataFrame = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'Isotopes.csv'),
    index_col='Unnamed: 0'
)
isotope_properties.loc['+[0]'] = [0, '+e', 0, -m_e, 1]
isotope_properties.loc['-[0]'] = [0, '-e', 0, m_e, 1]

# create table only containing the most common isotope for each element
mask_most_common = isotope_properties.groupby(by=c_el)[c_comp].transform('max') == isotope_properties[c_comp]
most_common_isotopes: pd.DataFrame = isotope_properties.loc[mask_most_common, :]
most_common_isotopes.index = most_common_isotopes[c_el]
most_common_isotopes_to_mass_number: dict[str, int] = (
    dict(zip(most_common_isotopes[c_el], most_common_isotopes[c_mnum])))
most_common_isotopes_to_mass_number['+'] = 0
most_common_isotopes_to_mass_number['-'] = 0

element_to_most_common_isotope_notation: dict[str, str] = {row.Element: f'{row.Element}[{row.loc['Mass number']}]' for
                                                           _, row in most_common_isotopes.iterrows()}


class Isotope:
    """Container for isotope properties."""

    def __init__(self, element: str, mass_number: int | str | None = None):
        if (mass_number is None) and ('[' in element):
            element, mass_number = element.split('[', 1)
            mass_number = mass_number.strip(']')
        elif mass_number is None:
            mass_number = most_common_isotopes_to_mass_number[element]
        mass_number = int(mass_number)
        self.index = f'{element}[{mass_number}]'
        row = isotope_properties.loc[self.index]
        self.number = row['Atomic Number']
        self.element = element
        self.mass_number = mass_number
        self.mass = row['Relative Atomic Mass']
        self.ratio = row['Isotopic Composition']

    def __repr__(self):
        return str(self.__dict__)


def get_all_isotopes(element: str) -> list[Isotope]:
    mass_numbers = isotope_properties.loc[isotope_properties[c_el] == element, c_mnum]
    isotopes = [Isotope(element, mass_number) for mass_number in mass_numbers]
    return isotopes


def isotope_mass(element: str, monoisotopic: bool = True) -> float:
    if monoisotopic:
        iso = Isotope(element)
        return iso.mass
    else:
        assert ('[' not in element) and (']' not in element), \
            'cannot specify isotope when not using monoisotopic elements'
        isos = get_all_isotopes(element)
        return sum([iso.mass * iso.ratio for iso in isos])
