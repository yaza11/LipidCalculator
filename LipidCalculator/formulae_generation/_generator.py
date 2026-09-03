"""Module for generating a list of possible Formulae given some constraints."""
import itertools
import re

import pandas as pd

data_table: pd.DataFrame = pd.read_pickle("data_table")

ELECTRON = 0.000549  # difference in m/z caused by electron (if z=1)

# relative double bond equivalent values for DBE calcs
dbe_vals = {
    'C': 1,
    'H': -0.5,
    'O': 0,
    'N': 0.5,
    'S': 0,
    'P': 0.5,
    'Na': -0.5,
    'K': -0.5,
    'F': -0.5,
    'Cl': -0.5,
    'Br': -0.5,
    'I': -0.5,
    'Si': 1
}


class Formula:
    # from https://github.com/vinnyjim/Mass-Spectrometry-Chemical-Formula-Assigner/blob/main/Formula%20Calculator%201.1.ipynb

    def __init__(
            self,
            input_formula: str,
            error_type: str = 'ppm',
            max_return: int = 10,
            max_delta: float | int = 1,
            nitrogen_rule: str = 'ignore',
            DBE_min: float | int = -1,
            DBE_max: float | int = 10,
            decimal_places: int = 4,
            charge_state: int = 1
    ) -> None:
        """Class for assigning formulae to masses

        Parameters
        ----------
        input_formula: str
            passed to make_formulae function. String which represents
            numbers of each element. Should take the form: "Cx-y Hm-n" etc.,
            for example "C20-22 H38-42 N0-6 O3-4 Na0-1"
        error_type: str
            Either "mmu" (milli mass units),
            "ppm" (parts per million),
            or "amu" (atomic mass units)
        max_return: int
            Number of formulae to return per m/z value in subsequent function calls
        max_delta: float | int
            Largest acceptable mass error. No formulae with a mass error larger
            than this will be returned
        nitrogen_rule: str
            Determines the kind of ions to consider, based on DBE values.
            Default is "ignore". "even e ions" considers even-electron ions,
            i.e. odd-DBE ions. "odd e ions" considers odd-electron ions,
            i.e. even-DBE ions (radical ions)
        decimal_places: Specifies the number of decimal places returned values are rounded to.
        charge_state: Charge state of ions to consider. Positive or negative non-zero integer.


        """
        self.error_type: str = error_type
        self.max_return: int = max_return
        self.max_delta: float | int = max_delta
        self.formula: str = input_formula
        self.nitrogen_rule: str = nitrogen_rule
        self.DBE_min: float | int = DBE_min
        self.DBE_max: float | int = DBE_max
        self.decimal_places: int = decimal_places
        self.charge_state: int = charge_state
        self.formulae_combinations = self.make_formulae()

        if self.error_type not in ['mmu', 'amu', 'ppm']:
            raise Exception(f"error_type provided is '{error_type}', should be either 'mmu', 'amu' or 'ppm'")

        if self.nitrogen_rule not in ['ignore', 'even e ions', 'odd e ions']:
            raise Exception(
                f"error_type provided is '{nitrogen_rule}', should be either 'ignore', 'even e ions' or 'odd e ions'")

        if self.charge_state == 0 or type(self.charge_state) == float:
            raise Exception(
                f"charge_state provided is {charge_state} - {type(charge_state)}, should be a non-zero integer'")

    def make_formulae(self):
        """Function to return df of all possible formulae and their masses, given a formula range. Adherance to the nitrogen rule can also be specified.

        Parameters
        ----------
        input_string: string which represents numbers of each element. Should take the form:
        "Cx-y Hm-n" etc., for example "C20-22 H38-42 N0-6 O3-4 Na0-1"

        """
        list_of_substrings: list[str | None] = []

        # this regex captures groups separated by a "space" character
        for substring in re.findall(re.compile(r'([\S]*)'), self.formula):
            list_of_substrings.append(substring)
        # filter out the None from regex - why does this occur?
        parsed_list_of_substrings: list[str] = list(filter(None, list_of_substrings))

        parsed_formula: dict[str, tuple[str, str]] = {}
        list_of_elements: list[str] = []

        for item in parsed_list_of_substrings:
            if "-" in item:
                # this regex splits substring into element and lower/upper bounds for range
                for components in re.findall(re.compile(r'([A-Za-z]*)([\d]*)-([\d]*)'), item):
                    parsed_formula[components[0]] = (components[1], components[2])
                    list_of_elements.append(components[0])

        # For elements parsed out of the above formula string, look up accurate monoisotopic masses and DBE values
        element_masses = {}
        element_dbe = {}

        for i in list_of_elements:
            subset = data_table[data_table['symbol'] == i]
            element_masses[i] = float(
                subset.loc[[subset['abundance'].idxmax()]]['mass'])  # this grabs the mass of the most abundant isotope
            element_dbe[i] = dbe_vals[i]  # this grabs the corresponding DBE value for each element

        parsed_formula_ranges = {}
        for k, v in parsed_formula.items():
            parsed_formula_ranges[k] = (range(int(v[0]), int(
                v[1]) + 1))  # populate a dict with key=element and values=lower and upper bounds of element count

        list_of_combinations = []
        for v in itertools.product(*list(parsed_formula_ranges.values())):
            list_of_combinations.append(v)  # add each possible combination of each element number to a list

        df_of_combinations = pd.DataFrame(list_of_combinations,
                                          columns=list_of_elements)  # make a df of element combinations

        masses_of_combinations = pd.DataFrame(df_of_combinations * list(element_masses.values())).sum(axis=1).rename(
            'Mass')
        dbes_of_combinations = (pd.DataFrame(df_of_combinations * list(element_dbe.values())).sum(axis=1).rename(
            'DBE')) + 1  # add 1 to DBE to account for +2H in DBE calc for saturation
        combos_with_appended_mass = pd.concat([df_of_combinations, masses_of_combinations, dbes_of_combinations],
                                              axis=1)

        # drop either side of DBE limits
        combos_with_appended_mass = combos_with_appended_mass[
            ~((combos_with_appended_mass['DBE'] < self.DBE_min) | (combos_with_appended_mass['DBE'] > self.DBE_max))]
        combos_with_appended_mass = combos_with_appended_mass.sort_values('Mass')

        if self.nitrogen_rule == 'ignore':
            pass
        elif self.nitrogen_rule == 'even e ions':
            combos_with_appended_mass = combos_with_appended_mass[combos_with_appended_mass['DBE'].apply(
                lambda x: x % 1 == 0.5)]  # select only rows where DBE isn't whole number
        elif self.nitrogen_rule == 'odd e ions':
            combos_with_appended_mass = combos_with_appended_mass[combos_with_appended_mass['DBE'].apply(
                lambda x: x % 1 == 0)]  # select only rows where DBE is a whole number
        else:
            raise NotImplementedError(f"{self.nitrogen_rule} not a valid nitrogen rule")

        combos_with_appended_mass['Mass'] = (combos_with_appended_mass['Mass'] / abs(self.charge_state)) - (
                self.charge_state * ELECTRON)
        combos_with_appended_mass.rename(columns={'Mass': 'm/z'}, inplace=True)

        return combos_with_appended_mass

    def assign_from_mass(self, input_mass=float, pass_to_list_func=False):
        """Function to assign formulae to a given mass or m/z value, ranked by mass error

        Function takes in an accurate mass or m/z and calculates a given type of mass error against masses contained within a formulae table.
        Formulae are then ranked by ascending absolute mass error (Delta) and n lowest mass error candidate formulae are returned.

        Parameters
        ----------
        input_mass: Accurate measured mass
        pass_to_list_func: Control behind-the-scenes behaviour based on whether output is used directly by mass_list_formulae_assigner function.
        Default is False for calls of assign_from_mass function, and overwritten with True when called from within mass_list_formulae_assigner function.


        """
        formula_combinations = self.formulae_combinations
        error_type = self.error_type
        max_return = self.max_return
        max_delta = self.max_delta
        decimal_places = self.decimal_places

        # trim combinations df by: calculating absolute mass error, sorting the result, taking top n (lowest error) values
        formula_combinations = formula_combinations.iloc[
            (formula_combinations['m/z'] - input_mass).abs().argsort()[:max_return]].reset_index(drop=True)

        if error_type == "mmu":
            formula_combinations['Delta'] = (input_mass - formula_combinations['m/z']) * 1000
        if error_type == "amu":
            formula_combinations['Delta'] = (input_mass - formula_combinations['m/z'])
        if error_type == "ppm":
            formula_combinations['Delta'] = ((input_mass - formula_combinations['m/z']) / input_mass) * 1000000

        formula_combinations = formula_combinations[formula_combinations[
                                                        'Delta'].abs() <= max_delta]  # only take forwards results with error less than max_delta

        formula_strings = []
        col_names = formula_combinations.columns[
                    :-1]  # select column names except last - this corresponds to list of elements

        for i in range(len(formula_combinations)):
            row = formula_combinations.iloc[i, :-3]
            formula_tuple = (list(zip(col_names, row.astype(int))))
            formula_string = (' '.join(map(lambda x: str(x[0]) + '' + str(x[1]), formula_tuple)))
            formula_strings.append(formula_string)  # nice print version of formula

        formula_strings = pd.Series(formula_strings, dtype='object').rename(
            'Formula')  # .reset_index(drop=True) #dtype specified as object here - is this ok?
        formula_combinations = pd.concat([formula_combinations, formula_strings], axis=1)
        dbe_col = formula_combinations['DBE']
        formula_combinations = formula_combinations.drop(['DBE'], axis=1)
        formula_combinations = pd.concat([formula_combinations, dbe_col],
                                         axis=1)  # this and above 2 lines are used to reorder the df for nice print

        formula_combinations = formula_combinations.round(decimals={
            'm/z': decimal_places,
            'Delta': 1
        })  # round the mass value to 'decimal_places' DP, and Delta to 1 DP (no need to assign as variable?)

        if pass_to_list_func == True:  # this is to ensure the right column name is present if passed to the mass_list function
            pass
        elif error_type == "mmu":
            formula_combinations.rename(columns={'Delta': 'Delta (mmu)'}, inplace=True)
        elif error_type == "amu":
            formula_combinations.rename(columns={'Delta': 'Delta (amu)'}, inplace=True)
        elif error_type == "ppm":
            formula_combinations.rename(columns={'Delta': 'Delta (ppm)'}, inplace=True)

        return formula_combinations

    def mass_list_formulae_assigner(self, xy_data, multiindex=False):
        """Function to assign formulae to a given list of masses, ranked by mass error

        Function provides an ease-of-use variant of assign_from_mass function for assigning formulae to each m/z value in a mass list.
        Each m/z value in the mass list provided is supplied to assign_from_mass function and results are assembled into a DataFrame.

        Parameters
        ----------
        xy_data: Mass list data to perform formulae assignment on. Accepted types for xy_data are list or pd.DataFrame.
        For pd.DataFrame, it is assumed the first column contains m/z values.
        multiindex: Sets behaviour for the returned DataFrame. Default behaviour is False, which returns unique rows for each m/z-formula pair. Setting to true will return
        the resultant DataFrame with a multiindex of m/z and formula.


        """
        formula_combinations = self.formulae_combinations
        error_type = self.error_type
        max_return = self.max_return
        max_delta = self.max_delta
        decimal_places = self.decimal_places

        df = pd.DataFrame()

        ##Add in some peak-picking functionality?
        if isinstance(xy_data, pd.DataFrame):
            for i in range(len(xy_data)):
                mz = xy_data.iloc[i, 0]
                mz_assignment = self.assign_from_mass(input_mass=mz, pass_to_list_func=True)
                mz_assignment['Exp. m/z'] = mz
                df = pd.concat([df, mz_assignment], axis=0)

            return_data = df.copy()
            return_data = return_data[['Exp. m/z', 'Formula', 'm/z', 'Delta', 'DBE']].copy()
            return_data.rename(columns={'m/z': 'Theo. m/z'}, inplace=True)
            return_data = return_data.round(decimals={
                'Exp. m/z': decimal_places,
                'Theo. m/z': decimal_places
            })
            if multiindex == False:
                return_data = return_data.reset_index(drop=True)
            elif multiindex == True:
                return_data = return_data.set_index(['Exp. m/z', 'Formula'])

        if isinstance(xy_data, list):
            for i in range(len(xy_data)):
                mz = xy_data[i]
                mz_assignment = self.assign_from_mass(input_mass=mz, pass_to_list_func=True)
                mz_assignment['Exp. m/z'] = mz
                df = pd.concat([df, mz_assignment], axis=0)

            return_data = df.copy()
            return_data = return_data[['Exp. m/z', 'Formula', 'm/z', 'Delta', 'DBE']].copy()
            return_data.rename(columns={'m/z': 'Theo. m/z'}, inplace=True)
            return_data = return_data.round(decimals={
                'Exp. m/z': decimal_places,
                'Theo. m/z': decimal_places
            })

            if multiindex == False:
                return_data = return_data.reset_index(drop=True)
            elif multiindex == True:
                return_data = return_data.set_index(['Exp. m/z', 'Formula'])

        if type(xy_data) not in [pd.DataFrame, list]:
            raise Exception(f"data type of data provided is '{type(xy_data)}', should be either pd.DataFrame or list'")

        if error_type == "mmu":
            return_data.rename(columns={'Delta': 'Delta (mmu)'}, inplace=True)
        elif error_type == "amu":
            return_data.rename(columns={'Delta': 'Delta (amu)'}, inplace=True)
        elif error_type == "ppm":
            return_data.rename(columns={'Delta': 'Delta (ppm)'}, inplace=True)

        return (return_data)


formula_assigner = Formula("C0-40 H0-80 N0-6 O1-10 Na0-1", error_type='ppm', max_delta=5, nitrogen_rule='even e ions',
                           DBE_max=20, charge_state=1)
# let's see if we can get a match from the reserpine [M+H]+ ion (C33 H41 N2 O9) with a monoisotopic mass from chemdraw of 609.2807
formula_assigner.assign_from_mass(609.2807)
