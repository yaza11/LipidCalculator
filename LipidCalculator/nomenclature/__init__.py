"""Find lipids by names of several conventions and name them in a standardized way (see names.py)."""
from LipidCalculator.nomenclature.lipid import Lipid, NotSupported
from LipidCalculator.nomenclature.names import CONVENTIONS, all_names, lipid_name, parse_name, standard_name
