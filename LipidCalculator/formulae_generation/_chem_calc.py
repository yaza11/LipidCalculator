"""
from https://www.chemcalc.org/ws:

Molecule finder API example (from www.chemcalc.org)

Short example on how to use ChemCalc API to look for a molecular formula
from its monoisotopic mass (returns all closest possibilities). It is
useful for me on the interpretation of High Resolution Mass Spectra data.

The options dictionary does not need all parameters, as it has default
values for each of them. You can perform a 'naive' search, however it
is often a good idea to tune these parameters a bit further.

The data is retrieved as a dictonary, with 'options' entry with the used
parameters (including default values) as a dictionary and 'results',
a list with one entry (dictionary) for each molecular structure.

Personal tip: I like using Pandas to format results output to a table:
table = pd.DataFrame(mf_finder(mz)['results'])

If you use it, do not forget to cite the authors from the web API:
ChemCalc: a building block for tomorrow's chemical infrastructure.
Patiny, Luc; Borel, Alain Journal of Chemical Information and Modeling 2013.
DOI:10.1021/ci300563h
"""
import requests

chemcalcURL = 'https://www.chemcalc.org/chemcalc/em'


def mf_finder(mz):
    options = {
        'mfRange': 'C0-100H0-202N0-10O0-10F0-3Cl0-3Br0-1',
        'numberOfResultsOnly': False,
        'typedResult': False,
        'useUnsaturation': False,
        'minUnsaturation': 0,
        'maxUnsaturation': 50,
        'jcampBaseURL': 'http://www.chemcalc.org/service/jcamp/',
        'monoisotopicMass': mz,
        'jcampLink': True,
        # The 'jcamplink' returns a link to a file containing isotopic
        # distribution of retrieved molecular structure.
        'integerUnsaturation': False,
        # Ions/Radicals can have non-integer unsaturation
        'referenceVersion': '2013',
        'massRange': 0.5
        #              'minMass': -0.5,
        #              'maxMass': 0.5,
    }
    return requests.get(chemcalcURL, options).json()


if __name__ == '__main__':
    import pandas as pd

    res = mf_finder(180.063390)
    df = pd.DataFrame(res['results'])
