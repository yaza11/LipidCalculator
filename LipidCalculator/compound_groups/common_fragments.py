import pandas as pd

from LipidCalculator.compound_creation.formula_parser import CompoundDict

# map compound group to common fragments & losses (positive ion mode)
# fragments are m/z of singly charged ions, losses are neutral masses
# keys use the abbreviations from package.json, '<head>-<core>' if the value depends on the core lipid
# loss keys also name the precursor adduct, losses from [M+NH4]+ include the NH3 of the adduct
COMMON_DIAGNOSTIC_FRAGMENTS: dict[str, str] = {
    # phospholipids
    'PC': 'C5H15NO4P+',  # phosphocholine, 184.0733
    # aminolipids
    'DGTS': 'C10H22NO5+',  # glyceryl-trimethylhomoserine, 236.1492
    'BL': 'C10H22NO5+',  # 236.1492
    'TMHS-DEG': 'C7H14NO2+',  # trimethylhomoserine, 144.1019
    'OL': 'C5H11N2O+',  # ornithine - H2O, 115.0866
    'TM-OL': 'C5H10NO2+',  # 116.0706
    # archaeal core lipids
    '1G-AR': 'C43H89O3+',  # [AR + H]+, 653.6806
    '2G-AR': 'C43H89O3+',  # [AR + H]+, 653.6806
    'P-AR': 'C23H50O6P+',  # 453.3340
    # sphingolipids
    'CER': 'C18H34N+',  # d18:1 sphingosine - 2 H2O, 264.2686
    # quinones
    'MK': 'C12H11O2+',  # 187.0754
    'UQ': 'C10H13O4+',  # 197.0808
}
COMMON_DIAGNOSTIC_LOSSES: dict[str, str] = {
    # phospholipids
    'PA-DAG [M+NH4]+': 'H6NO4P',  # H3PO4 + NH3, 115.0034
    'PG-DAG [M+NH4]+': 'C3H12NO6P',  # 189.0402
    'PI-DAG [M+NH4]+': 'C6H16NO9P',  # 277.0563
    'PE-DAG [M+H]+': 'C2H8NO4P',  # 141.0191
    'PE-DEG [M+H]+': 'C2H5N',  # aziridine, 43.0422
    'PME-DAG [M+H]+': 'C3H10NO4P',  # 155.0347
    'PME-DEG [M+H]+': 'C3H7N',  # 57.0578
    'PDME-DAG [M+H]+': 'C4H12NO4P',  # 169.0504
    'PnE-DAG [M+H]+': 'C2H8NO3P',  # 125.0242
    'PS-DAG [M+H]+': 'C3H8NO6P',  # 185.0089
    # aminolipids
    'OL [M+H]+': 'C5H9NO2',  # 115.0633
    'APT-DAG [M+H]+': 'C5H14NO7P',  # phospho-APT, 231.0508
    'APT-DEG [M+H]+': 'C5H11NO3',  # APT - H2O, 133.0739
    'N,N,N-3Me-APT-DAG [M+H]+': 'C8H20NO7P',  # 273.0977
    'N,N,N-3Me-APT-DEG [M+H]+': 'C8H17NO3',  # 175.1208
    # glycolipids
    '1G-DAG [M+NH4]+': 'C6H15NO6',  # hexose + NH3, 197.0899
    '2G-DAG [M+NH4]+': 'C12H23NO10',  # 2 hexose - H2O + NH3, 341.1322
    'SQ-DAG [M+NH4]+': 'C6H15NO8S',  # sulfoquinovose + NH3, 261.0518
    '1G-AR [M+NH4]+': 'C6H13NO5',  # hexose - H2O + NH3, 179.0794
    '2G-AR [M+NH4]+': 'C12H23NO10',  # 2 hexose - H2O + NH3, 341.1322
}

DIAGNOSTIC_FRAGMENTS_AND_LOSSES: pd.DataFrame = pd.DataFrame(
    data={
        'formula_fragment': COMMON_DIAGNOSTIC_FRAGMENTS, 'formula_loss': COMMON_DIAGNOSTIC_LOSSES}
)
DIAGNOSTIC_FRAGMENTS_AND_LOSSES.loc[:, 'mz_fragment'] = DIAGNOSTIC_FRAGMENTS_AND_LOSSES.formula_fragment.apply(
    lambda x: CompoundDict(x).mass if x == x else None)
DIAGNOSTIC_FRAGMENTS_AND_LOSSES.loc[:, 'mz_loss'] = DIAGNOSTIC_FRAGMENTS_AND_LOSSES.formula_loss.apply(
    lambda x: CompoundDict(x).mass if x == x else None)

if __name__ == '__main__':
    pass
