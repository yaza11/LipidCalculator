# calculating the double bond equivalent of a formula
dbs: dict[float, list[str]] = {
    1: 'C Si'.split(),
    -1 / 2: 'H F Cl Br I'.split(),
    1 / 2: 'N P'.split(),
    0: 'O S'.split(),
}
EL_TO_DBE = {}
for db, els in dbs.items():
    for el in els:
        EL_TO_DBE[el] = db

# Valence electron contributions for the Lewis check
# (how many electrons can participate in a bond)
VALENCE_ELECTRONS = {
    "C": 4, "H": 1, "N": 5, "O": 6,
    "P": 5, "S": 6, "F": 7, "Cl": 7,
    "Br": 7, "I": 7, "Si": 4
}

# Conventional valences for the Senior check
# (how many bonds an atom usually has in a molecule)
SENIOR_VALENCES = {
    "C": 4, "H": 1, "N": 3, "O": 2,
    "P": 3, "S": 2, "F": 1, "Cl": 1,
    "Br": 1, "I": 1, "Si": 4
}
