"""From https://fiehnlab.ucdavis.edu/downloads/projects/Seven_Golden_Rules/1471-2105-8-105.pdf"""
import pandas as pd
import numpy as np

element_limits = pd.DataFrame(
    data=[
        [(0, 500), "DNP", 29, 72, 10, 18, 4, 7, 15, 8, 5, np.nan],
        [(0, 500), "Wiley", 39, 72, 20, 20, 9, 10, 16, 10, 4, 8],
        [(500, 1000), "DNP", 66, 126, 25, 27, 6, 8, 16, 11, 8, np.nan],
        [(500, 1000), "Wiley", 78, 126, 20, 27, 9, 14, 34, 12, 8, 14],
        [(1000, 2000), "DNP", 115, 236, 32, 63, 6, 8, 16, 11, 8, np.nan],
        [(1000, 2000), "Wiley", 156, 180, 20, 40, 9, 14, 48, 12, 10, 15],
        [(2000, 3000), "DNP", 162, 208, 48, 78, 6, 9, 16, 11, 8, np.nan],
    ],
    columns=[
        "Mass_Range_Da", "Library",
        "C_max", "H_max", "N_max", "O_max", "P_max",
        "S_max", "F_max", "Cl_max", "Br_max", "Si_max"
    ]
).drop(columns='Library').groupby("Mass_Range_Da").max()

# common range: 99.7%
# extended range: 99.99%
element_ratios = pd.DataFrame({
    "Element_Ratio": [
        "H/C", "F/C", "Cl/C", "Br/C", "N/C",
        "O/C", "P/C", "S/C", "Si/C"
    ],
    "Common_Min": [
        0.2, 0, 0, 0, 0, 0, 0, 0, 0
    ],
    "Common_Max": [
        3.1, 1.5, 0.8, 0.8, 1.3, 1.2, 0.3, 0.8, 0.5
    ],
    "Extended_Min": [
        0.1, 0, 0, 0, 0, 0, 0, 0, 0
    ],
    "Extended_Max": [
        6, 6, 2, 2, 4, 3, 2, 3, 1
    ],
    "Extreme_Range": [
        "<0.1 and 6-9",
        ">1.5",
        ">0.8",
        ">0.8",
        ">1.3",
        ">1.2",
        ">0.3",
        ">0.8",
        ">0.5"
    ]
}).drop(columns="Extreme_Range")
