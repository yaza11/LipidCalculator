import os
import pandas as pd

df = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'elements.txt'),
    sep='\t',
    names=['Z', 'Abbreviation'],
    index_col=None,
    usecols=[0, 1]
)

elements = df.Abbreviation.tolist()
atomic_numbers = df.Z.tolist()
