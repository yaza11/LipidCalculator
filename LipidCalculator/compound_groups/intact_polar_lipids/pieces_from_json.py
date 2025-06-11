import os
import json

import pandas as pd

PLACEHOLDER_ELEMENTS: list[str] = ['Fr', 'Cs', 'Rb']

# C atoms beyond placeholders have to be included in chain
CORE_IMPLICIT_CHAIN: dict[tuple[str, str], tuple[int, int]] = {
    ('DGTS', 'Rb'): (4, 0),
    ('DGTS', 'Cs'): (1, 0),
    ('BL', 'Rb'): (4, 0),
    ('BL', 'Cs'): (1, 0),
    ('OL', 'Cs'): (4, 0),
    ('OL', 'Rb'): (1, 0),
    ('DAG', 'Cs'): (1, 0),
    ('DAG', 'Rb'): (1, 0),
    ('DEG', 'Cs'): (1, 0),
    ('DEG', 'Rb'): (1, 0),
    ('CER', 'Cs'): (1, 0),
    ('CER', 'Rb'): (5, 1),
    ('AEG', 'Cs'): (1, 0),
    ('AEG', 'Rb'): (1, 0)
}

path_file_blocks = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'package.json')
with open(path_file_blocks, 'r') as f:
    IPL_PIECES: dict[str, str | list] = json.load(f)

ABBREVIATIONS: list[str] = []
GROUPS: list[str] = []
_entries = []
for name_type, members in IPL_PIECES.items():
    if name_type == 'group':
        continue
    for struct in members:
        ABBREVIATIONS.append(struct['abbreviation'])
        GROUPS.append(name_type)
        _entries.append(struct | dict(group=name_type))
DATA_FRAME_IPL_PIECES = pd.DataFrame(_entries)

ABBREVIATION_TO_GROUP: dict[str, str] = dict(zip(ABBREVIATIONS, GROUPS))


def get_entry_by_abbreviation(name: str) -> pd.Series:
    match = DATA_FRAME_IPL_PIECES.abbreviation == name
    assert match.sum() == 1, f'got {match.sum()} matches for {name}'
    return DATA_FRAME_IPL_PIECES.loc[match, :]


def get_smiles(name: str) -> str:
    e = get_entry_by_abbreviation(name)
    return e.SMILES.values[0]


if __name__ == '__main__':
    print(get_entry_by_abbreviation('DAG'))
    print(get_smiles('DAG'))
