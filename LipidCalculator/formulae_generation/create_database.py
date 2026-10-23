"""
Build an annotated, filterable DuckDB table from the candidate files written by create_list.py

Example for querying the table afterward:
>>> import duckdb
>>> con = duckdb.connect(path_db, read_only=True)
>>> df = con.sql('SELECT * FROM formulas WHERE M BETWEEN 760.5 AND 760.6 AND Senior AND nP = 1').df()
"""
import glob
import os

import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm

from LipidCalculator.formulae_generation.constants_for_checks import EL_TO_DBE, VALENCE_ELECTRONS, SENIOR_VALENCES
from LipidCalculator.formulae_generation.create_list import ELEMENTS, MASSES

TABLE_NAME: str = 'formulas'
COUNT_COLUMNS: dict[str, str] = {el: f'n{el}' for el in ELEMENTS}
# H can exceed 255 (H/C up to 6 for up to 130 C), all other counts are small
COUNT_TYPES: dict[str, str] = {el: 'USMALLINT' if el == 'H' else 'UTINYINT' for el in ELEMENTS}


def _weighted_sum(weights: dict[str, float]) -> str:
    """SQL expression for sum(weight_el * n_el)."""
    # .item() turns numpy scalars into python numbers, whose repr is valid SQL
    return ' + '.join(
        f'{getattr(weights[el], "item", lambda: weights[el])()!r} * {COUNT_COLUMNS[el]}' for el in ELEMENTS
    )


def _formula_expression() -> str:
    """SQL expression for the formula string, e.g. C40H80NO8P (counts of 1 are omitted)."""
    parts: list[str] = [
        f"CASE WHEN {col} = 0 THEN '' WHEN {col} = 1 THEN '{el}' ELSE '{el}' || CAST({col} AS VARCHAR) END"
        for el, col in COUNT_COLUMNS.items()
    ]
    return ' || '.join(parts)


def _senior_expression() -> str:
    """SQL expression mirroring CompoundDict.is_senior."""
    v: str = f'({_weighted_sum(SENIOR_VALENCES)})'
    # highest valence among the elements present in the formula
    v_max: str = 'GREATEST(' + ', '.join(
        f'CASE WHEN {COUNT_COLUMNS[el]} > 0 THEN {SENIOR_VALENCES[el]} ELSE 0 END' for el in ELEMENTS
    ) + ')'
    n_atoms: str = '(' + ' + '.join(COUNT_COLUMNS.values()) + ')'
    return f'({v} % 2 = 0 AND {v} >= 2 * {v_max} AND {v} >= 2 * ({n_atoms} - 1))'


def _select_expression() -> str:
    """Columns of the annotated table, computed from the element counts."""
    n_c: str = COUNT_COLUMNS['C']
    columns: list[str] = [
        f'CAST({_weighted_sum(dict(zip(ELEMENTS, MASSES)))} AS DOUBLE) AS M',
        f'{_formula_expression()} AS formula',
        *[f'CAST({COUNT_COLUMNS[el]} / {n_c} AS FLOAT) AS {el}_C' for el in ELEMENTS if el != 'C'],
        *[f'CAST({col} AS {COUNT_TYPES[el]}) AS {col}' for el, col in COUNT_COLUMNS.items()],
        f'CAST(1 + {_weighted_sum(EL_TO_DBE)} AS FLOAT) AS DBE',
        f'({_weighted_sum(VALENCE_ELECTRONS)}) % 2 = 0 AS Lewis',
        f'{_senior_expression()} AS Senior',
    ]
    return ',\n    '.join(columns)


def create_database(path_folder_candidates: str, path_db: str) -> int:
    """
    Read all C{n}.npy files from path_folder_candidates and write the annotated table to the DuckDB file path_db.

    The table is sorted by mass, so filtering by mass only has to read the matching part of the table.
    Returns the number of rows.
    """
    paths: list[str] = sorted(
        glob.glob(os.path.join(path_folder_candidates, 'C*.npy')),
        key=lambda p: int(os.path.basename(p)[1:-4])
    )
    assert len(paths) > 0, f'found no candidate files in {path_folder_candidates}'
    # replacing tables leaves unused space in the file, so always start from a new file
    if os.path.exists(path_db):
        raise FileExistsError(f'{path_db} already exists, delete it first')
    # the unsorted table lives in a separate file, so its space is not left behind in path_db
    path_staging: str = os.path.splitext(path_db)[0] + '_staging.duckdb'

    select: str = _select_expression()
    con = duckdb.connect(path_db)
    try:
        con.execute(f"ATTACH '{path_staging}' AS staging_db")
        # load everything into an unsorted staging table first
        con.execute(f'CREATE OR REPLACE TABLE staging_db.staging AS SELECT {select} FROM (SELECT '
                    + ', '.join(f'0 AS {col}' for col in COUNT_COLUMNS.values()) + ') LIMIT 0')
        for path in tqdm(paths, desc='inserting'):
            # int32 so that the weighted sums cannot overflow
            chunk = pd.DataFrame(np.load(path).astype(np.int32), columns=list(COUNT_COLUMNS.values()))
            con.register('chunk', chunk)
            con.execute(f'INSERT INTO staging_db.staging SELECT {select} FROM chunk')
            con.unregister('chunk')

        print('sorting by mass ...')
        con.execute(f'CREATE TABLE {TABLE_NAME} AS SELECT * FROM staging_db.staging ORDER BY M')
        con.execute('DETACH staging_db')
        con.execute('CHECKPOINT')
        n_rows: int = con.execute(f'SELECT COUNT(*) FROM {TABLE_NAME}').fetchone()[0]
    finally:
        con.close()
        for path in (path_staging, path_staging + '.wal'):
            if os.path.exists(path):
                os.remove(path)
    return n_rows


if __name__ == '__main__':
    path_folder_candidates = r"C:\Users\Yannick Zander\Downloads\formulas"
    path_db = r"C:\Users\Yannick Zander\Downloads\formulas.duckdb"

    n: int = create_database(path_folder_candidates, path_db)
    print(f'wrote {n} rows to {path_db}')
