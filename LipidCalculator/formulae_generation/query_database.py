"""
Query the annotated formula table created by create_database.py

Conditions are given as {column: value} for exact matches (e.g. Senior=True, nP=1)
or {column: (lower, upper)} for inclusive ranges (e.g. H_C=(1.5, 2.2)); None as bound means unbounded.
Keys can also be sums of element counts, e.g. {'N+P+S': (0, 5)}.

Examples:
>>> df = query_formulas(path_db, mass_range=(760.5, 760.6), conditions=dict(Senior=True, nP=1, DBE=(1, 10)))
>>> df = query_formulas(path_db, mass_range=(700, 800), conditions={'N+P': (0, 3), 'S+P': (0, 2)})
>>> masses = get_masses(path_db, conditions=dict(Lewis=True))
"""
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from LipidCalculator.formulae_generation.create_database import TABLE_NAME

Condition = Any | tuple[Any | None, Any | None]


def _get_columns(con: duckdb.DuckDBPyConnection) -> list[str]:
    return [row[0] for row in con.execute(f'SELECT column_name FROM (DESCRIBE {TABLE_NAME})').fetchall()]


def _to_expression(key: str, columns: list[str]) -> str:
    """
    SQL expression for a condition key: either a column (e.g. 'nP', 'DBE')
    or a sum of element counts (e.g. 'N+P+S' or 'nN+nP').
    """
    if '+' not in key:
        # column names cannot be passed as parameters, so make sure they exist
        if key not in columns:
            raise KeyError(f'unknown column {key}, available columns: {columns}')
        return key

    parts: list[str] = []
    for part in key.replace(' ', '').split('+'):
        # allow element symbols as shortcut for their count column
        col: str = part if part.startswith('n') else f'n{part}'
        if col not in columns:
            raise KeyError(f'unknown element count {part} in {key}, available columns: {columns}')
        # cast so that the sum of small integer types cannot overflow
        parts.append(f'CAST({col} AS INTEGER)')
    return '(' + ' + '.join(parts) + ')'


def _build_where(
        con: duckdb.DuckDBPyConnection,
        mass_range: tuple[float | None, float | None] | None,
        conditions: dict[str, Condition] | None
) -> tuple[str, list[Any]]:
    """WHERE clause with placeholders and the corresponding parameters."""
    conditions = dict(conditions or {})
    if mass_range is not None:
        conditions['M'] = mass_range

    columns: list[str] = _get_columns(con)
    clauses: list[str] = []
    params: list[Any] = []
    for key, cond in conditions.items():
        col: str = _to_expression(key, columns)
        if isinstance(cond, tuple):
            lower, upper = cond
            if lower is not None:
                clauses.append(f'{col} >= ?')
                params.append(lower)
            if upper is not None:
                clauses.append(f'{col} <= ?')
                params.append(upper)
        else:
            clauses.append(f'{col} = ?')
            params.append(cond)

    where: str = ('WHERE ' + ' AND '.join(clauses)) if clauses else ''
    return where, params


def query_formulas(
        path_db: str,
        mass_range: tuple[float | None, float | None] | None = None,
        conditions: dict[str, Condition] | None = None,
        columns: list[str] | None = None,
        limit: int | None = None
) -> pd.DataFrame:
    """
    Retrieve all formulas satisfying the conditions, sorted by mass.

    Parameters
    ----------
    path_db: str
        Path to the DuckDB file.
    mass_range: tuple[float | None, float | None], optional
        Inclusive (lower, upper) bounds for M, shortcut for conditions=dict(M=(lower, upper)).
    conditions: dict[str, value | tuple], optional
        {column: value} for exact matches or {column: (lower, upper)} for inclusive ranges.
        Keys can also be sums of element counts, e.g. 'N+P' or 'N+P+S'.
    columns: list[str], optional
        Columns to return, defaults to all.
    limit: int, optional
        Maximum number of rows to return.
    """
    with duckdb.connect(path_db, read_only=True) as con:
        where, params = _build_where(con, mass_range, conditions)
        available: list[str] = _get_columns(con)
        if columns is None:
            columns = available
        if unknown := set(columns) - set(available):
            raise KeyError(f'unknown columns {unknown}, available columns: {available}')
        query: str = f'SELECT {", ".join(columns)} FROM {TABLE_NAME} {where} ORDER BY M'
        if limit is not None:
            query += f' LIMIT {int(limit)}'
        return con.execute(query, params).df()


def get_masses(
        path_db: str,
        mass_range: tuple[float | None, float | None] | None = None,
        conditions: dict[str, Condition] | None = None
) -> np.ndarray:
    """
    Retrieve the (sorted) masses of all formulas satisfying the conditions.

    Without conditions, this returns every mass in the database (8 bytes per row, so this can be several GB).
    """
    with duckdb.connect(path_db, read_only=True) as con:
        where, params = _build_where(con, mass_range, conditions)
        return con.execute(f'SELECT M FROM {TABLE_NAME} {where} ORDER BY M', params).fetchnumpy()['M']


def count_formulas(
        path_db: str,
        mass_range: tuple[float | None, float | None] | None = None,
        conditions: dict[str, Condition] | None = None
) -> int:
    """Number of formulas satisfying the conditions, useful to check the size before retrieving them."""
    with duckdb.connect(path_db, read_only=True) as con:
        where, params = _build_where(con, mass_range, conditions)
        return con.execute(f'SELECT COUNT(*) FROM {TABLE_NAME} {where}', params).fetchone()[0]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    path_db = r"C:\Users\Yannick Zander\Downloads\formulas.duckdb"

    with duckdb.connect(path_db, read_only=True) as con:
        print(con.sql('DESCRIBE formulas'))

    masses = get_masses(
        path_db,
        # mass_range=(100, 500),
        conditions=dict(Senior=True, Lewis=True, DBE=(0, 36),
                        H_C=(0.5454545454545454, 2.3333333333333335), O_C=(0, 1.1666),
                        nP=(0, 4), nS=(0, 2), nN=(0, 7)) | {'N+P': (0, 7), 'N+P+S': (0, 8), 'S+P': (0, 4),
                                                            'S+N': (0, 7)}
    )
    dmasses = np.diff(masses)
    res = masses[1:] / dmasses
    ppms = 1 / res * 1e6

    plt.figure()
    # plt.semilogy(masses[1:], res, 'o', markersize=1)
    plt.semilogy(masses[1:], ppms, 'o', markersize=1)
    # plt.xlim(1000, 1200)
    plt.ylabel('Distance to next (ppm)')
    # plt.ylabel('Required resolution')
    plt.xlabel('Mass (Da)')
    plt.show()
