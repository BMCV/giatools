from . import typing as _typing

try:
    import pandas as _pd
    class DataFrame(_pd.DataFrame): ...  # noqa: E701
except ImportError:
    class DataFrame: ...  # noqa: E701


def find_column(df: DataFrame, candidates: _typing.Iterable[str]) -> str:
    """
    Returns the column name present in `df` and the list of `candidates`.

    Raises:
        KeyError: If there is no candidate column name present in `df`, or more than one.
    """
    intersection = frozenset(df.columns) & frozenset(candidates)
    if len(intersection) == 0:
        raise KeyError(f'No such column: {", ".join(candidates)}')
    elif len(intersection) > 1:
        raise KeyError(f'The column names {", ".join(intersection)} are ambiguous')
    else:
        return next(iter(intersection))
