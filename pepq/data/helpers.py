"""
data.helpers
------------

Small utility helpers and common type aliases.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union
import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target

ArrayLike = Union[np.ndarray, pd.DataFrame]


def _ensure_dataframe(
    X: ArrayLike, feature_names: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """
    Coerce an array-like object into a pandas DataFrame.

    If ``X`` is already a DataFrame, it is copied and returned.
    If ``X`` is a NumPy array, column names are taken from ``feature_names`` if provided,
    otherwise synthetic names ``f0``, ``f1``, ...

    :param X: Input array-like (pandas.DataFrame or numpy.ndarray).
    :type X: ArrayLike
    :param feature_names: Optional feature names used when ``X`` is a NumPy array.
    :type feature_names: Sequence[str] or None
    :returns: Copy of the data as a pandas.DataFrame.
    :rtype: pandas.DataFrame

    Example
    -------

    >>> import numpy as np
    >>> from data.helpers import _ensure_dataframe
    >>> arr = np.arange(6).reshape(2,3)
    >>> df = _ensure_dataframe(arr)
    >>> list(df.columns)
    ['f0', 'f1', 'f2']
    """
    if isinstance(X, pd.DataFrame):
        return X.copy()

    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    n_cols = arr.shape[1]
    if feature_names is not None:
        if len(feature_names) != n_cols:
            raise ValueError("feature_names length must equal number of columns in X")
        cols = list(feature_names)
    else:
        cols = [f"f{i}" for i in range(n_cols)]

    return pd.DataFrame(arr, columns=cols)


def _is_classification_target(y: Sequence) -> bool:
    """
    Heuristic to determine if the target should be treated as classification.

    :param y: Target vector.
    :type y: Sequence
    :returns: ``True`` if classification, ``False`` otherwise.
    :rtype: bool
    """
    y_arr = np.asarray(y)
    try:
        t = type_of_target(y_arr)
        return t in {
            "binary",
            "multiclass",
            "multiclass-multioutput",
            "multilabel-indicator",
        }
    except Exception:
        if np.issubdtype(y_arr.dtype, np.integer):
            return np.unique(y_arr).size < 20
        return False
