"""
data.steps
----------

Concrete step transformers used by the DataPreprocessor.

Each step is a scikit-learn style transformer (BaseEstimator, TransformerMixin)
so they can be inspected and (de)serialized independently.
"""

from __future__ import annotations

from typing import Any, List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

from .helpers import ArrayLike, _ensure_dataframe
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class RemoveDuplicatesStep(BaseEstimator, TransformerMixin):
    """
    Step that removes duplicate rows and/or duplicate columns (by content).

    :param remove_rows: Drop duplicated rows (keep first) when True.
    :type remove_rows: bool
    :param remove_columns: Drop duplicated columns (identical content) when True.
    :type remove_columns: bool

    Example
    -------

    >>> import pandas as pd, numpy as np
    >>> from data.steps import RemoveDuplicatesStep
    >>> df = pd.DataFrame({"a":[1,1,2], "b":[1,1,2]})
    >>> step = RemoveDuplicatesStep(remove_rows=True, remove_columns=True)
    >>> step.fit(df)
    >>> transformed = step.transform(df)
    """

    def __init__(self, remove_rows: bool = True, remove_columns: bool = False) -> None:
        self.remove_rows = bool(remove_rows)
        self.remove_columns = bool(remove_columns)
        # store plain lists (avoids ambiguous pandas.Index truthiness)
        self.removed_row_index_: Optional[List[object]] = None
        self.removed_columns_: Optional[List[str]] = None

    def fit(self, X: ArrayLike, y: Any = None) -> "RemoveDuplicatesStep":
        Xdf = _ensure_dataframe(X)

        if self.remove_rows:
            dup_mask = Xdf.duplicated(keep="first")
            self.removed_row_index_ = list(Xdf.index[dup_mask])
        else:
            self.removed_row_index_ = []

        if self.remove_columns:
            dup_cols_mask = Xdf.T.duplicated(keep="first")
            self.removed_columns_ = list(Xdf.columns[dup_cols_mask])
        else:
            self.removed_columns_ = []

        return self

    def transform(self, X: ArrayLike) -> pd.DataFrame:
        Xdf = _ensure_dataframe(X)
        out = Xdf.copy()

        if (
            self.remove_rows
            and (self.removed_row_index_ is not None)
            and len(self.removed_row_index_) > 0
        ):
            out = out.drop(index=self.removed_row_index_, errors="ignore")

        if (
            self.remove_columns
            and (self.removed_columns_ is not None)
            and len(self.removed_columns_) > 0
        ):
            out = out.drop(columns=self.removed_columns_, errors="ignore")

        return out

    def fit_transform(self, X: ArrayLike, y: Any = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def __repr__(self) -> str:
        return f"RemoveDuplicatesStep(remove_rows={self.remove_rows}, remove_columns={self.remove_columns})"


class VarianceFilterStep(BaseEstimator, TransformerMixin):
    """
    Remove numeric columns with population variance below ``threshold``.
    Non-numeric columns are preserved.

    :param threshold: Minimum population variance (ddof=0) to keep a numeric column.
    :type threshold: float
    """

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = float(threshold)
        self.kept_numeric_cols_: Optional[List[str]] = None
        self.removed_numeric_cols_: Optional[List[str]] = None

    def fit(self, X: ArrayLike, y: Any = None) -> "VarianceFilterStep":
        Xdf = _ensure_dataframe(X)
        numeric = Xdf.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            self.kept_numeric_cols_ = []
            self.removed_numeric_cols_ = []
            return self

        variances = numeric.var(axis=0, ddof=0).fillna(0.0)
        keep_mask = variances >= self.threshold
        self.kept_numeric_cols_ = list(variances.index[keep_mask])
        self.removed_numeric_cols_ = list(variances.index[~keep_mask])
        return self

    def transform(self, X: ArrayLike) -> pd.DataFrame:
        Xdf = _ensure_dataframe(X)
        non_numeric = Xdf.select_dtypes(exclude=[np.number]).columns.tolist()
        numeric_keep = list(self.kept_numeric_cols_ or [])
        cols_to_keep = [c for c in Xdf.columns if c in non_numeric + numeric_keep]
        return Xdf[cols_to_keep].copy()

    def __repr__(self) -> str:
        return f"VarianceFilterStep(threshold={self.threshold})"


class ScalerStep(BaseEstimator, TransformerMixin):
    """
    Scale numeric columns using a selected sklearn scaler; non-numeric columns are preserved.

    :param scaler: One of ``'standard'``, ``'minmax'``, ``'robust'`` or ``None``.
    :type scaler: str or None
    """

    def __init__(self, scaler: Optional[str] = "standard") -> None:
        self.scaler = scaler
        self._scaler_obj = None
        self.numeric_cols_: Optional[List[str]] = None

    def _make_scaler(self):
        if self.scaler is None:
            return None
        s = self.scaler.lower()
        if s == "standard":
            return StandardScaler()
        if s == "minmax":
            return MinMaxScaler()
        if s == "robust":
            return RobustScaler()
        raise ValueError("scaler must be one of {'standard','minmax','robust', None}")

    def fit(self, X: ArrayLike, y: Any = None) -> "ScalerStep":
        Xdf = _ensure_dataframe(X)
        numeric = Xdf.select_dtypes(include=[np.number])
        self.numeric_cols_ = numeric.columns.tolist()
        self._scaler_obj = self._make_scaler()
        if self._scaler_obj is not None and len(self.numeric_cols_ or []) > 0:
            self._scaler_obj.fit(numeric.values)
        return self

    def transform(self, X: ArrayLike) -> pd.DataFrame:
        Xdf = _ensure_dataframe(X)
        out = Xdf.copy()
        if self._scaler_obj is None or not (self.numeric_cols_):
            return out
        present = [c for c in self.numeric_cols_ if c in out.columns]
        if not present:
            return out
        out.loc[:, present] = self._scaler_obj.transform(out[present].values)
        return out

    def __repr__(self) -> str:
        return f"ScalerStep(scaler={self.scaler})"
