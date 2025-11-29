from __future__ import annotations

from typing import List, Optional, Sequence, Union
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline


class RemoveDuplicates(BaseEstimator, TransformerMixin):
    """
    Transformer to remove duplicate rows and/or duplicate columns (by content).

    :param remove_rows: If True, drop duplicated rows (keep first).
    :param remove_columns: If True, drop duplicated columns (by identical content,
    keep first).
    """

    def __init__(self, remove_rows: bool = True, remove_columns: bool = False) -> None:
        self.remove_rows = bool(remove_rows)
        self.remove_columns = bool(remove_columns)
        # populated in fit
        self._removed_row_index: Optional[pd.Index] = None
        self._removed_columns: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit stores what will be removed (computed on X).

        :param X: input DataFrame
        :returns: self
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)  # attempt conversion

        if self.remove_rows:
            # rows to be removed (duplicate by all columns)
            dup_rows = X.duplicated(keep="first")
            self._removed_row_index = X.index[dup_rows]
        else:
            self._removed_row_index = pd.Index([])

        if self.remove_columns:
            # find duplicate columns by checking duplicated rows on transposed frame
            dup_cols_mask = X.T.duplicated(keep="first")
            self._removed_columns = list(X.columns[dup_cols_mask])
        else:
            self._removed_columns = []

        return self

    def transform(self, X: pd.DataFrame):
        """
        Apply removal to X and return new DataFrame.
        Non-destructive to original X.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_out = X.copy()

        if self.remove_rows and (self._removed_row_index is not None):
            if len(self._removed_row_index) > 0:
                X_out = X_out.drop(index=self._removed_row_index, errors="ignore")

        if self.remove_columns and (self._removed_columns is not None):
            if len(self._removed_columns) > 0:
                X_out = X_out.drop(columns=self._removed_columns, errors="ignore")

        return X_out

    def fit_transform(self, X: pd.DataFrame, y=None):
        return self.fit(X, y).transform(X)

    def get_removed_row_index(self) -> pd.Index:
        """Return index of rows that would be removed (or were removed after fit)."""
        return (
            self._removed_row_index
            if self._removed_row_index is not None
            else pd.Index([])
        )

    @property
    def removed_columns_(self) -> List[str]:
        """Return list of removed columns (empty list if none)."""
        return list(self._removed_columns or [])

    def __repr__(self):
        return (
            f"RemoveDuplicates(remove_rows={self.remove_rows}, "
            f"remove_columns={self.remove_columns})"
        )


class LowVarianceFilter(BaseEstimator, TransformerMixin):
    """
    Keep numeric columns whose variance >= threshold. Non-numeric columns
    are preserved untouched.

    :param threshold: variance threshold (features with var < threshold are removed)
    """

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = float(threshold)
        self._kept_numeric_cols: Optional[List[str]] = None
        self._removed_numeric_cols: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        numeric = X.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            # nothing numeric -> nothing to remove
            self._kept_numeric_cols = []
            self._removed_numeric_cols = []
            return self

        # use population variance (ddof=0) to match sklearn.VarianceThreshold
        variances = numeric.var(axis=0, ddof=0)
        keep_mask = variances >= self.threshold
        self._kept_numeric_cols = list(variances.index[keep_mask])
        self._removed_numeric_cols = list(variances.index[~keep_mask])
        return self

    def transform(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # keep non-numeric columns unchanged
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        numeric_keep = self._kept_numeric_cols or []
        cols_to_return = non_numeric + numeric_keep
        # preserve original column order where possible
        cols_to_return = [c for c in X.columns if c in cols_to_return]
        return X[cols_to_return].copy()

    def fit_transform(self, X: pd.DataFrame, y=None):
        return self.fit(X, y).transform(X)

    @property
    def kept_numeric_cols_(self) -> List[str]:
        return list(self._kept_numeric_cols or [])

    @property
    def removed_numeric_cols_(self) -> List[str]:
        return list(self._removed_numeric_cols or [])

    def __repr__(self):
        return f"LowVarianceFilter(threshold={self.threshold})"


class DataFrameScaler(BaseEstimator, TransformerMixin):
    """
    Wrap a sklearn scaler to operate on pandas DataFrames.
    Only numeric columns are scaled; non-numeric columns are preserved.

    :param scaler: one of 'standard', 'minmax', 'robust', or None
    """

    def __init__(self, scaler: Optional[str] = "standard") -> None:
        self.scaler = scaler
        self._scaler_obj = None
        self._numeric_cols: Optional[List[str]] = None

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

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        numeric = X.select_dtypes(include=[np.number])
        self._numeric_cols = numeric.columns.tolist()
        self._scaler_obj = self._make_scaler()
        if self._scaler_obj is not None and len(self._numeric_cols) > 0:
            self._scaler_obj.fit(numeric.values)
        return self

    def transform(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_out = X.copy()
        if self._scaler_obj is None or len(self._numeric_cols or []) == 0:
            return X_out

        numeric_vals = X_out[self._numeric_cols].values
        scaled = self._scaler_obj.transform(numeric_vals)
        X_out[self._numeric_cols] = scaled
        return X_out

    def fit_transform(self, X: pd.DataFrame, y=None):
        return self.fit(X, y).transform(X)

    def __repr__(self):
        return f"DataFrameScaler(scaler={self.scaler})"


@dataclass
class DataPreprocessor:
    """
    Convenience wrapper that constructs the sklearn Pipeline and exposes fit/transform/
    fit_transform/predict/fir_predict methods while preserving DataFrame inputs/outputs.

    :param remove_dup_rows: drop duplicated rows if True
    :param remove_dup_cols: drop duplicated columns (by content) if True
    :param var_threshold: numeric variance threshold
    (features with var < threshold are removed)
    :param scaler: scaler option, one of 'standard','minmax','robust', or None
    """

    remove_dup_rows: bool = True
    remove_dup_cols: bool = False
    var_threshold: float = 0.05
    scaler: Optional[str] = "standard"

    def __post_init__(self):
        # build pipeline
        steps = []
        steps.append(
            (
                "dedup",
                RemoveDuplicates(
                    remove_rows=self.remove_dup_rows,
                    remove_columns=self.remove_dup_cols,
                ),
            )
        )
        steps.append(("var", LowVarianceFilter(threshold=self.var_threshold)))
        # add scaler only if requested (DataFrameScaler accepts None)
        steps.append(("scaler", DataFrameScaler(scaler=self.scaler)))
        self.pipeline = Pipeline(steps)

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the entire preprocessing pipeline.

        :param X: pandas DataFrame
        :returns: self
        """
        self.pipeline.fit(X)
        return self

    def transform(self, X: pd.DataFrame):
        """
        Transform X (apply fitted pipeline). Returns a pandas DataFrame.
        """
        Xt = self.pipeline.transform(X)
        # ensure DataFrame output
        if isinstance(Xt, np.ndarray):
            # try to reconstruct columns from last transformer
            # but our transformers return DataFrame so this is unlikely
            return pd.DataFrame(Xt)
        return Xt

    def fit_transform(self, X: pd.DataFrame, y=None):
        """Fit and transform in one call."""
        return self.pipeline.fit_transform(X)

    # convenience aliases
    def fit_predict(self, X: pd.DataFrame, y=None):
        """Alias for fit_transform (user requested API)."""
        return self.fit_transform(X, y)

    def predict(self, X: pd.DataFrame):
        """Alias for transform (user requested API)."""
        return self.transform(X)

    # accessors for internal information
    @property
    def removed_rows_(self) -> pd.Index:
        return self.pipeline.named_steps["dedup"].get_removed_row_index()

    @property
    def removed_columns_by_content_(self) -> List[str]:
        return self.pipeline.named_steps["dedup"].removed_columns_

    @property
    def removed_by_variance_(self) -> List[str]:
        return self.pipeline.named_steps["var"].removed_numeric_cols_

    @property
    def kept_numeric_cols_(self) -> List[str]:
        return self.pipeline.named_steps["var"].kept_numeric_cols_

    def __repr__(self):
        return (
            f"DataPreprocessor(remove_dup_rows={self.remove_dup_rows}, "
            f"remove_dup_cols={self.remove_dup_cols}, "
            f"var_threshold={self.var_threshold}, scaler={self.scaler})"
        )
