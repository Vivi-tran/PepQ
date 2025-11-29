"""
data.preprocessor
-----------------

Composed DataPreprocessor with explicit step instances and PreprocessorConfig support.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from .helpers import ArrayLike, _ensure_dataframe
from .steps import RemoveDuplicatesStep, VarianceFilterStep, ScalerStep
from .display import NiceDisplayMixin
from .config import PreprocessorConfig


@dataclass
class DataPreprocessor(NiceDisplayMixin, BaseEstimator, TransformerMixin):
    """
    Composed preprocessor that accepts either explicit step instances or a PreprocessorConfig.

    :param dedup_step: Optional custom RemoveDuplicatesStep instance
    (if provided it overrides config).
    :param var_step: Optional custom VarianceFilterStep instance
    (if provided it overrides config).
    :param scaler_step: Optional custom ScalerStep instance
    (if provided it overrides config).
    :param config: Optional PreprocessorConfig instance used to construct steps
    if explicit steps not provided.

    Example
    -------

    >>> import pandas as pd, numpy as np
    >>> from data.preprocessor import DataPreprocessor
    >>> X = pd.DataFrame({"a":[1,1,2],"b":[0.0,0.0,0.0],"c":[1,2,3]})
    >>> prep = DataPreprocessor()
    >>> prep.fit(X)
    >>> prep.summary()
    """

    dedup_step: Optional[RemoveDuplicatesStep] = None
    var_step: Optional[VarianceFilterStep] = None
    scaler_step: Optional[ScalerStep] = None
    config: Optional[PreprocessorConfig] = None

    # internals
    feature_names_in_: Optional[List[str]] = field(default=None, init=False)
    feature_names_out_: Optional[List[str]] = field(default=None, init=False)
    _return_dataframe: bool = field(default=False, init=False)

    @classmethod
    def from_config(cls, config: PreprocessorConfig) -> "DataPreprocessor":
        ded, var, scl = config.build_steps()
        inst = cls(dedup_step=ded, var_step=var, scaler_step=scl, config=config)
        inst._return_dataframe = config.return_dataframe
        return inst

    def set_config(self, config: PreprocessorConfig) -> "DataPreprocessor":
        ded, var, scl = config.build_steps()
        self.dedup_step = ded
        self.var_step = var
        self.scaler_step = scl
        self.config = config
        self._return_dataframe = config.return_dataframe
        return self

    def fit(self, X: ArrayLike, y: Any = None) -> "DataPreprocessor":
        Xdf = _ensure_dataframe(X)
        self.feature_names_in_ = list(Xdf.columns)

        # ensure steps exist (materialize from config if necessary)
        if self.dedup_step is None or self.var_step is None or self.scaler_step is None:
            if self.config is None:
                self.dedup_step = RemoveDuplicatesStep()
                self.var_step = VarianceFilterStep()
                self.scaler_step = ScalerStep()
            else:
                ded, var, scl = self.config.build_steps()
                self.dedup_step = ded
                self.var_step = var
                self.scaler_step = scl

        # sequential fit/transform so later steps see the transformed data
        self.dedup_step.fit(Xdf)
        after_dedup = self.dedup_step.transform(Xdf)

        self.var_step.fit(after_dedup)
        after_var = self.var_step.transform(after_dedup)

        self.scaler_step.fit(after_var)
        after_scale = self.scaler_step.transform(after_var)

        self.feature_names_out_ = list(after_scale.columns)

        if self.config is not None:
            self._return_dataframe = self.config.return_dataframe

        return self

    def transform(self, X: ArrayLike) -> Union[list, pd.DataFrame]:
        if self.feature_names_in_ is None:
            raise ValueError("DataPreprocessor is not fitted yet. Call fit(...) first.")
        Xdf = _ensure_dataframe(
            X,
            feature_names=(
                None if isinstance(X, pd.DataFrame) else self.feature_names_in_
            ),
        )
        out = self.dedup_step.transform(Xdf)
        out = self.var_step.transform(out)
        out = self.scaler_step.transform(out)
        if self._return_dataframe:
            return out
        return out.values

    def transform_df(self, X: ArrayLike) -> pd.DataFrame:
        res = self.transform(X)
        if isinstance(res, pd.DataFrame):
            return res
        return pd.DataFrame(
            res,
            columns=self.get_feature_names_out(),
            index=_ensure_dataframe(X, self.feature_names_in_).index,
        )

    def fit_transform(self, X: ArrayLike, y: Any = None):
        return self.fit(X, y).transform(X)

    def fit_predict(self, X: ArrayLike, y: Any = None):
        return self.fit_transform(X, y)

    def get_feature_names_out(self) -> List[str]:
        if self.feature_names_out_ is None:
            raise ValueError("DataPreprocessor is not fitted yet.")
        return list(self.feature_names_out_)

    # display helpers used by NiceDisplayMixin
    def _repr_params_dict(self) -> dict:
        if self.config is not None:
            return {"config": self.config.to_dict()}
        return {
            "dedup": {
                "remove_rows": getattr(self.dedup_step, "remove_rows", None),
                "remove_columns": getattr(self.dedup_step, "remove_columns", None),
            },
            "variance": {"threshold": getattr(self.var_step, "threshold", None)},
            "scaler": {"scaler": getattr(self.scaler_step, "scaler", None)},
            "return_dataframe": self._return_dataframe,
        }

    def _repr_feature_sets(self) -> dict:
        kept = getattr(self.var_step, "kept_numeric_cols_", []) or []
        removed = getattr(self.var_step, "removed_numeric_cols_", []) or []
        return {
            "Kept numeric features": kept,
            "Removed (low-variance) features": removed,
        }

    def _ascii_diagram_lines(self) -> List[str]:
        status = "fitted" if self.feature_names_in_ is not None else "unfitted"
        return [
            f"{type(self).__name__} [{status}]",
            f"  in_features: {len(self.feature_names_in_) if self.feature_names_in_ is not None else 'None'}",
            f"  out_features: {len(self.feature_names_out_) if self.feature_names_out_ is not None else 'None'}",
            "  pipeline:",
            f"    [X] -> {self.dedup_step} -> {self.var_step} -> {self.scaler_step} -> [X']",
            f"  return_dataframe={self._return_dataframe}",
        ]

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "n_input_features": (
                        len(self.feature_names_in_)
                        if self.feature_names_in_ is not None
                        else None
                    ),
                    "n_output_features": (
                        len(self.feature_names_out_)
                        if self.feature_names_out_ is not None
                        else None
                    ),
                    "n_removed_by_variance": len(
                        getattr(self.var_step, "removed_numeric_cols_", []) or []
                    ),
                    "n_removed_by_dedup_cols": len(
                        getattr(self.dedup_step, "removed_columns_", []) or []
                    ),
                    "return_dataframe": self._return_dataframe,
                }
            ]
        )
