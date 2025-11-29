"""
regression.py
=============

DockQ / continuous-score regression model with conformal intervals.

This module defines :class:`DockQRegressor`, a voting ensemble regressor
with optional prediction intervals from :mod:`mapie`.

Two fitting modes are supported:

* **DataFrame mode**::

      reg.fit(df, target_col="dockq")

* **Array mode**::

      reg.fit(X, y)

Example
-------

.. code-block:: python

   import pandas as pd
   import numpy as np
   from pepq.model.regression import DockQRegressor

   df_train = ...  # DataFrame with features + "dockq" column
   reg = DockQRegressor(random_state=0)
   reg.fit(df_train, target_col="dockq")

   print(reg.cv_summary_)                       # MAE / RMSE / R²
   y_pred = reg.predict(df_train)

   # Array mode
   X = df_train.drop(columns=["dockq"]).values
   y = df_train["dockq"].values
   reg2 = DockQRegressor().fit(X, y)
   y_pred2 = reg2.predict(X)

   # If MAPIE is available:
   y_pred, y_lo, y_hi = reg.predict_with_interval(df_train)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .base import _BaseDockQModel
from .metrics import regression_cv_summary
from .helpers import ArrayLike, MAPIERegressor, has_mapie_regression


@dataclass
class DockQRegressor(_BaseDockQModel, BaseEstimator, RegressorMixin):
    """
    Ensemble regressor + conformal prediction intervals for DockQ-like scores.

    The ensemble is a :class:`VotingRegressor` over:

    * StandardScaler + Ridge regression
    * GradientBoostingRegressor
    * RandomForestRegressor

    :class:`DockQRegressor` follows the sklearn estimator API and accepts
    both DataFrame and array inputs for :meth:`fit` and :meth:`predict`.

    :param n_splits: Number of CV splits per repeat.
    :type n_splits: int
    :param n_repeats: Number of CV repeats.
    :type n_repeats: int
    :param random_state: Random seed used throughout.
    :type random_state: int or None
    :param default_alpha: Miscoverage level for conformal intervals.
    :type default_alpha: float
    """

    _mapie: Optional[MAPIERegressor] = field(default=None, init=False)
    _reg: Optional[VotingRegressor] = field(default=None, init=False)
    _cp_method_used: Optional[str] = field(default=None, init=False)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.__class__.__name__}("
            f"n_splits={self.n_splits}, "
            f"n_repeats={self.n_repeats}, "
            f"random_state={self.random_state}, "
            f"default_alpha={self.default_alpha}, "
            f"cp_method_used={self._cp_method_used}, "
            f"is_fitted={self._is_fitted})"
        )

    # ------------------------------------------------------------------
    # Base learners
    # ------------------------------------------------------------------
    def _make_ridge_pipeline(self) -> Pipeline:
        """
        Construct a StandardScaler + Ridge regression pipeline.

        :returns: Configured pipeline.
        :rtype: sklearn.pipeline.Pipeline
        """
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0, random_state=self.random_state)),
            ]
        )

    def _make_ensemble_estimator(self) -> VotingRegressor:
        """
        Build the voting regressor ensemble.

        :returns: Configured :class:`VotingRegressor` instance.
        :rtype: sklearn.ensemble.VotingRegressor
        """
        ridge = self._make_ridge_pipeline()
        gb = GradientBoostingRegressor(random_state=self.random_state)
        rf = RandomForestRegressor(
            n_estimators=500,
            random_state=self.random_state,
            n_jobs=-1,
        )
        return VotingRegressor(
            estimators=[
                ("ridge", ridge),
                ("gb", gb),
                ("rf", rf),
            ]
        )

    # ------------------------------------------------------------------
    # Internal helpers for X, y
    # ------------------------------------------------------------------
    def _prepare_xy(
        self,
        X: ArrayLike,
        y: Optional[Sequence] = None,
        target_col: str = "target",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalise X, y input for fit into numpy arrays and set feature names.

        Two modes:

        * If ``X`` is a DataFrame and ``y`` is ``None``:
          use ``target_col`` as target and all other columns as features.
        * Otherwise: use ``X`` as features and ``y`` as targets.

        :param X: Feature matrix or combined DataFrame.
        :type X: ArrayLike
        :param y: Target values or ``None`` in DataFrame mode.
        :type y: Sequence or None
        :param target_col: Target column name when ``X`` is a DataFrame.
        :type target_col: str
        :returns: Tuple ``(X_arr, y_arr)`` of numpy arrays.
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
        :raises ValueError: If ``y`` is missing in array mode.
        """
        if isinstance(X, pd.DataFrame) and y is None:
            if target_col not in X.columns:
                raise ValueError(
                    f"Target column '{target_col}' not found in dataframe."
                )
            self._feature_names = [c for c in X.columns if c != target_col]
            X_arr = X[self._feature_names].values
            y_arr = X[target_col].astype(float).values
            return X_arr, y_arr

        if y is None:
            raise ValueError(
                "When y is None, X must be a DataFrame with target_col present."
            )

        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            n_features = X_arr.shape[1]
            self._feature_names = [f"f{i}" for i in range(n_features)]

        y_arr = np.asarray(y).astype(float).ravel()
        return X_arr, y_arr

    def _ensure_feature_array(self, X: ArrayLike) -> np.ndarray:
        """
        Convert input X at prediction time to a numpy array with correct shape.

        :param X: Input feature data (DataFrame or ndarray).
        :type X: ArrayLike
        :returns: Feature matrix of shape ``(n_samples, n_features)``.
        :rtype: numpy.ndarray
        :raises ValueError: If the number of columns does not match.
        """
        self._check_is_fitted()
        n_features = len(self._feature_names)

        if isinstance(X, pd.DataFrame):
            missing = [c for c in self._feature_names if c not in X.columns]
            if missing:
                raise ValueError(
                    f"Missing feature columns at prediction time: {missing}"
                )
            X_arr = X[self._feature_names].values
        else:
            X_arr = np.asarray(X)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, n_features)
            if X_arr.shape[1] != n_features:
                raise ValueError(
                    f"Expected X with {n_features} features, got shape {X_arr.shape}"
                )
        return X_arr

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        X: ArrayLike,
        y: Optional[Sequence] = None,
        target_col: str = "target",
    ) -> "DockQRegressor":
        """
        Fit the regression ensemble (and MAPIE, if available).

        Two calling conventions are supported:

        1. **DataFrame mode**::

               reg.fit(df, target_col="dockq")

        2. **Array mode**::

               reg.fit(X, y)

        :param X: Input data (DataFrame or array-like).
        :type X: ArrayLike
        :param y: Target values for array mode; ignored in DataFrame mode.
        :type y: Sequence or None
        :param target_col: Target column name in DataFrame mode.
        :type target_col: str
        :returns: The fitted estimator (for chaining).
        :rtype: DockQRegressor
        """
        X_arr, y_arr = self._prepare_xy(X, y=y, target_col=target_col)

        base_estimator = self._make_ensemble_estimator()
        self._cv_summary = regression_cv_summary(
            base_estimator,
            X_arr,
            y_arr,
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )

        # MAPIE conformal regressor (optional)
        self._mapie = None
        self._cp_method_used = None
        if has_mapie_regression():
            try:
                mapie = MAPIERegressor(
                    estimator=self._make_ensemble_estimator(),
                    cv=self.n_splits,
                    method="plus",
                    n_jobs=-1,
                )
                mapie.fit(X_arr, y_arr)
                self._cp_method_used = "plus"
            except Exception:
                mapie = MAPIERegressor(
                    estimator=self._make_ensemble_estimator(),
                    cv=self.n_splits,
                    method="naive",
                    n_jobs=-1,
                )
                mapie.fit(X_arr, y_arr)
                self._cp_method_used = "naive"
            self._mapie = mapie

        reg = self._make_ensemble_estimator()
        reg.fit(X_arr, y_arr)
        self._reg = reg

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def mapie_model_(self) -> MAPIERegressor:
        """
        Underlying MAPIE regressor.

        :returns: Fitted :class:`MAPIERegressor` instance.
        :rtype: MAPIERegressor
        :raises RuntimeError: If MAPIE is not available or not fitted.
        """
        self._check_is_fitted()
        if self._mapie is None:
            raise RuntimeError("MAPIE is not available or was not fitted.")
        return self._mapie

    @property
    def cp_method_used_(self) -> Optional[str]:
        """
        Conformal prediction method used by MAPIE (e.g. ``'plus'`` or ``'naive'``).

        :returns: Method name or ``None``.
        :rtype: str or None
        """
        return self._cp_method_used

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------
    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict continuous scores for new samples.

        :param X: Feature matrix (DataFrame or ndarray).
        :type X: ArrayLike
        :returns: Predicted values of shape ``(n_samples,)``.
        :rtype: numpy.ndarray
        """
        self._check_is_fitted()
        if self._reg is None:
            raise RuntimeError("Internal regressor is not fitted.")
        X_arr = self._ensure_feature_array(X)
        return self._reg.predict(X_arr)

    def fit_predict(
        self,
        X: ArrayLike,
        y: Optional[Sequence] = None,
        target_col: str = "target",
    ) -> np.ndarray:
        """
        Fit the model and immediately return predictions on the same data.

        :param X: Input data (DataFrame or array-like).
        :type X: ArrayLike
        :param y: Target vector in array mode; ignored in DataFrame mode.
        :type y: Sequence or None
        :param target_col: Target column name in DataFrame mode.
        :type target_col: str
        :returns: Predicted values for the training data.
        :rtype: numpy.ndarray
        """
        self.fit(X, y=y, target_col=target_col)
        return self.predict(X)

    def predict_with_interval(
        self,
        X: ArrayLike,
        alpha: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict values and conformal prediction intervals via MAPIE.

        :param X: Feature matrix (DataFrame or ndarray).
        :type X: ArrayLike
        :param alpha: Miscoverage level; if ``None``, uses ``default_alpha``.
        :type alpha: float or None
        :returns: Tuple ``(y_pred, y_lower, y_upper)`` with arrays of
            shape ``(n_samples,)``.
        :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
        :raises RuntimeError: If MAPIE is not available or not fitted.
        """
        self._check_is_fitted()
        if self._mapie is None:
            raise RuntimeError("MAPIE is not available or was not fitted.")

        if alpha is None:
            alpha = self.default_alpha

        X_arr = self._ensure_feature_array(X)
        y_pred, y_interval = self._mapie.predict(X_arr, alpha=[alpha])
        y_lower = y_interval[:, 0, 0]
        y_upper = y_interval[:, 1, 0]
        return y_pred, y_lower, y_upper

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def evaluate(
        self,
        X: ArrayLike,
        y: Optional[Sequence] = None,
        *,
        df_mode_target_col: str = "target",
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate regression metrics and interval coverage.

        Supports both DataFrame and array calling conventions:

        * ``evaluate(df, df_mode_target_col="dockq")``
        * ``evaluate(X, y)``

        :param X: Input data (DataFrame or feature matrix).
        :type X: ArrayLike
        :param y: Target values in array mode; ignored in DataFrame mode.
        :type y: Sequence or None
        :param df_mode_target_col: Target column name in DataFrame mode.
        :type df_mode_target_col: str
        :param alpha: Miscoverage level for prediction intervals. Uses
            :attr:`default_alpha` if ``None``.
        :type alpha: float or None
        :returns: Dictionary with MAE, RMSE, R² and interval statistics.
        :rtype: dict[str, Any]
        """
        self._check_is_fitted()
        if alpha is None:
            alpha = self.default_alpha

        if isinstance(X, pd.DataFrame) and y is None:
            if df_mode_target_col not in X.columns:
                raise ValueError(
                    f"Target column '{df_mode_target_col}' not found in dataframe."
                )
            y_true = X[df_mode_target_col].astype(float).values
            X_feat = X[self._feature_names]
        else:
            if y is None:
                raise ValueError("y must be provided in array mode.")
            X_feat = X
            y_true = np.asarray(y).astype(float).ravel()

        y_pred = self.predict(X_feat)
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))

        results: Dict[str, Any] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "n_samples": int(len(y_true)),
        }

        if self._mapie is not None:
            _, y_lo, y_hi = self.predict_with_interval(X_feat, alpha=alpha)
            within = (y_true >= y_lo) & (y_true <= y_hi)
            results["interval_coverage"] = float(np.mean(within))
            results["interval_width_mean"] = float(np.mean(y_hi - y_lo))
        else:
            results["interval_coverage"] = float("nan")
            results["interval_width_mean"] = float("nan")

        return results
