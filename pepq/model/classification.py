"""
classification.py
=================

DockQ classification model and conformal prediction wrapper.

This module defines :class:`DockQClassifier`, a soft-voting ensemble
classifier with optional conformal prediction via :mod:`mapie`.

Two fitting modes are supported:

* **DataFrame mode** ::

    clf.fit(df, label_col="label")

  Here ``df`` contains both features and label column.

* **Array mode** ::

    clf.fit(X, y)

  Here ``X`` is an array-like of shape ``(n_samples, n_features)`` and
  ``y`` is a 1D array-like of labels.

Example
-------

.. code-block:: python

   import pandas as pd
   import numpy as np
   from pepq.model.classification import DockQClassifier

   # DataFrame mode
   df_train = ...  # DataFrame with features + "label" column (0/1)
   clf = DockQClassifier(random_state=0)
   clf.fit(df_train, label_col="label")

   print(clf.cv_summary_)                 # CV accuracy/ROC
   p = clf.predict_proba(df_train)        # class-1 probabilities
   y_hat = clf.predict(df_train)          # hard labels

   # Array mode
   X = np.asarray(df_train.drop(columns=["label"]))
   y = df_train["label"].values
   clf2 = DockQClassifier().fit(X, y)
   y_hat2 = clf2.predict(X)

   # If MAPIE is available:
   y_pred, y_sets = clf.predict_with_confidence(df_train)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from .base import _BaseDockQModel
from .metrics import classification_cv_summary
from .helpers import ArrayLike, MAPIEClassifier, has_mapie_classification


@dataclass
class DockQClassifier(_BaseDockQModel, BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier + conformal prediction for DockQ confidence.

    Internally uses a soft-voting ensemble consisting of:

    * StandardScaler + LogisticRegression
    * StandardScaler + SVC (RBF kernel)
    * StandardScaler + KNeighborsClassifier
    * RandomForestClassifier

    Optionally wraps the ensemble with :class:`MAPIEClassifier` to
    generate prediction sets with approximate coverage.

    The class follows the sklearn estimator API:

    * :meth:`fit` accepts either a :class:`pandas.DataFrame` or an
      array-like ``X`` plus ``y``.
    * :meth:`predict`, :meth:`predict_proba` and :meth:`fit_predict`
      work on DataFrames or ndarrays.

    :param n_splits: Number of stratified CV splits per repeat.
    :type n_splits: int
    :param n_repeats: Number of CV repeats.
    :type n_repeats: int
    :param random_state: Random seed used throughout.
    :type random_state: int or None
    :param max_iter: Maximum iterations for LogisticRegression.
    :type max_iter: int
    :param default_alpha: Miscoverage level for conformal prediction.
    :type default_alpha: float
    """

    max_iter: int = 1000

    _mapie: Optional[MAPIEClassifier] = field(default=None, init=False)
    _clf: Optional[VotingClassifier] = field(default=None, init=False)
    _cp_method_used: Optional[str] = field(default=None, init=False)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.__class__.__name__}("
            f"n_splits={self.n_splits}, "
            f"n_repeats={self.n_repeats}, "
            f"random_state={self.random_state}, "
            f"max_iter={self.max_iter}, "
            f"default_alpha={self.default_alpha}, "
            f"cp_method_used={self._cp_method_used}, "
            f"is_fitted={self._is_fitted})"
        )

    # ------------------------------------------------------------------
    # Base learners
    # ------------------------------------------------------------------
    def _make_logreg_pipeline(self) -> Pipeline:
        """StandardScaler + LogisticRegression pipeline."""
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                        class_weight="balanced",
                        C=2.0,
                    ),
                ),
            ]
        )

    def _make_svm_pipeline(self) -> Pipeline:
        """StandardScaler + RBF SVC pipeline."""
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        C=2.0,
                        gamma="scale",
                        class_weight="balanced",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def _make_knn_pipeline(self) -> Pipeline:
        """StandardScaler + KNN pipeline."""
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "knn",
                    KNeighborsClassifier(
                        n_neighbors=5,
                        weights="distance",
                    ),
                ),
            ]
        )

    def _make_ensemble_estimator(self) -> VotingClassifier:
        """
        Construct the soft-voting ensemble classifier.

        :returns: Configured :class:`VotingClassifier` instance.
        :rtype: sklearn.ensemble.VotingClassifier
        """
        lr_pipe = self._make_logreg_pipeline()
        svm_pipe = self._make_svm_pipeline()
        knn_pipe = self._make_knn_pipeline()
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            random_state=self.random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        return VotingClassifier(
            estimators=[
                ("lr", lr_pipe),
                ("svm", svm_pipe),
                ("knn", knn_pipe),
                ("rf", rf),
            ],
            voting="soft",
            n_jobs=-1,
        )

    # ------------------------------------------------------------------
    # Internal helpers for handling X, y
    # ------------------------------------------------------------------
    def _prepare_xy(
        self,
        X: ArrayLike,
        y: Optional[Sequence] = None,
        label_col: str = "label",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalise X, y input for fit into numpy arrays and set feature names.

        Two modes:

        * If ``X`` is a DataFrame and ``y`` is ``None``:
          use ``label_col`` as target and all other columns as features.
        * Otherwise: use ``X`` as features and ``y`` as labels.

        :param X: Feature matrix or combined DataFrame.
        :type X: ArrayLike
        :param y: Target labels or ``None`` in DataFrame mode.
        :type y: Sequence or None
        :param label_col: Label column name when ``X`` is a DataFrame.
        :type label_col: str
        :returns: Tuple ``(X_arr, y_arr)`` of numpy arrays.
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
        :raises ValueError: If ``y`` is missing in array mode.
        """
        # DataFrame mode (features + label in one df)
        if isinstance(X, pd.DataFrame) and y is None:
            if label_col not in X.columns:
                raise ValueError(f"Label column '{label_col}' not found in dataframe.")
            self._feature_names = [c for c in X.columns if c != label_col]
            X_arr = X[self._feature_names].values
            y_arr = X[label_col].astype(int).values
            return X_arr, y_arr

        # Array mode
        if y is None:
            raise ValueError(
                "When y is None, X must be a DataFrame with label_col present."
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

        y_arr = np.asarray(y).astype(int).ravel()
        return X_arr, y_arr

    def _ensure_feature_array(self, X: ArrayLike) -> np.ndarray:
        """
        Convert input X at prediction time to a numpy array with correct shape.

        :param X: Input feature data (DataFrame or ndarray).
        :type X: ArrayLike
        :returns: Feature matrix of shape ``(n_samples, n_features)``.
        :rtype: numpy.ndarray
        :raises ValueError: If the number of columns does not match the
            fitted feature space.
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
        label_col: str = "label",
    ) -> "DockQClassifier":
        """
        Fit the classification ensemble (and MAPIE if available).

        There are two supported calling conventions:

        1. **DataFrame mode**::

               clf.fit(df, label_col="label")

        2. **Array mode**::

               clf.fit(X, y)

        :param X: Input data (DataFrame or array-like).
        :type X: ArrayLike
        :param y: Target labels for array mode; ignore for DataFrame mode.
        :type y: Sequence or None
        :param label_col: Name of label column in DataFrame mode.
        :type label_col: str
        :returns: The fitted estimator (for chaining).
        :rtype: DockQClassifier
        """
        X_arr, y_arr = self._prepare_xy(X, y=y, label_col=label_col)

        base_estimator = self._make_ensemble_estimator()
        self._cv_summary = classification_cv_summary(
            base_estimator,
            X_arr,
            y_arr,
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )

        # MAPIE conformal classifier (optional)
        self._mapie = None
        self._cp_method_used = None
        if has_mapie_classification():
            try:
                mapie = MAPIEClassifier(
                    estimator=self._make_ensemble_estimator(),
                    cv=self.n_splits,
                    method="lac",
                    n_jobs=-1,
                    random_state=self.random_state,
                )
                mapie.fit(X_arr, y_arr)
                self._cp_method_used = "lac"
            except Exception:
                mapie = MAPIEClassifier(
                    estimator=self._make_ensemble_estimator(),
                    cv=self.n_splits,
                    method="score",
                    n_jobs=-1,
                    random_state=self.random_state,
                )
                mapie.fit(X_arr, y_arr)
                self._cp_method_used = "score"
            self._mapie = mapie

        # Fit final ensemble on all data
        clf = self._make_ensemble_estimator()
        clf.fit(X_arr, y_arr)
        self._clf = clf

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def mapie_model_(self) -> MAPIEClassifier:
        """
        Underlying MAPIE classifier.

        :returns: Fitted :class:`MAPIEClassifier` instance.
        :rtype: MAPIEClassifier
        :raises RuntimeError: If MAPIE is not available or not fitted.
        """
        self._check_is_fitted()
        if self._mapie is None:
            raise RuntimeError("MAPIE is not available or was not fitted.")
        return self._mapie

    @property
    def cp_method_used_(self) -> Optional[str]:
        """
        Conformal prediction method used by MAPIE (``'lac'`` or ``'score'``).

        :returns: Method name or ``None``.
        :rtype: str or None
        """
        return self._cp_method_used

    # ------------------------------------------------------------------
    # Prediction helpers (sklearn-style)
    # ------------------------------------------------------------------
    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Predict class-1 probabilities for new samples.

        :param X: Feature matrix (DataFrame or ndarray).
        :type X: ArrayLike
        :returns: Probability for the positive class of shape ``(n_samples,)``.
        :rtype: numpy.ndarray
        """
        self._check_is_fitted()
        if self._clf is None:
            raise RuntimeError("Internal classifier is not fitted.")
        X_arr = self._ensure_feature_array(X)
        proba = self._clf.predict_proba(X_arr)[:, 1]
        return proba

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict hard labels from probabilities using a 0.5 threshold.

        This mirrors the usual sklearn :meth:`predict` semantics.

        :param X: Feature matrix (DataFrame or ndarray).
        :type X: ArrayLike
        :returns: Predicted labels of shape ``(n_samples,)``.
        :rtype: numpy.ndarray
        """
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def fit_predict(
        self,
        X: ArrayLike,
        y: Optional[Sequence] = None,
        label_col: str = "label",
    ) -> np.ndarray:
        """
        Fit the model and immediately return predictions on the same data.

        :param X: Input data (DataFrame or array-like).
        :type X: ArrayLike
        :param y: Target labels in array mode; ignored in DataFrame mode.
        :type y: Sequence or None
        :param label_col: Label column name in DataFrame mode.
        :type label_col: str
        :returns: Predicted labels for the training data.
        :rtype: numpy.ndarray
        """
        self.fit(X, y=y, label_col=label_col)
        return self.predict(X)

    def predict_with_confidence(
        self,
        X: ArrayLike,
        alpha: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels and MAPIE prediction sets.

        :param X: Feature matrix (DataFrame or ndarray).
        :type X: ArrayLike
        :param alpha: Miscoverage level; if ``None``, uses ``default_alpha``.
        :type alpha: float or None
        :returns: Tuple ``(y_pred, y_sets)`` where ``y_sets`` is a boolean
            array of shape ``(n_samples, n_classes)``.
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
        :raises RuntimeError: If MAPIE is not available or not fitted.
        """
        self._check_is_fitted()
        if self._mapie is None:
            raise RuntimeError("MAPIE is not available or was not fitted.")

        if alpha is None:
            alpha = self.default_alpha

        X_arr = self._ensure_feature_array(X)
        y_pred, y_psets = self._mapie.predict(X_arr, alpha=[alpha])
        y_psets = np.squeeze(y_psets, axis=-1)
        return y_pred, y_psets

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _confidence_level_from_sets(y_psets: np.ndarray) -> np.ndarray:
        """
        Convert MAPIE prediction sets to qualitative confidence labels.

        For binary problems, the rule is:

        * if both classes are present in the set → ``"low"``
        * otherwise → ``"high"``

        :param y_psets: Boolean prediction sets of shape ``(n_samples, 2)``.
        :type y_psets: numpy.ndarray
        :returns: Array of strings (``"low"`` or ``"high"``).
        :rtype: numpy.ndarray
        """
        conf = []
        for row in y_psets:
            if row[0] and row[1]:
                conf.append("low")
            else:
                conf.append("high")
        return np.array(conf)

    def evaluate_selective(
        self,
        X: ArrayLike,
        y: Optional[Sequence] = None,
        *,
        df_mode_label_col: str = "label",
        alpha: Optional[float] = None,
        prob_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Evaluate performance overall and on a high-confidence subset.

        This is a slightly more general version of the earlier
        :meth:`evaluate_selective` that accepts either:

        * ``X`` = DataFrame with labels in ``df_mode_label_col`` and
          ``y is None``; or
        * ``X`` = features, ``y`` = label array.

        :param X: Input data (DataFrame or feature matrix).
        :type X: ArrayLike
        :param y: Label vector in array mode; ignored in DataFrame mode.
        :type y: Sequence or None
        :param df_mode_label_col: Label column name in DataFrame mode.
        :type df_mode_label_col: str
        :param alpha: Miscoverage level for MAPIE; defaults to ``default_alpha``.
        :type alpha: float or None
        :param prob_threshold: Probability threshold for hard labels.
        :type prob_threshold: float
        :returns: Dictionary with metrics for all samples and high-confidence
            subset.
        :rtype: dict[str, Any]
        """
        self._check_is_fitted()
        if alpha is None:
            alpha = self.default_alpha

        # Prepare X/y similarly to fit, but do not change feature_names_
        if isinstance(X, pd.DataFrame) and y is None:
            if df_mode_label_col not in X.columns:
                raise ValueError(
                    f"Label column '{df_mode_label_col}' not found in dataframe."
                )
            y_true = X[df_mode_label_col].astype(int).values
            X_feat = X[self._feature_names]
        else:
            if y is None:
                raise ValueError("y must be provided in array mode.")
            X_feat = X
            y_true = np.asarray(y).astype(int).ravel()

        probs = self.predict_proba(X_feat)
        y_pred = (probs >= prob_threshold).astype(int)

        metrics_all: Dict[str, Any] = {}
        metrics_all["accuracy"] = float(accuracy_score(y_true, y_pred))
        try:
            metrics_all["roc_auc"] = float(roc_auc_score(y_true, probs))
        except ValueError:
            metrics_all["roc_auc"] = float("nan")
        metrics_all["f1"] = float(f1_score(y_true, y_pred))

        if self._mapie is not None:
            _, y_sets = self.predict_with_confidence(X_feat, alpha=alpha)
            conf_level = self._confidence_level_from_sets(y_sets)
            mask_high = conf_level == "high"
            metrics_all["coverage_high"] = float(np.mean(mask_high))
        else:
            mask_high = np.ones_like(y_true, dtype=bool)
            metrics_all["coverage_high"] = 1.0

        if mask_high.any():
            y_true_high = y_true[mask_high]
            y_pred_high = y_pred[mask_high]
            probs_high = probs[mask_high]
            metrics_high: Dict[str, Any] = {}
            metrics_high["accuracy"] = float(accuracy_score(y_true_high, y_pred_high))
            try:
                metrics_high["roc_auc"] = float(roc_auc_score(y_true_high, probs_high))
            except ValueError:
                metrics_high["roc_auc"] = float("nan")
            metrics_high["f1"] = float(f1_score(y_true_high, y_pred_high))
            metrics_high["n_high"] = int(mask_high.sum())
        else:
            metrics_high = {
                "accuracy": float("nan"),
                "roc_auc": float("nan"),
                "f1": float("nan"),
                "n_high": 0,
            }

        return {
            "all": metrics_all,
            "high_conf": metrics_high,
        }
