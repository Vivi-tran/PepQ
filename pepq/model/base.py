"""
base.py
=======

Common base mixin for DockQ models.

The :class:`_BaseDockQModel` class is not intended to be used directly,
but provides shared configuration and properties for classification and
regression models in :mod:`pepq.model.classification` and
:mod:`pepq.model.regression`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class _BaseDockQModel:
    """
    Common configuration and utilities shared by DockQ models.

    :param n_splits: Number of cross-validation splits per repeat.
    :type n_splits: int
    :param n_repeats: Number of cross-validation repeats.
    :type n_repeats: int
    :param random_state: Random seed used for all internal estimators.
    :type random_state: int or None
    :param default_alpha: Default miscoverage level for conformal methods
        (e.g. 0.05 corresponds to 95\\% coverage).
    :type default_alpha: float
    """

    n_splits: int = 5
    n_repeats: int = 5
    random_state: Optional[int] = 42
    default_alpha: float = 0.05

    _feature_names: List[str] = field(default_factory=list, init=False)
    _cv_summary: Dict[str, float] = field(default_factory=dict, init=False)
    _is_fitted: bool = field(default=False, init=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _check_is_fitted(self) -> None:
        """
        Raise :class:`RuntimeError` if the model has not been fitted.

        This method is used by public properties and prediction helpers.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call `fit(...)` first.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def feature_names_(self) -> List[str]:
        """
        Names of input features used during fitting.

        :returns: List of feature names.
        :rtype: list[str]
        """
        return self._feature_names

    @property
    def cv_summary_(self) -> Dict[str, float]:
        """
        Cross-validation summary statistics.

        The concrete keys depend on the subclass (classification vs
        regression), but typically include means and standard deviations
        for main metrics along with ``"n_samples"``.

        :returns: Dictionary with cross-validation metrics.
        :rtype: dict[str, float]
        :raises RuntimeError: If the model has not been fitted.
        """
        self._check_is_fitted()
        return self._cv_summary
