"""
helpers.py
==========

Small helpers and shared types for the :mod:`pepq.model` package.

This module centralises:

* Numpy/pandas ``ArrayLike`` type alias.
* Version-safe imports for :mod:`mapie`.
"""

from __future__ import annotations

from typing import Optional, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.DataFrame]

# ---------------------------------------------------------------------------
# Version-safe MAPIE imports
# ---------------------------------------------------------------------------

try:  # classification
    from mapie.classification import MAPIEClassifier  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - optional dependency
    try:
        # Older versions used different class naming
        from mapie.classification import MapieClassifier as MAPIEClassifier  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        MAPIEClassifier = None  # type: ignore[assignment]

try:  # regression
    from mapie.regression import MAPIERegressor  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    try:
        from mapie.regression import MapieRegressor as MAPIERegressor  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        MAPIERegressor = None  # type: ignore[assignment]


def has_mapie_classification() -> bool:
    """
    Check whether :class:`MAPIEClassifier` is available.

    :returns: ``True`` if classification MAPIE is importable.
    :rtype: bool
    """
    return MAPIEClassifier is not None


def has_mapie_regression() -> bool:
    """
    Check whether :class:`MAPIERegressor` is available.

    :returns: ``True`` if regression MAPIE is importable.
    :rtype: bool
    """
    return MAPIERegressor is not None
