from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .base import BaseEDA


def train_default_rf_importance(
    eda: BaseEDA,
    task: Literal["regression", "classification"] = "regression",
    feature_cols: Sequence[str] | None = None,
) -> pd.Series:
    """
    Train a default RandomForest model on the EDA object's dataframe
    and return feature importances as a :class:`pandas.Series`.

    This is a convenience helper; for production you will typically
    configure and cross-validate your own model, then call
    :meth:`BaseEDA.set_feature_importance_from_model`.

    :param eda: EDA instance providing the dataframe and target column.
    :type eda: pepq.eda.base.BaseEDA
    :param task: Learning task type, ``"regression"`` or ``"classification"``.
    :type task: Literal["regression", "classification"]
    :param feature_cols: Subset of features to use. If ``None``, all numeric
        features from ``eda`` are used.
    :type feature_cols: Optional[Sequence[str]]
    :return: Series of feature importances indexed by feature name.
    :rtype: pandas.Series
    :raises ValueError: If no target column is configured.
    """
    if eda.target_col is None or eda.target_col not in eda.df.columns:
        raise ValueError("BaseEDA.target_col must be set for importance training.")

    if feature_cols is None:
        feature_cols = eda._get_numeric_features()

    X = eda.df[list(feature_cols)].values
    y = eda.df[eda.target_col].values

    if task == "classification":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=0,
            n_jobs=-1,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=0,
            n_jobs=-1,
        )

    model.fit(X, y)
    importances = np.asarray(model.feature_importances_, dtype=float)
    return pd.Series(importances, index=list(feature_cols), dtype=float)
