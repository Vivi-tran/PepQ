"""
feature_importance.py
=====================

Unified feature-importance report and Nature-style visualisation.

This module defines :class:`FeatureImportanceReport`, which:

* Handles both **regression** and **classification** targets.
* Computes several feature-importance measures:

  - variance (population variance)
  - absolute Pearson correlation with the target
  - mutual information (regression / classification)
  - linear-model coefficients (Ridge / LogisticRegression on standardized X)
  - RandomForest model importances

* Aggregates them into a single tidy :attr:`summary_` table.
* Produces **publication-ready, Nature-style** barplot panels:

  - Single-panel via :meth:`plot_importance`
  - Multi-panel figure via :meth:`plot_all` with adaptive layout

Example
-------

.. code-block:: python

   import pandas as pd
   from feature_importance import FeatureImportanceReport

   # X_train: engineered features (DataFrame or ndarray)
   # y_train: binary or continuous target

   fir = FeatureImportanceReport(random_state=0, n_estimators=300)
   fir.fit(X_train, y_train)

   # 1) Numerical summary for selection
   print(fir.summary_.sort_values("agg_rank"))

   # 2) Single-panel: random forest importance (Top 5)
   ax = fir.plot_importance(kind="model", top_k=5)
   ax.figure.savefig("fi_model_importance.pdf", dpi=300, bbox_inches="tight")

   # 3) Multi-panel Nature-style figure (all measures)
   axes = fir.plot_all(top_k=5)
   axes["agg"].figure.savefig("fi_panel_full.pdf", dpi=300, bbox_inches="tight")

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression

from .helpers import _ensure_dataframe, _is_classification_target

ArrayLike = Union[np.ndarray, pd.DataFrame]


# ---------------------------------------------------------------------------
# Palette and plotting helpers
# ---------------------------------------------------------------------------

#: Default Nature-like palette (muted, high-contrast).
PALETTE_DEFAULT: Dict[str, str] = {
    "primary": "#0B4F6C",  # deep navy
    "secondary": "#0FA3A4",  # teal
    "accent": "#E57A00",  # warm gold highlight
    "muted": "#6B7280",  # neutral grey
    "bg": "#FFFFFF",  # background
    "soft": "#E6EEF3",  # pale primary for gradient low-end
}


def _setup_nature_style() -> None:
    """
    Configure matplotlib rcParams for a Nature/Review-like appearance.

    This function is idempotent and can be called before each plot.
    """
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE_DEFAULT["bg"],
            "axes.facecolor": PALETTE_DEFAULT["bg"],
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.titleweight": "600",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.grid": False,
            "lines.linewidth": 1.0,
            "text.color": PALETTE_DEFAULT["muted"],
            "axes.labelcolor": PALETTE_DEFAULT["muted"],
            "xtick.color": PALETTE_DEFAULT["muted"],
            "ytick.color": PALETTE_DEFAULT["muted"],
        }
    )


def _interp_color(color_from: str, color_to: str, t: float) -> str:
    """
    Linearly interpolate between two hex colors.

    :param color_from: Starting hex color (e.g. ``"#E6EEF3"``).
    :type color_from: str
    :param color_to: Ending hex color (e.g. ``"#0B4F6C"``).
    :type color_to: str
    :param t: Interpolation parameter in ``[0, 1]``.
    :type t: float
    :returns: Interpolated color as hex string.
    :rtype: str
    """
    c1 = np.array(mcolors.to_rgba(color_from))
    c2 = np.array(mcolors.to_rgba(color_to))
    c = (1.0 - t) * c1 + t * c2
    if c[3] >= 0.9999:
        return mcolors.to_hex(c[:3])
    return mcolors.to_hex(c)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


@dataclass
class FeatureImportanceReport:
    """
    Unified feature-importance report and visualisation.

    This class computes several importance measures on a given feature
    matrix ``X`` and target vector ``y``, aggregates them into a single
    ranking, and provides Nature-style barplot panels for inspection.

    Supported measures
    ------------------
    * ``variance_``          – population variance of each feature
    * ``corr_abs_``          – absolute Pearson correlation with target
    * ``mutual_info_``       – mutual information
    * ``linear_importance_`` – standardized linear coefficients
    * ``model_importance_``  – RandomForest importances

    The aggregated table :attr:`summary_` contains raw scores, normalized
    scores, and an overall ranking ``agg_rank``.

    :param random_state: Random seed for internal models.
    :type random_state: int
    :param n_estimators: Number of trees in the RandomForest model.
    :type n_estimators: int

    Attributes
    ----------
    feature_names_ : list of str
        Names of the numeric input features.
    is_classification_ : bool
        ``True`` if the target is treated as classification.
    variance_ : pandas.Series
        Variance scores per feature.
    corr_abs_ : pandas.Series
        Absolute correlation scores per feature.
    mutual_info_ : pandas.Series
        Mutual information scores per feature.
    linear_importance_ : pandas.Series
        Absolute linear coefficients per feature.
    model_importance_ : pandas.Series
        RandomForest feature importances per feature.
    summary_ : pandas.DataFrame
        Aggregated table with scores and ranks.

    Example
    -------

    .. code-block:: python

       fir = FeatureImportanceReport(random_state=0, n_estimators=300)
       fir.fit(X_train, y_train)
       print(fir.summary_.sort_values("agg_rank").head())

       # Panel with all measures
       axes = fir.plot_all(top_k=5)
       axes["agg"].figure.savefig("fi_panel.pdf", dpi=300, bbox_inches="tight")
    """

    random_state: int = 0
    n_estimators: int = 200

    # Learned attributes
    feature_names_: List[str] = field(default_factory=list, init=False)
    is_classification_: bool = field(default=False, init=False)

    variance_: pd.Series = field(default=None, init=False)
    corr_abs_: pd.Series = field(default=None, init=False)
    mutual_info_: pd.Series = field(default=None, init=False)
    linear_importance_: pd.Series = field(default=None, init=False)
    model_importance_: pd.Series = field(default=None, init=False)

    summary_: pd.DataFrame = field(default=None, init=False)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: Sequence) -> "FeatureImportanceReport":
        """
        Fit all importance measures on the provided dataset.

        :param X: Feature matrix of shape ``(n_samples, n_features)``.
                  Can be a DataFrame or ndarray. Non-numeric columns
                  are ignored with a warning.
        :type X: ArrayLike
        :param y: Target vector, binary/multiclass or continuous.
        :type y: Sequence
        :returns: The fitted instance (for chaining).
        :rtype: FeatureImportanceReport
        :raises ValueError: if no numeric features are present in ``X``.
        """
        Xdf = _ensure_dataframe(X)
        self.feature_names_ = list(Xdf.columns)
        y_arr = np.asarray(y).ravel()

        numeric = Xdf.select_dtypes(include=[np.number])
        non_numeric_cols = [c for c in Xdf.columns if c not in numeric.columns]
        if non_numeric_cols:
            warnings.warn(
                f"Non-numeric columns {non_numeric_cols} ignored in importance computations.",
                UserWarning,
            )
        if numeric.shape[1] == 0:
            raise ValueError(
                "No numeric features available in X for importance computation."
            )

        self.is_classification_ = _is_classification_target(y_arr)

        # 1) Variance
        self.variance_ = numeric.var(axis=0, ddof=0)

        # 2) Absolute correlation
        corr_vals: Dict[str, float] = {}
        for col in numeric.columns:
            xcol = numeric[col].values
            if np.std(xcol) == 0 or np.std(y_arr) == 0:
                corr_vals[col] = 0.0
            else:
                corr = np.corrcoef(xcol, y_arr)[0, 1]
                corr_vals[col] = float(abs(corr))
        self.corr_abs_ = pd.Series(corr_vals)

        # 3) Mutual information
        if self.is_classification_:
            mi_scores = mutual_info_classif(
                numeric.values, y_arr, random_state=self.random_state
            )
        else:
            mi_scores = mutual_info_regression(
                numeric.values, y_arr, random_state=self.random_state
            )
        self.mutual_info_ = pd.Series(mi_scores, index=numeric.columns)

        # 4) Linear-model importance (standardised coefficients)
        try:
            scaler = StandardScaler()
            X_std = scaler.fit_transform(numeric.values)
            if self.is_classification_:
                clf = LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    n_jobs=-1,
                    random_state=self.random_state,
                )
                clf.fit(X_std, y_arr)
                coefs = clf.coef_
                if coefs.ndim > 1:
                    coefs = np.mean(np.abs(coefs), axis=0)
                else:
                    coefs = np.abs(coefs)
            else:
                ridge = Ridge(alpha=1.0, random_state=self.random_state)
                ridge.fit(X_std, y_arr)
                coefs = np.abs(ridge.coef_)
            self.linear_importance_ = pd.Series(coefs, index=numeric.columns)
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(
                f"Linear model importance failed: {exc!r}. Filling zeros.",
                UserWarning,
            )
            self.linear_importance_ = pd.Series(0.0, index=numeric.columns)

        # 5) RandomForest importance
        if self.is_classification_:
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
                max_features="sqrt",
            )
        else:
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
                max_features=None,
            )
        rf.fit(numeric.values, y_arr)
        self.model_importance_ = pd.Series(
            rf.feature_importances_, index=numeric.columns
        )

        # Build aggregated summary
        self.summary_ = self._build_summary(numeric.columns)
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self, s: pd.Series) -> pd.Series:
        """
        Normalize a score series to the range [0, 1].

        :param s: Input series of scores.
        :type s: pandas.Series
        :returns: Normalized series.
        :rtype: pandas.Series
        """
        arr = s.values.astype(float)
        if arr.size == 0 or np.all(arr == arr[0]):
            return pd.Series(0.0, index=s.index)
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return pd.Series(0.0, index=s.index)
        return (s - mn) / (mx - mn)

    def _build_summary(self, numeric_cols: Sequence[str]) -> pd.DataFrame:
        """
        Construct the aggregated summary table over all measures.

        :param numeric_cols: Names of numeric columns used during fitting.
        :type numeric_cols: Sequence[str]
        :returns: Summary table with scores and ranks.
        :rtype: pandas.DataFrame
        """
        idx = list(numeric_cols)
        df = pd.DataFrame(index=idx)

        df["variance"] = self.variance_.reindex(idx)
        df["abs_corr"] = self.corr_abs_.reindex(idx)
        df["mutual_info"] = self.mutual_info_.reindex(idx)
        df["linear_importance"] = self.linear_importance_.reindex(idx)
        df["model_importance"] = self.model_importance_.reindex(idx)

        df["variance_norm"] = self._normalize(df["variance"])
        df["abs_corr_norm"] = self._normalize(df["abs_corr"])
        df["mutual_info_norm"] = self._normalize(df["mutual_info"])
        df["linear_importance_norm"] = self._normalize(df["linear_importance"])
        df["model_importance_norm"] = self._normalize(df["model_importance"])

        norm_cols = [c for c in df.columns if c.endswith("_norm")]
        df["agg_score"] = df[norm_cols].sum(axis=1)
        df = df.sort_values("agg_score", ascending=False)
        df["agg_rank"] = (
            df["agg_score"].rank(method="dense", ascending=False).astype(int)
        )
        df = df.reset_index().rename(columns={"index": "feature"})
        return df

    def _barplot_on_ax(
        self,
        df: pd.DataFrame,
        score_col: str,
        title: str,
        top_k: int,
        ax: plt.Axes,
        palette: Optional[Mapping[str, str]] = None,
    ) -> plt.Axes:
        """
        Internal helper to draw a Nature-style horizontal barplot.

        :param df: Summary DataFrame.
        :type df: pandas.DataFrame
        :param score_col: Column name containing scores to plot.
        :type score_col: str
        :param title: Panel title.
        :type title: str
        :param top_k: Number of features to show.
        :type top_k: int
        :param ax: Axes to draw on.
        :type ax: matplotlib.axes.Axes
        :param palette: Optional palette overriding :data:`PALETTE_DEFAULT`.
        :type palette: Mapping[str, str] or None
        :returns: The Axes object (for chaining).
        :rtype: matplotlib.axes.Axes
        """
        if palette is None:
            palette = PALETTE_DEFAULT

        df = df.sort_values(score_col, ascending=True)
        top = df.tail(top_k)

        features = top["feature"].astype(str)
        scores = top[score_col].astype(float)

        # color gradient soft -> primary
        if len(scores) > 1:
            norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        else:
            norm = np.zeros_like(scores)
        colors = [
            _interp_color(palette["soft"], palette["primary"], float(v)) for v in norm
        ]

        ax.barh(
            features, scores, height=0.72, color=colors, edgecolor="none", alpha=0.98
        )
        ax.set_xlabel(score_col.replace("_", " ").title(), color=palette["muted"])
        ax.set_ylabel("")  # feature labels are on the y-axis
        ax.set_title(
            title, loc="left", fontsize=11, fontweight="bold", color=palette["muted"]
        )

        max_score = scores.max() if len(scores) > 0 else 0.0
        offset = max(1e-8, max_score * 0.02)
        for i, (s, f) in enumerate(zip(scores, features)):
            txt_color = (
                palette["accent"] if i == (len(scores) - 1) else palette["muted"]
            )
            ax.text(s + offset, f, f"{s:.3f}", va="center", fontsize=8, color=txt_color)

        # tidy axis styling
        ax.spines["left"].set_color(palette["muted"])
        ax.spines["bottom"].set_color(palette["muted"])
        ax.tick_params(axis="x", colors=palette["muted"])
        ax.tick_params(axis="y", colors=palette["muted"], pad=6)
        return ax

    # ------------------------------------------------------------------
    # Public plotting API
    # ------------------------------------------------------------------

    def plot_importance(
        self,
        kind: str = "model",
        top_k: int = 20,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[Tuple[float, float]] = None,
        palette: Optional[Mapping[str, str]] = None,
    ) -> plt.Axes:
        """
        Plot a single importance measure as a horizontal bar chart.

        :param kind: Which measure to plot. One of
            ``{"variance", "corr", "mutual_info", "linear", "model", "agg"}``.
        :type kind: str
        :param top_k: Number of top-ranked features to display.
        :type top_k: int
        :param ax: Optional Axes to draw on. If ``None``, a new figure/Axes is created.
        :type ax: matplotlib.axes.Axes or None
        :param figsize: Optional figure size (width, height in inches) when creating a new figure.
        :type figsize: tuple(float, float) or None
        :param palette: Optional color palette overriding :data:`PALETTE_DEFAULT`.
        :type palette: Mapping[str, str] or None
        :returns: Axes containing the plot.
        :rtype: matplotlib.axes.Axes
        :raises ValueError: if the report has not been fitted.

        Example
        -------

        .. code-block:: python

           ax = fir.plot_importance(kind="agg", top_k=5)
           ax.figure.savefig("fi_agg.pdf", dpi=300, bbox_inches="tight")
        """
        if self.summary_ is None:
            raise ValueError(
                "FeatureImportanceReport is not fitted. Call fit(X, y) first."
            )

        _setup_nature_style()
        df = self.summary_.copy()

        if kind == "variance":
            score_col = "variance"
            title = "Feature variance"
        elif kind == "corr":
            score_col = "abs_corr"
            title = "Absolute correlation with target"
        elif kind == "mutual_info":
            score_col = "mutual_info"
            title = "Mutual information with target"
        elif kind == "linear":
            score_col = "linear_importance"
            title = "Linear-model coefficients"
        elif kind == "model":
            score_col = "model_importance"
            title = "Random forest feature importance"
        elif kind == "agg":
            score_col = "agg_score"
            title = "Aggregate feature importance"
        else:
            raise ValueError(
                "kind must be one of {'variance','corr','mutual_info','linear','model','agg'}"
            )

        if ax is None:
            fig_height = max(2.4, 0.30 * min(top_k, len(df)) + 0.8)
            if figsize is None:
                figsize = (6.8, fig_height)
            _, ax = plt.subplots(figsize=figsize)

        ax = self._barplot_on_ax(
            df, score_col, title, top_k=top_k, ax=ax, palette=palette
        )
        plt.tight_layout()
        return ax

    def plot_all(
        self,
        top_k: int = 20,
        include: Optional[Sequence[str]] = None,
        n_cols: Optional[int] = None,
        figsize: Optional[Tuple[float, float]] = None,
        palette: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, plt.Axes]:
        """
        Create a multi-panel figure with several importance measures.

        Layout rule
        -----------
        If ``n_cols`` is not provided:

        * If ``top_k < 4``: use 2 columns.
        * If ``top_k >= 4``: use 3 columns.

        Measures
        --------
        By default, the following measures are included:

        ``["variance", "corr", "mutual_info", "linear", "model", "agg"]``

        You can restrict this via the ``include`` argument.

        :param top_k: Number of top-ranked features to display in each panel.
        :type top_k: int
        :param include: Optional subset of measure names to include.
            Each element must be one of
            ``{"variance", "corr", "mutual_info", "linear", "model", "agg"}``.
        :type include: Sequence[str] or None
        :param n_cols: Optional number of columns in the subplot grid.
        :type n_cols: int or None
        :param figsize: Optional overall figure size (width, height in inches).
        :type figsize: tuple(float, float) or None
        :param palette: Optional colour palette overriding :data:`PALETTE_DEFAULT`.
        :type palette: Mapping[str, str] or None
        :returns: Mapping from measure name to its Axes.
        :rtype: Dict[str, matplotlib.axes.Axes]
        :raises ValueError: If the report has not been fitted or if
            ``include`` does not contain any valid measure names.

        Example
        -------

        .. code-block:: python

           axes = fir.plot_all(top_k=5, include=["model", "agg"], n_cols=2)
           axes["model"].figure.savefig("fi_panel_model_agg.pdf",
                                        dpi=300, bbox_inches="tight")
        """
        if self.summary_ is None:
            raise ValueError(
                "FeatureImportanceReport is not fitted. Call fit(X, y) first."
            )

        _setup_nature_style()

        measures_meta = [
            ("variance", "A. Feature variance", "variance"),
            ("corr", "B. Absolute correlation with target", "abs_corr"),
            ("mutual_info", "C. Mutual information with target", "mutual_info"),
            ("linear", "D. Linear-model coefficients", "linear_importance"),
            ("model", "E. Random forest feature importance", "model_importance"),
            ("agg", "F. Aggregate feature importance", "agg_score"),
        ]

        if include is not None:
            include_set = set(include)
            measures_meta = [m for m in measures_meta if m[0] in include_set]
            if not measures_meta:
                raise ValueError("include did not contain any valid measure names.")

        n_measures = len(measures_meta)
        if n_cols is None:
            n_cols = 2 if top_k < 4 else 3
        n_cols = max(1, min(n_cols, n_measures))
        n_rows = int(np.ceil(n_measures / n_cols))

        if figsize is None:
            base_height = 0.35 * min(top_k, len(self.summary_)) + 1.1
            fig_height = n_rows * base_height
            fig_width = n_cols * 3.0 + 1.0
            figsize = (fig_width, fig_height)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        df = self.summary_.copy()
        axes_dict: Dict[str, plt.Axes] = {}

        for idx, (name, title, score_col) in enumerate(measures_meta):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row][col]
            axes_dict[name] = ax
            self._barplot_on_ax(
                df,
                score_col=score_col,
                title=title,
                top_k=top_k,
                ax=ax,
                palette=palette,
            )

        # Turn off unused axes
        for idx in range(n_measures, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row][col].axis("off")

        plt.tight_layout()
        return axes_dict
