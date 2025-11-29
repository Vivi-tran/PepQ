from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from sklearn.decomposition import PCA


Numeric = Union[int, float, np.number]


@dataclass
class BaseEDA:
    """
    Generic exploratory data analysis helper with Nature-style visualisations.

    This class handles:

    * storage of the underlying dataframe,
    * feature / target bookkeeping,
    * a lightweight Nature-like plotting style,
    * basic numeric summaries (missingness, descriptive stats, correlations),
    * optional feature importance values,
    * figure management.

    Domain-specific subclasses (e.g. :class:`DockEDA`) add concrete plots
    such as feature–target grids or custom overview panels.

    :param df: Input dataframe.
    :type df: pandas.DataFrame
    :param target_col: Main target column name, if any.
    :type target_col: Optional[str]
    :param secondary_target_col: Optional secondary target column
        (e.g. classification label derived from the main target).
    :type secondary_target_col: Optional[str]
    :param feature_cols: Optional list of feature columns. If ``None``,
        all numeric columns except targets are used.
    :type feature_cols: Optional[Sequence[str]]
    :param apply_style: Whether to apply a Nature-like matplotlib / seaborn style.
    :type apply_style: bool
    """

    df: pd.DataFrame
    target_col: Optional[str] = None
    secondary_target_col: Optional[str] = None
    feature_cols: Optional[Sequence[str]] = None
    apply_style: bool = True

    # Cached numeric summaries
    _basic_stats: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _corr_matrix: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _target_corr: Optional[pd.Series] = field(default=None, init=False, repr=False)
    _group_stats: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _missing_summary: Optional[pd.DataFrame] = field(
        default=None, init=False, repr=False
    )
    _outliers: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    # Feature importance (optional, model-based)
    _feature_importance: Optional[pd.Series] = field(
        default=None, init=False, repr=False
    )

    # Figures registry
    _figures: Dict[str, Figure] = field(default_factory=dict, init=False, repr=False)

    # Palette / style colours
    _primary_color: str = field(default="#355C7D", init=False, repr=False)
    _accent_color: str = field(default="#E15A97", init=False, repr=False)
    _muted_fill: str = field(default="#C6D8D3", init=False, repr=False)
    _muted_gray: str = field(default="#6B6B6B", init=False, repr=False)

    # ------------------------------------------------------------------
    # Core initialisation + style
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        """
        Copy dataframe, apply plotting style, and infer feature columns.
        """
        self.df = self.df.copy()

        if self.apply_style:
            self._set_nature_style()

        if self.feature_cols is None:
            self.feature_cols = self._infer_feature_cols()
        else:
            self.feature_cols = [c for c in self.feature_cols if c in self.df.columns]

    def __repr__(self) -> str:
        """
        Short textual summary for debugging / logging.

        :return: String representation with dataset dimensions and key options.
        :rtype: str
        """
        n_rows, n_cols = self.df.shape
        features = list(self.feature_cols or [])
        return (
            f"{self.__class__.__name__}("
            f"rows={n_rows}, cols={n_cols}, "
            f"target={self.target_col!r}, "
            f"secondary_target={self.secondary_target_col!r}, "
            f"n_features={len(features)})"
        )

    def _set_nature_style(self) -> None:
        """
        Apply a Nature-like plotting style.

        Clean white background, muted colours, thin spines and small fonts.
        """
        sns.set_theme(style="white", context="paper")
        sns.set_context("paper", font_scale=1.1)

        plt.rcParams.update(
            {
                "figure.dpi": 150,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.linewidth": 0.7,
                "axes.labelsize": 9,
                "axes.titlesize": 10,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "legend.fontsize": 8,
                "legend.frameon": False,
                "grid.color": "#E5E5E5",
                "grid.linestyle": "-",
                "grid.linewidth": 0.4,
            }
        )

        sns.set_palette([self._primary_color, self._accent_color, "#99B898", "#F8B195"])

    # ------------------------------------------------------------------
    # Public properties (numeric results + figures)
    # ------------------------------------------------------------------
    @property
    def basic_stats(self) -> Optional[pd.DataFrame]:
        """
        Descriptive statistics per feature.

        :return: Dataframe of basic statistics, or ``None`` if not computed.
        :rtype: Optional[pandas.DataFrame]
        """
        return self._basic_stats

    @property
    def corr_matrix(self) -> Optional[pd.DataFrame]:
        """
        Feature–feature correlation matrix.

        :return: Correlation matrix, or ``None`` if not computed.
        :rtype: Optional[pandas.DataFrame]
        """
        return self._corr_matrix

    @property
    def target_correlations(self) -> Optional[pd.Series]:
        """
        Feature–target correlation series.

        :return: Correlation with the main target, or ``None`` if not computed.
        :rtype: Optional[pandas.Series]
        """
        return self._target_corr

    @property
    def group_stats(self) -> Optional[pd.DataFrame]:
        """
        Per-group summary statistics for numeric features.

        :return: Grouped statistics, or ``None`` if not computed.
        :rtype: Optional[pandas.DataFrame]
        """
        return self._group_stats

    @property
    def missing_summary(self) -> Optional[pd.DataFrame]:
        """
        Missing value counts and ratios per column.

        :return: Missing value summary, or ``None`` if not computed.
        :rtype: Optional[pandas.DataFrame]
        """
        return self._missing_summary

    @property
    def outliers(self) -> Optional[pd.DataFrame]:
        """
        Boolean dataframe marking outliers per feature.

        :return: Outlier mask, or ``None`` if not computed.
        :rtype: Optional[pandas.DataFrame]
        """
        return self._outliers

    @property
    def feature_importance(self) -> Optional[pd.Series]:
        """
        Feature importance scores (typically model-based).

        :return: Series indexed by feature name, or ``None`` if not attached.
        :rtype: Optional[pandas.Series]
        """
        return self._feature_importance

    @property
    def figures(self) -> Dict[str, Figure]:
        """
        Access generated figures by name.

        Figures are stored under keys such as ``"overview_nature"``,
        ``"distributions"``, etc.

        :return: Mapping from figure name to :class:`matplotlib.figure.Figure`.
        :rtype: Dict[str, matplotlib.figure.Figure]
        """
        return self._figures

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _infer_feature_cols(self) -> List[str]:
        """
        Infer feature columns as numeric columns excluding any targets.

        :return: List of inferred feature names.
        :rtype: List[str]
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = {c for c in [self.target_col, self.secondary_target_col] if c}
        return [c for c in numeric_cols if c not in exclude]

    def _get_numeric_features(self) -> List[str]:
        """
        Return numeric feature columns intersected with the dataframe.

        :return: List of numeric feature names.
        :rtype: List[str]
        """
        cols = self.feature_cols or []
        numeric = self.df.select_dtypes(include=[np.number]).columns
        return [c for c in cols if c in numeric]

    def _is_classification_target(self) -> bool:
        """
        Decide if the main target looks like a classification label.

        Heuristic: low-cardinality numeric or non-floating dtype.

        :return: ``True`` if classification-like, else ``False``.
        :rtype: bool
        """
        if self.target_col is None or self.target_col not in self.df.columns:
            return False
        series = self.df[self.target_col].dropna()
        if series.empty:
            return False
        return series.nunique() <= 10

    def _store_figure(self, name: str, fig: Figure) -> None:
        """
        Store a figure internally under a given name.

        :param name: Key for the figure (e.g. ``"overview_nature"``).
        :type name: str
        :param fig: Matplotlib figure object.
        :type fig: matplotlib.figure.Figure
        """
        self._figures[name] = fig

    def _add_panel_label(
        self,
        ax: Axes,
        label: str,
        x: float = -0.12,
        y: float = 1.08,
    ) -> None:
        """
        Add a bold panel label (e.g. 'A', 'B') to an axis.

        :param ax: Axis to annotate.
        :type ax: matplotlib.axes.Axes
        :param label: Panel label, typically an uppercase letter.
        :type label: str
        :param x: Horizontal position in axis coordinates.
        :type x: float
        :param y: Vertical position in axis coordinates.
        :type y: float
        """
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=11,
            va="bottom",
            ha="left",
        )

    def _get_label_column(self) -> Optional[str]:
        """
        Decide which column to treat as a class label for colouring.

        Prefers :pyattr:`secondary_target_col` if it exists and looks
        categorical; otherwise uses :pyattr:`target_col` if that looks
        categorical.

        :return: Name of label column, or ``None``.
        :rtype: Optional[str]
        """
        candidates: List[str] = []
        if self.secondary_target_col and self.secondary_target_col in self.df.columns:
            candidates.append(self.secondary_target_col)
        if self.target_col and self.target_col in self.df.columns:
            candidates.append(self.target_col)

        for col in candidates:
            series = self.df[col].dropna()
            if series.empty:
                continue
            if series.nunique() <= 10:
                return col
        return None

    def _top_k_features_by_target_corr(self, top_k: int) -> List[str]:
        """
        Select top-k features by absolute correlation with the target.

        :param top_k: Number of features to select.
        :type top_k: int
        :return: List of feature names (possibly shorter if not enough features).
        :rtype: List[str]
        """
        if self._target_corr is None:
            self.compute_correlations(with_target_only=True)

        if self._target_corr is None or self._target_corr.empty:
            return []

        return (
            self._target_corr.abs()
            .sort_values(ascending=False)
            .head(top_k)
            .index.tolist()
        )

    # ------------------------------------------------------------------
    # Feature-importance helpers
    # ------------------------------------------------------------------
    def set_feature_importance(
        self,
        importance: Union[pd.Series, Dict[str, float], np.ndarray],
        feature_names: Optional[Sequence[str]] = None,
    ) -> "BaseEDA":
        """
        Attach externally computed feature importances.

        The importances are stored as a :class:`pandas.Series` in
        :pyattr:`feature_importance` and used by plotting methods such
        as :meth:`DockEDA.plot_overview_nature`.

        :param importance: Feature importance values. Can be:

            * a :class:`pandas.Series` with index = feature names,
            * a mapping ``{feature_name: importance}``,
            * a 1-D :class:`numpy.ndarray`, in which case
              ``feature_names`` must be provided.
        :type importance: Union[pandas.Series, Dict[str, float], numpy.ndarray]
        :param feature_names: Feature names if ``importance`` is an array.
        :type feature_names: Optional[Sequence[str]]
        :return: Self for fluent chaining.
        :rtype: BaseEDA
        """
        if isinstance(importance, pd.Series):
            series = importance.astype(float)
        elif isinstance(importance, dict):
            series = pd.Series(importance, dtype=float)
        else:
            arr = np.asarray(importance, dtype=float)
            if feature_names is None:
                raise ValueError(
                    "feature_names must be provided when importance is an array."
                )
            if len(arr) != len(feature_names):
                raise ValueError(
                    "Length of importance array and feature_names must match."
                )
            series = pd.Series(arr, index=list(feature_names), dtype=float)

        numeric_feats = set(self._get_numeric_features())
        series = series[series.index.intersection(numeric_feats)].dropna()

        self._feature_importance = series
        return self

    def set_feature_importance_from_model(self, model: Any) -> "BaseEDA":
        """
        Attach feature importances from a fitted scikit-learn model.

        The model must expose a ``feature_importances_`` attribute
        (e.g. :class:`RandomForestRegressor`,
        :class:`RandomForestClassifier`,
        :class:`GradientBoostingRegressor`, ...).

        :param model: Fitted model with ``feature_importances_`` attribute.
        :type model: Any
        :return: Self for fluent chaining.
        :rtype: BaseEDA
        :raises AttributeError: If the model has no ``feature_importances_``.
        :raises ValueError: If the length of ``feature_importances_`` does
            not match the number of numeric features.
        """
        if not hasattr(model, "feature_importances_"):
            raise AttributeError(
                "Model does not have 'feature_importances_' attribute."
            )

        feats = self._get_numeric_features()
        importances = np.asarray(model.feature_importances_, dtype=float)
        if len(importances) != len(feats):
            raise ValueError(
                "Length of model.feature_importances_ does not match "
                "number of numeric features."
            )

        series = pd.Series(importances, index=feats, dtype=float)
        return self.set_feature_importance(series)

    # ------------------------------------------------------------------
    # Numeric summaries (fluent API)
    # ------------------------------------------------------------------
    def compute_missing_summary(self) -> "BaseEDA":
        """
        Compute missing value counts and ratios per column.

        :return: Self for fluent chaining.
        :rtype: BaseEDA
        """
        total = len(self.df)
        counts = self.df.isna().sum()
        ratio = counts / float(total) if total > 0 else 0.0
        self._missing_summary = pd.DataFrame(
            {"missing_count": counts, "missing_ratio": ratio}
        ).sort_values("missing_ratio", ascending=False)
        return self

    def compute_basic_stats(
        self,
        percentiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    ) -> "BaseEDA":
        """
        Compute descriptive statistics for numeric features.

        :param percentiles: Percentiles to compute.
        :type percentiles: Sequence[float]
        :return: Self for fluent chaining.
        :rtype: BaseEDA
        """
        cols = self._get_numeric_features()
        if not cols:
            self._basic_stats = None
            return self

        desc = self.df[cols].describe(percentiles=list(percentiles)).T
        self._basic_stats = desc
        return self

    def compute_correlations(
        self,
        method: str = "pearson",
        with_target_only: bool = False,
    ) -> "BaseEDA":
        """
        Compute correlation matrix and feature–target correlations.

        :param method: Correlation method (``"pearson"`` or ``"spearman"``).
        :type method: str
        :param with_target_only: If ``True``, only correlations with target
            are needed (matrix is still cached).
        :type with_target_only: bool
        :return: Self for fluent chaining.
        :rtype: BaseEDA
        """
        cols = self._get_numeric_features()
        if self.target_col and self.target_col in self.df.columns:
            cols = list(dict.fromkeys(cols + [self.target_col]))

        if not cols:
            self._corr_matrix = None
            self._target_corr = None
            return self

        numeric_df = self.df[cols].dropna()
        corr = numeric_df.corr(method=method)
        self._corr_matrix = corr

        if self.target_col and self.target_col in corr:
            self._target_corr = corr[self.target_col].drop(self.target_col)
        else:
            self._target_corr = None

        return self

    def compute_group_stats(
        self,
        group_col: Optional[str] = None,
    ) -> "BaseEDA":
        """
        Compute per-group summary statistics for numeric features.

        :param group_col: Column to group by; defaults to :pyattr:`target_col`.
        :type group_col: Optional[str]
        :return: Self for fluent chaining.
        :rtype: BaseEDA
        """
        group_col = group_col or self.target_col
        if group_col is None or group_col not in self.df.columns:
            self._group_stats = None
            return self

        cols = self._get_numeric_features()
        if not cols:
            self._group_stats = None
            return self

        agg = {c: ["mean", "std", "median"] for c in cols}
        self._group_stats = self.df.groupby(group_col)[cols].agg(agg)
        return self

    def detect_outliers_iqr(self, k: Numeric = 1.5) -> "BaseEDA":
        """
        Flag outliers per feature via the IQR rule.

        A value is an outlier if it lies outside
        ``[Q1 - k * IQR, Q3 + k * IQR]`` for that feature.

        :param k: IQR multiplier for outlier fences.
        :type k: float
        :return: Self for fluent chaining.
        :rtype: BaseEDA
        """
        cols = self._get_numeric_features()
        mask = pd.DataFrame(False, index=self.df.index, columns=cols)

        for c in cols:
            series = self.df[c].dropna()
            if series.empty:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - k * iqr
            upper = q3 + k * iqr
            mask[c] = (self.df[c] < lower) | (self.df[c] > upper)

        self._outliers = mask
        return self

    # ------------------------------------------------------------------
    # Generic visual panels
    # ------------------------------------------------------------------
    def plot_distributions(
        self,
        cols: Optional[Sequence[str]] = None,
        max_cols: int = 3,
        hue: Optional[str] = None,
    ) -> "BaseEDA":
        """
        Plot distributions (histogram + KDE) for numeric features.

        Optionally colour histograms by a label column (e.g. ``"label"``)
        using the ``hue`` parameter. This is particularly useful for
        binary labels (0/1) to visualise class separation.

        :param cols: Columns to plot; if ``None``, uses all numeric features.
        :type cols: Optional[Sequence[str]]
        :param max_cols: Maximum number of subplots per row.
        :type max_cols: int
        :param hue: Optional column name to use as hue in the histograms.
            If provided, it must be present in :pyattr:`df`.
        :type hue: Optional[str]
        :return: Self for fluent chaining.
        :rtype: BaseEDA
        """
        cols = list(cols) if cols is not None else self._get_numeric_features()
        if not cols:
            return self

        if hue is not None and hue not in self.df.columns:
            raise KeyError(f"hue column {hue!r} not found in dataframe.")

        n = len(cols)
        n_cols = min(max_cols, n)
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.0 * n_cols, 2.3 * n_rows),
        )
        axes = np.atleast_1d(axes).flatten()

        for ax, col in zip(axes, cols):
            if hue is None:
                sns.histplot(
                    data=self.df,
                    x=col,
                    kde=True,
                    stat="density",
                    bins=40,
                    color=self._primary_color,
                    alpha=0.7,
                    edgecolor="white",
                    linewidth=0.3,
                    ax=ax,
                )
            else:
                sns.histplot(
                    data=self.df,
                    x=col,
                    hue=hue,
                    kde=True,
                    stat="density",
                    common_norm=False,
                    bins=40,
                    alpha=0.5,
                    edgecolor="white",
                    linewidth=0.3,
                    ax=ax,
                )

            ax.set_title(col, pad=4)
            ax.set_ylabel("Density")
            ax.grid(axis="y", alpha=0.3)

        for ax in axes[n:]:
            ax.axis("off")

        fig.tight_layout()
        self._store_figure("distributions", fig)
        return self

    def plot_correlation_triangle(
        self,
        annot: bool = False,
    ) -> "BaseEDA":
        """
        Plot an upper-triangle correlation heatmap.

        :param annot: Whether to annotate correlation values.
        :type annot: bool
        :return: Self for fluent chaining.
        :rtype: BaseEDA
        """
        if self._corr_matrix is None:
            self.compute_correlations()

        corr = self._corr_matrix
        if corr is None or corr.empty:
            return self

        mask = np.tril(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=(4.5, 4.2))
        sns.heatmap(
            corr,
            mask=mask,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.4,
            cbar_kws={"shrink": 0.7, "label": "r"},
            annot=annot,
            fmt=".2f" if annot else "",
            ax=ax,
        )
        ax.set_title("Feature correlation", pad=6)

        fig.tight_layout()
        self._store_figure("correlation_triangle", fig)
        return self

    def plot_pca(
        self,
        n_components: int = 2,
        color: Optional[str] = None,
        standardize: bool = True,
        sample: Optional[int] = None,
        figsize: Tuple[float, float] = (6.0, 5.0),
    ) -> "BaseEDA":
        """
        Plot a 2D PCA scatter of samples coloured by a column.

        Useful for visualising global structure, clusters and outliers.

        :param n_components: Number of PCA components to compute.
            Currently only the first two are visualised.
        :type n_components: int
        :param color: Optional column name to use for point colours
            (e.g. ``"label"`` or ``"dockq"``). If ``None``, a single
            colour is used.
        :type color: Optional[str]
        :param standardize: Whether to standardise features to zero mean
            and unit variance before PCA.
        :type standardize: bool
        :param sample: Optional maximum number of samples to plot.
            If provided and smaller than the dataset, a random subset of
            rows is used.
        :type sample: Optional[int]
        :param figsize: Figure size in inches ``(width, height)``.
        :type figsize: Tuple[float, float]
        :return: Self for fluent chaining.
        :rtype: BaseEDA
        """
        feats = self._get_numeric_features()
        if not feats:
            return self

        df_plot = self.df[feats].dropna()
        if sample is not None and len(df_plot) > sample:
            df_plot = df_plot.sample(sample, random_state=0)

        X = df_plot.values
        if standardize:
            X = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=0) + 1e-8)

        pca = PCA(n_components=n_components, random_state=0)
        X_pca = pca.fit_transform(X)
        pc1, pc2 = X_pca[:, 0], X_pca[:, 1]

        fig, ax = plt.subplots(figsize=figsize)

        if color is not None and color in self.df.columns:
            # align colour series with df_plot index
            cvals = self.df.loc[df_plot.index, color]
            scatter = ax.scatter(pc1, pc2, c=cvals, cmap="viridis", s=15, alpha=0.8)
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label(color)
        else:
            ax.scatter(pc1, pc2, s=15, alpha=0.8, color=self._primary_color)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        var_exp = pca.explained_variance_ratio_
        if var_exp.size >= 2:
            ax.set_title(
                f"PCA (PC1 {var_exp[0]:.1%}, PC2 {var_exp[1]:.1%})",
                pad=6,
            )
        else:
            ax.set_title("PCA", pad=6)

        fig.tight_layout()
        self._store_figure("pca", fig)
        return self

    def plot_pairwise(
        self,
        features: Sequence[str],
        kind: str = "reg",
        hue: Optional[str] = None,
        sample: Optional[int] = None,
    ) -> "BaseEDA":
        """
        Plot a seaborn PairGrid / pairplot over selected features.

        This is useful for inspecting pairwise relationships and marginal
        distributions for a small subset of features.

        :param features: List of feature columns to include.
        :type features: Sequence[str]
        :param kind: Kind of pairplot (e.g. ``"reg"``, ``"scatter"``,
            ``"kde"``, ``"hist"``). Passed through to
            :func:`seaborn.pairplot`.
        :type kind: str
        :param hue: Optional column name for colouring points by class
            (e.g. ``"label"``).
        :type hue: Optional[str]
        :param sample: Optional maximum number of samples to plot. If specified
            and smaller than the dataset, a random subset of rows is used.
        :type sample: Optional[int]
        :return: Self for fluent chaining.
        :rtype: BaseEDA
        """
        cols = [c for c in features if c in self.df.columns]
        if not cols:
            return self

        df_plot = self.df[
            cols + ([hue] if hue is not None and hue in self.df.columns else [])
        ]
        df_plot = df_plot.dropna()
        if sample is not None and len(df_plot) > sample:
            df_plot = df_plot.sample(sample, random_state=0)

        g = sns.pairplot(
            df_plot,
            vars=cols,
            hue=hue,
            kind=kind,
            diag_kind="kde" if kind in {"reg", "scatter"} else "hist",
            corner=False,
        )
        g.fig.suptitle("Pairwise feature relationships", y=1.02)

        self._store_figure("pairwise", g.fig)
        return self

    def plot_corr_clustermap(
        self,
        method: str = "average",
        metric: str = "euclidean",
        figsize: Tuple[float, float] = (6.0, 6.0),
    ) -> "BaseEDA":
        """
        Plot a clustered heatmap of the feature correlation matrix.

        Hierarchical clustering helps identify groups of correlated
        features, which is useful for feature selection and redundancy
        analysis.

        :param method: Linkage method for hierarchical clustering
            (e.g. ``"average"``, ``"single"``, ``"complete"``, ``"ward"``).
        :type method: str
        :param metric: Distance metric for clustering
            (e.g. ``"euclidean"``, ``"correlation"``).
        :type metric: str
        :param figsize: Figure size in inches ``(width, height)``.
        :type figsize: Tuple[float, float]
        :return: Self for fluent chaining.
        :rtype: BaseEDA
        """
        if self.corr_matrix is None:
            self.compute_correlations()

        corr = self.corr_matrix
        if corr is None or corr.empty:
            return self

        cg = sns.clustermap(
            corr,
            method=method,
            metric=metric,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            linewidths=0.3,
            figsize=figsize,
            cbar_kws={"label": "r"},
        )
        cg.ax_heatmap.set_title("Clustered feature correlation", pad=6)

        self._store_figure("corr_clustermap", cg.fig)
        return self

    def plot_hexbin(
        self,
        x: str,
        y: str,
        c: Optional[str] = None,
        gridsize: int = 40,
        figsize: Tuple[float, float] = (5.0, 4.0),
    ) -> "BaseEDA":
        """
        Plot a 2D hexbin density (optionally coloured by a third variable).

        This is useful when scatter plots are too dense to interpret and
        you want to visualise joint distributions or feature interactions.

        :param x: Column name for the x-axis.
        :type x: str
        :param y: Column name for the y-axis.
        :type y: str
        :param c: Optional column name whose values determine colour via
            aggregation (mean). If ``None``, point counts are shown.
        :type c: Optional[str]
        :param gridsize: Approximate number of hexagons across the x-axis.
        :type gridsize: int
        :param figsize: Figure size in inches ``(width, height)``.
        :type figsize: Tuple[float, float]
        :return: Self for fluent chaining.
        :rtype: BaseEDA
        """
        if x not in self.df.columns or y not in self.df.columns:
            return self

        fig, ax = plt.subplots(figsize=figsize)

        if c is None or c not in self.df.columns:
            hb = ax.hexbin(
                self.df[x],
                self.df[y],
                gridsize=gridsize,
                cmap="viridis",
                mincnt=1,
            )
            cbar = fig.colorbar(hb, ax=ax)
            cbar.set_label("Count")
        else:
            # use mean aggregation over c
            hb = ax.hexbin(
                self.df[x],
                self.df[y],
                C=self.df[c],
                gridsize=gridsize,
                reduce_C_function=np.mean,
                cmap="viridis",
                mincnt=1,
            )
            cbar = fig.colorbar(hb, ax=ax)
            cbar.set_label(f"Mean {c}")

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title("Hexbin density", pad=6)

        fig.tight_layout()
        self._store_figure(f"hexbin_{x}_vs_{y}", fig)
        return self
