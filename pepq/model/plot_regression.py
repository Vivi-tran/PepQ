"""
plot_regression.py
========

Visualisation utilities for classification and regression metrics.

This module provides plotting helpers that build on numeric outputs
from :mod:`pepq.model.metrics` and return Matplotlib Figures:

* Regression:
  - :func:`plot_pred_vs_true`
  - :func:`plot_residuals_hist`
  - :func:`plot_regression_report` (multi-panel figure with configurable panels)

All functions are light wrappers around Matplotlib and scikit-learn and
are designed for notebook use or manuscript-quality export.

Examples
--------

.. code-block:: python

   import matplotlib.pyplot as plt
   from pepq.model.plot_regression import plot_regression_report

   fig, axes = plot_regression_report(
       y_true=y_true,
       y_pred=y_pred,
       include=["pred_vs_true", "resid_hist", "qq", "tolerance"],
       title="DockQ regression"
   )
   fig.savefig("regression_report.pdf", dpi=300, bbox_inches="tight")
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Sequence, List

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)


# ---------------------------------------------------------------------------
# Regression: shared helpers
# ---------------------------------------------------------------------------


def _ensure_reg_arrays(
    y_true: Sequence,
    y_pred: Sequence,
) -> Tuple[np.ndarray, np.ndarray]:
    y_true_arr = np.asarray(y_true, dtype=float).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=float).ravel()
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have same length (got {y_true_arr.shape} vs {y_pred_arr.shape})."
        )
    return y_true_arr, y_pred_arr


def _basic_reg_stats(
    y_true_arr: np.ndarray, y_pred_arr: np.ndarray
) -> Dict[str, float]:
    """
    Compute basic regression statistics for annotation (no SciPy dependency).
    """
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    # Pearson r
    if y_true_arr.size < 2:
        pearson_r = float("nan")
    else:
        cov = np.cov(y_true_arr, y_pred_arr, ddof=0)
        denom = np.sqrt(np.diag(cov).prod())
        pearson_r = cov[0, 1] / denom if denom != 0 else float("nan")

    # Spearman rho via rank transformation
    y_true_rank = pd.Series(y_true_arr).rank(method="average").values
    y_pred_rank = pd.Series(y_pred_arr).rank(method="average").values
    if y_true_rank.size < 2:
        spearman_rho = float("nan")
    else:
        cov_r = np.cov(y_true_rank, y_pred_rank, ddof=0)
        denom_r = np.sqrt(np.diag(cov_r).prod())
        spearman_rho = cov_r[0, 1] / denom_r if denom_r != 0 else float("nan")

    return {
        "rmse": rmse,
        "r2": r2,
        "pearson_r": float(pearson_r),
        "spearman_rho": float(spearman_rho),
    }


# --- panel drawing helpers (work on a given Axes) -------------------------


def _panel_pred_vs_true(
    ax: plt.Axes,
    y_true_arr: np.ndarray,
    y_pred_arr: np.ndarray,
    stats: Dict[str, float],
    *,
    show_stats: bool = True,
    accent: str = "#FFB000",
    scatter_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    scatter_kwargs = {} if scatter_kwargs is None else dict(scatter_kwargs)

    ax.scatter(
        y_true_arr,
        y_pred_arr,
        alpha=0.7,
        s=24,
        edgecolor="none",
        **scatter_kwargs,
    )

    vmin = float(min(y_true_arr.min(), y_pred_arr.min()))
    vmax = float(max(y_true_arr.max(), y_pred_arr.max()))
    ax.plot(
        [vmin, vmax],
        [vmin, vmax],
        linestyle="--",
        linewidth=1.0,
        color="#7F7F7F",
        label="identity",
    )

    # linear fit
    try:
        slope, intercept = np.polyfit(y_true_arr, y_pred_arr, 1)
        x_line = np.array([vmin, vmax])
        ax.plot(
            x_line,
            slope * x_line + intercept,
            linestyle="-.",
            linewidth=1.25,
            color=accent,
            label="linear fit",
        )
    except Exception:
        slope, intercept = np.nan, np.nan

    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs True")

    if show_stats:
        stats_txt = (
            f"Pearson $r$ = {stats['pearson_r']:.3f}\n"
            f"Spearman $\\rho$ = {stats['spearman_rho']:.3f}\n"
            f"$R^2$ = {stats['r2']:.3f}\n"
            f"RMSE = {stats['rmse']:.3f}"
        )
        bbox_props = dict(
            boxstyle="round,pad=0.4",
            fc="white",
            ec="#374151",
            alpha=0.9,
        )
        ax.text(
            0.02,
            0.98,
            stats_txt,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=bbox_props,
            color="#111827",
        )

    ax.legend(frameon=False, fontsize=9)


def _panel_residuals_hist(
    ax: plt.Axes,
    y_true_arr: np.ndarray,
    y_pred_arr: np.ndarray,
    *,
    bins: int = 30,
    accent: str = "#FFB000",
    hist_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    hist_kwargs = {} if hist_kwargs is None else dict(hist_kwargs)
    residuals = y_true_arr - y_pred_arr
    mean_r = float(np.mean(residuals))
    std_r = float(np.std(residuals, ddof=1) if residuals.size > 1 else 0.0)

    ax.hist(
        residuals,
        bins=bins,
        density=True,
        alpha=0.85,
        edgecolor="black",
        **hist_kwargs,
    )

    ax.axvline(
        mean_r,
        color=accent,
        linestyle="--",
        linewidth=1.25,
        label=f"mean = {mean_r:.3f}",
    )
    ax.axvline(
        mean_r - std_r,
        color="#6B7280",
        linestyle=":",
        linewidth=1.0,
        label=f"std = {std_r:.3f}",
    )
    ax.axvline(
        mean_r + std_r,
        color="#6B7280",
        linestyle=":",
        linewidth=1.0,
    )

    ax.set_xlabel("Residual (true - pred)")
    ax.set_title("Residuals distribution")
    ax.legend(frameon=False, fontsize=9)

    n = int(residuals.size)
    stats_small = f"n = {n}\nmean = {mean_r:.3f}\nstd = {std_r:.3f}"
    bbox_props = dict(
        boxstyle="round,pad=0.3",
        fc="white",
        ec="#374151",
        alpha=0.9,
    )
    ax.text(
        0.98,
        0.98,
        stats_small,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=bbox_props,
    )


def _panel_residuals_vs_pred(
    ax: plt.Axes,
    y_true_arr: np.ndarray,
    y_pred_arr: np.ndarray,
    *,
    accent: str = "#FFB000",
) -> None:
    residuals = y_true_arr - y_pred_arr
    ax.scatter(
        y_pred_arr,
        residuals,
        alpha=0.7,
        s=20,
        edgecolor="none",
    )
    ax.axhline(0.0, color="#6B7280", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (true - pred)")
    ax.set_title("Residuals vs Predicted")

    # simple trend line (linear) to show heteroscedasticity / bias
    try:
        slope, intercept = np.polyfit(y_pred_arr, residuals, 1)
        x_line = np.linspace(y_pred_arr.min(), y_pred_arr.max(), 100)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            linestyle="-.",
            linewidth=1.0,
            color=accent,
            label="trend",
        )
        ax.legend(frameon=False, fontsize=9)
    except Exception:
        pass


def _panel_qq(
    ax: plt.Axes,
    y_true_arr: np.ndarray,
    y_pred_arr: np.ndarray,
) -> None:
    residuals = y_true_arr - y_pred_arr
    residuals = np.sort(residuals)
    n = residuals.size
    if n < 2:
        ax.text(
            0.5,
            0.5,
            "Not enough points for QQ-plot",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return

    # Approximate normal quantiles using inverse CDF formula for a standard normal
    # without SciPy: use np.percentile on a standard normal sample as a proxy.
    # For robustness, sample a dense standard normal and take matched quantiles.
    grid = np.random.RandomState(0).normal(size=100000)
    probs = (np.arange(1, n + 1) - 0.5) / n * 100.0
    theor = np.percentile(grid, probs)

    ax.scatter(theor, residuals, s=18, alpha=0.7, edgecolor="none")
    # line y=x
    lo = min(theor.min(), residuals.min())
    hi = max(theor.max(), residuals.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#6B7280", linewidth=1.0)
    ax.set_xlabel("Theoretical quantiles (N(0,1))")
    ax.set_ylabel("Sample residual quantiles")
    ax.set_title("Residuals QQ-plot")


def _panel_density(
    ax: plt.Axes,
    y_true_arr: np.ndarray,
    y_pred_arr: np.ndarray,
) -> None:
    ax.hist(
        y_true_arr,
        bins=30,
        density=True,
        alpha=0.45,
        edgecolor="none",
        label="True",
    )
    ax.hist(
        y_pred_arr,
        bins=30,
        density=True,
        alpha=0.45,
        edgecolor="none",
        label="Predicted",
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Marginal distributions")
    ax.legend(frameon=False, fontsize=9)


def _panel_tolerance_curve(
    ax: plt.Axes,
    y_true_arr: np.ndarray,
    y_pred_arr: np.ndarray,
    *,
    tolerances: Optional[Sequence[float]] = None,
) -> None:
    residuals = np.abs(y_true_arr - y_pred_arr)
    if tolerances is None:
        max_err = float(np.quantile(residuals, 0.99))
        tolerances = np.linspace(0.0, max_err, 30)[1:]
    tol = np.asarray(tolerances, dtype=float)
    frac = [(residuals <= t).mean() for t in tol]

    ax.plot(tol, frac, marker="o", linewidth=1.0)
    ax.set_xlabel("Tolerance |true - pred| ≤ ε")
    ax.set_ylabel("Fraction of samples")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Within-tolerance curve")


def _panel_interval_coverage(
    ax: plt.Axes,
    y_true_arr: np.ndarray,
    y_lower_arr: np.ndarray,
    y_upper_arr: np.ndarray,
) -> None:
    if y_lower_arr.shape != y_upper_arr.shape or y_true_arr.shape != y_lower_arr.shape:
        raise ValueError("y_true, y_lower and y_upper must have the same shape.")

    within = (y_true_arr >= y_lower_arr) & (y_true_arr <= y_upper_arr)
    coverage = float(np.mean(within))
    widths = y_upper_arr - y_lower_arr
    mean_width = float(np.mean(widths))

    ax.hist(widths, bins=30, density=True, alpha=0.8, edgecolor="black")
    ax.set_xlabel("Interval width")
    ax.set_ylabel("Density")
    ax.set_title("Prediction interval width")

    txt = f"Coverage = {coverage:.3f}\nMean width = {mean_width:.3f}"
    bbox_props = dict(
        boxstyle="round,pad=0.3",
        fc="white",
        ec="#374151",
        alpha=0.9,
    )
    ax.text(
        0.98,
        0.98,
        txt,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=bbox_props,
    )


# ---------------------------------------------------------------------------
# Regression plots (standalone single-panel helpers)
# ---------------------------------------------------------------------------


def plot_pred_vs_true(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    figsize: Tuple[float, float] = (5, 5),
    show_stats: bool = True,
    scatter_kwargs: Optional[Dict[str, Any]] = None,
    accent: str = "#FFB000",
) -> plt.Figure:
    """
    Scatter plot of predicted vs true values with identity and linear-fit lines.

    Annotates Pearson r, Spearman rho, R² and RMSE on the plot when
    ``show_stats`` is ``True``.

    :param y_true: True target values.
    :type y_true: sequence
    :param y_pred: Predicted target values.
    :type y_pred: sequence
    :param figsize: Figure size (width, height) in inches.
    :type figsize: tuple(float, float)
    :param show_stats: Whether to annotate statistics on the plot.
    :type show_stats: bool
    :param scatter_kwargs: Optional keyword arguments passed to ``ax.scatter``.
    :type scatter_kwargs: dict or None
    :param accent: Accent colour for the linear-fit line.
    :type accent: str
    :returns: Matplotlib Figure object.
    :rtype: matplotlib.figure.Figure
    """
    y_true_arr, y_pred_arr = _ensure_reg_arrays(y_true, y_pred)
    stats = _basic_reg_stats(y_true_arr, y_pred_arr)

    fig, ax = plt.subplots(figsize=figsize)
    _panel_pred_vs_true(
        ax,
        y_true_arr,
        y_pred_arr,
        stats,
        show_stats=show_stats,
        accent=accent,
        scatter_kwargs=scatter_kwargs,
    )
    fig.tight_layout()
    return fig


def plot_residuals_hist(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    bins: int = 30,
    figsize: Tuple[float, float] = (5, 3),
    hist_kwargs: Optional[Dict[str, Any]] = None,
    accent: str = "#FFB000",
) -> plt.Figure:
    """
    Histogram of residuals (true - pred) with mean and std annotations.

    :param y_true: True target values.
    :type y_true: sequence
    :param y_pred: Predicted target values.
    :type y_pred: sequence
    :param bins: Number of histogram bins.
    :type bins: int
    :param figsize: Figure size (width, height) in inches.
    :type figsize: tuple(float, float)
    :param hist_kwargs: Optional keyword arguments for ``ax.hist``.
    :type hist_kwargs: dict or None
    :param accent: Accent colour for the mean residual marker.
    :type accent: str
    :returns: Matplotlib Figure object.
    :rtype: matplotlib.figure.Figure
    """
    y_true_arr, y_pred_arr = _ensure_reg_arrays(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    _panel_residuals_hist(
        ax,
        y_true_arr,
        y_pred_arr,
        bins=bins,
        accent=accent,
        hist_kwargs=hist_kwargs,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Regression multi-panel report
# ---------------------------------------------------------------------------


def plot_regression_report(
    df: Optional[pd.DataFrame] = None,
    *,
    y_true: Optional[Sequence] = None,
    y_pred: Optional[Sequence] = None,
    y_lower: Optional[Sequence] = None,
    y_upper: Optional[Sequence] = None,
    target_col: str = "target",
    y_pred_col: str = "y_pred",
    y_lower_col: str = "y_lower",
    y_upper_col: str = "y_upper",
    include: Optional[Sequence[str]] = None,
    include_titles: Optional[Sequence[str]] = None,
    layout: Optional[Tuple[int, int]] = None,
    figsize: Tuple[float, float] = (10, 4.5),
    bins: int = 30,
    tolerances: Optional[Sequence[float]] = None,
    show_stats: bool = True,
    accent: str = "#FFB000",
    title: Optional[str] = None,
    dpi: int = 150,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Multi-panel regression report figure with configurable panels and optional per-panel titles.

    Two input modes are supported:

    * DataFrame mode:
      Provide ``df`` and set column names via ``target_col``, ``y_pred_col``
      and optionally ``y_lower_col``, ``y_upper_col`` for intervals.
    * Array mode:
      Provide ``y_true`` and ``y_pred`` (and optionally ``y_lower``,
      ``y_upper``).

    Panels
    ------

    The ``include`` parameter controls which panels are plotted. Allowed
    names are:

    * ``"pred_vs_true"``      – predicted vs true scatter with stats.
    * ``"resid_hist"``        – residual histogram.
    * ``"resid_vs_pred"``     – residuals vs predicted.
    * ``"qq"``                – QQ-plot of residuals.
    * ``"density"``           – marginal distributions of true vs predicted.
    * ``"tolerance"``         – within-tolerance curve vs |true - pred|.
    * ``"interval_coverage"`` – interval width + coverage (requires intervals).

    If ``include`` is ``None``, the default is ``["pred_vs_true", "resid_hist"]``.

    Per-panel titles
    ---------------

    Use ``include_titles`` to supply a sequence of strings used as titles
    for each included panel (in the same order as ``include``). If an
    element in ``include_titles`` is ``None`` or the list is shorter than
    ``include``, the function falls back to a sensible default panel title.

    Layout
    ------

    Layout is controlled as follows:

    * If ``layout`` is not ``None``, it must be a tuple ``(n_rows, n_cols)``
      and is used as-is.
    * Otherwise, the number of columns is chosen automatically:
      - 2 columns if ``len(include) <= 4``
      - 3 columns otherwise
      and the number of rows is ``ceil(n_panels / n_cols)``.

    :param df: Input DataFrame (DataFrame mode) or ``None``.
    :type df: pandas.DataFrame or None
    :param y_true: True values (array mode).
    :type y_true: sequence or None
    :param y_pred: Predicted values (array mode).
    :type y_pred: sequence or None
    :param y_lower: Lower prediction intervals (array mode).
    :type y_lower: sequence or None
    :param y_upper: Upper prediction intervals (array mode).
    :type y_upper: sequence or None
    :param target_col: Column name for true values in DataFrame mode.
    :type target_col: str
    :param y_pred_col: Column name for predicted values in DataFrame mode.
    :type y_pred_col: str
    :param y_lower_col: Column name for interval lower bounds in DataFrame mode.
    :type y_lower_col: str
    :param y_upper_col: Column name for interval upper bounds in DataFrame mode.
    :type y_upper_col: str
    :param include: List of panel names to include
        (e.g. ``["pred_vs_true", "resid_hist", "qq"]``).
    :type include: sequence of str or None
    :param include_titles: Optional list of panel titles that override defaults.
    :type include_titles: sequence of str or None
    :param layout: Optional manual layout as ``(n_rows, n_cols)``.
    :type layout: tuple(int, int) or None
    :param figsize: Figure size (width, height) in inches.
    :type figsize: tuple(float, float)
    :param bins: Number of bins for residual histogram.
    :type bins: int
    :param tolerances: Tolerance grid for the within-tolerance curve; if
        ``None``, a grid is chosen automatically.
    :type tolerances: sequence of float or None
    :param show_stats: Whether to show statistics in the ``"pred_vs_true"`` panel.
    :type show_stats: bool
    :param accent: Accent colour for fit/mean lines.
    :type accent: str
    :param title: Optional overall figure title.
    :type title: str or None
    :param dpi: Figure DPI.
    :type dpi: int
    :returns: Tuple ``(fig, axes)`` where ``axes`` is a flat list of
        :class:`matplotlib.axes.Axes` in the order of ``include``.
    :rtype: (matplotlib.figure.Figure, list[matplotlib.axes.Axes])
    """
    # --- unpack inputs ----------------------------------------------------
    if df is not None:
        if y_true is not None or y_pred is not None:
            raise ValueError("When df is provided, y_true and y_pred must be None.")
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not found in DataFrame.")
        if y_pred_col not in df.columns:
            raise ValueError(f"y_pred_col '{y_pred_col}' not found in DataFrame.")
        y_true_arr = np.asarray(df[target_col].values, dtype=float).ravel()
        y_pred_arr = np.asarray(df[y_pred_col].values, dtype=float).ravel()
        y_lower_arr = (
            None
            if y_lower_col not in df.columns
            else np.asarray(df[y_lower_col].values, dtype=float).ravel()
        )
        y_upper_arr = (
            None
            if y_upper_col not in df.columns
            else np.asarray(df[y_upper_col].values, dtype=float).ravel()
        )
    else:
        if y_true is None or y_pred is None:
            raise ValueError("Provide either df or both y_true and y_pred.")
        y_true_arr, y_pred_arr = _ensure_reg_arrays(y_true, y_pred)
        y_lower_arr = (
            None if y_lower is None else np.asarray(y_lower, dtype=float).ravel()
        )
        y_upper_arr = (
            None if y_upper is None else np.asarray(y_upper, dtype=float).ravel()
        )

    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    # default panels
    if include is None:
        include = ["pred_vs_true", "resid_hist"]
    include = list(include)

    allowed = {
        "pred_vs_true",
        "resid_hist",
        "resid_vs_pred",
        "qq",
        "density",
        "tolerance",
        "interval_coverage",
    }
    unknown = [p for p in include if p not in allowed]
    if unknown:
        raise ValueError(f"Unknown panel names in include: {unknown}")

    # interval-dependent checks
    if "interval_coverage" in include:
        if y_lower_arr is None or y_upper_arr is None:
            raise ValueError(
                "Panels 'interval_coverage' requested but y_lower/y_upper are missing."
            )

    # prepare per-panel title mapping and defaults
    default_titles = {
        "pred_vs_true": "Predicted vs True",
        "resid_hist": "Residuals distribution",
        "resid_vs_pred": "Residuals vs Predicted",
        "qq": "Residuals QQ-plot",
        "density": "Marginal distributions",
        "tolerance": "Within-tolerance curve",
        "interval_coverage": "Prediction interval coverage",
    }

    # sanitize include_titles -> list aligned with include
    if include_titles is None:
        include_titles = []
    include_titles = list(include_titles)

    # compute core stats once
    stats = _basic_reg_stats(y_true_arr, y_pred_arr)

    # layout
    n_panels = len(include)
    if layout is not None:
        n_rows, n_cols = layout
    else:
        n_cols = 2 if n_panels <= 4 else 3
        n_rows = int(math.ceil(n_panels / n_cols))

    rc = {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.edgecolor": "#111827",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }

    with plt.rc_context(rc):
        fig, axes_grid = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            dpi=dpi,
        )
        # axes_grid can be scalar if n_rows=n_cols=1
        if isinstance(axes_grid, plt.Axes):
            axes_list = [axes_grid]
        else:
            axes_list = list(np.ravel(axes_grid))

        # mapping of panel name -> drawing function
        for idx, name in enumerate(include):
            ax = axes_list[idx]
            if name == "pred_vs_true":
                _panel_pred_vs_true(
                    ax,
                    y_true_arr,
                    y_pred_arr,
                    stats,
                    show_stats=show_stats,
                    accent=accent,
                    scatter_kwargs=None,
                )
            elif name == "resid_hist":
                _panel_residuals_hist(
                    ax,
                    y_true_arr,
                    y_pred_arr,
                    bins=bins,
                    accent=accent,
                    hist_kwargs=None,
                )
            elif name == "resid_vs_pred":
                _panel_residuals_vs_pred(
                    ax,
                    y_true_arr,
                    y_pred_arr,
                    accent=accent,
                )
            elif name == "qq":
                _panel_qq(
                    ax,
                    y_true_arr,
                    y_pred_arr,
                )
            elif name == "density":
                _panel_density(
                    ax,
                    y_true_arr,
                    y_pred_arr,
                )
            elif name == "tolerance":
                _panel_tolerance_curve(
                    ax,
                    y_true_arr,
                    y_pred_arr,
                    tolerances=tolerances,
                )
            elif name == "interval_coverage":
                _panel_interval_coverage(
                    ax,
                    y_true_arr,
                    y_lower_arr,
                    y_upper_arr,
                )

            # override panel title if include_titles provides one, otherwise keep
            # panel-internal title or default mapping
            custom_title = None
            if idx < len(include_titles):
                t = include_titles[idx]
                if t is not None and t != "":
                    custom_title = str(t)
            if custom_title is None:
                # try to use default mapping (some helpers already set titles)
                custom_title = default_titles.get(name, None)
            if custom_title:
                # set/override the axis title with consistent padding
                ax.set_title(custom_title, pad=6)

        # hide unused axes
        for j in range(n_panels, len(axes_list)):
            axes_list[j].set_axis_off()

        if title:
            fig.suptitle(title, fontsize=12, y=1.02)

        fig.tight_layout()
        # only return the axes actually used, in the order of include
        used_axes = axes_list[:n_panels]
        return fig, used_axes
