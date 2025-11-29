"""
plot_classification.py
======================

Visualisation utilities for classification metrics.

This module provides plotting helpers that build on numeric outputs
from :mod:`pepq.model.metrics` (or directly from scikit-learn) and return
Matplotlib `Figure` (and optionally `Axes`) objects ready for notebook
display or manuscript-quality export.

Functions
---------
The module exposes the following high-level helpers:

- :func:`plot_roc_curve`
  Plot a receiver-operating-characteristic (ROC) curve and report AUC.

- :func:`plot_precision_recall_curve`
  Plot precision–recall curve and report average precision (AP).

- :func:`plot_confusion_matrix`
  Plot a labelled confusion matrix with optional normalisation and
  annotations (counts, percentages).

- :func:`plot_calibration_curve`
  Plot calibration (reliability) curve and optional histogram of
  predicted probabilities.

- :func:`plot_threshold_metrics`
  Sweep decision thresholds and plot precision / recall / F1 (and other
  threshold-dependent metrics) as functions of threshold.

- :func:`plot_classification_report`
  Multi-panel summary figure that can include any combination of the
  above panels (ROC, PR, confusion matrix, calibration, threshold
  metrics) and a compact textual summary (precision / recall / F1 /
  support). Suitable as a single-page diagnostics figure for model
  evaluation.

Design notes
------------
- All functions are lightweight wrappers around Matplotlib and
  scikit-learn utilities; they do **not** refit models (they consume
  numeric outputs such as `y_true`, `y_score`/`y_prob`, and `y_pred`).
- Return value is a `matplotlib.figure.Figure`. When helpful, functions
  also return the Axes array `(fig, axes)` for downstream programmatic
  manipulation.
- Defaults use muted, publication-ready styling (thin spines, small
  fonts, minimal legends). Functions accept `figsize` and `ax` to
  allow embedding into larger panels.

Common parameters
-----------------
Most functions accept the following parameters (where applicable):

: param y_true: Array-like of true binary labels (0/1) or multi-class labels.
: type y_true: Sequence[int] | numpy.ndarray
: param y_score: Predicted scores or probabilities (higher = more likely positive).
: type y_score: Sequence[float] | numpy.ndarray
: param y_pred: Predicted class labels (optional — required for confusion matrix).
: type y_pred: Sequence[int] | numpy.ndarray
: param pos_label: Label considered positive (default 1).
: type pos_label: int | str
: param labels: Iterable of label names / order for confusion matrices.
: type labels: Sequence[str]
: param normalize: Whether to normalise the confusion matrix per row.
: type normalize: bool
: param sample_weight: Optional sample weights for metric computations.
: type sample_weight: Optional[Sequence[float]]
: param figsize: Figure size in inches (width, height).
: type figsize: Tuple[float, float]
: param ax: Optional Matplotlib Axes to plot into.
: type ax: Optional[matplotlib.axes.Axes]

Examples
--------
.. code-block:: python

   import matplotlib.pyplot as plt
   from pepq.model.plot_classification import (
       plot_roc_curve,
       plot_precision_recall_curve,
       plot_confusion_matrix,
       plot_classification_report,
   )

   # basic ROC + PR
   fig_roc = plot_roc_curve(y_true=y_true, y_score=y_probs, pos_label=1)
   fig_roc.savefig("roc_curve.pdf", dpi=300, bbox_inches="tight")

   fig_pr = plot_precision_recall_curve(y_true=y_true, y_score=y_probs)
   fig_pr.savefig("pr_curve.pdf", dpi=300, bbox_inches="tight")

   # confusion matrix and multi-panel report
   fig_cm = plot_confusion_matrix(y_true=y_true, y_pred=y_pred, labels=class_names, normalize=True)
   fig_cm.savefig("confmat.pdf", dpi=300, bbox_inches="tight")

   fig_report = plot_classification_report(
       y_true=y_true,
       y_pred=y_pred,
       y_score=y_probs,
       panels=["roc", "pr", "confusion", "calibration"],
       title="Model diagnostics — DockQ classifier",
       figsize=(10, 6),
   )
   fig_report.savefig("classification_report.pdf", dpi=300, bbox_inches="tight")

See also
--------
- scikit-learn plotting helpers: :mod:`sklearn.metrics` (roc_curve,
  precision_recall_curve, calibration_curve) for the underlying metrics.
- :mod:`pepq.model.metrics` for convenience wrappers that compute numeric
  summaries (AUC, AP, precision@k, etc.) which can be fed into the
  plotting helpers.
"""

from __future__ import annotations

from typing import Optional, Tuple, Sequence, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve

# Class colours (use consistently: 0 -> NEG, 1 -> POS)
COLOR_NEG = "#0B4F6C"   # deep slate / navy  (class 0)
COLOR_POS = "#FF6B6B"   # coral / warm pink   (class 1)

ACCENT    = "#FFB000"   # warm gold highlight

# Diagnostics curve colours (muted / complementary)
COLOR_ROC = "#225E8E"   # muted cobalt / prussian-blue for ROC
COLOR_PR  = "#1E9A82"   # muted teal/green for PR (complements navy+coral)
COLOR_BASELINE = "#9CA3AF"


# -------------------------
# Helpers
# -------------------------
def _unpack_inputs(
    df: Optional[pd.DataFrame],
    *,
    y_true: Optional[Sequence] = None,
    y_pred: Optional[Sequence] = None,
    y_proba: Optional[Sequence] = None,
    label_col: str = "label",
    y_pred_col: Optional[str] = None,
    y_proba_col: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Normalize inputs: DataFrame mode or array mode -> numpy arrays.
    Returns (y_true_arr, y_pred_arr_or_None, y_proba_arr_or_None).
    """
    if df is not None:
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame.")
        y_true_arr = np.asarray(df[label_col].values)
        y_pred_arr = None if y_pred_col is None else np.asarray(df[y_pred_col].values)
        y_proba_arr = (
            None
            if y_proba_col is None
            else np.asarray(df[y_proba_col].values, dtype=float)
        )
        return y_true_arr, y_pred_arr, y_proba_arr

    if y_true is None:
        raise ValueError("y_true must be provided in array mode.")
    y_true_arr = np.asarray(y_true)
    y_pred_arr = None if y_pred is None else np.asarray(y_pred)
    y_proba_arr = None if y_proba is None else np.asarray(y_proba, dtype=float)
    return y_true_arr, y_pred_arr, y_proba_arr


def _positive_score_from_proba(y_proba: np.ndarray) -> np.ndarray:
    """
    Produce 1D positive-class scores from y_proba.

    - 1D input -> returned as-is
    - 2D with shape[1]==2 -> return column index 1
    - otherwise raise

    No flipping or inference attempted.
    """
    arr = np.asarray(y_proba)
    if arr.ndim == 1:
        return arr.ravel()
    if arr.ndim == 2:
        if arr.shape[1] == 2:
            return arr[:, 1]
        raise ValueError(
            "y_proba has more than 2 columns: pass a 1D positive-score array."
        )
    raise ValueError(f"Unsupported y_proba shape: {arr.shape}")


# -------------------------
# Panel drawing helpers
# -------------------------
def _draw_confusion(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalize: Optional[str] = None,
    class_names: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
) -> None:
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    cmap = plt.cm.Blues
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, aspect="auto")
    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion matrix")
    # block grid + remove spines
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
                fontweight="bold",
            )
    plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02)


def _draw_score_hist(
    ax: plt.Axes,
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    bins: int = 20,
    title: Optional[str] = None,
) -> None:
    mask_pos = np.asarray(y_true) == 1
    mask_neg = ~mask_pos
    ax.hist(
        scores[mask_neg],
        bins=bins,
        range=(0, 1),
        density=True,
        alpha=0.65,
        color=COLOR_NEG,
        label="0",
        edgecolor="none",
    )
    ax.hist(
        scores[mask_pos],
        bins=bins,
        range=(0, 1),
        density=True,
        alpha=0.65,
        color=COLOR_POS,
        label="1",
        edgecolor="none",
    )
    ax.set_xlabel("Predicted probability (class 1)")
    ax.set_ylabel("Density")
    ax.set_title(title or "Score distribution by true class")
    # ensure legend is simple '0' and '1'
    ax.legend(frameon=False, title="label")


def _draw_calibration(
    ax: plt.Axes,
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    n_bins: int = 10,
    title: Optional[str] = None,
) -> None:
    prob_true, prob_pred = calibration_curve(
        y_true, scores, n_bins=n_bins, strategy="uniform"
    )
    ax.plot(
        prob_pred, prob_true, marker="o", linewidth=1.5, color=COLOR_ROC, label="Model"
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=COLOR_BASELINE,
        linewidth=1.2,
        label="Perfect",
    )
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title or "Calibration curve")
    ax.legend(frameon=False)


def _draw_roc(
    ax: plt.Axes,
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    color: str = COLOR_ROC,
    title: Optional[str] = None,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    ax.plot(fpr, tpr, lw=2, color=color, label=f"Model (AUC = {auc:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.12, color=color, step="pre")
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=COLOR_BASELINE,
        linewidth=1.25,
        label="Random baseline (AUC = 0.5)",
    )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title or "ROC curve")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", frameon=False)


def _draw_pr(
    ax: plt.Axes,
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    color: str = COLOR_PR,
    pos_label: int = 1,
    title: Optional[str] = None,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores, pos_label=pos_label)
    pos_rate = float(np.mean(np.asarray(y_true) == pos_label))
    ax.plot(recall, precision, lw=2, color=color, label=f"Model (AP = {ap:.3f})")
    ax.fill_between(recall, precision, alpha=0.12, color=color, step="pre")
    ax.hlines(
        pos_rate,
        0.0,
        1.0,
        linestyle="--",
        color=COLOR_BASELINE,
        linewidth=1.2,
        label=f"Random baseline (π = {pos_rate:.2f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title or "Precision–Recall curve")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", frameon=False)


# -------------------------
# Public function: dashboard
# -------------------------
def plot_classification_report(
    df: Optional[pd.DataFrame] = None,
    *,
    y_true: Optional[Sequence] = None,
    y_pred: Optional[Sequence] = None,
    y_proba: Optional[Sequence] = None,
    label_col: str = "label",
    y_pred_col: Optional[str] = None,
    y_proba_col: Optional[str] = None,
    include: Optional[Sequence[str]] = None,
    include_titles: Optional[Sequence[str]] = None,
    pos_label: int = 1,
    figsize: Tuple[float, float] = (10, 7),
    n_bins: int = 20,
    calibration_bins: int = 10,
    normalize_confusion: Optional[str] = None,
    class_names: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    dpi: int = 150,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Build a 2x2 classification diagnostics figure.

    :param df: Optional DataFrame mode; if provided use label_col / y_pred_col / y_proba_col.
    :param y_true: True labels (array mode).
    :param y_pred: Predicted labels (array mode). If omitted but y_proba present, preds are derived at threshold 0.5.
    :param y_proba: Predicted probabilities/scores. If 2D with 2 cols -> use column 1 as positive score.
    :param label_col: Label column name for DataFrame mode.
    :param y_pred_col: Prediction column name for DataFrame mode.
    :param y_proba_col: Probability column name for DataFrame mode.
    :param include: panels to include; allowed: ['confusion','score_hist','calibration','roc','pr'].
                    Top-right panel will show 'score_hist' if present otherwise 'calibration' if present.
    :param include_titles: Optional list of titles for each panel in the same order as `include`.
                           If provided, len(include_titles) must equal len(include). Missing -> default titles used.
    :param pos_label: ONLY passed to average_precision_score (AP). NOT used to select probability columns.
    :param figsize: Figure size (width, height).
    :param n_bins: bins for score histogram.
    :param calibration_bins: bins for calibration curve.
    :param normalize_confusion: passed to sklearn.confusion_matrix normalize.
    :param class_names: tick labels for confusion matrix.
    :param title: optional figure title.
    :param dpi: DPI for figure.
    :returns: (fig, [axes]) axes order: [confusion, top-right, ROC, PR]
    """
    y_true_arr, y_pred_arr, y_proba_arr = _unpack_inputs(
        df,
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        label_col=label_col,
        y_pred_col=y_pred_col,
        y_proba_col=y_proba_col,
    )

    # derive predictions if not provided but probabilities are present
    if y_pred_arr is None and y_proba_arr is not None:
        scores_tmp = _positive_score_from_proba(y_proba_arr)
        y_pred_arr = (scores_tmp >= 0.5).astype(int)

    if y_pred_arr is None:
        raise ValueError("y_pred must be provided either directly or via y_proba.")

    if include is None:
        include = ["confusion", "score_hist", "roc", "pr"]
    include = list(include)
    allowed = {"confusion", "score_hist", "calibration", "roc", "pr"}
    unknown = [p for p in include if p not in allowed]
    if unknown:
        raise ValueError(f"Unknown panels requested: {unknown}")

    if include_titles is not None:
        if len(include_titles) != len(include):
            raise ValueError(
                "include_titles must have the same length as include when provided."
            )
        include_titles = list(include_titles)

    needs_proba = any(p in {"score_hist", "calibration", "roc", "pr"} for p in include)
    if needs_proba and y_proba_arr is None:
        raise ValueError(
            "Requested panels require probabilities but y_proba is missing."
        )

    scores = None if y_proba_arr is None else _positive_score_from_proba(y_proba_arr)

    # plotting rc for cleaner look
    rc = {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
    with plt.rc_context(rc):
        fig, axes_grid = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
        ax_conf, ax_topright, ax_roc, ax_pr = axes_grid.ravel()
        axes: List[plt.Axes] = [ax_conf, ax_topright, ax_roc, ax_pr]

        # top-left: confusion
        if "confusion" in include:
            idx = include.index("confusion")
            t = include_titles[idx] if include_titles is not None else None
            _draw_confusion(
                ax_conf,
                np.asarray(y_true_arr),
                np.asarray(y_pred_arr),
                normalize=normalize_confusion,
                class_names=class_names,
                title=t,
            )
        else:
            ax_conf.set_axis_off()

        # top-right: preference score_hist then calibration
        if "score_hist" in include and scores is not None:
            idx = include.index("score_hist")
            t = include_titles[idx] if include_titles is not None else None
            _draw_score_hist(
                ax_topright, np.asarray(y_true_arr), scores, bins=n_bins, title=t
            )
        elif "calibration" in include and scores is not None:
            idx = include.index("calibration")
            t = include_titles[idx] if include_titles is not None else None
            _draw_calibration(
                ax_topright,
                np.asarray(y_true_arr),
                scores,
                n_bins=calibration_bins,
                title=t,
            )
        else:
            ax_topright.set_axis_off()

        # bottom-left: ROC
        if "roc" in include and scores is not None:
            idx = include.index("roc")
            t = include_titles[idx] if include_titles is not None else None
            _draw_roc(ax_roc, np.asarray(y_true_arr), scores, color=COLOR_ROC, title=t)
        else:
            ax_roc.set_axis_off()

        # bottom-right: PR (AP uses pos_label)
        if "pr" in include and scores is not None:
            idx = include.index("pr")
            t = include_titles[idx] if include_titles is not None else None
            _draw_pr(
                ax_pr,
                np.asarray(y_true_arr),
                scores,
                pos_label=pos_label,
                color=COLOR_PR,
                title=t,
            )
        else:
            ax_pr.set_axis_off()

        if title:
            fig.suptitle(title, fontsize=14, y=1.02)

        fig.tight_layout()
        return fig, axes
