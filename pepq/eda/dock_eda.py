from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

import seaborn as sns
import matplotlib.pyplot as plt


from .base import BaseEDA


class DockEDA(BaseEDA):
    """
    EDA helper specialised for protein–peptide docking summaries.

    This subclass adds domain-oriented visualisations on top of
    :class:`pepq.eda.base.BaseEDA`, in particular a **Nature-style 2×2
    overview panel** and label-aware feature–target plots.

    Expected columns include, for example::

        prot_plddt, pep_plddt, PTM, PAE, iptm, composite_ptm,
        actifptm, label, dockq

    The interface is fluent: computation and plotting methods return
    ``self`` and results are retrieved via properties on the base class.

    Typical usage
    -------------

    .. code-block:: python

        eda = (
            DockEDA(df, target_col="dockq", secondary_target_col="label")
            .compute_missing_summary()
            .compute_basic_stats()
            .compute_correlations(with_target_only=True)
        )

        # Optionally, attach model-based feature importances
        eda.set_feature_importance_from_model(fitted_rf)

        eda.plot_overview_nature(top_k=6, figsize=(9.0, 7.0))
        eda.figures["overview_nature"].savefig(
            "eda_overview_nature.png", dpi=300, bbox_inches="tight"
        )
    """

    # ------------------------------------------------------------------
    # Label-aware plots
    # ------------------------------------------------------------------
    def plot_feature_target_grid(
        self,
        top_k: int = 4,
        show_stats: bool = True,
    ) -> "DockEDA":
        """
        Plot a grid of feature–target relationships for the top-k features.

        - Continuous target: scatter + LOWESS regression line, with optional
          overlay of Pearson correlation, R² and RMSE.
        - Classification target: boxplot + jitter overlay.

        :param top_k: Number of features to show.
        :type top_k: int
        :param show_stats: Whether to display correlation / R² / RMSE for
            continuous targets in a small annotation box.
        :type show_stats: bool
        :return: Self for fluent chaining.
        :rtype: DockEDA
        """
        if self.target_col is None or self.target_col not in self.df.columns:
            return self

        feats = self._top_k_features_by_target_corr(top_k)
        if not feats:
            feats = self._get_numeric_features()[:top_k]
        if not feats:
            return self

        is_class = self._is_classification_target()
        target = self.target_col

        n = len(feats)
        # Layout: up to 2 cols for <=4, then 3 cols for more
        if n <= 2:
            n_cols = n
        elif n <= 4:
            n_cols = 2
        else:
            n_cols = 3
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.2 * n_cols, 2.6 * n_rows),
        )
        axes = np.atleast_1d(axes).flatten()

        for ax, feat in zip(axes, feats):
            x = self.df[feat]
            y = self.df[target]

            if is_class:
                sns.boxplot(
                    data=self.df,
                    x=target,
                    y=feat,
                    ax=ax,
                    color=self._muted_fill,
                    linewidth=0.7,
                    fliersize=0,
                )
                sns.stripplot(
                    data=self.df,
                    x=target,
                    y=feat,
                    ax=ax,
                    color=self._primary_color,
                    size=2.5,
                    alpha=0.6,
                )
                ax.set_xlabel(target)
                ax.set_ylabel(feat)
            else:
                sns.regplot(
                    data=self.df,
                    x=feat,
                    y=target,
                    ax=ax,
                    scatter_kws={
                        "s": 10,
                        "alpha": 0.5,
                        "color": self._primary_color,
                    },
                    line_kws={"lw": 1.2, "color": self._accent_color},
                    lowess=True,
                )
                ax.set_xlabel(feat)
                ax.set_ylabel(target)

                if show_stats:
                    # Drop NaNs for metrics
                    mask = x.notna() & y.notna()
                    xv = x[mask].to_numpy()
                    yv = y[mask].to_numpy()
                    if xv.size > 1:
                        r = np.corrcoef(xv, yv)[0, 1]
                        r2 = r**2
                        # simple linear fit y = a x + b
                        a, b = np.polyfit(xv, yv, 1)
                        y_pred = a * xv + b
                        rmse = float(np.sqrt(np.mean((yv - y_pred) ** 2)))
                        txt = f"r = {r: .2f}\nR² = {r2: .2f}\nRMSE = {rmse: .2f}"
                        ax.text(
                            0.97,
                            0.03,
                            txt,
                            transform=ax.transAxes,
                            ha="right",
                            va="bottom",
                            fontsize=7,
                            bbox={
                                "boxstyle": "round,pad=0.2",
                                "facecolor": "white",
                                "alpha": 0.7,
                                "edgecolor": "none",
                            },
                        )

            ax.set_title(f"{feat} vs {target}", pad=4)

        for ax in axes[n:]:
            ax.axis("off")

        fig.tight_layout()
        self._store_figure("feature_target_grid", fig)
        return self

    def plot_label_violins(
        self,
        features: Optional[Sequence[str]] = None,
        label_col: Optional[str] = None,
    ) -> "DockEDA":
        """
        Plot violin + strip plots for a categorical label (e.g. ``label``).

        Unlike :meth:`plot_feature_target_grid`, this method does not use
        :pyattr:`target_col`; instead it focuses on a label column which
        can be supplied explicitly or inferred via :meth:`_get_label_column`.

        :param features: Subset of numeric features to plot. If ``None``,
            all numeric features are used.
        :type features: Optional[Sequence[str]]
        :param label_col: Column name to use as label (x-axis). If ``None``,
            :meth:`_get_label_column` is used to choose a suitable column.
        :type label_col: Optional[str]
        :return: Self for fluent chaining.
        :rtype: DockEDA
        """
        label_col = label_col or self._get_label_column()
        if label_col is None or label_col not in self.df.columns:
            return self

        feats = list(features) if features is not None else self._get_numeric_features()
        if not feats:
            return self

        n = len(feats)
        n_cols = 3 if n > 2 else n
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.0 * n_cols, 2.4 * n_rows),
        )
        axes = np.atleast_1d(axes).flatten()

        for ax, feat in zip(axes, feats):
            sns.violinplot(
                data=self.df,
                x=label_col,
                y=feat,
                ax=ax,
                inner="quartile",
                cut=0,
                linewidth=0.7,
                color=self._muted_fill,
            )
            sns.stripplot(
                data=self.df,
                x=label_col,
                y=feat,
                ax=ax,
                size=2.0,
                alpha=0.5,
                color=self._primary_color,
            )
            ax.set_title(feat, pad=4)
            ax.set_xlabel(label_col)
            ax.set_ylabel(None)

        for ax in axes[n:]:
            ax.axis("off")

        fig.tight_layout()
        self._store_figure("label_violins", fig)
        return self

    # ------------------------------------------------------------------
    # Nature-style 2×2 overview (with feature importance in panel C)
    # ------------------------------------------------------------------
    def plot_overview_nature(
        self,
        top_k: int = 4,
        standardize: bool = True,
        annot_corr: bool = False,
        figsize: Tuple[float, float] = (8.0, 6.0),
    ) -> "DockEDA":
        """
        Generate a compact, Nature-style 2×2 overview figure.

        Panels
        ------

        - **A**: Overlaid KDEs of top-k features (optionally z-scored).
        - **B**: Upper-triangle correlation heatmap.
        - **C**: Model-based feature importance ranking (fallback: |r|).
        - **D**: Best-correlated feature vs target with regression, coloured
          by label where available.

        :param top_k: Number of strongest target-correlated features to consider.
        :type top_k: int
        :param standardize: If ``True``, plot z-scored features in panel A so that
            all curves share a common scale.
        :type standardize: bool
        :param annot_corr: Whether to annotate correlation heatmap cells in panel B.
        :type annot_corr: bool
        :param figsize: Figure size in inches (width, height).
        :type figsize: Tuple[float, float]
        :return: Self for fluent chaining.
        :rtype: DockEDA
        """
        # Ensure correlations exist
        if self.corr_matrix is None or self.target_correlations is None:
            self.compute_correlations(with_target_only=True)

        feats = self._top_k_features_by_target_corr(top_k)
        if not feats:
            feats = self._get_numeric_features()[:top_k]
        if not feats:
            return self

        target = self.target_col
        label_col = self._get_label_column()

        fig, axes = plt.subplots(
            2,
            2,
            figsize=figsize,
            gridspec_kw={
                "height_ratios": [1.0, 1.0],
                "wspace": 0.45,
                "hspace": 0.55,
            },
        )
        axA, axB, axC, axD = axes.flatten()

        # --------------------------------------------------------------
        # Panel A – Feature distributions (z-score KDEs)
        # --------------------------------------------------------------
        plot_df = self.df[feats].dropna()
        if standardize:
            plot_df = (plot_df - plot_df.mean()) / plot_df.std(ddof=0)
            x_label = "Standardised value (z-score)"
        else:
            x_label = "Value"

        for feat in feats:
            sns.kdeplot(
                plot_df[feat],
                ax=axA,
                lw=1.4,
                fill=False,
                label=feat,
            )
        axA.set_xlabel(x_label)
        axA.set_ylabel("Density")
        axA.legend(frameon=False, fontsize=7)
        self._add_panel_label(axA, "A")
        axA.set_title("Feature distributions", pad=6, loc="left")

        # --------------------------------------------------------------
        # Panel B – Correlation structure (upper triangle)
        # --------------------------------------------------------------
        corr = self.corr_matrix
        if corr is not None and not corr.empty:
            mask = np.tril(np.ones_like(corr, dtype=bool))
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
                annot=annot_corr,
                fmt=".2f" if annot_corr else "",
                ax=axB,
            )
        self._add_panel_label(axB, "B")
        axB.set_title("Correlation structure", pad=6, loc="left")

        # --------------------------------------------------------------
        # Panel C – Feature importance ranking (fallback: |r|)
        # --------------------------------------------------------------
        if self.feature_importance is not None and not self.feature_importance.empty:
            imp = (
                self.feature_importance.copy()
                .rename("importance")
                .to_frame()
                .sort_values("importance", ascending=True)
            )
            imp = imp.tail(top_k)
            sns.barplot(
                data=imp,
                x="importance",
                y=imp.index,
                ax=axC,
                color=self._primary_color,
                orient="h",
            )
            axC.set_xlabel("Feature importance")
            axC.set_ylabel("")
            self._add_panel_label(axC, "C")
            axC.set_title("Model-based feature importance", pad=6, loc="left")

        elif self.target_correlations is not None:
            feature_list = self._get_numeric_features()
            corr_series = self.target_correlations.reindex(feature_list).dropna()
            if not corr_series.empty:
                corr_df = (
                    corr_series.rename("r")
                    .to_frame()
                    .assign(abs_r=lambda d: d["r"].abs())
                    .sort_values("abs_r", ascending=True)
                )
                corr_df = corr_df.tail(top_k)
                sns.barplot(
                    data=corr_df,
                    x="abs_r",
                    y=corr_df.index,
                    ax=axC,
                    color=self._primary_color,
                    orient="h",
                )
                axC.set_xlabel("|r(feature, target)|")
                axC.set_ylabel("")
                self._add_panel_label(axC, "C")
                axC.set_title("Feature–target correlations", pad=6, loc="left")
            else:
                axC.axis("off")
                self._add_panel_label(axC, "C")
        else:
            axC.axis("off")
            self._add_panel_label(axC, "C")

        # --------------------------------------------------------------
        # Panel D – Best feature vs target, coloured by label
        # --------------------------------------------------------------
        if target and target in self.df.columns:
            best_feat = feats[0]
            plot_df2 = self.df[[best_feat, target]].dropna()
            if label_col and label_col in self.df.columns:
                plot_df2[label_col] = self.df.loc[plot_df2.index, label_col]

            if np.issubdtype(self.df[target].dtype, np.number):
                if label_col and label_col in plot_df2.columns:
                    sns.scatterplot(
                        data=plot_df2,
                        x=best_feat,
                        y=target,
                        hue=label_col,
                        ax=axD,
                        s=12,
                        alpha=0.7,
                    )
                    sns.regplot(
                        data=plot_df2,
                        x=best_feat,
                        y=target,
                        scatter=False,
                        ax=axD,
                        line_kws={"lw": 1.4, "color": self._accent_color},
                        lowess=True,
                    )
                else:
                    sns.regplot(
                        data=plot_df2,
                        x=best_feat,
                        y=target,
                        ax=axD,
                        scatter_kws={"s": 12, "alpha": 0.6},
                        line_kws={"lw": 1.4, "color": self._accent_color},
                        lowess=True,
                    )
                axD.set_xlabel(best_feat)
                axD.set_ylabel(target)
                self._add_panel_label(axD, "D")
                axD.set_title("Best feature vs target", pad=6, loc="left")
            else:
                axD.axis("off")
                self._add_panel_label(axD, "D")
        else:
            axD.axis("off")
            self._add_panel_label(axD, "D")

        fig.tight_layout()
        self._store_figure("overview_nature", fig)
        return self

    def plot_residuals(
        self,
        model,
        features: Optional[Sequence[str]] = None,
        figsize: Tuple[float, float] = (8.0, 3.5),
    ) -> "DockEDA":
        """
        Plot residual diagnostics for a regression model on the current data.

        Two panels are shown:

        - predicted vs residuals (bias / heteroscedasticity),
        - histogram of residuals.

        :param model: Fitted scikit-learn-style model exposing ``predict``.
        :type model: Any
        :param features: Feature columns to use. If ``None``, all numeric
            features inferred by :meth:`BaseEDA._get_numeric_features` are used.
        :type features: Optional[Sequence[str]]
        :param figsize: Figure size in inches ``(width, height)``.
        :type figsize: Tuple[float, float]
        :return: Self for fluent chaining.
        :rtype: DockEDA
        """
        if self.target_col is None or self.target_col not in self.df.columns:
            return self

        feats = list(features) if features is not None else self._get_numeric_features()
        if not feats:
            return self

        X = self.df[feats].values
        y = self.df[self.target_col].values
        y_pred = model.predict(X)

        residuals = y - y_pred

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        ax1.scatter(y_pred, residuals, s=10, alpha=0.6, color=self._primary_color)
        ax1.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Residual (y - ŷ)")
        ax1.set_title("Residuals vs predicted", pad=6)

        sns.histplot(
            residuals,
            bins=40,
            kde=True,
            stat="density",
            color=self._primary_color,
            alpha=0.7,
            edgecolor="white",
            linewidth=0.3,
            ax=ax2,
        )
        ax2.set_xlabel("Residual")
        ax2.set_ylabel("Density")
        ax2.set_title("Residual distribution", pad=6)

        fig.tight_layout()
        self._store_figure("residuals", fig)
        return self

    def plot_enrichment(
        self,
        y_true_col: str,
        score_col: str,
        top_fracs: Sequence[float] = (0.01, 0.05, 0.10),
        figsize: Tuple[float, float] = (5.0, 4.0),
    ) -> "DockEDA":
        """
        Plot an enrichment / accumulation curve for ranking performance.

        The curve shows the fraction of positives recovered as a function
        of the screened fraction of the library (sorted by score). Enrichment
        factors at selected top fractions are annotated.

        :param y_true_col: Column with binary ground-truth labels (0/1).
        :type y_true_col: str
        :param score_col: Column with ranking scores (higher = better).
        :type score_col: str
        :param top_fracs: Fractions of the dataset (e.g. 0.01, 0.05, 0.10)
            at which enrichment factors are computed.
        :type top_fracs: Sequence[float]
        :param figsize: Figure size in inches ``(width, height)``.
        :type figsize: Tuple[float, float]
        :return: Self for fluent chaining.
        :rtype: DockEDA
        """
        if y_true_col not in self.df.columns or score_col not in self.df.columns:
            return self

        df = self.df[[y_true_col, score_col]].dropna()
        if df.empty:
            return self

        # ensure binary labels 0/1
        y_true = df[y_true_col].astype(int).values
        scores = df[score_col].values

        # sort by descending score
        order = np.argsort(-scores)
        y_true_sorted = y_true[order]

        n = len(y_true_sorted)
        cum_pos = np.cumsum(y_true_sorted)
        total_pos = y_true_sorted.sum()
        if total_pos == 0:
            return self

        frac_screened = np.arange(1, n + 1) / float(n)
        frac_recalled = cum_pos / float(total_pos)

        # compute enrichment factors at requested fractions
        ef_lines = []
        for f in top_fracs:
            if f <= 0 or f > 1:
                continue
            idx = max(1, int(np.round(f * n)))
            recall_at_f = frac_recalled[idx - 1]
            ef = recall_at_f / f
            ef_lines.append((f, ef))

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(
            frac_screened,
            frac_recalled,
            label="Model",
            color=self._primary_color,
            lw=1.5,
        )
        ax.plot(
            frac_screened,
            frac_screened,
            label="Random",
            color=self._muted_gray,
            lw=1.0,
            linestyle="--",
        )

        for f, ef in ef_lines:
            ax.axvline(f, color="#BBBBBB", lw=0.8, linestyle=":")
            ax.text(
                f,
                0.05,
                f"EF@{int(100*f)}% = {ef:.1f}",
                rotation=90,
                va="bottom",
                ha="right",
                fontsize=7,
            )

        ax.set_xlabel("Fraction of library screened")
        ax.set_ylabel("Fraction of positives recovered")
        ax.set_title("Enrichment curve", pad=6)
        ax.legend(frameon=False)

        fig.tight_layout()
        self._store_figure("enrichment", fig)
        return self

    def plot_roc_pr(
        self,
        y_true_col: str,
        score_col: str,
        figsize: Tuple[float, float] = (10.0, 4.0),
    ) -> "DockEDA":
        """
        Plot ROC and Precision–Recall curves for a binary classifier.

        :param y_true_col: Column with binary ground-truth labels (0/1).
        :type y_true_col: str
        :param score_col: Column with predicted scores or probabilities
            (higher = more likely positive).
        :type score_col: str
        :param figsize: Figure size in inches ``(width, height)``.
        :type figsize: Tuple[float, float]
        :return: Self for fluent chaining.
        :rtype: DockEDA
        """
        if y_true_col not in self.df.columns or score_col not in self.df.columns:
            return self

        df = self.df[[y_true_col, score_col]].dropna()
        if df.empty:
            return self

        y_true = df[y_true_col].astype(int).values
        scores = df[score_col].values

        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # ROC
        ax1.plot(
            fpr, tpr, color=self._primary_color, lw=1.5, label=f"AUC = {roc_auc:.3f}"
        )
        ax1.plot([0, 1], [0, 1], color=self._muted_gray, lw=1.0, linestyle="--")
        ax1.set_xlabel("False positive rate")
        ax1.set_ylabel("True positive rate")
        ax1.set_title("ROC curve", pad=6)
        ax1.legend(frameon=False)

        # PR
        ax2.plot(
            recall, precision, color=self._primary_color, lw=1.5, label=f"AP = {ap:.3f}"
        )
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision–Recall curve", pad=6)
        ax2.legend(frameon=False)

        fig.tight_layout()
        self._store_figure("roc_pr", fig)
        return self

    def plot_threshold_metrics(
        self,
        y_true_col: str,
        score_col: str,
        metrics: Sequence[str] = ("precision", "recall", "f1"),
        n_points: int = 50,
        figsize: Tuple[float, float] = (6.0, 4.0),
    ) -> "DockEDA":
        """
        Plot classifier metrics as a function of decision threshold.

        :param y_true_col: Column with binary ground-truth labels (0/1).
        :type y_true_col: str
        :param score_col: Column with predicted scores or probabilities
            (higher = more likely positive).
        :type score_col: str
        :param metrics: Sequence of metric names to plot. Supported:
            ``"precision"``, ``"recall"``, ``"f1"``.
        :type metrics: Sequence[str]
        :param n_points: Number of thresholds to evaluate between min and max
            observed scores.
        :type n_points: int
        :param figsize: Figure size in inches ``(width, height)``.
        :type figsize: Tuple[float, float]
        :return: Self for fluent chaining.
        :rtype: DockEDA
        """
        if y_true_col not in self.df.columns or score_col not in self.df.columns:
            return self

        df = self.df[[y_true_col, score_col]].dropna()
        if df.empty:
            return self

        y_true = df[y_true_col].astype(int).values
        scores = df[score_col].values

        thr_vals = np.linspace(scores.min(), scores.max(), n_points)
        curves = {m: [] for m in metrics}

        for thr in thr_vals:
            y_pred = (scores >= thr).astype(int)
            if "precision" in metrics:
                curves["precision"].append(
                    precision_score(y_true, y_pred, zero_division=0)
                )
            if "recall" in metrics:
                curves["recall"].append(recall_score(y_true, y_pred, zero_division=0))
            if "f1" in metrics:
                curves["f1"].append(f1_score(y_true, y_pred, zero_division=0))

        fig, ax = plt.subplots(figsize=figsize)
        for m in metrics:
            ax.plot(thr_vals, curves[m], label=m)

        ax.set_xlabel("Decision threshold")
        ax.set_ylabel("Metric value")
        ax.set_ylim(0.0, 1.05)
        ax.set_title("Threshold-dependent metrics", pad=6)
        ax.legend(frameon=False)

        fig.tight_layout()
        self._store_figure("threshold_metrics", fig)
        return self
