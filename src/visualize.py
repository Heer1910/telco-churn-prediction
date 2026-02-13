"""
Visualization module for churn model evaluation.

Generates publication-quality charts saved to outputs/figures/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from pathlib import Path

from src.config import OUTPUT_DIR


FIGURES_DIR = OUTPUT_DIR / "figures"

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FAFAFA",
    "axes.edgecolor": "#CCCCCC",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#CCCCCC",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})

CHURN_COLOR = "#E74C3C"
STAY_COLOR = "#2ECC71"
ACCENT = "#3498DB"


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def plot_roc_curve(y_true, y_prob):
    """ROC curve with AUC shaded."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.fill_between(fpr, tpr, alpha=0.15, color=ACCENT)
    ax.plot(fpr, tpr, color=ACCENT, lw=2.5, label=f"Logistic Regression  (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    return _save(fig, "roc_curve")


def plot_pr_curve(y_true, y_prob):
    """Precision-Recall curve with AUC shaded."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.fill_between(recall, precision, alpha=0.15, color=CHURN_COLOR)
    ax.plot(recall, precision, color=CHURN_COLOR, lw=2.5, label=f"Logistic Regression  (PR-AUC = {pr_auc:.3f})")
    ax.axhline(baseline, ls="--", color="#999999", lw=1, label=f"Baseline (churn rate = {baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.05)
    return _save(fig, "pr_curve")


def plot_feature_importance(drivers_df):
    """Horizontal bar chart of top coefficient magnitudes."""
    df = drivers_df.sort_values("abs_coefficient", ascending=True).copy()
    colors = [CHURN_COLOR if c > 0 else STAY_COLOR for c in df["coefficient"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df["feature"], df["coefficient"], color=colors, edgecolor="white", height=0.65)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlabel("Coefficient (positive = increases churn)")
    ax.set_title("Top 10 Churn Drivers")

    # Add value labels
    for i, (val, feat) in enumerate(zip(df["coefficient"], df["feature"])):
        offset = 0.02 if val >= 0 else -0.02
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, i, f"{val:+.3f}", va="center", ha=ha, fontsize=9, color="#333333")

    ax.set_xlim(df["coefficient"].min() - 0.15, df["coefficient"].max() + 0.15)
    return _save(fig, "feature_importance")


def plot_probability_distribution(y_true, y_prob):
    """Overlapping histograms of predicted probabilities by actual class."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(y_prob[y_true == 0], bins=40, alpha=0.6, color=STAY_COLOR,
            label="Did not churn", edgecolor="white", density=True)
    ax.hist(y_prob[y_true == 1], bins=40, alpha=0.6, color=CHURN_COLOR,
            label="Churned", edgecolor="white", density=True)

    ax.set_xlabel("Predicted Churn Probability")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution by Actual Outcome")
    ax.legend(framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    return _save(fig, "score_distribution")


def plot_precision_by_decile(y_true, y_prob):
    """Bar chart showing precision in each risk decile."""
    df = pd.DataFrame({"actual": y_true, "prob": y_prob})
    df["decile"] = pd.qcut(df["prob"], 10, labels=False, duplicates="drop")
    summary = df.groupby("decile").agg(
        n=("actual", "size"),
        churners=("actual", "sum"),
        avg_prob=("prob", "mean"),
    )
    summary["precision"] = summary["churners"] / summary["n"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(summary.index, summary["precision"], color=ACCENT, edgecolor="white", width=0.7)

    # Highlight top decile
    bars[-1].set_color(CHURN_COLOR)

    ax.set_xlabel("Risk Decile (0 = lowest, 9 = highest)")
    ax.set_ylabel("Actual Churn Rate")
    ax.set_title("Churn Rate by Risk Decile")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Label bars
    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{int(row.churners)}/{int(row.n)}", ha="center", fontsize=8, color="#555555")

    return _save(fig, "precision_by_decile")


def generate_all_figures(y_true, y_prob, drivers_df):
    """Generate all figures. Returns list of saved paths."""
    y_true_arr = np.array(y_true)
    paths = [
        plot_roc_curve(y_true_arr, y_prob),
        plot_pr_curve(y_true_arr, y_prob),
        plot_feature_importance(drivers_df),
        plot_probability_distribution(y_true_arr, y_prob),
        plot_precision_by_decile(y_true_arr, y_prob),
    ]
    return paths
