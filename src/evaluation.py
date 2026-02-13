"""
Evaluation metrics and output writers for churn predictions.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from src.config import PRECISION_AT_K


def compute_metrics(y_true, y_prob):
    """
    Compute evaluation metrics.

    Returns
    -------
    dict with keys: roc_auc, pr_auc, precision_at_10pct,
                    n_true_positives_top10, n_false_positives_top10,
                    top10_cutoff
    """
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    # Precision @ top 10 % risk
    n = len(y_true)
    k = int(np.ceil(n * PRECISION_AT_K))
    top_idx = np.argsort(y_prob)[::-1][:k]
    top_actual = np.array(y_true)[top_idx]

    tp = int(top_actual.sum())
    fp = int(k - tp)
    precision_at_k = tp / k

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision_at_10pct": round(precision_at_k, 4),
        "n_true_positives_top10": tp,
        "n_false_positives_top10": fp,
        "top10_cutoff": round(float(y_prob[top_idx[-1]]), 4),
        "top10_count": k,
        "total_test": n,
    }


def save_metrics(metrics, path):
    """Write metrics dict to a one-row CSV."""
    pd.DataFrame([metrics]).to_csv(path, index=False)


def save_predictions(ids, y_prob, path):
    """Write customer_id + churn_probability to CSV."""
    pd.DataFrame({
        "customer_id": ids.values,
        "churn_probability": np.round(y_prob, 6),
    }).sort_values("churn_probability", ascending=False).to_csv(
        path, index=False
    )
