"""Tests for evaluation module."""

import numpy as np
import pytest
from src.evaluation import compute_metrics


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
        m = compute_metrics(y_true, y_prob)
        assert m["roc_auc"] == 1.0
        assert m["pr_auc"] == 1.0

    def test_random_predictions(self):
        rng = np.random.RandomState(0)
        y_true = rng.randint(0, 2, size=1000)
        y_prob = rng.rand(1000)
        m = compute_metrics(y_true, y_prob)
        # Random should be near 0.5 AUC
        assert 0.4 <= m["roc_auc"] <= 0.6

    def test_all_keys_present(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])
        m = compute_metrics(y_true, y_prob)
        expected_keys = {
            "roc_auc", "pr_auc", "precision_at_10pct",
            "n_true_positives_top10", "n_false_positives_top10",
            "top10_cutoff", "top10_count", "total_test",
        }
        assert set(m.keys()) == expected_keys

    def test_precision_at_k_in_range(self):
        rng = np.random.RandomState(1)
        y_true = rng.randint(0, 2, size=500)
        y_prob = rng.rand(500)
        m = compute_metrics(y_true, y_prob)
        assert 0.0 <= m["precision_at_10pct"] <= 1.0
