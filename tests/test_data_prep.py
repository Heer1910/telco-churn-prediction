"""Tests for data_prep module."""

import pandas as pd
import numpy as np
import pytest
from src.data_prep import load_and_clean, split_data
from src.config import RAW_CSV


@pytest.fixture(scope="module")
def loaded_data():
    """Load once for all tests in this module."""
    return load_and_clean()


class TestLoadAndClean:
    def test_returns_three_items(self, loaded_data):
        X, y, ids = loaded_data
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(ids, pd.Series)

    def test_no_missing_values(self, loaded_data):
        X, y, _ = loaded_data
        assert X.isnull().sum().sum() == 0
        assert y.isnull().sum() == 0

    def test_target_is_binary(self, loaded_data):
        _, y, _ = loaded_data
        assert set(y.unique()) == {0, 1}

    def test_all_features_numeric(self, loaded_data):
        X, _, _ = loaded_data
        assert all(np.issubdtype(dt, np.number) for dt in X.dtypes)

    def test_expected_row_count(self, loaded_data):
        X, y, ids = loaded_data
        assert len(X) == len(y) == len(ids)
        assert len(X) == 7043  # known dataset size

    def test_feature_count_in_range(self, loaded_data):
        X, _, _ = loaded_data
        # After one-hot encoding we expect roughly 30+ columns
        assert 12 <= X.shape[1] <= 50


class TestSplitData:
    def test_split_preserves_total(self, loaded_data):
        X, y, ids = loaded_data
        split = split_data(X, y, ids)
        assert len(split["X_train"]) + len(split["X_test"]) == len(X)

    def test_stratification(self, loaded_data):
        X, y, ids = loaded_data
        split = split_data(X, y, ids)
        train_rate = split["y_train"].mean()
        test_rate = split["y_test"].mean()
        # Rates should be close (within 2 pct points)
        assert abs(train_rate - test_rate) < 0.02

    def test_reproducibility(self, loaded_data):
        X, y, ids = loaded_data
        s1 = split_data(X, y, ids)
        s2 = split_data(X, y, ids)
        pd.testing.assert_frame_equal(s1["X_train"], s2["X_train"])
