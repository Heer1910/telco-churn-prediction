"""Tests for model module."""

import numpy as np
import pytest
from src.data_prep import load_and_clean, split_data
from src.model import train, predict


@pytest.fixture(scope="module")
def fitted():
    """Train model once for all tests."""
    X, y, ids = load_and_clean()
    split = split_data(X, y, ids)
    model, scaler = train(split["X_train"], split["y_train"])
    return model, scaler, split


class TestTrain:
    def test_model_is_fitted(self, fitted):
        model, _, _ = fitted
        assert hasattr(model, "coef_")
        assert hasattr(model, "classes_")

    def test_two_classes(self, fitted):
        model, _, _ = fitted
        assert len(model.classes_) == 2


class TestPredict:
    def test_probabilities_in_range(self, fitted):
        model, scaler, split = fitted
        y_prob = predict(model, scaler, split["X_test"])
        assert np.all(y_prob >= 0)
        assert np.all(y_prob <= 1)

    def test_output_length(self, fitted):
        model, scaler, split = fitted
        y_prob = predict(model, scaler, split["X_test"])
        assert len(y_prob) == len(split["X_test"])

    def test_not_all_same(self, fitted):
        model, scaler, split = fitted
        y_prob = predict(model, scaler, split["X_test"])
        assert y_prob.std() > 0.01  # predictions should vary
