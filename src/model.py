"""
Logistic regression model for churn prediction.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED, MAX_ITER, C, SOLVER


def train(X_train, y_train):
    """
    Fit a logistic regression model with standard-scaled features.

    Returns
    -------
    model   : fitted LogisticRegression
    scaler  : fitted StandardScaler (needed at prediction time)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        random_state=RANDOM_SEED,
        max_iter=MAX_ITER,
        C=C,
        solver=SOLVER,
    )
    model.fit(X_scaled, y_train)

    return model, scaler


def predict(model, scaler, X):
    """Return P(Churn = 1) for each row."""
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1]
