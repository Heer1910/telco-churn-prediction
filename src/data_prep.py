"""
Data loading, cleaning, and feature encoding for Telco churn dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_CSV,
    TARGET_COL,
    ID_COL,
    BINARY_COLS,
    NUMERIC_FLAG_COLS,
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    RANDOM_SEED,
    TEST_SIZE,
)


def load_and_clean(csv_path=None):
    """
    Load the Telco CSV and return cleaned feature matrix, target, and IDs.

    Steps:
        1. Read CSV
        2. Fix TotalCharges blanks → 0
        3. Map binary Yes/No columns → 1/0
        4. Map target Churn → 1/0
        5. One-hot encode categoricals
        6. Combine all features

    Returns
    -------
    X : pd.DataFrame   – feature matrix (all numeric)
    y : pd.Series       – binary target (1 = churned)
    ids : pd.Series     – customerID column for output
    """
    path = csv_path or RAW_CSV
    df = pd.read_csv(path)

    # ── Fix TotalCharges: blank strings → NaN → 0 ──────────────────
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # ── Target encoding ─────────────────────────────────────────────
    y = df[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)
    ids = df[ID_COL]

    # ── Binary columns: Yes/No → 1/0 ───────────────────────────────
    binary_df = df[BINARY_COLS].replace({"Yes": 1, "No": 0}).astype(int)

    # ── Numeric flag columns (already 0/1) ──────────────────────────
    flag_df = df[NUMERIC_FLAG_COLS].astype(int)

    # ── Continuous numeric columns ──────────────────────────────────
    numeric_df = df[NUMERIC_COLS].astype(float)

    # ── One-hot encode categoricals ─────────────────────────────────
    cat_df = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=True)
    # Ensure column names are strings and bool columns become int
    cat_df.columns = cat_df.columns.astype(str)
    cat_df = cat_df.astype(int)

    # ── Combine ─────────────────────────────────────────────────────
    X = pd.concat([numeric_df, flag_df, binary_df, cat_df], axis=1)

    return X, y, ids


def split_data(X, y, ids):
    """
    Stratified 80/20 train/test split.

    Returns
    -------
    dict with keys: X_train, X_test, y_train, y_test, ids_test
    """
    X_train, X_test, y_train, y_test, _, ids_test = train_test_split(
        X, y, ids,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "ids_test": ids_test,
    }
