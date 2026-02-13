"""
Central configuration for Telco Customer Churn pipeline.

All paths, feature lists, and model settings defined here.
"""

from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "telco_churn.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ── Random seed ────────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.20

# ── Target ─────────────────────────────────────────────────────────────
TARGET_COL = "Churn"
ID_COL = "customerID"

# ── Feature column groups ──────────────────────────────────────────────
# Binary yes/no columns → mapped to 1/0
BINARY_COLS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

# Already numeric (0/1) – no transformation needed
NUMERIC_FLAG_COLS = [
    "SeniorCitizen",
]

# Continuous numeric columns
NUMERIC_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# Categorical columns → one-hot encoded
CATEGORICAL_COLS = [
    "gender",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

# ── Model ──────────────────────────────────────────────────────────────
MAX_ITER = 1000
C = 1.0
SOLVER = "lbfgs"

# ── Evaluation ─────────────────────────────────────────────────────────
PRECISION_AT_K = 0.10  # top 10 %
