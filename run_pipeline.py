"""
Main entry point for telco churn prediction pipeline.

Run: python run_pipeline.py
"""

import sys
import logging

from src.config import OUTPUT_DIR, REPORTS_DIR
from src.data_prep import load_and_clean, split_data
from src.model import train, predict
from src.evaluation import compute_metrics, save_metrics, save_predictions
from src.interpret import get_top_drivers, generate_report
from src.visualize import generate_all_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    # ── 1. Load & clean ────────────────────────────────────────────
    log.info("Loading and cleaning data …")
    X, y, ids = load_and_clean()
    log.info(f"  Rows: {len(X):,}  |  Features: {X.shape[1]}  |  Churn rate: {y.mean():.1%}")

    # ── 2. Split ───────────────────────────────────────────────────
    log.info("Splitting 80/20 (stratified) …")
    split = split_data(X, y, ids)
    log.info(f"  Train: {len(split['X_train']):,}  |  Test: {len(split['X_test']):,}")

    # ── 3. Train ───────────────────────────────────────────────────
    log.info("Training logistic regression …")
    model, scaler = train(split["X_train"], split["y_train"])

    # ── 4. Predict ─────────────────────────────────────────────────
    log.info("Generating predictions on test set …")
    y_prob = predict(model, scaler, split["X_test"])

    # ── 5. Evaluate ────────────────────────────────────────────────
    log.info("Computing metrics …")
    metrics = compute_metrics(split["y_test"], y_prob)
    for k, v in metrics.items():
        log.info(f"  {k}: {v}")

    # ── 6. Save outputs ───────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    save_metrics(metrics, OUTPUT_DIR / "metrics.csv")
    save_predictions(split["ids_test"], y_prob, OUTPUT_DIR / "predictions.csv")
    log.info(f"  Saved metrics.csv and predictions.csv → {OUTPUT_DIR}")

    # ── 7. Interpret & report ──────────────────────────────────────
    log.info("Generating report …")
    drivers = get_top_drivers(model, list(X.columns), top_n=10)
    generate_report(metrics, drivers, REPORTS_DIR / "report.md")
    log.info(f"  Saved report.md → {REPORTS_DIR}")

    # ── 8. Visualizations ──────────────────────────────────────────
    log.info("Generating figures …")
    fig_paths = generate_all_figures(split["y_test"], y_prob, drivers)
    for p in fig_paths:
        log.info(f"  → {p.name}")

    print("\n✅ Pipeline completed successfully!")
    print(f"\nOutputs:")
    print(f"  • outputs/metrics.csv")
    print(f"  • outputs/predictions.csv")
    print(f"  • outputs/figures/  (5 charts)")
    print(f"  • reports/report.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())
