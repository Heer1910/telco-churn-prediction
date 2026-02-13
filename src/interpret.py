"""
Model interpretation and report generation.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def get_top_drivers(model, feature_names, top_n=10):
    """
    Extract logistic regression coefficients sorted by absolute magnitude.

    Returns
    -------
    pd.DataFrame with columns: feature, coefficient, abs_coefficient, direction
    """
    coefs = model.coef_[0]
    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": np.round(coefs, 4),
        "abs_coefficient": np.round(np.abs(coefs), 4),
    })
    df["direction"] = df["coefficient"].apply(
        lambda c: "increases churn" if c > 0 else "decreases churn"
    )
    df = df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    return df.head(top_n)


def generate_report(metrics, drivers, output_path):
    """
    Write a plain-language markdown report summarising model results.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    top3 = drivers.head(3)
    driver_bullets = "\n".join(
        f"  - **{row.feature}** (coef = {row.coefficient:+.3f}): "
        f"{row.direction}"
        for _, row in top3.iterrows()
    )

    all_driver_rows = "\n".join(
        f"| {i+1} | {row.feature} | {row.coefficient:+.4f} | {row.direction} |"
        for i, (_, row) in enumerate(drivers.iterrows())
    )

    report = f"""# Telco Customer Churn – Model Report

_Generated: {now}_

---

## 1. Problem

Predict which customers are likely to cancel their telecom service so the
business can target retention offers before they leave.

**Model**: Logistic Regression (sklearn)
**Train/Test split**: 80 / 20, stratified by churn, seed = 42

---

## 2. Evaluation Metrics

| Metric | Value |
|---|---|
| ROC-AUC | {metrics['roc_auc']:.4f} |
| PR-AUC | {metrics['pr_auc']:.4f} |
| Precision @ Top 10 % | {metrics['precision_at_10pct']:.4f} |

---

## 3. Top-10 % Segment Breakdown

When we rank all {metrics['total_test']:,} test customers by predicted risk and
look at the **top {metrics['top10_count']}** (≈ 10 %):

- **{metrics['n_true_positives_top10']}** are actual churners (true positives)
- **{metrics['n_false_positives_top10']}** are non-churners (false positives)
- Probability cutoff for this segment: **{metrics['top10_cutoff']:.2f}**

**Plain-language interpretation**: If the retention team contacts the top 10 %
highest-risk customers, roughly **{metrics['precision_at_10pct']*100:.0f} %** of
those contacts will be genuine churners. This is substantially better than
treating all customers equally (base churn rate ≈ 27 %).

---

## 4. Top Churn Drivers

| Rank | Feature | Coefficient | Effect |
|---|---|---|---|
{all_driver_rows}

### Key takeaways

{driver_bullets}

---

## 5. Business Recommendations

Based on the coefficient analysis:

1. **Month-to-month contracts** are the strongest risk factor. Offer incentives
   for customers to switch to annual or two-year contracts (e.g., a discount
   or loyalty perk).

2. **Customers without online security or tech support** show elevated churn.
   Bundle these services into plans or offer a free trial.

3. **Tenure** is protective – longer-tenured customers are less likely to leave.
   Focus retention campaigns on newer customers (tenure < 12 months) who also
   have month-to-month contracts.

4. **High monthly charges** correlate with churn. Review pricing tiers and
   consider value-adds rather than price cuts.

---

## 6. Reproducibility

To reproduce these results:

```bash
source venv/bin/activate
python run_pipeline.py
```

Outputs:
- `outputs/metrics.csv`
- `outputs/predictions.csv`
- `reports/report.md` (this file)
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
