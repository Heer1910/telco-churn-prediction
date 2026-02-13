# Telco Customer Churn – Model Report

_Generated: 2026-02-12 13:30_

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
| ROC-AUC | 0.8418 |
| PR-AUC | 0.6324 |
| Precision @ Top 10 % | 0.7376 |

---

## 3. Top-10 % Segment Breakdown

When we rank all 1,409 test customers by predicted risk and
look at the **top 141** (≈ 10 %):

- **104** are actual churners (true positives)
- **37** are non-churners (false positives)
- Probability cutoff for this segment: **0.66**

**Plain-language interpretation**: If the retention team contacts the top 10 %
highest-risk customers, roughly **74 %** of
those contacts will be genuine churners. This is substantially better than
treating all customers equally (base churn rate ≈ 27 %).

---

## 4. Top Churn Drivers

| Rank | Feature | Coefficient | Effect |
|---|---|---|---|
| 1 | tenure | -1.2365 | decreases churn |
| 2 | MonthlyCharges | -0.9202 | decreases churn |
| 3 | InternetService_Fiber optic | +0.7762 | increases churn |
| 4 | Contract_Two year | -0.5869 | decreases churn |
| 5 | TotalCharges | +0.5143 | increases churn |
| 6 | Contract_One year | -0.2855 | decreases churn |
| 7 | StreamingMovies_Yes | +0.2572 | increases churn |
| 8 | StreamingTV_Yes | +0.2571 | increases churn |
| 9 | MultipleLines_Yes | +0.2162 | increases churn |
| 10 | PaperlessBilling | +0.1820 | increases churn |

### Key takeaways

  - **tenure** (coef = -1.236): decreases churn
  - **MonthlyCharges** (coef = -0.920): decreases churn
  - **InternetService_Fiber optic** (coef = +0.776): increases churn

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
