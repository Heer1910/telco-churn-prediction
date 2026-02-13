# Telco Customer Churn – Predictive Model

A minimum-viable, interpretable churn prediction pipeline using logistic regression on the [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) dataset.

## Results

| Metric | Value |
|---|---|
| ROC-AUC | 0.8418 |
| PR-AUC | 0.6324 |
| Precision @ Top 10% | 73.8% |

When the retention team contacts the top 10% highest-risk customers, ~74% of those contacts are genuine churners — far better than the overall 27% base rate.

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset (already included)
python download_data.py

# 4. Run the pipeline
python run_pipeline.py

# 5. Run tests
pytest tests/ -v
```

## Project Structure

```
churn-decision-support/
├── data/raw/telco_churn.csv    # 7,043 customers × 21 columns
├── src/
│   ├── config.py               # Paths, feature lists, model settings
│   ├── data_prep.py            # Load, clean, encode, split
│   ├── model.py                # Logistic regression train / predict
│   ├── evaluation.py           # ROC-AUC, PR-AUC, Precision@K
│   └── interpret.py            # Coefficient analysis, report gen
├── tests/                      # Unit tests (18 passing)
├── outputs/
│   ├── metrics.csv             # Evaluation metrics
│   └── predictions.csv         # customer_id + churn_probability
├── reports/
│   └── report.md               # Business-readable 1-page report
├── run_pipeline.py             # Entry point
├── download_data.py            # Dataset downloader
├── requirements.txt
└── Makefile
```

## Approach

1. **Features**: ~30 features after one-hot encoding (tenure, charges, contract type, services, demographics)
2. **Model**: Logistic Regression with StandardScaler, no hyperparameter tuning
3. **Split**: 80/20 stratified holdout, seed = 42
4. **Evaluation**: ROC-AUC, PR-AUC, Precision @ top 10% risk segment
5. **Interpretation**: Coefficient-based feature importance with business recommendations

## Key Findings

- **Month-to-month contracts** are the strongest churn driver
- **Lack of online security / tech support** elevates risk
- **Longer tenure** is protective
- **Higher monthly charges** correlate with churn

## Business Implications

### Why This Matters

With a base churn rate of ~27%, losing customers is the single largest drag on recurring revenue. This model identifies **who is most likely to leave before they do**, enabling the retention team to act proactively rather than reactively.

### How to Use the Results

| Action | Detail |
|---|---|
| **Prioritise outreach** | Sort `predictions.csv` by `churn_probability` descending. The top 10% segment captures ~74% true churners — focus retention spend here first. |
| **Set a risk threshold** | Customers above the 0.66 probability cutoff are the highest-risk segment. Flag them in CRM for immediate follow-up. |
| **Measure ROI** | If the average customer lifetime value is \$3,000 and a retention offer costs \$50, saving even 20% of the top-risk segment pays for itself many times over. |

### Recommended Retention Strategies (Tied to Model Drivers)

1. **Contract migration incentives** — Month-to-month customers churn at the highest rate. Offer a discount or bonus (e.g., free month, device credit) for switching to a 1-year or 2-year plan.

2. **Service bundling for fiber optic users** — Fiber optic internet customers show elevated risk, likely due to higher monthly bills without matching perceived value. Bundle online security or tech support at no extra cost to increase stickiness.

3. **Early-tenure engagement program** — Tenure is the strongest protective factor. Customers in their first 6–12 months are most vulnerable. Implement onboarding check-ins, usage tips, and satisfaction surveys during this window.

4. **Billing review for high-spend customers** — High monthly charges correlate with churn. Proactively review bills for customers in the top charge bracket and offer plan optimisation or loyalty pricing before they comparison-shop.

### Deploying in Practice

This model is designed as a **batch scoring** tool — run it monthly or weekly to refresh risk scores, then feed `predictions.csv` into the CRM or marketing automation platform to trigger retention workflows automatically.

See [reports/report.md](reports/report.md) for the full analysis.
