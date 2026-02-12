# BI Dashboard Guide

Export pipeline runs into **dashboard-ready CSVs** and use them in Power BI or Tableau.

## 1. Export data (run from project root)

```bash
# Run pipeline first
python -m src.pipeline --csv data/raw/campaign_data.csv --target response

# Export all dashboard inputs to dashboards/exports/
python -m dashboards.export_dashboard_data
python -m dashboards.segment_dashboard
python -m dashboards.feature_importance_dashboard
```

## 2. Outputs in `dashboards/exports/`

| File | Description |
|------|-------------|
| `dashboard_predictions.csv` | Predictions with `prob_bucket`, `is_high_prob` |
| `dashboard_metrics.csv` | ROC-AUC, precision, recall, F1 |
| `dashboard_confusion_matrix.csv` | Confusion matrix |
| `segment_response_rates.csv` | Actual response rate by job, marital, education, contact, month |
| `segment_high_prob_rates.csv` | Predicted high-probability rate by segment (test set) |
| `feature_importance.csv` | Feature importance from model |
| `plots/` | PNGs: segment_response_*.png, feature_importance.png |

## 3. Power BI / Tableau

- **Power BI**: See [POWER_BI_TEMPLATE.md](POWER_BI_TEMPLATE.md) for step-by-step report pages and visuals.
- **Tableau**: Connect to the same CSVs via **Text file** and build Campaign Overview, Segmentation, and Feature Importance sheets.

## 4. Optional: use raw run folder

```bash
python -m dashboards.export_dashboard_data data/processed/run_20250101_120000
python -m dashboards.segment_dashboard data/processed/run_20250101_120000
python -m dashboards.feature_importance_dashboard data/processed/run_20250101_120000
```
