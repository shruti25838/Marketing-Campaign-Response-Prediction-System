# Marketing Campaign Response Prediction System

End-to-end ML pipeline that predicts customer campaign response (yes/no), explains drivers with SHAP, and exports dashboard-ready outputs for targeting and ROI. Built with Python, scikit-learn, LightGBM/XGBoost, and SMOTE for class imbalance.

## Run (3 steps)

From the project root:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m src.pipeline --csv data/raw/campaign_data.csv --target response
```

You should see metrics (ROC-AUC, precision, recall, F1) and a confusion matrix in the terminal. Artifacts are saved under `data/processed/run_YYYYMMDD_HHMMSS/` (model, metrics, predictions, plots including SHAP).

## Results

On the default campaign dataset with LightGBM:

| Metric    | Holdout (single split) | 5-fold CV (mean +/- std) |
|-----------|------------------------|--------------------------|
| ROC-AUC   | ~0.93                  | ~0.925 +/- 0.002         |
| Precision | ~0.83                  | ~0.83 +/- 0.004         |
| Recall    | ~0.88                  | ~0.88 +/- 0.006         |
| F1        | ~0.86                  | ~0.85 +/- 0.005         |

LightGBM improves ROC-AUC by about 2% over the logistic regression baseline. The model ranks likely responders well and keeps a strong precision-recall balance, so marketing can target top-scoring customers and reduce wasted contacts. Use `--cv` to print full 5-fold cross-validation and the baseline comparison.

## Options

- `--model logistic|xgboost|lightgbm` (default: lightgbm)
- `--cv` — run 5-fold CV and baseline comparison before training
- `--no-shap` — skip SHAP plots (faster)
- `--db-url` and `--table` — load from PostgreSQL instead of CSV

See `python -m src.pipeline --help` for all flags.

## Outputs

Each run writes to `data/processed/run_YYYYMMDD_HHMMSS/`:

| Artifact            | Description                                      |
|---------------------|--------------------------------------------------|
| model.joblib        | Fitted pipeline (preprocessor + SMOTE + model)  |
| metrics.json        | ROC-AUC, precision, recall, F1                   |
| predictions.csv     | row_index, y_true, y_pred, y_prob                |
| confusion_matrix.csv| 2x2 confusion matrix                             |
| plots/              | ROC curve, PR curve, confusion matrix, SHAP      |

Optional dashboard exports: run `python -m dashboards.export_dashboard_data`, `segment_dashboard`, and `feature_importance_dashboard`. Outputs go to `dashboards/exports/` for use in Power BI or Tableau. See [dashboards/README.md](dashboards/README.md) and [dashboards/POWER_BI_TEMPLATE.md](dashboards/POWER_BI_TEMPLATE.md).

## Structure

- `src/` — pipeline, models (logistic, XGBoost, LightGBM), services (data load, preprocessing, feature engineering, trainer, evaluator), views (plots, SHAP)
- `notebooks/` — EDA, feature engineering, training, evaluation
- `dashboards/` — scripts to export run artifacts for BI
- `data/raw/` — input CSV; `data/processed/` — run folders
- `sql/schema.sql` — example table schema
- `docs/API.md` — module and API reference
