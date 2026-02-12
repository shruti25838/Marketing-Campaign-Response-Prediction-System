# Power BI Dashboard Template – Campaign Response Prediction

Use this guide to build a **Campaign Response Prediction** dashboard in Power BI Desktop using the exports from this project.

## 1. Get the data

1. Run the pipeline and export dashboard data from the project root:
   ```bash
   python -m src.pipeline --csv data/raw/campaign_data.csv --target response
   python -m dashboards.export_dashboard_data
   python -m dashboards.segment_dashboard
   python -m dashboards.feature_importance_dashboard
   ```
2. Open Power BI Desktop → **Get data** → **Text/CSV**.
3. Load these files from `dashboards/exports/`:
   - `dashboard_predictions.csv`
   - `dashboard_metrics.csv`
   - `dashboard_confusion_matrix.csv`
   - `segment_response_rates.csv` (optional)
   - `segment_high_prob_rates.csv` (optional)
   - `feature_importance.csv` (optional)

## 2. Suggested report pages

### Page 1: Campaign response overview

- **Card**: Total contacts (Count of rows in `dashboard_predictions`).
- **Card**: Actual response rate (Average of `y_true`).
- **Card**: ROC-AUC (from `dashboard_metrics`).
- **Card**: Precision (from `dashboard_metrics`).
- **Clustered bar**: Count by `prob_bucket` (0–25%, 25–50%, 50–75%, 75–100%).
- **Clustered bar**: Count of `is_high_prob` (0 vs 1).

**Dataset**: `dashboard_predictions`, `dashboard_metrics`.

### Page 2: Segmentation performance

- **Table**: `segment_response_rates` – columns `segment_type`, `segment`, `rate`, `count`.
- **Bar chart**: X = `segment`, Y = `rate`, Legend or small multiples by `segment_type`.
- **Bar chart**: Same using `segment_high_prob_rates` for “Predicted high-probability rate by segment”.

**Dataset**: `segment_response_rates`, `segment_high_prob_rates`.

### Page 3: Feature importance

- **Table**: `feature_importance` – columns `feature`, `importance`.
- **Bar chart**: X = `importance`, Y = `feature` (sorted by importance descending).
- Optional: use the image `dashboards/exports/plots/feature_importance.png` as a static image.

**Dataset**: `feature_importance`.

### Page 4: Confusion matrix

- **Matrix visual**: Rows = row index 0/1 (from `dashboard_confusion_matrix`), Columns = `pred_0`, `pred_1`, Values = same values (or use a single value column).

Alternatively, load `dashboard_confusion_matrix.csv` and reshape in Power Query so you have rows “Actual” and “Predicted” and one value column.

## 3. Refreshing the dashboard

- Keep the CSV paths in Power BI pointing to `dashboards/exports/`.
- After each new pipeline run, run the three dashboard scripts (export, segment, feature importance). Optionally point them to the latest run folder.
- In Power BI: **Home** → **Refresh** to reload the CSVs.

## 4. Tableau

The same CSVs in `dashboards/exports/` can be connected in Tableau via **Text file** data source. Use the same logical breakdown: one sheet/dashboard for overview (predictions + metrics), one for segmentation (segment tables), one for feature importance.
