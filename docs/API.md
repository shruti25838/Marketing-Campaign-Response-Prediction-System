# API Reference

Summary of main modules and functions for the Marketing Campaign Response Prediction System.

---

## Pipeline

**`src.pipeline`**

- **`run_pipeline(csv_path=None, db_url=None, table=None, target_col='response', model_key='lightgbm', use_smote=True, output_dir=None, save_shap=True)`**  
  Loads data (CSV or PostgreSQL), runs feature engineering, train/test split, training, evaluation, saves artifacts and optional SHAP plots. Returns `{"pipeline", "metrics", "confusion_matrix"}`.

- **`MODEL_REGISTRY`** – Dict mapping `"logistic"`, `"xgboost"`, `"lightgbm"` to model classes.

- **`build_arg_parser()`** – Returns an `argparse.ArgumentParser` for CLI.

---

## Config

**`src.config.settings`**

- **`settings`** – Frozen dataclass: `db_url`, `default_table`, `target_column`, `random_state`, `test_size`. Reads from environment when set.

---

## Models (`src.models`)

All models extend **`BaseModel`** and implement **`build()`** returning a scikit-learn–style estimator.

- **`BaseModel(**model_params)`** – Abstract base; `model_params` are passed through to the underlying estimator.
- **`LogisticModel`** – Logistic regression (balanced class weight).
- **`XGBoostModel`** – XGBoost classifier.
- **`LightGBMModel`** – LightGBM classifier.

---

## Services

**`src.services.data_loader`**

- **`load_from_csv(csv_path: str) -> pd.DataFrame`**
- **`load_from_postgres(db_url: str, table: str, query: str | None = None) -> pd.DataFrame`**

**`src.services.preprocessing`**

- **`infer_feature_types(df, target_col) -> (numeric_cols, categorical_cols)`**  
  Returns lists of column names by type (excluding target).
- **`build_preprocessor(numeric_cols, categorical_cols)`**  
  Returns a `ColumnTransformer`: median impute + scale for numeric; most_frequent impute + one-hot for categorical; remainder dropped.

**`src.services.feature_engineering`**

- **`add_features(df: pd.DataFrame) -> pd.DataFrame`**  
  Adds tenure, aggregates, ratios, interaction features, age_group, previous_success/failure, engagement and log features. Safe when columns are missing.

**`src.services.trainer`**

- **`train_model(model: BaseModel, X_train, y_train, preprocessor, use_smote=True) -> (pipeline, y_pred_train)`**  
  Builds an imblearn pipeline: preprocess → optional SMOTE → model; fits and returns pipeline and training predictions.

**`src.services.evaluator`**

- **`evaluate_model(y_true, y_pred, y_prob=None) -> (metrics_dict, confusion_matrix_array)`**  
  Metrics: precision, recall, F1; ROC-AUC when `y_prob` is provided.

**`src.services.experiment_tracker`**

- **`log_experiment(log_path, config_dict, metrics_dict)`**  
  Appends one JSONL line with timestamp, config, and metrics.

---

## Views

**`src.views.visualization`**

- **`plot_confusion_matrix(cm, title=..., save_path=None)`**
- **`plot_metric_bar(metrics_dict, title=..., save_path=None)`**
- **`plot_roc_curve(y_true, y_prob, title=..., save_path=None)`**
- **`plot_pr_curve(y_true, y_prob, title=..., save_path=None)`**

**`src.views.shap_explainer`**

- **`shap_global_explanation(model, X_transformed, feature_names=None, save_path=None)`**  
  Bar plot of mean |SHAP|.
- **`shap_local_explanation(model, X_transformed, row_index=0, save_path=None)`**  
  Waterfall for one row.

---

## Dashboards

**`dashboards.export_dashboard_data`**

- **`export_dashboard_data(run_dir, output_dir)`**  
  Writes `dashboard_predictions.csv`, `dashboard_metrics.csv`, `dashboard_confusion_matrix.csv` under `output_dir`.

**`dashboards.segment_dashboard`**

- **`run_segment_dashboard(run_dir, raw_path, output_dir)`**  
  Writes segment response rates, optional high-prob rates, and PNGs in `output_dir` and `output_dir/plots/`.

**`dashboards.feature_importance_dashboard`**

- **`run_feature_importance_dashboard(run_dir, output_dir)`**  
  Writes `feature_importance.csv` and `plots/feature_importance.png` from the saved pipeline model.
