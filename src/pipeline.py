import argparse
import json
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
import pandas as pd
import joblib

from src.config.settings import settings
from src.models.logistic_model import LogisticModel
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.services.data_loader import load_from_csv, load_from_postgres
from src.services.feature_engineering import add_features
from src.services.preprocessing import build_preprocessor, infer_feature_types
from src.services.trainer import train_model, build_pipeline
from src.services.evaluator import evaluate_model
from src.services.experiment_tracker import log_experiment
from src.views.visualization import (
    plot_confusion_matrix,
    plot_metric_bar,
    plot_pr_curve,
    plot_roc_curve,
)
from src.views.shap_explainer import shap_global_explanation, shap_local_explanation


MODEL_REGISTRY = {
    "logistic": LogisticModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
}

CV_FOLDS = 5
CV_SCORING = ["roc_auc", "precision", "recall", "f1"]


def run_cross_validation(X, y, preprocessor, model_key: str, use_smote: bool):
    """Run 5-fold stratified CV; return dict of metric -> (mean, std)."""
    model_class = MODEL_REGISTRY.get(model_key)
    if not model_class:
        raise ValueError(f"Unknown model '{model_key}'.")
    pipeline = build_pipeline(model_class(), preprocessor, use_smote)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=settings.random_state)
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=CV_SCORING,
        n_jobs=1,
    )
    result = {}
    for name in CV_SCORING:
        key = f"test_{name}"
        if key in scores:
            result[name] = (float(scores[key].mean()), float(scores[key].std()))
    return result


def run_pipeline(
    csv_path: str | None,
    db_url: str | None,
    table: str | None,
    target_col: str,
    model_key: str,
    use_smote: bool = True,
    output_dir: str | None = None,
    save_shap: bool = True,
    run_cv: bool = False,
):
    if csv_path:
        df = load_from_csv(csv_path)
    elif db_url and table:
        df = load_from_postgres(db_url, table)
    else:
        raise ValueError("Provide either a CSV path or a database URL with table.")

    df = add_features(df)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_cols, categorical_cols = infer_feature_types(df, target_col)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    if run_cv:
        print(f"\n--- {CV_FOLDS}-Fold Cross-Validation ---")
        baseline_cv = run_cross_validation(X, y, preprocessor, "logistic", use_smote)
        model_cv = run_cross_validation(X, y, preprocessor, model_key, use_smote)
        for name in CV_SCORING:
            b_mean, b_std = baseline_cv.get(name, (0.0, 0.0))
            m_mean, m_std = model_cv.get(name, (0.0, 0.0))
            print(f"  {name}: baseline (logistic) {b_mean:.4f} +/- {b_std:.4f}  |  {model_key} {m_mean:.4f} +/- {m_std:.4f}")
        if "roc_auc" in baseline_cv and baseline_cv["roc_auc"][0] > 0:
            b_auc = baseline_cv["roc_auc"][0]
            m_auc = model_cv.get("roc_auc", (0, 0))[0]
            pct = 100 * (m_auc - b_auc) / b_auc
            print(f"  vs baseline (logistic): ROC-AUC {pct:+.1f}% improvement\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=y,
    )

    model_class = MODEL_REGISTRY.get(model_key)
    if not model_class:
        raise ValueError(f"Unknown model '{model_key}'. Use one of: {list(MODEL_REGISTRY)}")

    pipeline, _ = train_model(
        model=model_class(),
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        use_smote=use_smote,
    )

    y_pred = pipeline.predict(X_test)
    y_prob = None
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics, cm = evaluate_model(y_test, y_pred, y_prob)

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_root = Path(output_dir or f"data/processed/run_{run_id}")
    plots_dir = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame(
        {"row_index": X_test.index, "y_true": y_test.values, "y_pred": y_pred, "y_prob": y_prob if y_prob is not None else float("nan")}
    )
    pred_df.to_csv(output_root / "predictions.csv", index=False)
    pd.DataFrame(cm, columns=["pred_0", "pred_1"]).to_csv(
        output_root / "confusion_matrix.csv", index=False
    )
    with (output_root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    joblib.dump(pipeline, output_root / "model.joblib")

    plot_confusion_matrix(cm, save_path=str(plots_dir / "confusion_matrix.png"))
    plot_metric_bar(metrics, save_path=str(plots_dir / "metrics.png"))
    if y_prob is not None:
        plot_roc_curve(y_test, y_prob, save_path=str(plots_dir / "roc_curve.png"))
        plot_pr_curve(y_test, y_prob, save_path=str(plots_dir / "pr_curve.png"))

    config = {
        "csv_path": csv_path,
        "db_url": bool(db_url),
        "table": table,
        "target_col": target_col,
        "model_key": model_key,
        "use_smote": use_smote,
        "rows": len(df),
        "features": X.shape[1],
    }
    log_experiment("data/processed/experiments.jsonl", config, metrics)

    if save_shap:
        try:
            preprocessor = pipeline.named_steps.get("preprocess")
            model = pipeline.named_steps.get("model")
            X_transformed = preprocessor.transform(X_test) if preprocessor else X_test
            feature_names = (
                preprocessor.get_feature_names_out().tolist()
                if preprocessor and hasattr(preprocessor, "get_feature_names_out")
                else None
            )
            shap_global_explanation(
                model,
                X_transformed,
                feature_names=feature_names,
                save_path=str(plots_dir / "shap_global.png"),
            )
            shap_local_explanation(
                model,
                X_transformed,
                row_index=0,
                save_path=str(plots_dir / "shap_local_0.png"),
            )
        except Exception as exc:
            print(f"SHAP generation skipped: {exc}")

    print("Model:", model_key)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "confusion_matrix": cm,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Campaign Response Prediction Pipeline")
    parser.add_argument("--csv", help="Path to CSV file")
    parser.add_argument("--db-url", help="PostgreSQL connection string")
    parser.add_argument("--table", help="Database table name")
    parser.add_argument("--target", default=settings.target_column, help="Target column name")
    parser.add_argument("--model", default="lightgbm", choices=list(MODEL_REGISTRY))
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE")
    parser.add_argument("--output-dir", help="Output directory for artifacts")
    parser.add_argument("--no-shap", action="store_true", help="Disable SHAP plots")
    parser.add_argument("--cv", action="store_true", help="Run 5-fold CV and baseline comparison before training")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_pipeline(
        csv_path=args.csv,
        db_url=args.db_url or settings.db_url,
        table=args.table or settings.default_table,
        target_col=args.target,
        model_key=args.model,
        use_smote=not args.no_smote,
        output_dir=args.output_dir,
        save_shap=not args.no_shap,
        run_cv=args.cv,
    )
