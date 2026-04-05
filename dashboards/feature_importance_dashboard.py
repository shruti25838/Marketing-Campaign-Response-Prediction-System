"""
Export feature importance from a pipeline run (SHAP or model-based) for BI dashboards.
Run from project root: python -m dashboards.feature_importance_dashboard [run_dir]
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_latest_run(processed_dir: Path) -> Path | None:
    runs = sorted(processed_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def get_feature_importance_from_pipeline(run_dir: Path) -> tuple[list[str], np.ndarray] | None:
    run_dir = Path(run_dir)
    pipe_path = run_dir / "model.joblib"
    if not pipe_path.exists():
        return None
    pipeline = joblib.load(pipe_path)
    preprocessor = pipeline.named_steps.get("preprocess")
    model = pipeline.named_steps.get("model")
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        names = preprocessor.get_feature_names_out().tolist()
    else:
        names = []
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        if len(names) != len(imp):
            names = [f"f{i}" for i in range(len(imp))]
        return names, imp
    if hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        if len(names) != len(coef):
            names = [f"f{i}" for i in range(len(coef))]
        return names, np.abs(coef)
    return None


def run_feature_importance_dashboard(run_dir: Path, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    result = get_feature_importance_from_pipeline(run_dir)
    if result is None:
        print("Could not extract feature importance from model (no run or unsupported model).")
        return
    names, importance = result
    df = pd.DataFrame({"feature": names, "importance": importance}).sort_values("importance", ascending=False)
    df.to_csv(output_dir / "feature_importance.csv", index=False)

    # Top 20 bar chart
    top = df.head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(top)), top["importance"], color="teal")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Feature Importance (Top 20)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_importance.png", dpi=150)
    plt.close()
    print(f"Exported {output_dir / 'feature_importance.csv'} and plots/feature_importance.png")


def main():
    parser = argparse.ArgumentParser(description="Export feature importance for dashboards")
    parser.add_argument("run_dir", nargs="?", default=None, help="Pipeline run directory (default: latest)")
    parser.add_argument("-o", "--output-dir", default="dashboards/exports", help="Output directory")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    processed = base / "data" / "processed"
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run(processed)
    if not run_dir:
        print("No run directory found. Run the pipeline first.")
        return
    if not run_dir.is_absolute():
        run_dir = base / run_dir
    run_feature_importance_dashboard(run_dir, base / args.output_dir)


if __name__ == "__main__":
    main()
