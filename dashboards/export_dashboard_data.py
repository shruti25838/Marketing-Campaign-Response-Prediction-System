"""
Export pipeline run artifacts into dashboard-ready datasets for Power BI or Tableau.
Run from project root: python -m dashboards.export_dashboard_data [run_dir]
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def find_latest_run(processed_dir: Path) -> Path | None:
    runs = sorted(processed_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def export_dashboard_data(run_dir: Path, output_dir: Path) -> None:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load artifacts
    predictions_path = run_dir / "predictions.csv"
    metrics_path = run_dir / "metrics.json"
    confusion_path = run_dir / "confusion_matrix.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions not found: {predictions_path}")

    pred = pd.read_csv(predictions_path)
    prob = pred["y_prob"].fillna(0) if "y_prob" in pred.columns else 0
    pred["prob_bucket"] = pd.cut(
        prob,
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=["0-25%", "25-50%", "50-75%", "75-100%"],
    )
    pred["is_high_prob"] = (prob >= 0.5).astype(int)
    pred.to_csv(output_dir / "dashboard_predictions.csv", index=False)

    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        pd.DataFrame([metrics]).to_csv(output_dir / "dashboard_metrics.csv", index=False)

    if confusion_path.exists():
        cm = pd.read_csv(confusion_path)
        cm.to_csv(output_dir / "dashboard_confusion_matrix.csv", index=False)

    print(f"Exported to {output_dir}: dashboard_predictions.csv, dashboard_metrics.csv, dashboard_confusion_matrix.csv")


def main():
    parser = argparse.ArgumentParser(description="Export pipeline run to dashboard-ready CSVs")
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help="Path to pipeline run dir (e.g. data/processed/run_20250101_120000). Default: latest run",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="dashboards/exports",
        help="Output directory for CSV files",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    processed = base / "data" / "processed"

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = base / run_dir
    else:
        run_dir = find_latest_run(processed)
        if not run_dir:
            print("No run directory found. Run the pipeline first.")
            return
        print(f"Using latest run: {run_dir}")

    export_dashboard_data(run_dir, base / args.output_dir)


if __name__ == "__main__":
    main()
