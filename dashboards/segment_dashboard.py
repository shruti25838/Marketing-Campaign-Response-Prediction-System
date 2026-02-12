"""
Generate customer segmentation dashboard data and plots: response rate and
high-probability share by job, marital, education, contact, month.
Run from project root: python -m dashboards.segment_dashboard [run_dir]
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services.feature_engineering import add_features


def find_latest_run(processed_dir: Path) -> Path | None:
    runs = sorted(processed_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def load_test_with_predictions(run_dir: Path, raw_path: Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    pred = pd.read_csv(run_dir / "predictions.csv")
    if "row_index" not in pred.columns:
        return pd.DataFrame()
    df = pd.read_csv(raw_path)
    df = add_features(df)
    df = df.loc[df.index.isin(pred["row_index"])].copy()
    pred = pred.set_index("row_index")
    for col in ["y_true", "y_pred", "y_prob"]:
        if col in pred.columns:
            df[col] = pred.loc[df.index, col].values
    df["high_prob"] = (df["y_prob"].fillna(0) >= 0.5).astype(int)
    return df


def segment_rates(df: pd.DataFrame, segment_col: str, value_col: str = "response") -> pd.DataFrame:
    if segment_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    g = df.groupby(segment_col)[value_col].agg(["mean", "sum", "count"])
    g.columns = ["rate", "sum", "count"]
    g = g[g["count"] >= 10].sort_values("rate", ascending=True)
    return g.reset_index()


def run_segment_dashboard(run_dir: Path, raw_path: Path, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Actual response rate by segment (full raw data)
    df_raw = pd.read_csv(raw_path)
    df_raw = add_features(df_raw)
    target = "response"
    if target not in df_raw.columns:
        print("Target 'response' not in data. Skipping segment dashboard.")
        return

    segment_cols = [c for c in ["job", "marital", "education", "contact", "month", "age_group"] if c in df_raw.columns]
    all_rates = []
    for col in segment_cols:
        g = segment_rates(df_raw, col, target)
        if g.empty:
            continue
        g["segment_type"] = col
        g.rename(columns={col: "segment"}, inplace=True)
        all_rates.append(g[["segment_type", "segment", "rate", "count"]])
    if all_rates:
        pd.concat(all_rates, ignore_index=True).to_csv(output_dir / "segment_response_rates.csv", index=False)

    # Test set with predictions: high-prob rate by segment
    df_test = pd.DataFrame()
    if run_dir.exists() and (run_dir / "predictions.csv").exists():
        df_test = load_test_with_predictions(run_dir, raw_path)
    if df_test.empty:
        print("No row_index in predictions; segmentation by prediction skipped.")
    else:
        prob_rates = []
        for col in segment_cols:
            if col not in df_test.columns:
                continue
            g = segment_rates(df_test, col, "high_prob")
            if g.empty:
                continue
            g["segment_type"] = col
            g.rename(columns={col: "segment"}, inplace=True)
            prob_rates.append(g[["segment_type", "segment", "rate", "count"]])
        if prob_rates:
            pd.concat(prob_rates, ignore_index=True).to_csv(output_dir / "segment_high_prob_rates.csv", index=False)

    # Plots: response rate by job and by contact
    for col in ["job", "contact"]:
        if col not in df_raw.columns:
            continue
        g = segment_rates(df_raw, col, target)
        if g.empty or len(g) < 2:
            continue
        fig, ax = plt.subplots(figsize=(8, max(4, len(g) * 0.25)))
        g.plot(x=col, y="rate", kind="barh", ax=ax, legend=False, color="steelblue")
        ax.set_title(f"Response rate by {col}")
        ax.set_xlabel("Response rate")
        ax.set_ylabel(col)
        plt.tight_layout()
        plt.savefig(plots_dir / f"segment_response_{col}.png", dpi=150)
        plt.close()
    print(f"Segmentation outputs in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate segmentation dashboard data and plots")
    parser.add_argument("run_dir", nargs="?", default=None, help="Pipeline run directory (default: latest)")
    parser.add_argument("-o", "--output-dir", default="dashboards/exports", help="Output directory")
    parser.add_argument("--csv", default="data/raw/campaign_data.csv", help="Raw campaign CSV")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    raw_path = base / args.csv
    if not raw_path.exists():
        print(f"Raw data not found: {raw_path}")
        return
    processed = base / "data" / "processed"
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run(processed)
    if not run_dir:
        if args.run_dir:
            run_dir = base / args.run_dir
        else:
            run_dir = processed / "run_00000000_000000"  # dummy so we only use raw
    if not run_dir.is_absolute():
        run_dir = base / run_dir
    run_segment_dashboard(run_dir, raw_path, base / args.output_dir)


if __name__ == "__main__":
    main()
