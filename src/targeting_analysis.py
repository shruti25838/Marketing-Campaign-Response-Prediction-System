"""
Targeting ROI Analysis
======================
Reads a predictions.csv from any pipeline run, bins customers into deciles
by predicted response probability, and computes:
  - Cumulative response capture (gains)
  - Lift over random targeting
  - Cost savings from targeted vs. blanket campaigns

Outputs:
  - targeting_analysis.csv   (decile-level breakdown)
  - targeting_summary.json   (headline numbers for resume / README)
  - gains_chart.csv          (Power BI / Tableau ready)

Usage:
    python -m src.targeting_analysis --run-dir data/processed/run_YYYYMMDD_HHMMSS
    python -m src.targeting_analysis --predictions path/to/predictions.csv

If --run-dir is given, reads predictions.csv from that folder and writes
outputs there.  Otherwise writes to the same directory as --predictions.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def load_predictions(path: Path) -> pd.DataFrame:
    """Load predictions.csv and validate required columns."""
    df = pd.read_csv(path)
    required = {"y_true", "y_prob"}
    missing = required - set(df.columns)
    if missing:
        print(f"Error: predictions.csv missing columns: {missing}", file=sys.stderr)
        sys.exit(1)
    return df


def build_decile_table(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Bin customers into deciles by predicted probability (descending)."""
    df = df.copy()
    # Rank into deciles: 1 = highest predicted probability
    df["decile"] = pd.qcut(df["y_prob"], q=n_bins, labels=False, duplicates="drop")
    # Invert so decile 1 = top scores
    df["decile"] = df["decile"].max() - df["decile"] + 1

    summary = (
        df.groupby("decile")
        .agg(
            n_customers=("y_true", "count"),
            n_responders=("y_true", "sum"),
            mean_prob=("y_prob", "mean"),
        )
        .sort_values("decile")
        .reset_index()
    )

    total_responders = summary["n_responders"].sum()
    total_customers = summary["n_customers"].sum()

    summary["response_rate"] = summary["n_responders"] / summary["n_customers"]
    summary["cum_responders"] = summary["n_responders"].cumsum()
    summary["cum_customers"] = summary["n_customers"].cumsum()
    summary["cum_response_capture"] = summary["cum_responders"] / total_responders
    summary["cum_customer_pct"] = summary["cum_customers"] / total_customers
    summary["lift"] = summary["cum_response_capture"] / summary["cum_customer_pct"]

    return summary


def compute_headline_numbers(table: pd.DataFrame) -> dict:
    """Extract the key numbers for the resume bullet and README."""
    total_customers = table["n_customers"].sum()
    total_responders = table["n_responders"].sum()
    overall_response_rate = total_responders / total_customers

    results = {
        "total_customers": int(total_customers),
        "total_responders": int(total_responders),
        "overall_response_rate": round(overall_response_rate, 4),
    }

    # Find the optimal cutoff: fewest deciles to capture >= 70%, 75%, 80% of responders
    for threshold in [0.70, 0.75, 0.80]:
        row = table[table["cum_response_capture"] >= threshold].iloc[0]
        decile = int(row["decile"])
        pct_customers = round(row["cum_customer_pct"] * 100, 1)
        pct_captured = round(row["cum_response_capture"] * 100, 1)
        lift = round(row["lift"], 2)
        cost_reduction = round((1 - row["cum_customer_pct"]) * 100, 1)

        key = f"threshold_{int(threshold * 100)}"
        results[key] = {
            "top_n_deciles": decile,
            "pct_customers_contacted": pct_customers,
            "pct_responders_captured": pct_captured,
            "lift_over_random": lift,
            "cost_reduction_pct": cost_reduction,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Targeting ROI Analysis")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=str, help="Pipeline run directory containing predictions.csv")
    group.add_argument("--predictions", type=str, help="Direct path to predictions.csv")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        pred_path = run_dir / "predictions.csv"
        out_dir = run_dir
    else:
        pred_path = Path(args.predictions)
        out_dir = pred_path.parent

    if not pred_path.exists():
        print(f"Error: {pred_path} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading predictions from {pred_path}")
    df = load_predictions(pred_path)
    print(f"  {len(df)} customers, {int(df['y_true'].sum())} responders")

    print("Building decile analysis...")
    table = build_decile_table(df)

    headlines = compute_headline_numbers(table)

    # Save outputs
    table_path = out_dir / "targeting_analysis.csv"
    table.to_csv(table_path, index=False)
    print(f"  Saved decile table -> {table_path}")

    summary_path = out_dir / "targeting_summary.json"
    with open(summary_path, "w") as f:
        json.dump(headlines, f, indent=2)
    print(f"  Saved summary -> {summary_path}")

    # Gains chart for Power BI (cumulative % customers vs cumulative % responders)
    gains = table[["decile", "cum_customer_pct", "cum_response_capture", "lift"]].copy()
    gains.columns = ["decile", "pct_customers_contacted", "pct_responders_captured", "lift"]
    # Add the origin point for charting
    origin = pd.DataFrame([{"decile": 0, "pct_customers_contacted": 0.0, "pct_responders_captured": 0.0, "lift": 0.0}])
    gains = pd.concat([origin, gains], ignore_index=True)
    gains_path = out_dir / "gains_chart.csv"
    gains.to_csv(gains_path, index=False)
    print(f"  Saved gains chart -> {gains_path}")

    # Print headline results
    print("\n" + "=" * 60)
    print("TARGETING ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Total customers: {headlines['total_customers']}")
    print(f"Total responders: {headlines['total_responders']}")
    print(f"Overall response rate: {headlines['overall_response_rate']:.1%}")

    for threshold in [70, 75, 80]:
        key = f"threshold_{threshold}"
        if key in headlines:
            h = headlines[key]
            print(f"\nTo capture {h['pct_responders_captured']}% of responders:")
            print(f"  Contact top {h['top_n_deciles']} deciles ({h['pct_customers_contacted']}% of customers)")
            print(f"  Lift over random: {h['lift_over_random']}x")
            print(f"  Cost reduction vs. blanket campaign: {h['cost_reduction_pct']}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
