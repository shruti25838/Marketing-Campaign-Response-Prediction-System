"""
Export targeting analysis artifacts for Power BI / Tableau.

Reads the targeting_analysis.csv and targeting_summary.json from a pipeline
run and produces a single consolidated export with:
  - Decile-level metrics (response rate, cumulative capture, lift)
  - ROI simulation at configurable cost-per-contact and revenue-per-conversion

Usage:
    python -m dashboards.targeting_dashboard --run-dir data/processed/run_YYYYMMDD_HHMMSS

Outputs to dashboards/exports/targeting_roi.csv
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Targeting Dashboard Export")
    parser.add_argument("--run-dir", type=str, required=True, help="Pipeline run directory")
    parser.add_argument("--cost-per-contact", type=float, default=3.0, help="Cost per customer contacted ($)")
    parser.add_argument("--revenue-per-conversion", type=float, default=11.0, help="Revenue per converted customer ($)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    analysis_path = run_dir / "targeting_analysis.csv"
    summary_path = run_dir / "targeting_summary.json"

    if not analysis_path.exists():
        print(f"Error: {analysis_path} not found. Run targeting_analysis first.", file=sys.stderr)
        sys.exit(1)

    table = pd.read_csv(analysis_path)

    # Load summary for total counts
    with open(summary_path) as f:
        summary = json.load(f)

    cost = args.cost_per_contact
    revenue = args.revenue_per_conversion
    total_customers = summary["total_customers"]

    # Compute ROI columns
    table["cum_contact_cost"] = table["cum_customers"] * cost
    table["cum_revenue"] = table["cum_responders"] * revenue
    table["cum_profit"] = table["cum_revenue"] - table["cum_contact_cost"]
    table["roi_pct"] = ((table["cum_revenue"] - table["cum_contact_cost"]) / table["cum_contact_cost"] * 100).round(1)

    # Blanket campaign baseline
    blanket_cost = total_customers * cost
    blanket_revenue = summary["total_responders"] * revenue
    blanket_profit = blanket_revenue - blanket_cost

    table["blanket_cost"] = blanket_cost
    table["blanket_profit"] = blanket_profit
    table["savings_vs_blanket"] = blanket_cost - table["cum_contact_cost"]
    table["profit_lift_vs_blanket"] = table["cum_profit"] - blanket_profit

    # Export
    out_dir = Path("dashboards/exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "targeting_roi.csv"
    table.to_csv(out_path, index=False)
    print(f"Exported targeting ROI dashboard data -> {out_path}")

    # Print quick summary
    best_row = table.loc[table["cum_profit"].idxmax()]
    print(f"\nOptimal targeting: top {int(best_row['decile'])} deciles")
    print(f"  Contact {int(best_row['cum_customers'])} / {total_customers} customers ({best_row['cum_customer_pct']:.0%})")
    print(f"  Captures {best_row['cum_response_capture']:.0%} of responders")
    print(f"  Profit: ${best_row['cum_profit']:,.0f} vs ${blanket_profit:,.0f} blanket (${best_row['profit_lift_vs_blanket']:,.0f} lift)")
    print(f"  Cost savings: ${best_row['savings_vs_blanket']:,.0f}")


if __name__ == "__main__":
    main()
