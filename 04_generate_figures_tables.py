#!/usr/bin/env python3
"""Generate manuscript-ready result tables from saved CSV outputs."""

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/csv")
    parser.add_argument("--figures_dir", default="results/figures")
    parser.add_argument("--output_dir", default="manuscript_tables")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    all_results = pd.read_csv(results_dir / "all_model_results_v2.csv")
    cols = ["model_key", "test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1", "test_weighted_f1", "test_nll", "test_ece", "test_brier"]
    table = all_results[cols].sort_values("test_macro_f1", ascending=False)
    table.to_csv(out / "table_model_comparison.csv", index=False)

    best_key = table.iloc[0]["model_key"]
    report_path = results_dir / f"{best_key}_classification_report.csv"
    if report_path.exists():
        pd.read_csv(report_path).to_csv(out / "table_best_model_classwise_report.csv", index=False)

    class_dist = pd.read_csv(results_dir / "class_distribution_v2.csv")
    split_dist = pd.read_csv(results_dir / "split_distribution_v2.csv")
    class_dist.to_csv(out / "table_class_distribution.csv", index=False)
    split_dist.to_csv(out / "table_split_distribution.csv", index=False)

    print("Best model:", best_key)
    print("Tables saved to:", out)

if __name__ == "__main__":
    main()
