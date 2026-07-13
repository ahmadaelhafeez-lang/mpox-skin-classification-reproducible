#!/usr/bin/env python3
"""Final analysis from saved prediction CSVs.

Computes final model ranking, class-wise reports, bootstrap confidence intervals,
and pairwise McNemar tests from saved prediction files.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, classification_report, confusion_matrix

CLASSES = ["Monkeypox", "Chickenpox", "Measles", "Normal"]
PROB_COLS = [f"prob_{c}" for c in CLASSES]


def ece(y_true, probs, n_bins=10):
    pred = np.argmax(probs, axis=1)
    conf = probs.max(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    val = 0.0
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i+1])
        if mask.sum():
            val += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(val)


def brier(y_true, probs):
    return float(np.mean(np.sum((probs - np.eye(4)[y_true]) ** 2, axis=1)))


def metrics(y_true, probs):
    pred = probs.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "macro_precision": precision_score(y_true, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, pred, average="weighted", zero_division=0),
        "nll": log_loss(y_true, probs, labels=np.arange(4)),
        "ece": ece(y_true, probs),
        "brier": brier(y_true, probs),
    }


def bootstrap_ci(y_true, probs, metric_fn, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(metric_fn(y_true[idx], probs[idx]))
    return np.percentile(vals, [2.5, 50, 97.5])


def mcnemar(y_true, pred_a, pred_b):
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    b01 = int(np.sum((~a_correct) & b_correct))
    b10 = int(np.sum(a_correct & (~b_correct)))
    n = b01 + b10
    p = 1.0 if n == 0 else binomtest(min(b01, b10), n=n, p=0.5, alternative="two-sided").pvalue
    return b01, b10, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/csv")
    parser.add_argument("--output_dir", default="analysis_outputs")
    parser.add_argument("--n_boot", type=int, default=5000)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    prediction_files = sorted(results_dir.glob("*_predictions.csv"))
    rows, preds, probs_map = [], {}, {}

    for f in prediction_files:
        model_key = f.name.replace("_predictions.csv", "")
        df = pd.read_csv(f)
        y = df["label"].values.astype(int)
        probs = df[PROB_COLS].values.astype(float)
        probs_map[model_key] = probs
        preds[model_key] = probs.argmax(axis=1)
        row = {"model_key": model_key}
        row.update(metrics(y, probs))
        rows.append(row)

        pd.DataFrame(classification_report(y, preds[model_key], target_names=CLASSES, output_dict=True, zero_division=0)).T.to_csv(out / f"{model_key}_classification_report.csv")
        pd.DataFrame(confusion_matrix(y, preds[model_key], labels=np.arange(4)), index=CLASSES, columns=CLASSES).to_csv(out / f"{model_key}_confusion_matrix.csv")

    ranking = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    ranking.to_csv(out / "final_model_ranking.csv", index=False)
    print(ranking)

    best = ranking.iloc[0]["model_key"]
    first_file = pd.read_csv(results_dir / f"{best}_predictions.csv")
    y = first_file["label"].values.astype(int)
    probs = probs_map[best]
    ci_rows = []
    for name, fn in [
        ("accuracy", lambda yt, pr: accuracy_score(yt, pr.argmax(axis=1))),
        ("macro_f1", lambda yt, pr: f1_score(yt, pr.argmax(axis=1), average="macro", zero_division=0)),
        ("weighted_f1", lambda yt, pr: f1_score(yt, pr.argmax(axis=1), average="weighted", zero_division=0)),
    ]:
        low, mid, high = bootstrap_ci(y, probs, fn, n_boot=args.n_boot)
        ci_rows.append({"model_key": best, "metric": name, "median": mid, "ci95_low": low, "ci95_high": high})
    pd.DataFrame(ci_rows).to_csv(out / "bootstrap_ci_best_model.csv", index=False)

    mc_rows = []
    for a, b in itertools.combinations(preds.keys(), 2):
        b01, b10, p = mcnemar(y, preds[a], preds[b])
        mc_rows.append({"model_a": a, "model_b": b, "b01_a_wrong_b_correct": b01, "b10_a_correct_b_wrong": b10, "p_value": p})
    pd.DataFrame(mc_rows).to_csv(out / "mcnemar_pairwise_tests.csv", index=False)
    print("Best model:", best)

if __name__ == "__main__":
    main()
