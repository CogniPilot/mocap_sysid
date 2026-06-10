#!/usr/bin/env python3
"""Aggregate benchmark summaries into comparison tables and figures."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.metrics import aggregate_trajectory_score
from common.paths import FIG_DIR, RESULTS_DIR, TABLE_DIR
from common.plotting import save_figure


SUMMARY_FILES = [
    RESULTS_DIR / "oem_fit_summary.csv",
    RESULTS_DIR / "sindy_fit_summary.csv",
    RESULTS_DIR / "pinn_fit_summary.csv",
    RESULTS_DIR / "ude_fit_summary.csv",
]


def read_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in SUMMARY_FILES:
        if not path.exists():
            continue
        with path.open(newline="") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def value(row: dict[str, str], key: str, default: float = np.nan) -> float:
    try:
        return float(row.get(key, ""))
    except ValueError:
        return default


def normalize_row(row: dict[str, str]) -> dict[str, object]:
    rmse = np.array([value(row, f"rmse_{name}") for name in ["V", "alpha", "gamma", "Q"]])
    # No fallback: averaging raw RMSEs would mix m/s with radians.
    score = value(row, "train_score")
    err_keys = [key for key in row if key.startswith("errpct_")]
    parameter_error = float(np.nanmean([value(row, key) for key in err_keys])) if err_keys else np.nan
    return {
        "case": row.get("case", ""),
        "method": row.get("method", ""),
        "trajectory_score": score,
        "mean_parameter_error_pct": parameter_error,
        "elapsed_s": value(row, "elapsed_s"),
        "decision_variables": value(row, "decision_variables"),
        "rmse_V": rmse[0],
        "rmse_alpha": rmse[1],
        "rmse_gamma": rmse[2],
        "rmse_Q": rmse[3],
    }


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    output = TABLE_DIR / "benchmark_metrics.csv"
    if not rows:
        return
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_scores(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    cases = sorted(set(str(row["case"]) for row in rows))
    fig, axes = plt.subplots(1, len(cases), figsize=(8.4, 3.5), sharey=True)
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        case_rows = [row for row in rows if row["case"] == case]
        methods = [str(row["method"]) for row in case_rows]
        scores = [float(row["trajectory_score"]) for row in case_rows]
        colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"][: len(methods)]
        ax.bar(np.arange(len(methods)), scores, color=colors)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_title(case.replace("_", " "))
        ax.set_ylabel("trajectory score")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "method_trajectory_score_comparison")


def main() -> int:
    rows = [normalize_row(row) for row in read_rows()]
    write_table(rows)
    plot_scores(rows)
    print(f"Wrote {TABLE_DIR / 'benchmark_metrics.csv'}")
    print(f"Wrote {FIG_DIR / 'method_trajectory_score_comparison.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
