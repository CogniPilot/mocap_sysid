#!/usr/bin/env python3
"""Generate benchmark results and LaTeX-ready artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from benchmark.export import export_web_data
from benchmark.paths import (
    DATASET_TOOLS,
    LATEX,
    LATEX_FIG,
    LATEX_GENERATED,
    LATEX_TABLES,
    METHOD_CODE,
    RESULTS as METHOD_RESULTS,
    ROOT,
    SPORTCUB_DATASET_ID,
    WORK,
    WORK_DATA,
)
from benchmark.registry import gpu_method_names, heavy_method_names, method_training_modes, worker_method_names
from benchmark.scenarios import (
    DATASET_MODES,
    DATASET_OUTPUTS,
    DATASET_TITLES,
    SIX_DOF_DATASET_MODES,
    SIX_DOF_DATASET_OUTPUTS,
    SIX_DOF_DATASET_TITLES,
)


METHOD_FIG = LATEX_FIG
METHOD_TABLES = LATEX_TABLES
METHOD_WORKERS = worker_method_names()
HEAVY_METHOD_WORKERS = heavy_method_names()
GPU_METHOD_WORKERS = gpu_method_names()
METHOD_TRAINING_MODES = method_training_modes()

FIGURE_EXPORTS = {
    "shared_validation_score_comparison.svg": "generated_shared_validation_score_comparison.svg",
    "shared_validation_trajectory_overlay.svg": "generated_shared_validation_trajectory_overlay.svg",
    "shared_pinn_coeff_residual_validation.svg": "generated_shared_pinn_coeff_residual_validation.svg",
    "shared_frequency_validation_diagnostic.svg": "generated_shared_frequency_validation_diagnostic.svg",
    "shared_train_time_accuracy_tradeoff.svg": "generated_shared_train_time_accuracy_tradeoff.svg",
    "observation_rate_state_reconstruction.svg": "generated_observation_rate_state_reconstruction.svg",
}
SIX_DOF_FIGURE_EXPORTS = {
    "aircraft6dof_validation_score_comparison.svg": "generated_aircraft6dof_validation_score_comparison.svg",
    "aircraft6dof_validation_trajectory_overlay.svg": "generated_aircraft6dof_validation_trajectory_overlay.svg",
    "aircraft6dof_train_time_accuracy_tradeoff.svg": "generated_aircraft6dof_train_time_accuracy_tradeoff.svg",
    "aircraft6dof_method_score_heatmap_direct.svg": "generated_aircraft6dof_method_score_heatmap_direct.svg",
    "aircraft6dof_method_score_heatmap_mocap.svg": "generated_aircraft6dof_method_score_heatmap_mocap.svg",
}
ARCHIVED_FIGURES = [
    "shared_validation_score_comparison.svg",
    "shared_validation_trajectory_overlay.svg",
    "shared_pinn_coeff_residual_validation.svg",
    "shared_frequency_validation_diagnostic.svg",
    "shared_train_time_accuracy_tradeoff.svg",
]
ARCHIVED_RESULTS = [
    "shared_method_comparison.csv",
    "shared_method_traces.json",
    "shared_uq_diagnostics.csv",
    "shared_frequency_summary.csv",
    "shared_sindy_coefficients.csv",
    "shared_symbolic_coefficients.csv",
]
LATEX_GENERATED_PATTERNS = ("*.tex",)
LATEX_GENERATED_FIGURE_PATTERNS = ("generated_*",)
TRADEOFF_FAILURE_THRESHOLD = 1.0


def methods_python() -> str:
    venv_python = ROOT / ".venv" / "bin" / "python"
    return str(venv_python if venv_python.exists() else Path(sys.executable))


def run(command: list[str], cwd: Path = ROOT) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def run_with_env(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def relative_or_absolute(path: Path, root: Path = ROOT) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def worker_env(threads_per_worker: int = 1) -> dict[str, str]:
    env = os.environ.copy()
    value = str(max(1, int(threads_per_worker)))
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[key] = value
    return env


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def ffloat(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(number):
        return "--"
    if number == 0.0:
        return "0"
    if abs(number) < 1e-2 or abs(number) >= 1e3:
        return f"{number:.{digits}e}"
    return f"{number:.{digits}g}"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"missing required results file: {path}")
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def is_open_loop_model_row(row: dict[str, str]) -> bool:
    mode = row.get("evaluation_mode", "open_loop")
    if mode == "measurement_assimilated":
        return False
    if "evaluation_mode" not in row and row.get("method") == "FilterError-EKF":
        return False
    return True


def write_method_table(rows: list[dict[str, str]], output: Path, caption: str, label: str) -> None:
    rows = sorted(rows, key=lambda row: float(row["validation_score"]))
    source_name = rows[0].get("state_source", "") if rows else ""
    show_source = len({row.get("state_source", "") for row in rows}) > 1
    column_spec = r"p{0.30\linewidth}lrrrrp{0.18\linewidth}" if show_source else r"p{0.36\linewidth}rrrrp{0.22\linewidth}"
    header = (
        r"Method & Source & Score & Train [s] & Rollout [s] & $V$ RMSE & Backend \\"
        if show_source
        else r"Method & Score & Train [s] & Rollout [s] & $V$ RMSE & Backend \\"
    )
    with output.open("w") as stream:
        stream.write("% Generated by results.py latex-assets. Do not edit by hand.\n")
        stream.write(r"\begingroup\scriptsize\setlength{\tabcolsep}{2pt}" + "\n")
        stream.write(rf"\begin{{longtable}}{{{column_spec}}}" + "\n")
        stream.write(rf"\caption{{{caption}}}\label{{{label}}}\\" + "\n")
        stream.write(r"\toprule" + "\n")
        stream.write(header + "\n")
        stream.write(r"\midrule" + "\n")
        stream.write(r"\endfirsthead" + "\n")
        stream.write(r"\toprule" + "\n")
        stream.write(header + "\n")
        stream.write(r"\midrule" + "\n")
        stream.write(r"\endhead" + "\n")
        for row in rows:
            values = [
                latex_escape(row["method"].replace(f" ({source_name})", "")),
            ]
            if show_source:
                values.append(latex_escape(row.get("state_source", "")))
            values.extend(
                [
                    ffloat(row["validation_score"]),
                    ffloat(row["train_elapsed_s"]),
                    ffloat(row["rollout_elapsed_s"]),
                    ffloat(row["rmse_V"]),
                    latex_escape(row["backend"]),
                ]
            )
            stream.write(" & ".join(values) + r" \\" + "\n")
        stream.write(r"\bottomrule" + "\n")
        stream.write(r"\end{longtable}" + "\n")
        stream.write(r"\endgroup" + "\n")


def archived_path(mode: str, filename: str, directory: Path) -> Path:
    return directory / f"{mode}_{filename}"


def available_archived_modes() -> list[str]:
    return [mode for mode in DATASET_MODES if archived_path(mode, "shared_method_comparison.csv", METHOD_RESULTS).exists()]


def remove_matching(directory: Path, patterns: tuple[str, ...]) -> int:
    if not directory.exists():
        return 0
    removed = 0
    for pattern in patterns:
        for path in directory.glob(pattern):
            if path.is_file() or path.is_symlink():
                path.unlink()
                removed += 1
    return removed


def clean_latex_generated_assets() -> None:
    removed_tables = remove_matching(LATEX_GENERATED, LATEX_GENERATED_PATTERNS)
    removed_figures = remove_matching(LATEX_FIG, LATEX_GENERATED_FIGURE_PATTERNS)
    if removed_tables or removed_figures:
        print(f"Removed {removed_tables} generated LaTeX tables and {removed_figures} generated LaTeX figure files")


def clean_suite_artifacts(modes: list[str] | tuple[str, ...]) -> None:
    removed = 0
    for mode in modes:
        for filename in ARCHIVED_RESULTS:
            for directory in (METHOD_RESULTS, METHOD_TABLES):
                path = archived_path(mode, filename, directory)
                if path.exists():
                    path.unlink()
                    removed += 1
        for filename in ARCHIVED_FIGURES:
            for suffix in (".svg", ".png"):
                path = METHOD_FIG / f"{mode}_{Path(filename).with_suffix(suffix).name}"
                if path.exists():
                    path.unlink()
                    removed += 1
    for filename in ARCHIVED_RESULTS:
        for directory in (METHOD_RESULTS, METHOD_TABLES):
            path = directory / filename
            if path.exists():
                path.unlink()
                removed += 1
    for filename in ARCHIVED_FIGURES:
        for suffix in (".svg", ".png"):
            path = METHOD_FIG / Path(filename).with_suffix(suffix).name
            if path.exists():
                path.unlink()
                removed += 1
    if removed:
        print(f"Removed {removed} stale suite result artifacts before regeneration", flush=True)


def shared_results_path() -> Path:
    archived = archived_path("open_loop", "shared_method_comparison.csv", METHOD_RESULTS)
    return archived if archived.exists() else METHOD_RESULTS / "shared_method_comparison.csv"


def write_shared_method_table() -> None:
    open_loop_rows = [row for row in read_csv(shared_results_path()) if is_open_loop_model_row(row)]
    write_method_table(
        open_loop_rows,
        LATEX_GENERATED / "shared_method_comparison_table.tex",
        "Shared 3-DoF longitudinal open-loop model benchmark results. Lower validation score is better.",
        "tab:shared_method_comparison",
    )
    direct_rows = [row for row in open_loop_rows if row.get("state_source") == "direct"]
    mocap_rows = [row for row in open_loop_rows if row.get("state_source") == "mocap"]
    write_method_table(
        direct_rows,
        LATEX_GENERATED / "shared_method_comparison_direct_table.tex",
        "Direct-state benchmark results sorted by validation score. Lower is better.",
        "tab:shared_method_comparison_direct",
    )
    write_method_table(
        mocap_rows,
        LATEX_GENERATED / "shared_method_comparison_mocap_table.tex",
        "Motion-capture-derived benchmark results sorted by validation score. Lower is better.",
        "tab:shared_method_comparison_mocap",
    )


def write_experiment_method_tables() -> None:
    for mode in available_archived_modes():
        rows = [row for row in read_csv(archived_path(mode, "shared_method_comparison.csv", METHOD_RESULTS)) if is_open_loop_model_row(row)]
        title = DATASET_TITLES[mode]
        for source in ("direct", "mocap"):
            source_rows = [row for row in rows if row.get("state_source") == source]
            if not source_rows:
                continue
            write_method_table(
                source_rows,
                LATEX_GENERATED / f"{mode}_shared_method_comparison_{source}_table.tex",
                f"{title} {source}-state benchmark results sorted by validation score. Lower is better.",
                f"tab:{mode}_shared_method_comparison_{source}",
            )


def write_observation_rate_table() -> None:
    rows = [row for row in read_csv(METHOD_RESULTS / "observation_rate_study.csv") if row["split"] == "validation"]
    output = LATEX_GENERATED / "observation_rate_table.tex"
    with output.open("w") as stream:
        stream.write("% Generated by results.py latex-assets. Do not edit by hand.\n")
        stream.write(r"\begin{table}[hbt!]" + "\n")
        stream.write(r"\centering" + "\n")
        stream.write(r"\caption{Validation-state reconstruction error after decimating the 100 Hz motion-capture stream.}" + "\n")
        stream.write(r"\label{tab:observation_rate_study}" + "\n")
        stream.write(r"\begin{tabular}{rrrrrr}" + "\n")
        stream.write(r"\toprule" + "\n")
        stream.write(r"Rate [Hz] & Nyquist [Hz] & $V$ RMSE & $\alpha$ RMSE & $\gamma$ RMSE & $q$ RMSE \\" + "\n")
        stream.write(r"\midrule" + "\n")
        for row in rows:
            stream.write(
                " & ".join(
                    [
                        ffloat(row["observation_rate_hz"]),
                        ffloat(row["nyquist_hz"]),
                        ffloat(row["rmse_V"]),
                        ffloat(row["rmse_alpha"]),
                        ffloat(row["rmse_gamma"]),
                        ffloat(row["rmse_Q"]),
                    ]
                )
                + r" \\"
                + "\n"
            )
        stream.write(r"\bottomrule" + "\n")
        stream.write(r"\end{tabular}" + "\n")
        stream.write(r"\end{table}" + "\n")


def write_uq_table() -> None:
    rows = read_csv(METHOD_RESULTS / "shared_uq_diagnostics.csv")
    output = LATEX_GENERATED / "shared_uq_diagnostics_table.tex"
    with output.open("w") as stream:
        stream.write("% Generated by results.py latex-assets. Do not edit by hand.\n")
        stream.write(r"\begin{table}[hbt!]" + "\n")
        stream.write(r"\centering" + "\n")
        stream.write(r"\caption{Fisher-information uncertainty diagnostics for the equation-error aerodynamic parameter fit.}" + "\n")
        stream.write(r"\label{tab:shared_uq_diagnostics}" + "\n")
        stream.write(r"\begin{tabular}{llrrrr}" + "\n")
        stream.write(r"\toprule" + "\n")
        stream.write(r"Source & Parameter & Estimate & CRLB std. & Rel. std. & Max corr. \\" + "\n")
        stream.write(r"\midrule" + "\n")
        for row in rows:
            stream.write(
                " & ".join(
                    [
                        latex_escape(row["state_source"]),
                        latex_escape(row["parameter"]),
                        ffloat(row["estimate"]),
                        ffloat(row["crlb_std"]),
                        ffloat(row["relative_std"]),
                        ffloat(row["max_abs_correlation"]),
                    ]
                )
                + r" \\"
                + "\n"
            )
        stream.write(r"\bottomrule" + "\n")
        stream.write(r"\end{tabular}" + "\n")
        stream.write(r"\end{table}" + "\n")


def clean_method_name(name: str, source: str) -> str:
    return name.replace(f" ({source})", "")


def tradeoff_label(name: str, source: str) -> str:
    aliases = {
        "EquationError-LS": "EqErr-LS",
        "Symbolic-Stepwise": "Symbolic",
        "GP-CoeffClosure": "GP/RBF",
        "PINN-CoeffClosure": "PINN-closure",
        "UDE-Residual": "UDE-resid",
        "UDE-HiddenControl": "UDE-hidden",
        "PINN-HiddenElevator": "PINN-hidden",
        "NN-CoeffSurrogate": "NN-surrogate",
        "OEM-MocapOutput": "OEM-mocap",
        "OEM-HiddenController": "OEM-hidden",
        "Frequency-Welch": "Freq-Welch",
        "Frequency-Stitching": "Freq-Stitch",
        "Variational-Mocap": "Variational",
        "Subspace-Hankel": "Subspace",
        "Koopman-EDMD": "EDMD",
        "Model-Stitching": "Stitching",
    }
    return aliases.get(clean_method_name(name, source), clean_method_name(name, source))


def spread_log_labels(
    points: list[tuple[float, float, str]],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> list[tuple[float, float, str, float, float]]:
    if not points:
        return []
    log_x_min, log_x_max = np.log10(x_limits)
    log_y_min, log_y_max = np.log10(y_limits)
    x_mid = 0.5 * (log_x_min + log_x_max)
    min_sep = 0.045 * max(log_y_max - log_y_min, 1.0)
    placed: list[tuple[float, float, str, float, float]] = []
    for side in (-1, 1):
        side_points = [
            (np.log10(x), np.log10(y), label, x, y)
            for x, y, label in points
            if (-1 if np.log10(x) < x_mid else 1) == side
        ]
        side_points.sort(key=lambda item: item[1])
        previous_y = log_y_min - min_sep
        for log_x, log_y, label, x, y in side_points:
            near_top = log_y > log_y_max - 0.18 * (log_y_max - log_y_min)
            preferred_y = log_y - 0.055 if near_top else log_y + 0.035
            label_y = min(max(preferred_y, previous_y + min_sep), log_y_max - 0.04)
            previous_y = label_y
            horizontal_offset = 0.075 * (log_x_max - log_x_min)
            label_x = np.clip(log_x + side * horizontal_offset, log_x_min + 0.02, log_x_max - 0.02)
            placed.append((10**label_x, 10**label_y, label, x, y))
    return placed


def split_tradeoff_rows(rows: list[dict[str, str]], threshold: float = TRADEOFF_FAILURE_THRESHOLD) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    passed: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []
    for row in rows:
        if float(row["validation_score"]) > threshold:
            failed.append(row)
        else:
            passed.append(row)
    return passed, failed


def add_failure_callout(ax, failed_rows: list[dict[str, str]], source: str, threshold: float = TRADEOFF_FAILURE_THRESHOLD) -> None:
    if not failed_rows:
        return
    labels = [tradeoff_label(row["method"], source) for row in sorted(failed_rows, key=lambda row: float(row["validation_score"]), reverse=True)]
    shown = ", ".join(labels[:4])
    if len(labels) > 4:
        shown += f", +{len(labels) - 4}"
    ax.text(
        0.98,
        0.98,
        f"failed > {threshold:g}: {shown}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color="0.25",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.88},
    )


def plot_train_time_accuracy() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [row for row in read_csv(shared_results_path()) if is_open_loop_model_row(row)]
    groups = [
        ("direct", "Direct-state benchmark", "#4c78a8"),
        ("mocap", "Mocap-derived benchmark", "#f58518"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), sharey=True)
    passed_scores = [
        max(float(row["validation_score"]), 1e-4)
        for row in rows
        if row.get("state_source") in {"direct", "mocap"} and float(row["validation_score"]) <= TRADEOFF_FAILURE_THRESHOLD
    ]
    y_limits = (
        max(min(passed_scores) * 0.45, 6e-3),
        max(min(max(passed_scores) * 2.2, TRADEOFF_FAILURE_THRESHOLD * 1.1), 0.2),
    )
    for ax, (source, title, color) in zip(axes, groups):
        group_rows = sorted(
            [row for row in rows if row.get("state_source") == source and is_open_loop_model_row(row)],
            key=lambda row: float(row["validation_score"]),
        )
        nominal_rows = [row for row in group_rows if clean_method_name(row["method"], source) == "Nominal"]
        if nominal_rows:
            nominal_score = max(float(nominal_rows[0]["validation_score"]), 1e-4)
            ax.axhline(nominal_score, color="#d62728", linewidth=1.05, linestyle="--", alpha=0.85)
            ax.text(
                0.985,
                nominal_score,
                "Nominal",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=6.4,
                color="#9d1f1f",
                bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            )
        group_rows = [row for row in group_rows if clean_method_name(row["method"], source) != "Nominal"]
        group_rows, failed_rows = split_tradeoff_rows(group_rows)
        add_failure_callout(ax, failed_rows, source)
        if not group_rows:
            continue
        x = [max(float(row["train_elapsed_s"]), 1e-3) for row in group_rows]
        y = [max(float(row["validation_score"]), 1e-4) for row in group_rows]
        sizes = [42.0 + 18.0 * min(float(row["rollout_elapsed_s"]), 12.0) for row in group_rows]
        ax.scatter(x, y, s=sizes, color=color, alpha=0.72, edgecolor="black", linewidth=0.45)
        ax.set_xscale("log")
        ax.set_yscale("log")
        x_limits = (max(min(x) * 0.55, 5e-4), max(x) * 2.4)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        label_points = [(xi, yi, tradeoff_label(row["method"], source)) for row, xi, yi in zip(group_rows, x, y)]
        for label_x, label_y, label, xi, yi in spread_log_labels(label_points, x_limits, y_limits):
            ax.annotate(
                label,
                xy=(xi, yi),
                xytext=(label_x, label_y),
                textcoords="data",
                fontsize=6.1,
                alpha=0.92,
                arrowprops={"arrowstyle": "-", "color": "0.45", "linewidth": 0.35, "alpha": 0.55},
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
            )
        ax.set_title(title)
        ax.set_xlabel("training / solve time [s]")
        ax.text(
            0.02,
            0.98,
            "lower error is better",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("validation score: mean state NRMSE")
    fig.suptitle("Near-trim open-loop cost-error tradeoff")
    fig.tight_layout()
    output = METHOD_FIG / "shared_train_time_accuracy_tradeoff.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_validation_score_comparison() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [row for row in read_csv(shared_results_path()) if is_open_loop_model_row(row)]
    groups = [
        ("direct", "Direct-state validation", "#4c78a8"),
        ("mocap", "Mocap-derived validation", "#f58518"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.2), sharex=True)
    finite_scores = [
        max(float(row["validation_score"]), 1e-4)
        for row in rows
        if row.get("state_source") in {"direct", "mocap"}
    ]
    x_min = max(min(finite_scores) * 0.65, 1e-4)
    x_max = max(finite_scores) * 1.45
    for ax, (source, title, color) in zip(axes, groups):
        group_rows = sorted(
            [row for row in rows if row.get("state_source") == source and is_open_loop_model_row(row)],
            key=lambda row: float(row["validation_score"]),
            reverse=True,
        )
        methods = [clean_method_name(row["method"], source) for row in group_rows]
        scores = np.array([max(float(row["validation_score"]), 1e-4) for row in group_rows])
        y = np.arange(len(methods))
        for yi, score in zip(y, scores):
            ax.plot([x_min, score], [yi, yi], color="0.82", linewidth=1.0, zorder=1)
        ax.scatter(scores, y, s=42, color=color, edgecolor="black", linewidth=0.35, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels(methods, fontsize=7.8)
        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)
        ax.set_title(title)
        ax.grid(True, axis="x", which="both", alpha=0.25)
        ax.text(
            0.02,
            0.03,
            "left is better",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )
    axes[0].set_xlabel("validation score: mean state NRMSE")
    axes[1].set_xlabel("validation score: mean state NRMSE")
    fig.suptitle("Validation score by method and measurement assumption")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output = METHOD_FIG / "shared_validation_score_comparison.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_archived_train_time_accuracy(mode: str, rows: list[dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    groups = [
        ("direct", "Direct states", "#4c78a8"),
        ("mocap", "Mocap-derived states", "#f58518"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), sharey=True)
    passed_scores = [
        max(float(row["validation_score"]), 1e-4)
        for row in rows
        if row.get("state_source") in {"direct", "mocap"}
        and is_open_loop_model_row(row)
        and float(row["validation_score"]) <= TRADEOFF_FAILURE_THRESHOLD
    ]
    y_limits = (
        max(min(passed_scores) * 0.45, 6e-3),
        max(min(max(passed_scores) * 2.2, TRADEOFF_FAILURE_THRESHOLD * 1.1), 0.2),
    ) if passed_scores else (1e-2, TRADEOFF_FAILURE_THRESHOLD * 1.1)
    for ax, (source, title, color) in zip(axes, groups):
        group_rows = sorted(
            [row for row in rows if row.get("state_source") == source and is_open_loop_model_row(row)],
            key=lambda row: float(row["validation_score"]),
        )
        if not group_rows:
            ax.set_axis_off()
            continue
        nominal_rows = [row for row in group_rows if clean_method_name(row["method"], source) == "Nominal"]
        if nominal_rows:
            nominal_score = max(float(nominal_rows[0]["validation_score"]), 1e-4)
            ax.axhline(nominal_score, color="#d62728", linewidth=1.0, linestyle="--", alpha=0.85)
            ax.text(
                0.985,
                nominal_score,
                "Nominal",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=6.2,
                color="#9d1f1f",
                bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            )
        group_rows = [row for row in group_rows if clean_method_name(row["method"], source) != "Nominal"]
        group_rows, failed_rows = split_tradeoff_rows(group_rows)
        add_failure_callout(ax, failed_rows, source)
        if not group_rows:
            continue
        x = [max(float(row["train_elapsed_s"]), 1e-3) for row in group_rows]
        y = [max(float(row["validation_score"]), 1e-4) for row in group_rows]
        sizes = [42.0 + 18.0 * min(float(row["rollout_elapsed_s"]), 12.0) for row in group_rows]
        ax.scatter(x, y, s=sizes, color=color, alpha=0.72, edgecolor="black", linewidth=0.45)
        ax.set_xscale("log")
        ax.set_yscale("log")
        x_limits = (max(min(x) * 0.55, 5e-4), max(x) * 2.4)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        label_points = [(xi, yi, tradeoff_label(row["method"], source)) for row, xi, yi in zip(group_rows, x, y)]
        for label_x, label_y, label, xi, yi in spread_log_labels(label_points, x_limits, y_limits):
            ax.annotate(
                label,
                xy=(xi, yi),
                xytext=(label_x, label_y),
                textcoords="data",
                fontsize=5.9,
                alpha=0.92,
                arrowprops={"arrowstyle": "-", "color": "0.45", "linewidth": 0.32, "alpha": 0.5},
                bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
            )
        ax.set_title(title)
        ax.set_xlabel("training / solve time [s]")
        ax.text(
            0.02,
            0.98,
            "lower error is better",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("validation score: mean state NRMSE")
    fig.suptitle(f"{DATASET_TITLES[mode]} cost-error tradeoff")
    fig.tight_layout()
    output = METHOD_FIG / f"{mode}_shared_train_time_accuracy_tradeoff.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_score_panels() -> None:
    import matplotlib.pyplot as plt

    modes = available_archived_modes()
    if not modes:
        return
    fig, axes = plt.subplots(len(modes), 1, figsize=(9.8, 3.1 * len(modes)), sharex=False)
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        rows = [
            row
            for row in read_csv(archived_path(mode, "shared_method_comparison.csv", METHOD_RESULTS))
            if row.get("state_source") == "direct" and is_open_loop_model_row(row)
        ]
        ordered = sorted(rows, key=lambda row: float(row["validation_score"]))
        methods = [clean_method_name(row["method"], "direct") for row in ordered]
        scores = [float(row["validation_score"]) for row in ordered]
        colors = [
            "#e45756" if method in {"Frequency-LS", "Frequency-CIFER", "Frequency-Welch", "Frequency-Stitching"} else "#4c78a8"
            for method in methods
        ]
        ax.bar(range(len(methods)), scores, color=colors)
        ax.set_yscale("log")
        ax.set_ylabel("mean state NRMSE")
        ax.set_title(f"{DATASET_TITLES[mode]} direct-state validation")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=28, ha="right", fontsize=7.0)
        ax.grid(True, axis="y", which="both", alpha=0.25)
        ax.text(0.01, 0.95, "lower is better", transform=ax.transAxes, va="top", fontsize=8.5)
    fig.tight_layout()
    output = METHOD_FIG / "dataset_direct_validation_score_panels.svg"
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_maneuver_overview() -> None:
    import matplotlib.pyplot as plt

    modes = [mode for mode in DATASET_MODES if (DATASET_OUTPUTS[mode] / "validation.npz").exists()]
    if not modes:
        return
    colors = {
        "open_loop": "#4c78a8",
        "sine_sweep": "#59a14f",
        "aggressive": "#e45756",
        "open_loop_safe": "#9ecae9",
        "sine_sweep_safe": "#8cd17d",
        "aggressive_safe": "#ff9da6",
        "safe_loop": "#f58518",
    }
    fig, axes = plt.subplots(3, 2, figsize=(11.4, 9.0), constrained_layout=True)
    ax_path = axes[0, 0]
    ax_speed = axes[0, 1]
    ax_alpha = axes[1, 0]
    ax_throttle = axes[1, 1]
    ax_theta = axes[2, 0]
    ax_elevator = axes[2, 1]
    summary_rows: list[tuple[str, float, float, float, float, float]] = []
    for mode in modes:
        data = np.load(DATASET_OUTPUTS[mode] / "validation.npz")
        t = data["t"]
        x_true = data["x_true"]
        mocap = data["mocap_true"]
        u_cmd = data["u_cmd"]
        u_act = data["u_act"]
        alpha = x_true[:, :, 1]
        trial_idx = int(np.argmax(np.max(np.abs(alpha), axis=1)))
        path = mocap[trial_idx].copy()
        path[:, 0] -= path[0, 0]
        path[:, 1] -= path[0, 1]
        state = x_true[trial_idx]
        theta = state[:, 1] + state[:, 2]
        thrust_cmd = u_cmd[trial_idx, :, 0]
        thrust_act = u_act[trial_idx, :, 0]
        elevator_cmd = u_cmd[trial_idx, :, 1]
        elevator_act = u_act[trial_idx, :, 1]
        color = colors.get(mode, None)
        label = DATASET_TITLES[mode]
        safe_mode = mode.endswith("_safe") or mode in {"safe_loop", "proprietary_autopilot"}
        linestyle = "--" if safe_mode else "-"

        ax_path.plot(path[:, 0], path[:, 1], color=color, linestyle=linestyle, linewidth=1.25, label=label)
        ax_speed.plot(t, state[:, 0], color=color, linestyle=linestyle, linewidth=1.0)
        ax_alpha.plot(t, np.rad2deg(state[:, 1]), color=color, linestyle=linestyle, linewidth=1.0)
        ax_throttle.plot(t, thrust_cmd, color=color, linestyle=":", linewidth=0.9, alpha=0.85)
        ax_throttle.plot(t, thrust_act, color=color, linestyle=linestyle, linewidth=1.05)
        ax_theta.plot(t, np.rad2deg(theta), color=color, linestyle=linestyle, linewidth=1.0)
        ax_elevator.plot(t, np.rad2deg(elevator_cmd), color=color, linestyle=":", linewidth=0.9, alpha=0.85)
        ax_elevator.plot(t, np.rad2deg(elevator_act), color=color, linestyle=linestyle, linewidth=1.05)

        summary_rows.append(
            (
                label,
                float(np.rad2deg(np.max(np.abs(state[:, 1])))),
                float(np.rad2deg(np.max(np.abs(theta)))),
                float(np.min(state[:, 0])),
                float(np.max(state[:, 0])),
                float(np.max(path[:, 1]) - np.min(path[:, 1])),
            )
        )

    ax_path.set_title("Validation flight paths")
    ax_path.set_xlabel("$p_x-p_x(0)$ [m]")
    ax_path.set_ylabel("$p_z-p_z(0)$ [m]")
    ax_path.grid(True, alpha=0.25)
    ax_path.legend(fontsize=6.8, loc="best")

    ax_speed.set_title("Airspeed")
    ax_speed.set_xlabel("time [s]")
    ax_speed.set_ylabel("$V$ [m/s]")
    ax_speed.grid(True, alpha=0.25)

    ax_alpha.axhline(12.0, color="0.35", linestyle="--", linewidth=0.8, alpha=0.8)
    ax_alpha.axhline(-12.0, color="0.35", linestyle="--", linewidth=0.8, alpha=0.8)
    ax_alpha.text(0.99, 0.90, "stall onset", transform=ax_alpha.transAxes, ha="right", fontsize=7.5, color="0.25")
    ax_alpha.set_title("Angle of attack")
    ax_alpha.set_xlabel("time [s]")
    ax_alpha.set_ylabel(r"$\alpha$ [deg]")
    ax_alpha.grid(True, alpha=0.25)

    ax_throttle.set_title("Thrust command and realized input")
    ax_throttle.set_xlabel("time [s]")
    ax_throttle.set_ylabel("$T$ [N]")
    ax_throttle.grid(True, alpha=0.25)

    ax_theta.set_title("Pitch attitude")
    ax_theta.set_xlabel("time [s]")
    ax_theta.set_ylabel(r"$\theta=\alpha+\gamma$ [deg]")
    ax_theta.grid(True, alpha=0.25)

    ax_elevator.set_title("Elevator command and realized input")
    ax_elevator.set_xlabel("time [s]")
    ax_elevator.set_ylabel(r"$\delta_e$ [deg]")
    ax_elevator.grid(True, alpha=0.25)
    ax_elevator.text(
        0.01,
        0.04,
        "solid/dashed: actuator input, dotted: pilot command",
        transform=ax_elevator.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
    )

    fig.suptitle("Representative validation maneuvers used by the benchmark", fontsize=12.5)
    output = METHOD_FIG / "benchmark_maneuver_overview.svg"
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary_path = METHOD_RESULTS / "benchmark_maneuver_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mode", "max_abs_alpha_deg", "max_abs_theta_deg", "min_speed_mps", "max_speed_mps", "vertical_extent_m"])
        writer.writerows(summary_rows)


def plot_benchmark_problem_matrix() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 6.0)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="0.25", linewidth=1.0))
        ax.text(x + 0.18, y + h - 0.32, title, fontsize=10.5, weight="bold", va="top")
        ax.text(x + 0.18, y + h - 0.78, body, fontsize=8.5, va="top", linespacing=1.25)

    box(
        0.45,
        3.45,
        4.15,
        1.75,
        "Direct-state measurement",
        "Estimator is given noisy V, alpha, gamma, Q.\nUseful diagnostic upper bound.\nDerivative methods are not dominated by mocap preprocessing.",
        "#d8ecff",
    )
    box(
        5.35,
        3.45,
        4.15,
        1.75,
        "Motion-capture measurement",
        "Measured outputs are px, pz, theta at 100 Hz.\nV, alpha, gamma, Q are latent or reconstructed.\nDifferentiation noise and smoothing lag matter.",
        "#ffe8c7",
    )
    box(
        0.45,
        0.65,
        4.15,
        1.75,
        "SAFE off",
        "Pilot command passes through actuator lag, rate limits,\nand saturation before the plant.\nUnknown aerodynamics remain active.",
        "#e4f5d9",
    )
    box(
        5.35,
        0.65,
        4.15,
        1.75,
        "SAFE on",
        "External pilot command is reshaped by a hidden\npitch attitude/rate loop and recovery gate.\nEstimator sees u_cmd, plant sees hidden u_int.",
        "#f4d9ef",
    )

    arrows = [
        ((4.65, 4.32), (5.25, 4.32), "measurement realism"),
        ((4.65, 1.52), (5.25, 1.52), "hidden input dynamics"),
        ((2.52, 3.35), (2.52, 2.48), "same simulator"),
        ((7.42, 3.35), (7.42, 2.48), "same simulator"),
    ]
    for start, end, label in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color="0.25"))
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.12, label, fontsize=8.2, ha="center")

    ax.text(
        5.0,
        5.75,
        "Benchmark axes: what is measured, and whether the hidden SAFE-like loop is active",
        ha="center",
        va="top",
        fontsize=12.0,
        weight="bold",
    )
    ax.text(
        5.0,
        0.15,
        "The same train/validation protocol is applied across cells so method rankings can be interpreted by problem type.",
        ha="center",
        va="bottom",
        fontsize=8.8,
    )
    fig.tight_layout()
    output = METHOD_FIG / "benchmark_problem_matrix.svg"
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_method_score_heatmap() -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    modes = available_archived_modes()
    if not modes:
        return
    values_by_source: dict[str, dict[str, dict[str, float]]] = {"direct": {}, "mocap": {}}
    modes_by_source: dict[str, list[tuple[str, str]]] = {"direct": [], "mocap": []}
    for mode in modes:
        rows = [row for row in read_csv(archived_path(mode, "shared_method_comparison.csv", METHOD_RESULTS)) if is_open_loop_model_row(row)]
        for source in ["direct", "mocap"]:
            source_rows = [row for row in rows if row.get("state_source") == source]
            if not source_rows:
                continue
            modes_by_source[source].append((mode, DATASET_TITLES[mode]))
            for row in source_rows:
                method = clean_method_name(row["method"], source)
                values_by_source[source].setdefault(method, {})[mode] = max(float(row["validation_score"]), 1e-6)

    for source, source_title in [("direct", "Direct-state benchmark"), ("mocap", "Mocap-derived benchmark")]:
        source_values = values_by_source[source]
        source_modes = modes_by_source[source]
        if not source_values or not source_modes:
            continue
        mean_by_method = {
            method: float(np.mean(list(values.values())))
            for method, values in source_values.items()
            if values
        }
        methods = sorted(
            source_values,
            key=lambda method: (mean_by_method.get(method, float("inf")), method),
        )
        columns = [("__mean__", "Mean score")] + source_modes
        matrix = np.full((len(methods), len(columns)), np.nan)
        for row_idx, method in enumerate(methods):
            matrix[row_idx, 0] = mean_by_method.get(method, np.nan)
            for col_idx, (mode, _label) in enumerate(columns):
                if mode == "__mean__":
                    continue
                matrix[row_idx, col_idx] = source_values[method].get(mode, np.nan)

        finite = matrix[np.isfinite(matrix)]
        if finite.size == 0:
            continue
        vmin = max(float(np.nanpercentile(finite, 5)), 1e-4)
        vmax = max(float(np.nanpercentile(finite, 95)), vmin * 10.0)

        fig_height = max(5.8, 0.36 * len(methods) + 2.0)
        fig_width = max(8.8, 1.30 * len(columns) + 2.3)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        masked = np.ma.masked_invalid(matrix)
        image = ax.imshow(masked, aspect="auto", cmap="viridis_r", norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_yticklabels(methods, fontsize=7.6)
        ax.set_xticks(np.arange(len(columns)))
        ax.set_xticklabels([column[1] for column in columns], rotation=32, ha="right", fontsize=7.4)
        ax.set_title(f"Validation trajectory error: {source_title}")
        ax.set_xlabel("benchmark condition")
        ax.set_ylabel("method")
        ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8, alpha=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                if not np.isfinite(value):
                    continue
                text = f"{value:.2g}" if value >= 0.01 else f"{value:.1e}"
                ax.text(
                    col_idx,
                    row_idx,
                    text,
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    color="white" if value > np.sqrt(vmin * vmax) else "black",
                )
        cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("validation score: mean state NRMSE, lower is better")
        fig.tight_layout()
        output = METHOD_FIG / f"method_score_heatmap_{source}.svg"
        fig.savefig(output, bbox_inches="tight")
        fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
        plt.close(fig)


def copy_figure(source: Path, target: Path) -> None:
    # The paper includes generated figures via \includegraphics (no svg
    # package), so the rasterized sibling must travel with every .svg copy.
    shutil.copy2(source, target)
    png_source = source.with_suffix(".png")
    if source.suffix == ".svg" and png_source.exists():
        shutil.copy2(png_source, target.with_suffix(".png"))


def copy_figures() -> None:
    for source_name, target_name in FIGURE_EXPORTS.items():
        source = METHOD_FIG / source_name
        if not source.exists():
            raise SystemExit(f"missing required figure: {source}")
        copy_figure(source, LATEX_FIG / target_name)
    for mode in available_archived_modes():
        for source_name in ARCHIVED_FIGURES:
            source = METHOD_FIG / f"{mode}_{source_name}"
            if source.exists():
                copy_figure(source, LATEX_FIG / f"generated_{mode}_{source_name}")
        tradeoff = METHOD_FIG / f"{mode}_shared_train_time_accuracy_tradeoff.svg"
        if tradeoff.exists():
            copy_figure(tradeoff, LATEX_FIG / f"generated_{mode}_shared_train_time_accuracy_tradeoff.svg")
    panel = METHOD_FIG / "dataset_direct_validation_score_panels.svg"
    if panel.exists():
        copy_figure(panel, LATEX_FIG / "generated_dataset_direct_validation_score_panels.svg")
    for name in [
        "benchmark_problem_matrix.svg",
        "benchmark_maneuver_overview.svg",
        "method_score_heatmap_direct.svg",
        "method_score_heatmap_mocap.svg",
    ]:
        source = METHOD_FIG / name
        if source.exists():
            copy_figure(source, LATEX_FIG / f"generated_{name}")
    for source_name, target_name in SIX_DOF_FIGURE_EXPORTS.items():
        source = METHOD_FIG / source_name
        if source.exists():
            copy_figure(source, LATEX_FIG / target_name)


def latex_assets(_args: argparse.Namespace) -> None:
    LATEX_GENERATED.mkdir(parents=True, exist_ok=True)
    LATEX_FIG.mkdir(parents=True, exist_ok=True)
    clean_latex_generated_assets()
    write_shared_method_table()
    write_experiment_method_tables()
    write_observation_rate_table()
    write_uq_table()
    six_dof_table = METHOD_TABLES / "aircraft6dof_method_comparison.tex"
    if six_dof_table.exists():
        shutil.copy2(six_dof_table, LATEX_GENERATED / "aircraft6dof_method_comparison_table.tex")
    plot_train_time_accuracy()
    plot_validation_score_comparison()
    for mode in available_archived_modes():
        plot_archived_train_time_accuracy(mode, read_csv(archived_path(mode, "shared_method_comparison.csv", METHOD_RESULTS)))
    plot_dataset_score_panels()
    plot_benchmark_problem_matrix()
    plot_maneuver_overview()
    plot_method_score_heatmap()
    copy_figures()
    print(f"Wrote LaTeX tables to {LATEX_GENERATED}")
    print(f"Copied generated figures to {LATEX_FIG}")


# Unstabilized high-excitation maneuvers use short flight-test records: with
# the realistic (lightly damped phugoid) dynamics, long open-loop aggressive
# or sweep trials leave the flight envelope, exactly like a real aircraft
# flown hands-off. Stabilized (*_safe / autopilot) modes keep full duration.
SHORT_RECORD_DURATIONS = {"aggressive": 10.0, "sine_sweep": 10.0}


def simulate(args: argparse.Namespace) -> None:
    for mode in args.dataset_modes:
        duration = min(float(args.duration), SHORT_RECORD_DURATIONS.get(mode, float(args.duration)))
        command = [
            sys.executable,
            str(ROOT / "simulation" / "generate_dataset.py"),
            "--output",
            str(DATASET_OUTPUTS[mode]),
            "--dataset-mode",
            mode,
            "--train-trials",
            str(args.train_trials),
            "--validation-trials",
            str(args.validation_trials),
            "--duration",
            str(duration),
        ]
        if args.no_plot:
            command.append("--no-plot")
        run(command)


def compact_3dof(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-m",
        "dataset_tools.synthetic_3dof.compact",
        "--output",
        str(args.output),
        "--dataset-mode",
        args.dataset_mode,
        "--train-trials",
        str(args.train_trials),
        "--validation-trials",
        str(args.validation_trials),
        "--duration",
        str(args.duration),
        "--dt",
        str(args.dt),
        "--seed",
        str(args.seed),
    ]
    run(command)


def simulate_6dof(args: argparse.Namespace) -> None:
    modes = list(getattr(args, "dataset_modes", None) or [args.dataset_mode])
    for mode in modes:
        output = args.output if len(modes) == 1 and getattr(args, "output", None) is not None else SIX_DOF_DATASET_OUTPUTS[mode]
        command = [
            sys.executable,
            "-m",
            "models.aircraft6dof.generate_dataset",
            "--output",
            str(output),
            "--train-trials",
            str(args.train_trials),
            "--validation-trials",
            str(args.validation_trials),
            "--duration",
            str(args.duration),
            "--dt",
            str(args.dt),
            "--seed",
            str(args.seed),
            "--dataset-mode",
            mode,
        ]
        if args.no_plot:
            command.append("--no-plot")
        run(command, cwd=ROOT)


def suite_6dof(args: argparse.Namespace) -> None:
    dataset_modes = list(getattr(args, "dataset_modes", None) or [])
    command = [
        sys.executable,
        "-m",
        "models.aircraft6dof.comparison_suite",
        "--state-source",
        args.state_source,
        "--ridge",
        str(args.ridge),
        "--workers",
        str(args.workers),
        "--results-dir",
        str(METHOD_RESULTS),
        "--fig-dir",
        str(METHOD_FIG),
        "--table-dir",
        str(METHOD_TABLES),
    ]
    if dataset_modes:
        command.append("--datasets")
        command.extend(str(SIX_DOF_DATASET_OUTPUTS[mode]) for mode in dataset_modes)
    else:
        dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
        command.extend(["--dataset", str(dataset)])
    if args.no_plot:
        command.append("--no-plot")
    run(command, cwd=ROOT)


def all_6dof(args: argparse.Namespace) -> None:
    simulate_6dof(args)
    dataset_modes = list(getattr(args, "dataset_modes", None) or [])
    dataset = args.output if getattr(args, "output", None) is not None else SIX_DOF_DATASET_OUTPUTS[args.dataset_mode]
    suite_6dof(
        argparse.Namespace(
            dataset=dataset,
            dataset_modes=dataset_modes,
            state_source=args.state_source,
            ridge=args.ridge,
            workers=args.workers,
            no_plot=args.no_plot,
        )
    )
    latex_assets(args)
    web_data(argparse.Namespace(output=ROOT / "site" / "public" / "data"))
    if args.build:
        build_pdf(args)


def fetch_dataset(args: argparse.Namespace) -> None:
    command = [sys.executable, "-m", "dataset_tools.fetch", args.dataset_id]
    if args.output_dir is not None:
        command.extend(["--output-dir", str(args.output_dir)])
    if args.url is not None:
        command.extend(["--url", args.url])
    run(command)


def process_dataset(args: argparse.Namespace) -> None:
    if args.dataset_id != SPORTCUB_DATASET_ID:
        raise SystemExit(f"unknown dataset processor: {args.dataset_id}")
    command = [
        sys.executable,
        "-m",
        "dataset_tools.sportcub_mocap_4_17_26.process",
        "--data-root",
        str(args.data_root),
        "--steps",
        args.steps,
    ]
    if args.only_cases:
        command.extend(["--only-cases", args.only_cases])
    if args.no_plots:
        command.append("--no-plots")
    run(command)


def canonicalize_dataset(args: argparse.Namespace) -> None:
    if args.dataset_id != SPORTCUB_DATASET_ID:
        raise SystemExit(f"unknown dataset canonicalizer: {args.dataset_id}")
    command = [
        sys.executable,
        "-m",
        "dataset_tools.sportcub_mocap_4_17_26.canonicalize",
        "--data-root",
        str(args.data_root),
        "--output",
        str(args.output),
    ]
    run(command)


def check_data(args: argparse.Namespace) -> None:
    command = [sys.executable, "-m", "dataset_tools.validate_format"]
    if args.dataset:
        command.extend(args.dataset)
    if args.allow_empty:
        command.append("--allow-empty")
    run(command)


def _sportcub_data_root(args: argparse.Namespace) -> Path:
    if getattr(args, "data_root", None) is not None:
        return args.data_root
    standard = WORK_DATA / SPORTCUB_DATASET_ID / "raw"
    nested = standard / "Sports_Cub_Data_17April"
    if nested.exists():
        return nested
    return standard


def _run_sportcub_sysid(args: argparse.Namespace, data_root: Path) -> None:
    raise SystemExit(
        "The legacy Sport Cub scripts have been removed from this repository. "
        "Use the framework 6DOF grey-box implementation or pass --results-csv "
        "with an externally generated OEM result CSV to refresh the exported row."
    )


def _latest_sportcub_results() -> Path:
    candidates = sorted(METHOD_RESULTS.glob("sportcub_oem_6dof_results_*.csv"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise SystemExit(
            "sportcub-real requires --results-csv because the legacy Sport Cub "
            "scripts and their generated OEM CSVs are no longer committed."
        )
    return candidates[-1]


def _wrap_radians(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def _sportcub_result_metrics(path: Path) -> dict[str, float | int]:
    rows = read_csv(path)
    val_rows = [row for row in rows if row.get("split") == "val"]
    if not val_rows:
        raise SystemExit(f"{path} has no validation rows")
    pos_err = np.array(
        [
            [
                float(row["pN_sim"]) - float(row["pN_meas"]),
                float(row["pE_sim"]) - float(row["pE_meas"]),
                float(row["pD_sim"]) - float(row["pD_meas"]),
            ]
            for row in val_rows
        ],
        dtype=float,
    )
    att_err = _wrap_radians(
        np.array(
            [
                [
                    float(row["phi_sim_rad"]) - float(row["phi_meas_rad"]),
                    float(row["theta_sim_rad"]) - float(row["theta_meas_rad"]),
                    float(row["psi_sim_rad"]) - float(row["psi_meas_rad"]),
                ]
                for row in val_rows
            ],
            dtype=float,
        )
    )
    train_samples = sum(1 for row in rows if row.get("split") == "train")
    pos_rmse_axis = np.sqrt(np.mean(pos_err**2, axis=0))
    att_rmse_axis = np.sqrt(np.mean(att_err**2, axis=0))
    pos_rmse = float(np.sqrt(np.mean(pos_err**2)))
    att_rmse = float(np.sqrt(np.mean(att_err**2)))
    score = 0.5 * (pos_rmse / 1.0) + 0.5 * (att_rmse / np.deg2rad(10.0))
    return {
        "validation_score": score,
        "train_samples": train_samples,
        "validation_samples": len(val_rows),
        "rmse_position_m": pos_rmse,
        "rmse_mocap_position_m": pos_rmse,
        "rmse_quaternion": att_rmse,
        "rmse_mocap_quaternion": att_rmse,
        "rmse_pN_m": float(pos_rmse_axis[0]),
        "rmse_pE_m": float(pos_rmse_axis[1]),
        "rmse_pD_m": float(pos_rmse_axis[2]),
        "rmse_phi_rad": float(att_rmse_axis[0]),
        "rmse_theta_rad": float(att_rmse_axis[1]),
        "rmse_psi_rad": float(att_rmse_axis[2]),
    }


def export_sportcub_real(args: argparse.Namespace) -> None:
    from dataset_tools.registry import load_manifest, source_url

    data_root = _sportcub_data_root(args)
    if args.run_sysid:
        _run_sportcub_sysid(args, data_root)
    output = METHOD_RESULTS / "sportcub_mocap_4_17_26_method_comparison.csv"
    if args.results_csv is None and output.exists():
        web_data(argparse.Namespace())
        print(f"Using existing {output}")
        return
    result_path = Path(args.results_csv) if args.results_csv is not None else _latest_sportcub_results()
    if not result_path.is_absolute():
        result_path = ROOT / result_path
    result_path = result_path.resolve()
    manifest = load_manifest(SPORTCUB_DATASET_ID)
    metrics = _sportcub_result_metrics(result_path)
    source = source_url(manifest) or ""
    output.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "scenario": SPORTCUB_DATASET_ID,
        "scenario_title": manifest["title"],
        "model_family": "aircraft6dof",
        "method": "6DOF-GreyBoxOEM-EEMInit",
        "description": "6DOF grey-box OEM with equation-error warm start and mocap output residuals.",
        "implementation_status": "provisional",
        "backend": "numpy-casadi-ipopt",
        "state_source": "mocap",
        "input_channel": "u_cmd",
        "evaluation_mode": "held_out_measured_output",
        "training_scenario": f"{SPORTCUB_DATASET_ID}_train",
        "validation_scenario": f"{SPORTCUB_DATASET_ID}_val",
        "train_elapsed_s": "",
        "train_cpu_s": "",
        "train_gpu_s": "",
        "gpu_memory_mb": "",
        "rollout_elapsed_s": "",
        "total_elapsed_s": "",
        "train_loss_final": "",
        "decision_variables": 22 + 9 * 9,
        "rmse_V": "",
        "rmse_alpha": "",
        "rmse_gamma": "",
        "rmse_Q": "",
        "rmse_velocity_mps": "",
        "rmse_rates_rad_s": "",
        "mocap_rmse_x_pos": metrics["rmse_pN_m"],
        "mocap_rmse_z_pos": metrics["rmse_pD_m"],
        "mocap_rmse_theta": metrics["rmse_theta_rad"],
        "coeff_residual_rmse_C_L": "",
        "coeff_residual_rmse_C_D": "",
        "coeff_residual_rmse_C_M": "",
        "notes": (
            "Real Sport Cub MoCap provisional dataset; validation score is the mean of "
            "position RMSE normalized by 1 m and Euler-angle RMSE normalized by 10 deg."
        ),
        "dataset_status": manifest["status"],
        "dataset_source_url": source,
        "source_results_csv": relative_or_absolute(result_path, ROOT),
    }
    row.update(metrics)
    fieldnames = list(dict.fromkeys([*row.keys()]))
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    print(f"Wrote {output}")
    web_data(argparse.Namespace(output=ROOT / "site" / "public" / "data"))


def rates(_args: argparse.Namespace) -> None:
    # use the regenerated open_loop dataset; the unsuffixed default directory
    # is not produced by `simulate` and can hold stale data
    run([sys.executable, str(ROOT / "observation_rate_study.py"), "--dataset", str(DATASET_OUTPUTS["open_loop"])])


def suite_command(
    args: argparse.Namespace,
    mode: str,
    results_dir: Path = METHOD_RESULTS,
    table_dir: Path = METHOD_TABLES,
    fig_dir: Path = METHOD_FIG,
) -> list[str]:
    include_methods = getattr(args, "include_methods", None)
    train_mode = None
    if include_methods and len(include_methods) == 1 and include_methods[0] != "all":
        train_mode = METHOD_TRAINING_MODES.get(include_methods[0], "aggressive")
    command = [
        methods_python(),
        str(ROOT / "comparison_suite.py"),
        "--dataset",
        str(DATASET_OUTPUTS[mode]),
        "--input-channel",
        args.input_channel,
        "--state-source",
        args.state_source,
    ]
    if train_mode is not None:
        command.extend(["--train-dataset", str(DATASET_OUTPUTS[train_mode])])
    command.extend(
        [
            "--epochs",
        str(args.epochs),
        "--max-samples",
        str(args.max_samples),
        "--max-oem-trials",
        str(args.max_oem_trials),
        "--oem-stride",
        str(args.oem_stride),
        "--max-oem-nfev",
        str(args.max_oem_nfev),
        "--max-vi-trials",
        str(args.max_vi_trials),
        "--vi-stride",
        str(args.vi_stride),
        "--max-vi-nfev",
        str(args.max_vi_nfev),
        "--frequency-nperseg",
        str(args.frequency_nperseg),
        "--frequency-min-coherence",
        str(args.frequency_min_coherence),
        "--device",
        args.device,
        "--results-dir",
        str(results_dir),
        "--table-dir",
        str(table_dir),
        "--fig-dir",
        str(fig_dir),
        ]
    )
    if include_methods and include_methods != ["all"]:
        command.append("--include-methods")
        command.extend(include_methods)
    if args.skip_oem:
        command.append("--skip-oem")
    return command


def suite(args: argparse.Namespace) -> None:
    clean_suite_artifacts(args.dataset_modes)
    available_cores = os.cpu_count() or 2
    core_limited_jobs = max(1, available_cores - 1)
    task_count = len(build_suite_tasks(args))
    jobs = max(1, min(int(args.jobs), core_limited_jobs, task_count or 1))
    if jobs == 1 and not args.split_methods:
        for mode in args.dataset_modes:
            run(suite_command(args, mode))
            archive_suite_outputs(mode)
    else:
        run_suite_parallel(args, jobs)
    if args.dataset_modes:
        restore_shared_outputs(args.dataset_modes[0])


def run_suite_parallel(args: argparse.Namespace, jobs: int) -> None:
    work_root = WORK / "suite"
    work_root.mkdir(parents=True, exist_ok=True)
    pending = build_suite_tasks(args)
    total_tasks = len(pending)
    running: dict[int, tuple[str, str, str, subprocess.Popen, float, Path, Path, Path]] = {}
    completed: dict[str, dict[str, list[tuple[Path, Path, Path]]]] = {}
    rows: list[dict[str, object]] = []
    suite_start = time.perf_counter()
    next_progress = suite_start
    failed = 0
    try:
        while pending or running:
            while pending and len(running) < jobs:
                task_index = next_launchable_task_index(pending, running, args.max_heavy_workers, args.max_gpu_workers)
                if task_index is None:
                    break
                mode, source, method = pending.pop(task_index)
                mode_root = work_root / f"{mode}_{source}_{method.replace('-', '_')}"
                shutil.rmtree(mode_root, ignore_errors=True)
                results_dir = mode_root / "results"
                table_dir = mode_root / "tables"
                fig_dir = mode_root / "fig"
                results_dir.mkdir(parents=True, exist_ok=True)
                table_dir.mkdir(parents=True, exist_ok=True)
                fig_dir.mkdir(parents=True, exist_ok=True)
                local_args = argparse.Namespace(**vars(args))
                local_args.state_source = source
                local_args.include_methods = ["all"] if method == "all" else [method]
                command = suite_command(local_args, mode, results_dir, table_dir, fig_dir)
                print("+", " ".join(command), flush=True)
                log_path = mode_root / "worker.log"
                log_stream = log_path.open("w")
                log_stream.write("+ " + " ".join(command) + "\n")
                log_stream.flush()
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=worker_env(args.threads_per_worker),
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                running[process.pid] = (mode, source, method, process, time.perf_counter(), results_dir, table_dir, fig_dir, log_stream, log_path)

            now = time.perf_counter()
            if now >= next_progress:
                print_suite_progress(
                    total_tasks=total_tasks,
                    completed_tasks=len(rows),
                    failed_tasks=failed,
                    pending_tasks=len(pending),
                    running=running,
                    suite_start=suite_start,
                )
                next_progress = now + max(1.0, float(args.progress_interval))

            try:
                pid, status, usage = os.wait4(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if pid == 0:
                time.sleep(0.5)
                continue
            if pid not in running:
                continue
            mode, source, method, process, start, results_dir, table_dir, fig_dir, log_stream, log_path = running.pop(pid)
            log_stream.close()
            exit_code = os.waitstatus_to_exitcode(status)
            wall_s = time.perf_counter() - start
            cpu_s = usage.ru_utime + usage.ru_stime
            rows.append({"mode": mode, "source": source, "method": method, "exit_code": exit_code, "wall_s": wall_s, "child_cpu_s": cpu_s})
            if exit_code != 0:
                failed += 1
                print_suite_progress(
                    total_tasks=total_tasks,
                    completed_tasks=len(rows),
                    failed_tasks=failed,
                    pending_tasks=len(pending),
                    running=running,
                    suite_start=suite_start,
                )
                print_worker_log_tail(log_path)
                for _mode, _source, _method, other, *_rest in running.values():
                    other.terminate()
                raise subprocess.CalledProcessError(exit_code, process.args)
            if args.split_sources or args.split_methods:
                completed.setdefault(mode, {}).setdefault(source, []).append((results_dir, table_dir, fig_dir))
                print(f"finished {mode}/{source}/{method}: wall={wall_s:.1f}s child_cpu={cpu_s:.1f}s", flush=True)
            else:
                archive_suite_outputs(mode, results_dir, table_dir, fig_dir)
                print(f"archived {mode}: wall={wall_s:.1f}s child_cpu={cpu_s:.1f}s", flush=True)
        if args.split_sources or args.split_methods:
            for mode in args.dataset_modes:
                archive_split_source_outputs(mode, completed.get(mode, {}))
        print_suite_progress(
            total_tasks=total_tasks,
            completed_tasks=len(rows),
            failed_tasks=failed,
            pending_tasks=len(pending),
            running=running,
            suite_start=suite_start,
        )
    finally:
        for _mode, _source, _method, _process, *_rest, log_stream, _log_path in running.values():
            try:
                log_stream.close()
            except Exception:
                pass
        if rows:
            METHOD_RESULTS.mkdir(parents=True, exist_ok=True)
            with (METHOD_RESULTS / "suite_orchestration.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["mode", "source", "method", "exit_code", "wall_s", "child_cpu_s"])
                writer.writeheader()
                writer.writerows(rows)


def format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def print_worker_log_tail(log_path: Path, lines: int = 80) -> None:
    if not log_path.exists():
        return
    content = log_path.read_text(errors="replace").splitlines()
    print(f"--- tail of failed worker log: {log_path} ---", flush=True)
    for line in content[-lines:]:
        print(line, flush=True)
    print("--- end failed worker log ---", flush=True)


def print_suite_progress(
    *,
    total_tasks: int,
    completed_tasks: int,
    failed_tasks: int,
    pending_tasks: int,
    running: dict[int, tuple],
    suite_start: float,
) -> None:
    elapsed = time.perf_counter() - suite_start
    success_tasks = max(0, completed_tasks - failed_tasks)
    remaining_tasks = max(0, total_tasks - completed_tasks)
    rate = completed_tasks / elapsed if completed_tasks and elapsed > 0 else 0.0
    eta = remaining_tasks / rate if rate > 0 else None
    percent = 100.0 * completed_tasks / total_tasks if total_tasks else 100.0
    active = sorted(
        ((time.perf_counter() - start, mode, source, method) for mode, source, method, _process, start, *_rest in running.values()),
        reverse=True,
    )
    heavy_running = sum(1 for _age, _mode, _source, method in active if method in HEAVY_METHOD_WORKERS)
    gpu_running = sum(1 for _age, _mode, _source, method in active if method in GPU_METHOD_WORKERS)
    active_summary = ", ".join(f"{mode}/{source}/{method} {format_duration(age)}" for age, mode, source, method in active[:5])
    if len(active) > 5:
        active_summary += f", +{len(active) - 5} more"
    if not active_summary:
        active_summary = "none"
    print(
        "[suite] "
        f"{completed_tasks}/{total_tasks} complete ({percent:.1f}%), "
        f"ok={success_tasks}, failed={failed_tasks}, running={len(running)}, pending={pending_tasks}, "
        f"heavy={heavy_running}, gpu={gpu_running}, "
        f"elapsed={format_duration(elapsed)}, eta={format_duration(eta)}, "
        f"active={active_summary}",
        flush=True,
    )


def next_launchable_task_index(
    pending: list[tuple[str, str, str]],
    running: dict[int, tuple],
    max_heavy_workers: int,
    max_gpu_workers: int,
) -> int | None:
    if not pending:
        return None
    heavy_limit = max(1, int(max_heavy_workers))
    gpu_limit = max(1, int(max_gpu_workers))
    heavy_running = sum(1 for mode, source, method, *_rest in running.values() if method in HEAVY_METHOD_WORKERS)
    gpu_running = sum(1 for mode, source, method, *_rest in running.values() if method in GPU_METHOD_WORKERS)
    for index, (_mode, _source, method) in enumerate(pending):
        if method in HEAVY_METHOD_WORKERS and heavy_running >= heavy_limit:
            continue
        if method in GPU_METHOD_WORKERS and gpu_running >= gpu_limit:
            continue
        return index
    return None


def build_suite_tasks(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    sources = ["direct", "mocap"] if args.state_source == "both" and args.split_sources else [args.state_source]
    if args.split_methods:
        methods = list(METHOD_WORKERS) if args.include_methods == ["all"] else list(args.include_methods)
    else:
        methods = ["all"]
    tasks = [(mode, source, method) for mode in args.dataset_modes for source in sources for method in methods]
    if args.heavy_first:
        mode_order = {mode: index for index, mode in enumerate(args.dataset_modes)}
        source_order = {"mocap": 0, "direct": 1}
        method_order = {method: index for index, method in enumerate(METHOD_WORKERS)}
        tasks.sort(
            key=lambda task: (
                0 if task[2] in HEAVY_METHOD_WORKERS else 1,
                source_order.get(task[1], 2) if task[2] in HEAVY_METHOD_WORKERS else mode_order.get(task[0], 999),
                mode_order.get(task[0], 999) if task[2] in HEAVY_METHOD_WORKERS else source_order.get(task[1], 2),
                method_order.get(task[2], 999),
            )
        )
    return tasks


def archive_suite_outputs(
    mode: str,
    results_dir: Path = METHOD_RESULTS,
    table_dir: Path = METHOD_TABLES,
    fig_dir: Path = METHOD_FIG,
) -> None:
    for filename in ARCHIVED_RESULTS:
        source = results_dir / filename
        if source.exists():
            shutil.copy2(source, archived_path(mode, filename, METHOD_RESULTS))
        table_source = table_dir / filename
        if table_source.exists():
            shutil.copy2(table_source, archived_path(mode, filename, METHOD_TABLES))
    for filename in ARCHIVED_FIGURES:
        source = fig_dir / filename
        if source.exists():
            shutil.copy2(source, METHOD_FIG / f"{mode}_{filename}")
            png = source.with_suffix(".png")
            if png.exists():
                shutil.copy2(png, METHOD_FIG / f"{mode}_{Path(filename).with_suffix('.png').name}")


def write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def archive_split_source_outputs(mode: str, source_outputs: dict[str, list[tuple[Path, Path, Path]]]) -> None:
    if not source_outputs:
        return
    for filename in ["shared_method_comparison.csv", "shared_uq_diagnostics.csv"]:
        rows: list[dict[str, str]] = []
        table_rows: list[dict[str, str]] = []
        for source in ("direct", "mocap"):
            path_list = source_outputs.get(source, [])
            if not path_list:
                continue
            for results_dir, table_dir, _fig_dir in path_list:
                result_file = results_dir / filename
                table_file = table_dir / filename
                if result_file.exists():
                    rows.extend(read_csv(result_file))
                if table_file.exists():
                    table_rows.extend(read_csv(table_file))
        write_csv_rows(archived_path(mode, filename, METHOD_RESULTS), rows)
        write_csv_rows(archived_path(mode, filename, METHOD_TABLES), table_rows or rows)

    trace_rows: list[dict[str, object]] = []
    for source in ("direct", "mocap"):
        for results_dir, _table_dir, _fig_dir in source_outputs.get(source, []):
            trace_file = results_dir / "shared_method_traces.json"
            if not trace_file.exists():
                continue
            payload = json.loads(trace_file.read_text())
            rows = payload.get("traces", payload if isinstance(payload, list) else [])
            trace_rows.extend(row for row in rows if isinstance(row, dict))
    if trace_rows:
        trace_output = archived_path(mode, "shared_method_traces.json", METHOD_RESULTS)
        trace_output.parent.mkdir(parents=True, exist_ok=True)
        trace_output.write_text(json.dumps({"traces": trace_rows}, indent=2, sort_keys=True) + "\n")

    for filename in ["shared_frequency_summary.csv", "shared_sindy_coefficients.csv", "shared_symbolic_coefficients.csv"]:
        for source in ("mocap", "direct"):
            for paths in source_outputs.get(source, []):
                if (paths[0] / filename).exists():
                    shutil.copy2(paths[0] / filename, archived_path(mode, filename, METHOD_RESULTS))
                    break
            else:
                continue
            break

    for filename in ARCHIVED_FIGURES:
        for source in ("mocap", "direct"):
            for paths in source_outputs.get(source, []):
                fig_source = paths[2] / filename
                if fig_source.exists():
                    shutil.copy2(fig_source, METHOD_FIG / f"{mode}_{filename}")
                    png = fig_source.with_suffix(".png")
                    if png.exists():
                        shutil.copy2(png, METHOD_FIG / f"{mode}_{Path(filename).with_suffix('.png').name}")
                    break
            else:
                continue
            break
    print(f"archived {mode} from split source workers", flush=True)


def restore_shared_outputs(mode: str) -> None:
    if mode not in available_archived_modes():
        return
    for filename in ARCHIVED_RESULTS:
        archived = archived_path(mode, filename, METHOD_RESULTS)
        if archived.exists():
            shutil.copy2(archived, METHOD_RESULTS / filename)
        table_archived = archived_path(mode, filename, METHOD_TABLES)
        if table_archived.exists():
            shutil.copy2(table_archived, METHOD_TABLES / filename)
    for filename in ARCHIVED_FIGURES:
        archived = METHOD_FIG / f"{mode}_{filename}"
        if archived.exists():
            shutil.copy2(archived, METHOD_FIG / filename)
            png = archived.with_suffix(".png")
            if png.exists():
                shutil.copy2(png, METHOD_FIG / Path(filename).with_suffix(".png").name)


def build_pdf(_args: argparse.Namespace) -> None:
    run([sys.executable, str(LATEX / "paper.py"), "build"], cwd=LATEX)


def web_data(args: argparse.Namespace) -> None:
    manifest = export_web_data(
        root=ROOT,
        output_dir=args.output,
        results_dir=METHOD_RESULTS,
        dataset_modes=DATASET_MODES,
        dataset_titles=DATASET_TITLES,
        method_training_modes=METHOD_TRAINING_MODES,
    )
    print(f"Wrote web benchmark data to {args.output}")
    print(f"Exported {len(manifest['scenarios'])} scenarios at schema {manifest['schema_version']}")


def _plugin_dirs() -> list[Path]:
    plugin_root = METHOD_CODE / "plugins"
    if not plugin_root.exists():
        return []
    return sorted(path.parent for path in plugin_root.glob("*/method.json"))


def check_setup(_args: argparse.Namespace) -> None:
    """Run fast local checks for the website/plugin benchmark setup."""

    from benchmark.registry import all_method_metadata

    py_files = [
        "benchmark/export.py",
        "benchmark/method_api.py",
        "benchmark/registry.py",
        "benchmark/smoke_plugin.py",
        "dataset_tools/fetch.py",
        "dataset_tools/registry.py",
        "dataset_tools/validate_dataset.py",
        "dataset_tools/validate_format.py",
        "dataset_tools/synthetic_3dof/compact.py",
        "dataset_tools/sportcub_mocap_4_17_26/canonicalize.py",
        "dataset_tools/sportcub_mocap_4_17_26/process.py",
        "models/aircraft6dof/model.py",
        "models/aircraft6dof/comparison_suite.py",
        "models/aircraft6dof/smoke.py",
        "results.py",
    ]
    run([sys.executable, "-m", "py_compile", *py_files])
    run([sys.executable, "-m", "dataset_tools.validate_dataset", str(DATASET_TOOLS / SPORTCUB_DATASET_ID)])
    run([sys.executable, "-m", "dataset_tools.validate_format", "--allow-empty"])
    registered_methods = all_method_metadata(METHOD_CODE / "plugins")
    if not registered_methods:
        raise SystemExit("method registry is empty")
    print(f"Registered {len(registered_methods)} methods")
    for plugin_dir in _plugin_dirs():
        run([sys.executable, "-m", "benchmark.smoke_plugin", str(plugin_dir)])
    web_data(argparse.Namespace(output=ROOT / "site" / "public" / "data"))
    manifest_path = ROOT / "site" / "public" / "data" / "manifest.json"
    method_results_path = ROOT / "site" / "public" / "data" / "method_results.json"
    manifest = json.loads(manifest_path.read_text())
    method_results = json.loads(method_results_path.read_text())
    if not manifest.get("scenarios"):
        raise SystemExit("site manifest has no scenarios")
    if not method_results:
        raise SystemExit("site method_results.json has no method rows")
    run([sys.executable, "-m", "models.aircraft6dof.smoke"], cwd=ROOT)
    print("Setup check passed.")
    print(f"Site data: {len(manifest['scenarios'])} scenarios, {len(method_results)} method result rows")


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _choose_port(start: int) -> int:
    port = int(start)
    for candidate in range(port, port + 100):
        if _port_available(candidate):
            return candidate
    raise SystemExit(f"could not find an available port from {port} to {port + 99}")


def serve_site(args: argparse.Namespace) -> None:
    """Serve the static benchmark site locally."""

    web_data(argparse.Namespace(output=ROOT / "site" / "public" / "data"))
    port = _choose_port(args.port)
    print(f"Serving benchmark site at http://127.0.0.1:{port}")
    print("Press Ctrl-C to stop.")
    run([sys.executable, str(ROOT / "site" / "serve.py"), "--port", str(port), "--bind", "127.0.0.1", "--directory", str(ROOT / "site")])


def all_results(args: argparse.Namespace) -> None:
    if not args.skip_simulate:
        simulate(args)
    rates(args)
    suite(args)
    latex_assets(args)
    web_data(argparse.Namespace(output=ROOT / "site" / "public" / "data"))
    if args.build:
        build_pdf(args)


def add_shared_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-modes",
        nargs="+",
        choices=list(DATASET_MODES),
        default=list(DATASET_MODES),
        help="experiment datasets to generate or benchmark",
    )
    parser.add_argument("--train-trials", type=int, default=64)
    parser.add_argument("--validation-trials", type=int, default=16)
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--state-source", choices=["direct", "mocap", "both"], default="both")
    parser.add_argument("--input-channel", choices=["u_act", "u_cmd"], default="u_cmd")
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--max-oem-trials", type=int, default=2)
    parser.add_argument("--oem-stride", type=int, default=10)
    parser.add_argument("--max-oem-nfev", type=int, default=200)
    parser.add_argument("--max-vi-trials", type=int, default=2)
    parser.add_argument("--vi-stride", type=int, default=40)
    parser.add_argument("--max-vi-nfev", type=int, default=25)
    parser.add_argument("--frequency-nperseg", type=int, default=1024)
    parser.add_argument("--frequency-min-coherence", type=float, default=0.08)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-oem", action="store_true")
    parser.add_argument("--include-methods", nargs="*", default=["all"], help="methods to include when running the suite")
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(30, 2 * len(DATASET_MODES) * len(METHOD_WORKERS), (os.cpu_count() or 2) - 2)),
        help="parallel suite workers; default targets 30 workers on a 32-thread workstation and is capped internally",
    )
    parser.add_argument("--threads-per-worker", type=int, default=1, help="BLAS/OpenMP threads per worker process")
    parser.add_argument("--max-heavy-workers", type=int, default=1, help="maximum concurrent memory-heavy method workers")
    parser.add_argument("--max-gpu-workers", type=int, default=2, help="maximum concurrent GPU-training method workers")
    parser.add_argument("--progress-interval", type=float, default=30.0, help="seconds between global suite progress and ETA reports")
    parser.add_argument("--no-heavy-first", dest="heavy_first", action="store_false", help="do not force memory-heavy methods to run before the rest of the suite")
    parser.add_argument("--no-split-sources", dest="split_sources", action="store_false", help="do not split direct and mocap source runs into separate parallel workers")
    parser.add_argument("--no-split-methods", dest="split_methods", action="store_false", help="do not split individual methods into separate parallel workers")
    parser.set_defaults(heavy_first=True)
    parser.set_defaults(split_sources=True)
    parser.set_defaults(split_methods=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_all = sub.add_parser("all", help="generate simulation, rates, suite, LaTeX assets, and optionally the PDF")
    add_shared_options(p_all)
    p_all.add_argument("--skip-simulate", action="store_true")
    p_all.add_argument("--build", action="store_true")
    p_all.set_defaults(func=all_results)

    p_sim = sub.add_parser("simulate", help="generate the synthetic train/validation dataset")
    add_shared_options(p_sim)
    p_sim.set_defaults(func=simulate)

    p_compact3 = sub.add_parser("compact-3dof", help="generate an ignored compact 3DOF NPZ dataset under work/data/")
    p_compact3.add_argument("--output", type=Path, default=WORK_DATA / "longitudinal_3dof_nonlinear_open_loop")
    p_compact3.add_argument("--train-trials", type=int, default=64)
    p_compact3.add_argument("--validation-trials", type=int, default=16)
    p_compact3.add_argument("--duration", type=float, default=40.0)
    p_compact3.add_argument("--dt", type=float, default=0.01)
    p_compact3.add_argument("--seed", type=int, default=7)
    p_compact3.add_argument("--dataset-mode", choices=list(DATASET_MODES), default="open_loop")
    p_compact3.set_defaults(func=compact_3dof)

    p_sim6 = sub.add_parser("simulate-6dof", help="generate the 6DOF train/validation dataset")
    p_sim6.add_argument("--output", type=Path, default=None, help="single-mode output directory; ignored when multiple --dataset-modes are selected")
    p_sim6.add_argument("--train-trials", type=int, default=32)
    p_sim6.add_argument("--validation-trials", type=int, default=8)
    p_sim6.add_argument("--duration", type=float, default=12.0)
    p_sim6.add_argument("--dt", type=float, default=0.02)
    p_sim6.add_argument("--seed", type=int, default=17)
    p_sim6.add_argument("--dataset-mode", choices=list(SIX_DOF_DATASET_MODES), default="aggressive", help="single 6DOF mode used when --dataset-modes is omitted")
    p_sim6.add_argument("--dataset-modes", nargs="+", choices=list(SIX_DOF_DATASET_MODES), default=None, help="generate several 6DOF modes into their standard data directories")
    p_sim6.add_argument("--no-plot", action="store_true")
    p_sim6.set_defaults(func=simulate_6dof)

    p_suite6 = sub.add_parser("suite-6dof", help="run baseline methods on the 6DOF train/validation dataset")
    p_suite6.add_argument("--dataset", type=Path, default=SIX_DOF_DATASET_OUTPUTS["aggressive"])
    p_suite6.add_argument("--dataset-modes", nargs="+", choices=list(SIX_DOF_DATASET_MODES), default=None, help="standard generated 6DOF datasets to aggregate; omit to use --dataset")
    p_suite6.add_argument("--state-source", choices=["direct", "mocap", "both"], default="both")
    p_suite6.add_argument("--ridge", type=float, default=1e-5)
    p_suite6.add_argument("--workers", type=int, default=max(1, min(30, (os.cpu_count() or 2) - 2)))
    p_suite6.add_argument("--no-plot", action="store_true")
    p_suite6.set_defaults(func=suite_6dof)

    p_all6 = sub.add_parser("all-6dof", help="generate 6DOF data, run baseline methods, export LaTeX/site assets, and optionally build")
    p_all6.add_argument("--output", type=Path, default=None)
    p_all6.add_argument("--train-trials", type=int, default=256)
    p_all6.add_argument("--validation-trials", type=int, default=64)
    p_all6.add_argument("--duration", type=float, default=20.0)
    p_all6.add_argument("--dt", type=float, default=0.02)
    p_all6.add_argument("--seed", type=int, default=17)
    p_all6.add_argument("--dataset-mode", choices=list(SIX_DOF_DATASET_MODES), default="aggressive")
    p_all6.add_argument("--dataset-modes", nargs="+", choices=list(SIX_DOF_DATASET_MODES), default=list(SIX_DOF_DATASET_MODES))
    p_all6.add_argument("--state-source", choices=["direct", "mocap", "both"], default="both")
    p_all6.add_argument("--ridge", type=float, default=1e-5)
    p_all6.add_argument("--workers", type=int, default=max(1, min(30, (os.cpu_count() or 2) - 2)))
    p_all6.add_argument("--no-plot", action="store_true")
    p_all6.add_argument("--build", action="store_true")
    p_all6.set_defaults(func=all_6dof)

    p_fetch_dataset = sub.add_parser("fetch-dataset", help="download a contributed dataset payload into work/data")
    p_fetch_dataset.add_argument("dataset_id")
    p_fetch_dataset.add_argument("--output-dir", type=Path, default=None)
    p_fetch_dataset.add_argument("--url", default=None, help="override the manifest URL")
    p_fetch_dataset.set_defaults(func=fetch_dataset)

    p_process_dataset = sub.add_parser("process-dataset", help="run a contributed dataset's raw-data processing pipeline")
    p_process_dataset.add_argument("dataset_id")
    p_process_dataset.add_argument("--data-root", type=Path, default=WORK_DATA / SPORTCUB_DATASET_ID / "raw")
    p_process_dataset.add_argument("--steps", default="1,2,3")
    p_process_dataset.add_argument("--only-cases", default=None)
    p_process_dataset.add_argument("--no-plots", action="store_true")
    p_process_dataset.set_defaults(func=process_dataset)

    p_canonicalize_dataset = sub.add_parser("canonicalize-dataset", help="convert processed dataset segments to flat compact data/<dataset_id>_<split>.npz arrays")
    p_canonicalize_dataset.add_argument("dataset_id")
    p_canonicalize_dataset.add_argument("--data-root", type=Path, default=WORK_DATA / SPORTCUB_DATASET_ID / "raw")
    p_canonicalize_dataset.add_argument("--output", type=Path, default=ROOT / "data")
    p_canonicalize_dataset.set_defaults(func=canonicalize_dataset)

    p_check_data = sub.add_parser("check-data", help="validate committed compact datasets under data/")
    p_check_data.add_argument("dataset", nargs="*")
    p_check_data.add_argument("--allow-empty", action="store_true")
    p_check_data.set_defaults(func=check_data)

    p_sportcub = sub.add_parser("sportcub-real", help="export or rerun the provisional Sport Cub real-data benchmark row")
    p_sportcub.add_argument("--run-sysid", action="store_true", help="rerun EEM/OEM before exporting the result row")
    p_sportcub.add_argument("--data-root", type=Path, default=None, help="processed Sport Cub data root containing case folders")
    p_sportcub.add_argument("--results-csv", type=Path, default=None, help="specific Sport Cub OEM result CSV to summarize")
    p_sportcub.add_argument("--skip-eem", action="store_true", help="when --run-sysid is set, skip the EEM warm-start stage")
    p_sportcub.add_argument("--skip-oem", action="store_true", help="when --run-sysid is set, skip the OEM stage")
    p_sportcub.set_defaults(func=export_sportcub_real)

    p_rates = sub.add_parser("rates", help="run the observation-rate study")
    p_rates.set_defaults(func=rates)

    p_suite = sub.add_parser("suite", help="run the shared method comparison suite")
    add_shared_options(p_suite)
    p_suite.set_defaults(func=suite)

    p_assets = sub.add_parser("latex-assets", help="export current method results into latex/generated and latex/fig")
    p_assets.set_defaults(func=latex_assets)

    p_web = sub.add_parser("web-data", help="export current method results into site/public/data JSON")
    p_web.add_argument("--output", type=Path, default=ROOT / "site" / "public" / "data")
    p_web.set_defaults(func=web_data)

    p_check = sub.add_parser("check-setup", help="run fast checks for the plugin, website, and model setup")
    p_check.set_defaults(func=check_setup)

    p_serve = sub.add_parser("serve-site", help="serve the static benchmark website locally")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=serve_site)

    p_build = sub.add_parser("build", help="build latex/main.pdf")
    p_build.set_defaults(func=build_pdf)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
