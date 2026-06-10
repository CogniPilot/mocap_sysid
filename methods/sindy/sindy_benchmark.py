#!/usr/bin/env python3
"""SINDy benchmark for the shared longitudinal aircraft dataset."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.benchmark import STATE_LABELS, STATE_NAMES, Aircraft, TestCase, make_cases, make_validation_case
from common.metrics import aggregate_trajectory_score, finite_difference_derivative, rmse
from common.paths import FIG_DIR, RESULTS_DIR
from common.plotting import save_figure


@dataclass
class SindyModel:
    coefficients: np.ndarray
    active: np.ndarray
    feature_names: list[str]
    x_mean: np.ndarray
    x_scale: np.ndarray


def smooth_and_derivative(y: np.ndarray, dt: float, window: int, polyorder: int) -> tuple[np.ndarray, np.ndarray]:
    window = min(window, len(y) - (1 - len(y) % 2))
    if window % 2 == 0:
        window -= 1
    min_window = polyorder + 2
    if min_window % 2 == 0:
        min_window += 1
    window = max(window, min_window)
    smoothed = savgol_filter(y, window_length=window, polyorder=polyorder, axis=0, mode="interp")
    derivative = finite_difference_derivative(smoothed, dt)
    return smoothed, derivative


def library(x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, list[str]]:
    v, alpha, gamma, q_rate = x.T
    thrust, elevator = u.T
    columns = [
        np.ones_like(v),
        v,
        alpha,
        gamma,
        q_rate,
        thrust,
        elevator,
        v * alpha,
        v * gamma,
        alpha * q_rate,
        gamma * q_rate,
        q_rate * elevator,
        thrust * alpha,
        alpha**2,
        gamma**2,
        q_rate**2,
        # Trig features carry only the content beyond their low-order Taylor
        # terms; the raw sin/cos columns are numerically collinear with the
        # alpha/gamma polynomial features at benchmark angles.
        np.sin(alpha) - alpha,
        np.sin(gamma) - gamma,
        np.cos(alpha) - 1.0 + 0.5 * alpha**2,
        np.cos(gamma) - 1.0 + 0.5 * gamma**2,
    ]
    names = [
        "1",
        "V",
        "alpha",
        "gamma",
        "Q",
        "T",
        "delta_e",
        "V alpha",
        "V gamma",
        "alpha Q",
        "gamma Q",
        "Q delta_e",
        "T alpha",
        "alpha^2",
        "gamma^2",
        "Q^2",
        "sin(alpha)-alpha",
        "sin(gamma)-gamma",
        "cos(alpha)-1+alpha^2/2",
        "cos(gamma)-1+gamma^2/2",
    ]
    return np.column_stack(columns), names


def sequential_thresholded_ls(
    theta: np.ndarray,
    xdot: np.ndarray,
    threshold: float,
    ridge: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_features = theta.shape[1]
    eye = np.eye(n_features)
    coefficients = np.linalg.solve(theta.T @ theta + ridge * eye, theta.T @ xdot)
    active = np.ones_like(coefficients, dtype=bool)
    for _ in range(iterations):
        active = np.abs(coefficients) >= threshold
        for state_idx in range(xdot.shape[1]):
            keep = active[:, state_idx]
            if not np.any(keep):
                keep[np.argmax(np.abs(coefficients[:, state_idx]))] = True
            lhs = theta[:, keep].T @ theta[:, keep] + ridge * np.eye(np.count_nonzero(keep))
            rhs = theta[:, keep].T @ xdot[:, state_idx]
            coefficients[:, state_idx] = 0.0
            coefficients[keep, state_idx] = np.linalg.solve(lhs, rhs)
    return coefficients, active


def fit_sindy(case: TestCase, dt: float, threshold: float, ridge: float, window: int, polyorder: int) -> SindyModel:
    x_smooth, xdot = smooth_and_derivative(case.y_meas, dt, window, polyorder)
    theta, names = library(x_smooth, case.u_id)
    x_mean = theta.mean(axis=0)
    x_scale = theta.std(axis=0)
    x_scale[x_scale < 1e-12] = 1.0
    theta_scaled = (theta - x_mean) / x_scale
    coefficients_scaled, _ = sequential_thresholded_ls(theta_scaled, xdot, threshold, ridge, iterations=8)
    coefficients = coefficients_scaled / x_scale[:, None]
    intercept = -(x_mean / x_scale) @ coefficients_scaled
    coefficients[0, :] += intercept
    # Report the terms the simulated model actually uses (exact nonzero
    # pattern after de-standardization), not a re-threshold in unscaled units.
    return SindyModel(coefficients=coefficients, active=coefficients != 0.0, feature_names=names, x_mean=x_mean, x_scale=x_scale)


def rhs(model: SindyModel, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    theta, _ = library(x.reshape(1, -1), u.reshape(1, -1))
    return (theta @ model.coefficients).ravel()


def simulate_sindy(model: SindyModel, x0: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    x = np.empty((len(u), 4))
    x[0] = x0
    for k in range(len(u) - 1):
        u0, u1 = u[k], u[k + 1]
        umid = 0.5 * (u0 + u1)
        k1 = rhs(model, x[k], u0)
        k2 = rhs(model, x[k] + 0.5 * dt * k1, umid)
        k3 = rhs(model, x[k] + 0.5 * dt * k2, umid)
        k4 = rhs(model, x[k] + dt * k3, u1)
        x[k + 1] = x[k] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.all(np.isfinite(x[k + 1])) or np.linalg.norm(x[k + 1]) > 1e4:
            x[k + 1 :] = x[k]
            break
    return x


def plot_sindy_trajectories(rows: list[dict[str, object]], cases: list[TestCase], trajectories: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(4, len(cases), figsize=(8.2, 5.6), sharex=True)
    if len(cases) == 1:
        axes = axes[:, None]
    for col, case in enumerate(cases):
        y_hat = trajectories[case.name]
        for row, label in enumerate(STATE_LABELS):
            ax = axes[row, col]
            truth = case.x_true[:, row].copy()
            meas = case.y_meas[:, row].copy()
            pred = y_hat[:, row].copy()
            if row in (1, 2, 3):
                truth = np.rad2deg(truth)
                meas = np.rad2deg(meas)
                pred = np.rad2deg(pred)
            ax.plot(case.t, truth, color="black", linewidth=1.4, label="Truth")
            ax.plot(case.t, meas, color="0.75", linewidth=0.5, alpha=0.75, label="Measured")
            ax.plot(case.t, pred, color="#ff7f0e", linewidth=1.1, label="SINDy")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title(case.name.replace("_", " "))
            if row == 3:
                ax.set_xlabel("time [s]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, FIG_DIR / "sindy_trajectories")


def plot_active_terms(model_by_case: dict[str, SindyModel]) -> None:
    fig, axes = plt.subplots(1, len(model_by_case), figsize=(8.2, 3.6), sharey=True)
    if len(model_by_case) == 1:
        axes = [axes]
    for ax, (case_name, model) in zip(axes, model_by_case.items()):
        active_counts = np.count_nonzero(model.active, axis=1)
        ax.barh(np.arange(len(model.feature_names)), active_counts, color="#ff7f0e")
        ax.set_yticks(np.arange(len(model.feature_names)))
        ax.set_yticklabels(model.feature_names)
        ax.set_xlabel("active state equations")
        ax.set_title(case_name.replace("_", " "))
        ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "sindy_active_terms")


def write_coefficients(model_by_case: dict[str, SindyModel]) -> None:
    output = RESULTS_DIR / "sindy_coefficients.csv"
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "feature", *[f"d{name}/dt" for name in STATE_NAMES]])
        for case_name, model in model_by_case.items():
            for feature, row in zip(model.feature_names, model.coefficients):
                writer.writerow([case_name, feature, *row])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--polyorder", type=int, default=3)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cases = make_cases(args.duration, args.dt, args.seed)
    validation = make_validation_case(args.duration, args.dt, args.seed + 100)
    rows: list[dict[str, object]] = []
    model_by_case: dict[str, SindyModel] = {}
    trajectories: dict[str, np.ndarray] = {}
    for case in cases:
        start = time.perf_counter()
        model = fit_sindy(case, args.dt, args.threshold, args.ridge, args.smooth_window, args.polyorder)
        elapsed = time.perf_counter() - start
        y_hat = simulate_sindy(model, case.y_meas[0], case.u_id, args.dt)
        val_hat = simulate_sindy(model, validation.y_meas[0], validation.u_id, args.dt)
        model_by_case[case.name] = model
        trajectories[case.name] = y_hat
        row = {
            "case": case.name,
            "method": "SINDy",
            "elapsed_s": elapsed,
            "active_terms": int(np.count_nonzero(model.active)),
            "train_score": aggregate_trajectory_score(y_hat, case.x_true),
            "validation_score": aggregate_trajectory_score(val_hat, validation.x_true),
        }
        row.update({f"rmse_{name}": value for name, value in zip(STATE_NAMES, rmse(y_hat, case.x_true))})
        rows.append(row)
        print(f"{case.name}: score={row['train_score']:.4g}, validation={row['validation_score']:.4g}, active={row['active_terms']}")

    with (RESULTS_DIR / "sindy_fit_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_coefficients(model_by_case)
    plot_sindy_trajectories(rows, cases, trajectories)
    plot_active_terms(model_by_case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
