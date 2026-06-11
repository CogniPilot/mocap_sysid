#!/usr/bin/env python3
"""Fit the framework Sport Cub 6DOF grey-box model to the 2026-05-22 flights.

The browser free-runs and manual-segment predictions are limited by one-step
affine regressions; this fits the physical lumped-parameter grey-box
(``SportCubGreyboxConfig``) by multiple shooting over the benchmark's manual
training windows: each gap-free, bias-corrected chunk is rolled out with the
framework's RK4 from its measured initial state, and the 22 aerodynamic
parameters minimize the position/attitude output residuals across all chunks
simultaneously (scipy TRF within the spec's bounds, warm-started at the spec
initials).

Outputs ``results/sportcub_greybox_params.json`` (fitted parameters plus
validation scores) for export to the browser and reuse by the benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from benchmark.paths import RESULTS
from . import comparison_suite as suite
from .model import INPUT_NAMES
from .greybox import (
    CONTROL_NAMES_OEM,
    SPORTCUB_PARAMETER_NAMES,
    STATE_NAMES_EULER,
    build_casadi_dynamics,
    euler_from_quaternion,
    normalize_quaternion,
    quaternion_from_euler,
    sportcub_greybox_spec,
    wrap_angle_np,
)

TRAIN_DEFAULT = Path("data/sportcub_mocap_5_22_26_train.npz")
VALIDATION_DEFAULT = Path("data/sportcub_mocap_5_22_26_validation.npz")
MODEL_RATE_HZ = 60.0


# Benchmark u_cmd channels follow model.INPUT_NAMES (throttle, elevator,
# aileron, rudder) — not the dataset's canonical control_meas order.
OEM_CONTROL_ORDER = [INPUT_NAMES.index(name) for name in CONTROL_NAMES_OEM]


def quat_states_to_euler(x_quat: np.ndarray) -> np.ndarray:
    """Convert benchmark 13-state (quaternion) arrays to grey-box Euler states."""
    n, length = x_quat.shape[:2]
    x = np.empty((n, length, len(STATE_NAMES_EULER)))
    x[:, :, 0:6] = x_quat[:, :, 0:6]
    for trial in range(n):
        for k in range(length):
            x[trial, k, 6:9] = euler_from_quaternion(normalize_quaternion(x_quat[trial, k, 6:10]))
    x[:, :, 9:12] = x_quat[:, :, 10:13]
    return x


def euler_states_to_quat(x_euler: np.ndarray) -> np.ndarray:
    """Convert grey-box Euler states back to benchmark 13-state arrays."""
    n, length = x_euler.shape[:2]
    x = np.empty((n, length, 13))
    x[:, :, 0:6] = x_euler[:, :, 0:6]
    for trial in range(n):
        for k in range(length):
            x[trial, k, 6:10] = quaternion_from_euler(x_euler[trial, k, 6:9])
    x[:, :, 10:13] = x_euler[:, :, 9:12]
    return x


def chunk_rollouts(rk4, params_full: np.ndarray, x: np.ndarray, u: np.ndarray, stride: int) -> np.ndarray:
    n, length = x.shape[:2]
    steps = (length - 1) // stride
    pred = np.empty((n, steps + 1, x.shape[2]))
    for trial in range(n):
        state = x[trial, 0].copy()
        pred[trial, 0] = state
        for k in range(steps):
            state = np.asarray(rk4(state, u[trial, k * stride], params_full)).ravel()
            pred[trial, k + 1] = state
    return pred


def output_residuals(pred: np.ndarray, x: np.ndarray, stride: int, sigma: np.ndarray) -> np.ndarray:
    meas = x[:, ::stride][:, : pred.shape[1]]
    res = (pred[:, :, [0, 1, 2, 6, 7, 8]] - meas[:, :, [0, 1, 2, 6, 7, 8]]) / sigma
    res[:, :, 5] = wrap_angle_np(res[:, :, 5] * sigma[5]) / sigma[5]
    return res.ravel()


def rollout_nrmse(rk4, params_full: np.ndarray, x: np.ndarray, u: np.ndarray, stride: int) -> float:
    pred = chunk_rollouts(rk4, params_full, x, u, stride)
    meas = x[:, ::stride][:, : pred.shape[1]]
    pred_w = pred.copy()
    pred_w[:, :, 8] = meas[:, :, 8] + wrap_angle_np(pred[:, :, 8] - meas[:, :, 8])
    scale = np.ptp(meas.reshape(-1, meas.shape[-1]), axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    rmse = np.sqrt(np.mean((pred_w - meas) ** 2, axis=(0, 1)))
    return float(np.mean(rmse / scale))


def fit_greybox(x_quat: np.ndarray, u_cmd: np.ndarray, dt: float, max_nfev: int = 120) -> dict:
    """Fit the Sport Cub grey-box on benchmark chunks (quaternion 13-states).

    Returns the fitted free parameters plus everything needed to rebuild the
    model: the spec, scores on the training chunks, and the optimizer stats.
    """
    spec = sportcub_greybox_spec()
    x_train = quat_states_to_euler(x_quat)
    u_train = u_cmd[:, :, OEM_CONTROL_ORDER]
    stride = max(1, int(round((1.0 / MODEL_RATE_HZ) / dt)))
    _dynamics, rk4 = build_casadi_dynamics(spec, dt * stride)
    sigma = np.asarray(spec.output_sigma)

    setup = spec.default_parameter_setup()
    lower = np.array([row[1] for row in setup])
    theta0 = np.array([row[2] for row in setup])
    upper = np.array([row[3] for row in setup])

    def residual(theta: np.ndarray) -> np.ndarray:
        full = spec.full_parameter_vector(theta)
        pred = chunk_rollouts(rk4, full, x_train, u_train, stride)
        return output_residuals(pred, x_train, stride, sigma)

    fit = least_squares(residual, theta0, bounds=(lower, upper), max_nfev=max_nfev, verbose=1)
    theta = fit.x
    full = spec.full_parameter_vector(theta)
    return {
        "spec": spec,
        "theta": theta,
        "theta_full": full,
        "lower": lower,
        "upper": upper,
        "train_nrmse_initial": rollout_nrmse(rk4, spec.full_parameter_vector(theta0), x_train, u_train, stride),
        "train_nrmse": rollout_nrmse(rk4, full, x_train, u_train, stride),
        "model_rate_hz": 1.0 / (dt * stride),
        "cost": float(fit.cost),
        "nfev": int(fit.nfev),
    }


def greybox_rollout_quat(spec, theta_full: np.ndarray, x0_quat: np.ndarray, u_cmd: np.ndarray, dt: float) -> np.ndarray:
    """Roll the grey-box from quaternion initial states over every sample.

    Integrates at the data rate and returns benchmark 13-state predictions
    aligned with ``u_cmd`` so the suite can score it like the other methods.
    """
    _dynamics, rk4 = build_casadi_dynamics(spec, dt)
    n, length = u_cmd.shape[:2]
    u = u_cmd[:, :, OEM_CONTROL_ORDER]
    pred = np.empty((n, length, len(STATE_NAMES_EULER)))
    pred[:, 0, :] = quat_states_to_euler(x0_quat[:, None, :])[:, 0, :]
    for trial in range(n):
        state = pred[trial, 0]
        for k in range(length - 1):
            state = np.asarray(rk4(state, u[trial, k], theta_full)).ravel()
            if not np.all(np.isfinite(state)):
                pred[trial, k + 1 :, :] = pred[trial, k]
                break
            pred[trial, k + 1] = state
    return euler_states_to_quat(pred)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument("--validation", type=Path, default=VALIDATION_DEFAULT)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--max-nfev", type=int, default=120)
    args = parser.parse_args()

    segments, dt = suite.load_segments(args.train)
    train = suite.trim_to_input_onset(segments, dt, estimate_bias=True, label="greybox train")
    print(f"fitting on {train.x_true.shape[0]} chunks ({train.x_true.shape[1]} samples each)")
    fit = fit_greybox(train.x_true, train.u_cmd, dt, max_nfev=args.max_nfev)
    spec = fit["spec"]
    theta = fit["theta"]
    stride = max(1, int(round((1.0 / MODEL_RATE_HZ) / dt)))
    _dynamics, rk4 = build_casadi_dynamics(spec, dt * stride)

    val_segments, val_dt = suite.load_segments(args.validation)
    validation = suite.trim_to_input_onset(val_segments, val_dt, estimate_bias=True, label="greybox validation")
    x_val = quat_states_to_euler(validation.x_true)
    u_val = validation.u_cmd[:, :, OEM_CONTROL_ORDER]

    scores = {
        "train_nrmse_initial": fit["train_nrmse_initial"],
        "train_nrmse": fit["train_nrmse"],
        "validation_nrmse": rollout_nrmse(rk4, fit["theta_full"], x_val, u_val, stride),
    }
    for name, value in scores.items():
        print(f"  {name}: {value:.4f}")
    at_bounds = []
    for name, value, lo, hi in zip(SPORTCUB_PARAMETER_NAMES, theta, fit["lower"], fit["upper"]):
        # Within 1% of the box span counts as pinned: TRF stops just inside.
        pinned = min(value - lo, hi - value) < 0.01 * (hi - lo)
        if pinned:
            at_bounds.append(name)
        print(f"  {name:6s} = {value:+10.4f}{'  (at bound)' if pinned else ''}")
    if at_bounds:
        print(f"  WARNING: parameters pinned at bounds: {', '.join(at_bounds)} — widen the spec box")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    output = args.results_dir / "sportcub_greybox_params.json"
    output.write_text(json.dumps({
        "parameter_names": list(SPORTCUB_PARAMETER_NAMES),
        "parameters": [float(v) for v in theta],
        "fixed_parameters": spec.fixed_parameters,
        "max_deflection_deg": spec.max_deflection_deg,
        "model_rate_hz": fit["model_rate_hz"],
        "parameters_at_bounds": at_bounds,
        "scores": scores,
        "cost": fit["cost"],
        "nfev": fit["nfev"],
    }, indent=2) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
