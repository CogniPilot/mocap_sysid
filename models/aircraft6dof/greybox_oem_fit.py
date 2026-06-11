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
    STATE_NAMES_LAG,
    build_casadi_dynamics_lag,
    SPORTCUB_PARAMETER_NAMES,
    STATE_NAMES_EULER,
    build_casadi_dynamics,
    euler_from_quaternion,
    normalize_quaternion,
    quaternion_from_euler,
    sportcub_greybox_spec,
    wrap_angle_np,
)

LAG_PARAMETERS = ("TAUS", "TAUM", "KB")
# The production fit uses the 22 aerodynamic parameters; the lag/battery
# extension is gated out (see the paper) and its parameters are excluded so
# their zero-gradient columns cannot poison the covariance.
GREYBOX_FIT_NAMES = tuple(n for n in SPORTCUB_PARAMETER_NAMES if n not in LAG_PARAMETERS)

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

    setup = [row for row in spec.default_parameter_setup() if row[0] in GREYBOX_FIT_NAMES]
    lower = np.array([row[1] for row in setup])
    theta0 = np.array([row[2] for row in setup])
    upper = np.array([row[3] for row in setup])

    def full_vector(theta: np.ndarray) -> np.ndarray:
        values = dict(zip(GREYBOX_FIT_NAMES, theta))
        for name in LAG_PARAMETERS:
            values[name] = spec.default_parameter_bounds[name].initial
        return spec.full_parameter_vector(np.array([values[n] for n in SPORTCUB_PARAMETER_NAMES]))

    def residual(theta: np.ndarray) -> np.ndarray:
        pred = chunk_rollouts(rk4, full_vector(theta), x_train, u_train, stride)
        return output_residuals(pred, x_train, stride, sigma)

    fit = least_squares(residual, theta0, bounds=(lower, upper), max_nfev=max_nfev, verbose=1)
    theta = fit.x
    full = full_vector(theta)
    # Cramer-Rao bounds and parameter correlations from the Jacobian at the
    # solution (residual coloring uncorrected: optimistic lower bounds).
    jac = fit.jac
    dof = max(jac.shape[0] - jac.shape[1], 1)
    s2 = 2.0 * fit.cost / dof
    cr_std = np.full(len(theta), np.nan)
    couplings: list[dict[str, object]] = []
    try:
        cov = s2 * np.linalg.inv(jac.T @ jac + 1e-12 * np.eye(jac.shape[1]))
        cr_std = np.sqrt(np.maximum(np.diag(cov), 0.0))
        corr = cov / np.outer(cr_std, cr_std)
        for i in range(len(theta)):
            for j in range(i + 1, len(theta)):
                if abs(corr[i, j]) > 0.9:
                    couplings.append({"a": GREYBOX_FIT_NAMES[i], "b": GREYBOX_FIT_NAMES[j], "r": float(corr[i, j])})
        couplings.sort(key=lambda d: -abs(d["r"]))
    except np.linalg.LinAlgError:
        pass
    return {
        "spec": spec,
        "parameter_names": list(GREYBOX_FIT_NAMES),
        "theta": theta,
        "theta_full": full,
        "cr_std": cr_std,
        "couplings": couplings,
        "lower": lower,
        "upper": upper,
        "train_nrmse_initial": rollout_nrmse(rk4, full_vector(theta0), x_train, u_train, stride),
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


def make_greybox_quat_step(theta_full: np.ndarray, dt: float):
    """One-step grey-box map on benchmark 13-states (quaternion in/out)."""
    spec = sportcub_greybox_spec()
    _dynamics, rk4 = build_casadi_dynamics(spec, dt)

    def step(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        xe = quat_states_to_euler(np.asarray(x)[None, None, :])[0, 0]
        xe = np.asarray(rk4(xe, np.asarray(u)[OEM_CONTROL_ORDER], theta_full)).ravel()
        return euler_states_to_quat(xe[None, None, :])[0, 0]

    return step


def greybox_one_step(spec, theta_full: np.ndarray, xk: np.ndarray, uk: np.ndarray, dt: float) -> np.ndarray:
    """Batch one-step grey-box predictions for residual targets (n, T, 13)."""
    step = make_greybox_quat_step(np.asarray(theta_full), dt)
    out = np.empty((xk.shape[0], xk.shape[1], 13))
    for trial in range(xk.shape[0]):
        for k in range(xk.shape[1]):
            out[trial, k] = step(xk[trial, k], uk[trial, k])
    return out




def lag_chunk_inputs(split: suite.Split6DOF, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Augmented (16-state, 5-input) chunk arrays for the lag grey-box.

    Lag states initialize at the commanded values (the lead-in is
    quasi-steady) and the fifth input channel is the flight time at each
    sample, parsed from the chunk name's full-rate start index, for the
    battery thrust-decay term.
    """
    import re

    spec = sportcub_greybox_spec()
    deg = np.pi / 180.0
    x12 = quat_states_to_euler(split.x_true)
    u4 = split.u_cmd[:, :, OEM_CONTROL_ORDER]
    n, length = x12.shape[:2]
    x16 = np.concatenate([x12, np.zeros((n, length, 4))], axis=2)
    u5 = np.concatenate([u4, np.zeros((n, length, 1))], axis=2)
    throws = np.array([
        spec.max_deflection_deg["elevator"],
        spec.max_deflection_deg["aileron"],
        spec.max_deflection_deg["rudder"],
    ]) * deg
    for trial in range(n):
        # u4 order is CONTROL_NAMES_OEM = (aileron, elevator, throttle, rudder)
        ail0, elev0, thr0, rud0 = u4[trial, 0]
        x16[trial, :, 12] = throws[0] * elev0
        x16[trial, :, 13] = throws[1] * ail0
        x16[trial, :, 14] = throws[2] * rud0
        x16[trial, :, 15] = np.clip(thr0, 0.0, 1.0)
        name = str(split.segment_names[trial]) if split.segment_names is not None else ""
        match = re.search(r"__manual_(\d+)(?:_w(\d+))?", name)
        start = int(match.group(1)) if match else 0
        window = int(match.group(2)) - 1 if match and match.group(2) else 0
        t0 = (start + window * length) * dt
        u5[trial, :, 4] = t0 + np.arange(length) * dt
    return x16, u5


def fit_greybox_lag(split: suite.Split6DOF, dt: float, max_nfev: int = 150) -> dict:
    """Fit the 25-parameter lag-and-battery grey-box on benchmark chunks."""
    spec = sportcub_greybox_spec()
    x_train, u_train = lag_chunk_inputs(split, dt)
    stride = max(1, int(round((1.0 / MODEL_RATE_HZ) / dt)))
    _dynamics, rk4 = build_casadi_dynamics_lag(spec, dt * stride)
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
    full = spec.full_parameter_vector(fit.x)
    return {
        "spec": spec,
        "theta": fit.x,
        "theta_full": full,
        "lower": lower,
        "upper": upper,
        "rk4": rk4,
        "train_nrmse": rollout_nrmse(rk4, full, x_train, u_train, stride),
        "cost": float(fit.cost),
        "nfev": int(fit.nfev),
    }




def fit_greybox_ekf(x_quat: np.ndarray, u_cmd: np.ndarray, dt: float, stride: int = 4) -> dict:
    """Filter-error parameter estimation: augmented-state EKF over the grey-box.

    Port of the 3DOF filter-error EKF: the 12 Euler states are augmented with
    the 22 aerodynamic parameters, propagated chunk by chunk over the manual
    training windows with CasADi step and parameter Jacobians, and corrected
    by Joseph-form updates against the measured states. The parameter
    covariance carries across chunks; the state covariance resets at each
    chunk start (independent measured initial conditions). Validation is a
    frozen-theta open-loop rollout elsewhere -- no filtering touches
    validation data.
    """
    import casadi as ca

    spec = sportcub_greybox_spec()
    x_train = quat_states_to_euler(x_quat)
    u_train = u_cmd[:, :, OEM_CONTROL_ORDER]
    _dynamics, rk4 = build_casadi_dynamics(spec, dt * stride)

    n_x, n_th = 12, len(GREYBOX_FIT_NAMES)
    x_sym = ca.SX.sym("x", n_x)
    u_sym = ca.SX.sym("u", 4)
    th_sym = ca.SX.sym("th", n_th)
    lag_init = [spec.default_parameter_bounds[n].initial for n in LAG_PARAMETERS]
    p_full = ca.vertcat(ca.DM(spec.fixed_parameter_vector()), th_sym, ca.DM(lag_init))
    x_next = rk4(x_sym, u_sym, p_full)
    step_jac = ca.Function(
        "greybox_ekf_step", [x_sym, u_sym, th_sym],
        [x_next, ca.jacobian(x_next, x_sym), ca.jacobian(x_next, th_sym)],
    )

    bounds = [spec.default_parameter_bounds[n] for n in GREYBOX_FIT_NAMES]
    theta = np.array([b.initial for b in bounds])
    lower = np.array([b.lower for b in bounds])
    upper = np.array([b.upper for b in bounds])
    span = upper - lower

    # Measurement: all 12 mocap-derived Euler states. Stds reflect the
    # derived-state quality (positions mm-class, velocities/rates derived).
    meas_std = np.array([0.05, 0.05, 0.05, 0.15, 0.15, 0.15, 0.02, 0.02, 0.03, 0.10, 0.10, 0.10])
    measurement_cov = np.diag(meas_std**2)
    process_std = np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.005, 0.005, 0.005, 0.05, 0.05, 0.05])
    process_cov = np.diag(process_std**2)
    theta_process_cov = np.diag((1e-4 * span) ** 2)
    state_initial_cov = np.diag((2.0 * meas_std) ** 2)
    theta_cov = np.diag((span / 4.0) ** 2)

    n_aug = n_x + n_th
    h_mat = np.zeros((n_x, n_aug))
    h_mat[:, :n_x] = np.eye(n_x)
    eye_aug = np.eye(n_aug)
    updates = 0
    for trial in range(x_train.shape[0]):
        x = x_train[trial, 0].copy()
        p_cov = np.zeros((n_aug, n_aug))
        p_cov[:n_x, :n_x] = state_initial_cov
        p_cov[n_x:, n_x:] = theta_cov
        for k in range(0, x_train.shape[1] - stride, stride):
            x_pred_dm, f_x_dm, f_th_dm = step_jac(x, u_train[trial, k], theta)
            x_pred = np.asarray(x_pred_dm).ravel()
            transition = eye_aug.copy()
            transition[:n_x, :n_x] = np.asarray(f_x_dm)
            transition[:n_x, n_x:] = np.asarray(f_th_dm)
            q_aug = np.zeros((n_aug, n_aug))
            q_aug[:n_x, :n_x] = process_cov
            q_aug[n_x:, n_x:] = theta_process_cov
            p_pred = transition @ p_cov @ transition.T + q_aug
            innovation = x_train[trial, k + stride] - x_pred
            innovation[8] = np.arctan2(np.sin(innovation[8]), np.cos(innovation[8]))
            s_cov = h_mat @ p_pred @ h_mat.T + measurement_cov
            gain = np.linalg.solve(s_cov, h_mat @ p_pred).T
            correction = gain @ innovation
            x = x_pred + correction[:n_x]
            theta = np.clip(theta + correction[n_x:], lower, upper)
            joseph = eye_aug - gain @ h_mat
            p_cov = joseph @ p_pred @ joseph.T + gain @ measurement_cov @ gain.T
            updates += 1
        theta_cov = p_cov[n_x:, n_x:]

    values = dict(zip(GREYBOX_FIT_NAMES, theta))
    for name, init in zip(LAG_PARAMETERS, lag_init):
        values[name] = init
    theta_full = spec.full_parameter_vector(np.array([values[n] for n in SPORTCUB_PARAMETER_NAMES]))
    return {
        "spec": spec,
        "theta": theta,
        "theta_full": theta_full,
        "theta_std": np.sqrt(np.maximum(np.diag(theta_cov), 0.0)),
        "updates": updates,
    }


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
    for name, value, lo, hi in zip(GREYBOX_FIT_NAMES, theta, fit["lower"], fit["upper"]):
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
        "parameter_names": list(GREYBOX_FIT_NAMES),
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
