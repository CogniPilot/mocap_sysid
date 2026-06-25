#!/usr/bin/env python3
"""Fit the framework Sport Cub 6DOF grey-box model to the 2026-05-22 flights.

The browser free-runs and manual-segment predictions are limited by one-step
affine regressions; this fits the physical lumped-parameter grey-box
(``SportCubGreyboxConfig``) by multiple shooting over the benchmark's manual
training windows: each gap-free, bias-corrected chunk is rolled out with the
Rumoca-generated CasADi RK4 from its measured initial state, and the fixed-wing
plant parameters minimize the position/attitude output residuals across all
chunks simultaneously with Ipopt within the spec's bounds, warm-started at the
spec initials.

Outputs ``results/sportcub_greybox_params.json`` (fitted parameters plus
validation scores) for export to the browser and reuse by the benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import casadi as ca
import numpy as np

from benchmark.paths import RESULTS
from . import comparison_suite as suite
from .model import INPUT_NAMES, euler_from_quaternion, normalize_quaternion, quaternion_from_euler
from .greybox import (
    AERO_PARAMETER_NAMES,
    CONTROL_NAMES_OEM,
    FIXED_PARAMETER_NAMES,
    GREYBOX_ESTIMATED_PARAMETER_NAMES,
    GROUND_ESTIMATED_PARAMETER_NAMES,
    GROUND_PARAMETER_NAMES,
    INERTIA_PARAMETER_NAMES,
    SPORTCUB_PARAMETER_NAMES,
    STATE_NAMES_EULER,
    build_casadi_dynamics,
    sportcub_greybox_spec,
    wrap_angle_np,
)
from .modelica.export_identified import write_identified_modelica
from .segmentation import sample_labels, split_stabilized_windows, stabilized_windows

def write_identified_greybox_model(theta_full, out_path, *, provenance=None, new_model_name=None):
    """Write the identified Sport Cub grey-box as a standalone Modelica file.

    ``theta_full`` is the full parameter vector (fixed(10) + SPORTCUB
    fixed-wing plant coefficients); its values are baked into ``SportCubGreybox.mo``'s
    parameter defaults. Returns the written path.
    """
    full_names = FIXED_PARAMETER_NAMES + SPORTCUB_PARAMETER_NAMES
    values = dict(zip(full_names, (float(v) for v in np.asarray(theta_full).ravel())))
    return write_identified_modelica(
        "SportCubGreybox", values, out_path,
        new_model_name=new_model_name, provenance=provenance,
    )


def greybox_modelica_sources(theta_full, *, provenance=None) -> dict:
    """Return the baseline and identified Sport Cub grey-box Modelica source text.

    Used to ship the actual ``.mo`` source (single source of truth) into the
    website's Model Inspector, alongside the fitted parameter tables.
    """
    from .modelica.export_identified import HERE as _MO_DIR, identified_model_source

    base_mo = _MO_DIR / "SportCubGreybox.mo"
    full_names = FIXED_PARAMETER_NAMES + SPORTCUB_PARAMETER_NAMES
    values = dict(zip(full_names, (float(v) for v in np.asarray(theta_full).ravel())))
    return {
        "baseline_name": "SportCubGreybox",
        "baseline_source": base_mo.read_text(),
        "identified_name": "SportCubGreyboxIdentified",
        "identified_source": identified_model_source(
            base_mo, values, "SportCubGreyboxIdentified", provenance=provenance
        ),
    }


AIR_FIT_NAMES = INERTIA_PARAMETER_NAMES + AERO_PARAMETER_NAMES
# The segmented ground windows are only used for ground-contact parameters. Mass,
# geometry, thrust, inertia, and aero parameters are fitted from the air windows.
GROUND_FIT_NAMES = GROUND_ESTIMATED_PARAMETER_NAMES
GREYBOX_FIT_NAMES = GREYBOX_ESTIMATED_PARAMETER_NAMES

TRAIN_DEFAULT = Path("data/sportcub_mocap_5_22_26_train.npz")
VALIDATION_DEFAULT = Path("data/sportcub_mocap_5_22_26_validation.npz")
FLIGHTS_DEFAULT = Path("data/sportcub_mocap_5_22_26_flights.npz")
MODEL_RATE_HZ = 60.0
FIT_RATE_HZ = 20.0
FIT_MAX_MANUAL_CHUNKS = 24
FIT_MAX_MANUAL_STEPS = 12
FIT_MAX_GROUND_STEPS = 24
GROUND_MIN_SPEED_MPS = 0.5


# Benchmark u_cmd channels follow model.INPUT_NAMES (throttle, elevator,
# aileron, rudder) — not the dataset's canonical control_meas order.
OEM_CONTROL_ORDER = [INPUT_NAMES.index(name) for name in CONTROL_NAMES_OEM]


def _flight_name(segment_name: object) -> str:
    text = str(segment_name)
    return text.split("__manual_", 1)[0]


def _ground_windows(labels: np.ndarray, tracked: np.ndarray, dt: float) -> list[tuple[int, int]]:
    relabeled = np.where(labels == 0, 2, -1).astype(np.int8)
    return stabilized_windows(relabeled, tracked, dt)


def ground_heights_from_flights(data) -> dict[str, float]:
    """Per-flight runway down-coordinate from segmented, tracked ground samples."""
    heights: dict[str, float] = {}
    for fi, name in enumerate(str(s) for s in data["segment_names"]):
        mask = np.asarray(data["valid_mask"][fi], dtype=bool)
        x = np.asarray(data["x_meas"][fi][mask], dtype=float)
        mode = np.asarray(data["flight_mode"][fi][mask])
        tracked = (
            np.asarray(data["mocap_tracked"][fi][mask]) != 0
            if "mocap_tracked" in data.files
            else np.ones(len(x), dtype=bool)
        )
        labels = sample_labels(x, mode)
        keep = (labels == 0) & tracked
        if keep.any():
            heights[name] = float(np.median(x[keep, 2]))
    return heights


def shift_ground_relative(x_euler: np.ndarray, segment_names, ground_heights: dict[str, float] | None) -> np.ndarray:
    """Shift p_d so each flight's segmented runway is at p_d=0 for contact."""
    if not ground_heights or segment_names is None:
        return x_euler
    shifted = np.array(x_euler, copy=True)
    for i, name in enumerate(segment_names):
        height = ground_heights.get(_flight_name(name))
        if height is not None:
            shifted[i, :, 2] -= height
    return shifted


def _true_runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(np.asarray(mask, dtype=bool)):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        runs.append((start, len(mask)))
    return runs


def ground_training_chunks(
    data,
    *,
    max_seconds: float = 5.0,
    min_speed_mps: float = GROUND_MIN_SPEED_MPS,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, float]]:
    """Moving segmented ground train windows as ground-relative Euler chunks."""
    dt = float(data["sample_period_s"])
    max_len = max(2, int(round(min(max_seconds, 3.0) / dt)))
    min_len = max(20, int(round(0.5 / dt)))
    heights = ground_heights_from_flights(data)
    chunks: list[tuple[np.ndarray, np.ndarray]] = []
    for fi, name in enumerate(str(s) for s in data["segment_names"]):
        mask = np.asarray(data["valid_mask"][fi], dtype=bool)
        xq = np.asarray(data["x_meas"][fi][mask], dtype=float)
        sticks = np.asarray(data["u_cmd"][fi][mask], dtype=float)
        mode = np.asarray(data["flight_mode"][fi][mask])
        tracked = (
            np.asarray(data["mocap_tracked"][fi][mask]) != 0
            if "mocap_tracked" in data.files
            else np.ones(len(xq), dtype=bool)
        )
        labels = sample_labels(xq, mode)
        height = heights.get(name, 0.0)
        for a, b, split in split_stabilized_windows(_ground_windows(labels, tracked, dt)):
            if split != "train" or b - a < 20:
                continue
            ground_speed = np.linalg.norm(xq[a:b, 3:5], axis=1)
            for rel_a, rel_b in _true_runs(ground_speed >= min_speed_mps, min_len):
                run_a = a + rel_a
                run_b = a + rel_b
                for start in range(run_a, run_b - min_len + 1, max_len):
                    stop = min(run_b, start + max_len)
                    if stop - start < min_len:
                        continue
                    xe = quat_states_to_euler(xq[start:stop][None, :, :])[0]
                    xe[:, 2] -= height
                    chunks.append((xe, sticks[start:stop][:, OEM_CONTROL_ORDER]))
    return chunks, heights


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


def fit_greybox(
    x_quat: np.ndarray,
    u_cmd: np.ndarray,
    dt: float,
    max_nfev: int = 120,
    *,
    segment_names=None,
    ground_heights: dict[str, float] | None = None,
    ground_chunks: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict:
    """Fit the Sport Cub grey-box on benchmark chunks (quaternion 13-states).

    Returns the fitted free parameters plus everything needed to rebuild the
    model: the spec, scores on the training chunks, and the optimizer stats.
    """
    spec = sportcub_greybox_spec()
    x_train = shift_ground_relative(quat_states_to_euler(x_quat), segment_names, ground_heights)
    u_train = u_cmd[:, :, OEM_CONTROL_ORDER]
    stride = max(1, int(round((1.0 / MODEL_RATE_HZ) / dt)))
    fit_stride = max(stride, int(round((1.0 / FIT_RATE_HZ) / dt)))
    _dynamics, rk4 = build_casadi_dynamics(spec, dt * stride)
    _fit_dynamics, fit_rk4 = build_casadi_dynamics(spec, dt * fit_stride)
    sigma = np.asarray(spec.output_sigma)

    if x_train.shape[0] > FIT_MAX_MANUAL_CHUNKS:
        fit_trials = np.linspace(0, x_train.shape[0] - 1, FIT_MAX_MANUAL_CHUNKS, dtype=int)
    else:
        fit_trials = np.arange(x_train.shape[0])

    setup_all = spec.default_parameter_setup(GREYBOX_FIT_NAMES)
    lower_all = np.array([row[1] for row in setup_all])
    theta_initial = np.array([row[2] for row in setup_all])
    upper_all = np.array([row[3] for row in setup_all])
    theta_values = dict(zip(GREYBOX_FIT_NAMES, theta_initial))

    def full_vector_from_values(values: dict[str, float]) -> np.ndarray:
        return spec.full_parameter_vector_from_mapping(values)

    def solve_stage(
        stage_name: str,
        fit_names: tuple[str, ...],
        chunks: list[tuple[np.ndarray, np.ndarray, tuple[int, ...], tuple[float, ...], float, int]],
    ) -> dict:
        if not fit_names or not chunks:
            return {
                "cost": 0.0,
                "nfev": 0,
                "optimizer_status": "skipped",
                "optimizer_success": True,
                "cr_std": {},
                "couplings": [],
            }

        setup = spec.default_parameter_setup(fit_names)
        lower = np.array([row[1] for row in setup])
        theta0 = np.array([theta_values[row[0]] for row in setup])
        upper = np.array([row[3] for row in setup])
        theta_mid = 0.5 * (lower + upper)
        theta_scale = 0.5 * (upper - lower)
        z0 = (theta0 - theta_mid) / theta_scale

        z = ca.MX.sym(f"{stage_name}_theta_scaled", len(fit_names))
        stage_theta = {
            name: theta_values[name]
            for name in FIXED_PARAMETER_NAMES + SPORTCUB_PARAMETER_NAMES
            if name in theta_values
        }
        for i, name in enumerate(fit_names):
            stage_theta[name] = ca.DM(theta_mid[i]) + ca.DM(theta_scale[i]) * z[i]
        p_full = ca.vertcat(*[
            stage_theta.get(name, spec.default_full_parameter_mapping()[name])
            for name in FIXED_PARAMETER_NAMES + SPORTCUB_PARAMETER_NAMES
        ])

        residual_terms = []
        for x_chunk, u_chunk, output_idx, output_sigma, weight, max_steps in chunks:
            steps = min((len(x_chunk) - 1) // fit_stride, max_steps)
            state = ca.DM(x_chunk[0])
            for k in range(steps):
                state = fit_rk4(state, ca.DM(u_chunk[k * fit_stride]), p_full)
                measured = x_chunk[(k + 1) * fit_stride]
                for idx, sig in zip(output_idx, output_sigma):
                    err = state[idx] - float(measured[idx])
                    if idx == 8:
                        err = ca.atan2(ca.sin(err), ca.cos(err))
                    residual_terms.append((weight / sig) * err)

        residual_expr = ca.vertcat(*residual_terms)
        objective = 0.5 * ca.sumsqr(residual_expr)

        if stage_name == "ground" and len(fit_names) == 1:
            objective_fun = ca.Function(f"greybox_{stage_name}_objective", [z], [objective])
            residual_fun = ca.Function(f"greybox_{stage_name}_residual", [z], [residual_expr])
            evaluations = 0

            def eval_objective(z_value: float) -> float:
                nonlocal evaluations
                evaluations += 1
                try:
                    value = float(objective_fun(ca.DM([float(np.clip(z_value, -1.0, 1.0))])))
                except RuntimeError:
                    return float("inf")
                return value if np.isfinite(value) else float("inf")

            grid = np.linspace(-1.0, 1.0, max(17, 2*int(max_nfev) + 1))
            values = np.array([eval_objective(float(v)) for v in grid])
            best = int(np.nanargmin(values))
            lo = float(grid[max(0, best - 1)])
            hi = float(grid[min(len(grid) - 1, best + 1)])
            if lo == hi:
                lo, hi = max(-1.0, lo - 0.1), min(1.0, hi + 0.1)

            inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
            a, b = lo, hi
            c = b - inv_phi*(b - a)
            d = a + inv_phi*(b - a)
            fc = eval_objective(c)
            fd = eval_objective(d)
            for _ in range(max(12, 4*int(max_nfev))):
                if fc <= fd:
                    b, d, fd = d, c, fc
                    c = b - inv_phi*(b - a)
                    fc = eval_objective(c)
                else:
                    a, c, fc = c, d, fd
                    d = a + inv_phi*(b - a)
                    fd = eval_objective(d)

            z_fit = np.array([0.5*(a + b)], dtype=float)
            theta_fit = theta_mid + theta_scale * z_fit
            theta_values.update(dict(zip(fit_names, theta_fit)))

            residual_at_fit = np.asarray(residual_fun(ca.DM(z_fit))).ravel()
            cost = 0.5 * float(residual_at_fit @ residual_at_fit)
            dof = max(residual_at_fit.size - 1, 1)
            s2 = 2.0 * cost / dof
            cr_stage = dict.fromkeys(fit_names, np.nan)
            h = 1e-3
            z0_scalar = float(z_fit[0])
            z_minus = max(-1.0, z0_scalar - h)
            z_plus = min(1.0, z0_scalar + h)
            if z_plus > z_minus:
                h_eff = 0.5*(z_plus - z_minus)
                f0 = eval_objective(z0_scalar)
                fm = eval_objective(z_minus)
                fp = eval_objective(z_plus)
                curvature_z = (fp - 2.0*f0 + fm) / max(h_eff*h_eff, 1e-18)
                if np.isfinite(curvature_z) and curvature_z > 1e-18:
                    cr_stage[fit_names[0]] = float(theta_scale[0] * np.sqrt(s2 / curvature_z))

            return {
                "cost": cost,
                "nfev": evaluations,
                "optimizer_status": "scalar_grid_golden",
                "optimizer_success": bool(np.isfinite(cost)),
                "cr_std": cr_stage,
                "couplings": [],
            }

        solver = ca.nlpsol(
            f"greybox_{stage_name}_fit",
            "ipopt",
            {"x": z, "f": objective},
            {
                "print_time": False,
                "ipopt.print_level": 5,
                "ipopt.max_iter": int(max_nfev),
                "ipopt.sb": "yes",
                "ipopt.hessian_approximation": "limited-memory",
            },
        )
        fit = solver(x0=z0, lbx=-np.ones_like(z0), ubx=np.ones_like(z0))
        solver_stats = solver.stats()
        z_fit = np.asarray(fit["x"]).ravel()
        theta_fit = theta_mid + theta_scale * z_fit
        theta_values.update(dict(zip(fit_names, theta_fit)))

        residual_fun = ca.Function(f"greybox_{stage_name}_residual", [z], [residual_expr])
        jac_fun = ca.Function(f"greybox_{stage_name}_jacobian", [z], [ca.jacobian(residual_expr, z)])
        residual_at_fit = np.asarray(residual_fun(z_fit)).ravel()
        jac = np.asarray(jac_fun(z_fit))
        cost = 0.5 * float(residual_at_fit @ residual_at_fit)
        dof = max(jac.shape[0] - jac.shape[1], 1)
        s2 = 2.0 * cost / dof
        cr_stage = dict.fromkeys(fit_names, np.nan)
        couplings_stage: list[dict[str, object]] = []
        try:
            cov_z = s2 * np.linalg.inv(jac.T @ jac + 1e-12 * np.eye(jac.shape[1]))
            cov = np.diag(theta_scale) @ cov_z @ np.diag(theta_scale)
            cr_values = np.sqrt(np.maximum(np.diag(cov), 0.0))
            cr_stage.update(dict(zip(fit_names, cr_values)))
            corr = cov / np.outer(cr_values, cr_values)
            for i, name_i in enumerate(fit_names):
                for j, name_j in enumerate(fit_names[i + 1:], start=i + 1):
                    if abs(corr[i, j]) > 0.9:
                        couplings_stage.append({"a": name_i, "b": name_j, "r": float(corr[i, j]), "stage": stage_name})
            couplings_stage.sort(key=lambda d: -abs(d["r"]))
        except np.linalg.LinAlgError:
            pass
        return {
            "cost": cost,
            "nfev": int(solver_stats.get("iter_count", max_nfev)),
            "optimizer_status": str(solver_stats.get("return_status", "unknown")),
            "optimizer_success": bool(solver_stats.get("success", False)),
            "cr_std": cr_stage,
            "couplings": couplings_stage,
        }

    ground_stage_chunks = [
        (
            x_ground,
            u_ground,
            (0, 1, 2, 3, 4, 5, 6, 7, 8),
            (0.05, 0.05, 0.03, 0.20, 0.20, 0.20, 0.08, 0.08, 0.12),
            1.0,
            FIT_MAX_GROUND_STEPS,
        )
        for x_ground, u_ground in (ground_chunks or [])
    ]
    air_stage_chunks = [
        (x_train[trial], u_train[trial], (0, 1, 2, 6, 7, 8), tuple(float(v) for v in sigma), 1.0, FIT_MAX_MANUAL_STEPS)
        for trial in fit_trials
    ]

    ground_result = solve_stage("ground", GROUND_FIT_NAMES, ground_stage_chunks)
    air_result = solve_stage("air", AIR_FIT_NAMES, air_stage_chunks)

    theta = np.array([theta_values[name] for name in GREYBOX_FIT_NAMES], dtype=float)
    full = full_vector_from_values(theta_values)
    cr_map = {name: np.nan for name in GREYBOX_FIT_NAMES}
    cr_map.update(ground_result["cr_std"])
    cr_map.update(air_result["cr_std"])
    cr_std = np.array([cr_map[name] for name in GREYBOX_FIT_NAMES], dtype=float)
    couplings = ground_result["couplings"] + air_result["couplings"]
    cost = float(ground_result["cost"] + air_result["cost"])
    nfev = int(ground_result["nfev"] + air_result["nfev"])
    status = f"ground={ground_result['optimizer_status']}; air={air_result['optimizer_status']}"
    success = bool(ground_result["optimizer_success"] and air_result["optimizer_success"])
    return {
        "spec": spec,
        "parameter_names": list(GREYBOX_FIT_NAMES),
        "theta": theta,
        "theta_full": full,
        "cr_std": cr_std,
        "couplings": couplings,
        "lower": lower_all,
        "upper": upper_all,
        "train_nrmse_initial": rollout_nrmse(rk4, full_vector_from_values({}), x_train, u_train, stride),
        "train_nrmse": rollout_nrmse(rk4, full, x_train, u_train, stride),
        "ground_train_windows": len(ground_chunks or []),
        "model_rate_hz": 1.0 / (dt * stride),
        "cost": cost,
        "nfev": nfev,
        "optimizer_status": status,
        "optimizer_success": success,
        "stage_results": {"ground": ground_result, "air": air_result},
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

def fit_greybox_ekf(x_quat: np.ndarray, u_cmd: np.ndarray, dt: float, stride: int = 4) -> dict:
    """Filter-error parameter estimation: augmented-state EKF over the grey-box.

    The 12 Euler states are augmented with
    the fixed-wing plant parameters, propagated chunk by chunk over the manual
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
    p_full = ca.vertcat(ca.DM(spec.fixed_parameter_vector()), th_sym)
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

    theta_full = spec.full_parameter_vector(theta)
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
    parser.add_argument("--flights", type=Path, default=FLIGHTS_DEFAULT)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--max-nfev", type=int, default=120)
    args = parser.parse_args()

    flights = np.load(args.flights, allow_pickle=False)
    ground_chunks, ground_heights = ground_training_chunks(flights)
    print(
        f"ground-contact training: {len(ground_chunks)} moving ground windows "
        f"(speed >= {GROUND_MIN_SPEED_MPS:.2f} m/s) from {len(ground_heights)} flights"
    )
    segments, dt = suite.load_segments(args.train)
    train = suite.trim_to_input_onset(segments, dt, estimate_bias=True, label="greybox train")
    print(f"fitting on {train.x_true.shape[0]} chunks ({train.x_true.shape[1]} samples each)")
    fit = fit_greybox(
        train.x_true,
        train.u_cmd,
        dt,
        max_nfev=args.max_nfev,
        segment_names=train.segment_names,
        ground_heights=ground_heights,
        ground_chunks=ground_chunks,
    )
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
        # Within 1% of the box span counts as pinned by the bounded NLP solve.
        pinned = min(value - lo, hi - value) < 0.01 * (hi - lo)
        if pinned:
            at_bounds.append(name)
        print(f"  {name:6s} = {value:+10.4f}{'  (at bound)' if pinned else ''}")
    if at_bounds:
        print(f"  WARNING: parameters pinned at bounds: {', '.join(at_bounds)} — widen the spec box")
    print(f"  optimizer_status: {fit['optimizer_status']} (success={fit['optimizer_success']})")
    weak = []
    for name, value, sd in zip(GREYBOX_FIT_NAMES, theta, fit["cr_std"]):
        if np.isfinite(sd) and abs(value) > 1e-9 and sd / abs(value) > 0.25:
            weak.append(name)
    if weak:
        print(f"  weak uncertainty (>25% relative std): {', '.join(weak)}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    output = args.results_dir / "sportcub_greybox_params.json"
    cr_std = [
        None if not np.isfinite(sd) else float(sd)
        for sd in fit["cr_std"]
    ]
    rel_std = [
        None if sd is None or abs(value) <= 1e-9 else float(sd / abs(value))
        for value, sd in zip(theta, cr_std)
    ]
    output.write_text(json.dumps({
        "parameter_names": list(GREYBOX_FIT_NAMES),
        "parameters": [float(v) for v in theta],
        "cr_std": cr_std,
        "relative_std": rel_std,
        "couplings": fit["couplings"],
        "uncertainty_note": "Cramer-Rao lower bounds from each staged CasADi output-error Jacobian; residual coloring uncorrected, so these are optimistic.",
        "fixed_parameters": spec.fixed_parameters,
        "default_parameters": spec.default_sportcub_parameters(),
        "max_deflection_deg": spec.max_deflection_deg,
        "ground_heights": {k: round(float(v), 6) for k, v in ground_heights.items()},
        "ground_train_windows": fit["ground_train_windows"],
        "ground_min_speed_mps": GROUND_MIN_SPEED_MPS,
        "model_rate_hz": fit["model_rate_hz"],
        "parameters_at_bounds": at_bounds,
        "scores": scores,
        "cost": fit["cost"],
        "nfev": fit["nfev"],
        "optimizer_status": fit["optimizer_status"],
        "optimizer_success": fit["optimizer_success"],
    }, indent=2) + "\n")
    print(f"wrote {output}")

    # Round-trip the identified model back out as Modelica: the base grey-box
    # with the fitted parameter values baked into the parameter defaults. The
    # result is a self-contained, Rumoca-recompilable model of the found system.
    provenance = (
        f"fit: train_nrmse={scores['train_nrmse']:.4f}, "
        f"validation_nrmse={scores['validation_nrmse']:.4f}, "
        f"cost={fit['cost']:.6g}, nfev={fit['nfev']}, "
        f"status={fit['optimizer_status']}"
    )
    mo_out = write_identified_greybox_model(
        fit["theta_full"], args.results_dir / "SportCubGreyboxIdentified.mo",
        provenance=provenance,
    )
    print(f"wrote {mo_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
