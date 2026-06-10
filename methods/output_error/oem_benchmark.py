#!/usr/bin/env python3
"""Reproduce output-error-method figures for the longitudinal aircraft paper.

The script is intentionally self-contained so the numerical method can be
inspected and rerun independently of the LaTeX source.  It implements the
four-state longitudinal model from the paper, generates two synthetic test
cases, and fits the aerodynamic parameters with single-shooting and
multiple-shooting output-error least-squares estimators.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


METHOD_DIR = Path(__file__).resolve().parent
ROOT = METHOD_DIR.parents[1]
FIG_DIR = ROOT / "latex" / "fig"
RESULTS_DIR = ROOT / "results"
PARAMETER_NAMES = [
    "C_L0",
    "C_L_alpha",
    "C_D0",
    "k",
    "C_M0",
    "C_M_alpha",
    "C_M_Q",
    "C_M_delta_e",
]
STATE_NAMES = ["V", "alpha", "gamma", "Q"]
STATE_LABELS = [
    r"$V$ [m/s]",
    r"$\alpha$ [deg]",
    r"$\gamma$ [deg]",
    r"$Q$ [deg/s]",
]


@dataclass(frozen=True)
class Aircraft:
    mass: float = 1.0
    jy: float = 0.15
    wing_area: float = 0.25
    chord: float = 0.18
    rho: float = 1.225
    gravity: float = 9.81


@dataclass(frozen=True)
class Bounds:
    theta_lower: tuple[float, ...] = (-0.5, 0.0, 0.0, 0.0, -1.0, -5.0, -50.0, -3.0)
    theta_upper: tuple[float, ...] = (0.5, 6.0, 0.2, 0.5, 1.0, 3.0, 0.0, 3.0)
    # outer state box for shooting nodes/initial states; generous envelope so
    # feasible trajectories are never clipped
    state_lower: tuple[float, ...] = (3.0, -0.8, -1.25, -3.5)
    state_upper: tuple[float, ...] = (40.0, 0.8, 1.25, 3.5)


@dataclass
class FitResult:
    theta: np.ndarray
    trajectory: np.ndarray
    cost: float
    nfev: int
    elapsed_s: float
    success: bool
    message: str
    decision_variables: int
    residuals: np.ndarray


@dataclass
class TestCase:
    name: str
    t: np.ndarray
    u_truth: np.ndarray
    u_id: np.ndarray
    x_true: np.ndarray
    y_meas: np.ndarray
    noise_std: np.ndarray


def true_theta() -> np.ndarray:
    # Standard non-dimensional coefficients: M = Cm qbar S cbar with
    # Cm = Cm0 + Cma a + Cmq (cbar q / 2V) + Cmde de.
    return np.array([0.10, 3.00, 0.030, 0.10, 0.05, -0.60, -12.0, 0.60])


def initial_theta() -> np.ndarray:
    return np.array([-0.050, 4.800, 0.080, 0.350, 0.300, 0.300, -28.0, 0.120])


def trim_state(theta: np.ndarray, aircraft: Aircraft, speed: float = 15.0) -> np.ndarray:
    """Level-flight equilibrium: solve the lift equation for alpha (see common/benchmark.py)."""
    cl0, cla, cd0, k_drag = theta[0], theta[1], theta[2], theta[3]
    qbar_s = 0.5 * aircraft.rho * speed**2 * aircraft.wing_area
    weight = aircraft.mass * aircraft.gravity
    alpha = (weight / qbar_s - cl0) / cla
    for _ in range(20):
        cl = cl0 + cla * alpha
        cd = cd0 + k_drag * cl**2
        thrust = cd * qbar_s / max(np.cos(alpha), 0.2)
        alpha = ((weight - thrust * np.sin(alpha)) / qbar_s - cl0) / cla
    return np.array([speed, alpha, 0.0, 0.0])


def trim_controls(theta: np.ndarray, aircraft: Aircraft, x_trim: np.ndarray) -> np.ndarray:
    v, alpha, _, q_rate = x_trim
    cl0, cla, cd0, k_drag, cm0, cma, cmq, cme = theta
    qbar = 0.5 * aircraft.rho * v**2
    cl = cl0 + cla * alpha
    cd = cd0 + k_drag * cl**2
    drag = cd * qbar * aircraft.wing_area
    thrust = drag / max(np.cos(alpha), 0.2)
    q_hat = aircraft.chord * q_rate / (2.0 * max(v, 3.0))
    elevator = -(cm0 + cma * alpha + cmq * q_hat) / cme
    return np.array([thrust, elevator])


def make_time(duration: float = 30.0, dt: float = 0.1) -> np.ndarray:
    return np.arange(0.0, duration + 0.5 * dt, dt)


# Keeps the benchmark inside the linear-aero envelope (|alpha| < ~20 deg) with
# the realistic pitch damping of the non-dimensionalized model.
EXCITATION_SCALE = 0.35


def excitation(t: np.ndarray, u_trim: np.ndarray, scale: float = EXCITATION_SCALE) -> np.ndarray:
    thrust = (
        u_trim[0]
        + scale * 0.22 * np.sin(0.23 * t + 0.2)
        + scale * 0.12 * np.sin(0.91 * t)
        + scale * 0.10 * np.where((t > 8.0) & (t < 14.0), 1.0, 0.0)
        - scale * 0.08 * np.where((t > 20.0) & (t < 25.0), 1.0, 0.0)
    )
    elevator = (
        u_trim[1]
        + scale * 0.070 * np.sin(0.72 * t)
        + scale * 0.035 * np.sin(1.73 * t + 0.5)
        # short-period coverage (~6.8 rad/s) so the pitch derivatives are identifiable
        + scale * 0.030 * np.sin(4.5 * t + 0.3)
        + scale * 0.025 * np.sin(7.0 * t + 0.8)
        - scale * 0.055 * np.where((t > 10.0) & (t < 13.5), 1.0, 0.0)
        + scale * 0.045 * np.where((t > 19.0) & (t < 22.0), 1.0, 0.0)
    )
    return np.column_stack((np.clip(thrust, 0.0, 3.0), np.clip(elevator, -0.35, 0.35)))


def actuator_response(t: np.ndarray, u_cmd: np.ndarray, dt: float) -> np.ndarray:
    u_act = np.empty_like(u_cmd)
    u_act[0] = u_cmd[0]
    tau = np.array([0.10, 0.05])
    rate_limit = np.array([8.0, 2.5])
    lower = np.array([0.0, -0.35])
    upper = np.array([3.0, 0.35])
    for k in range(len(t) - 1):
        desired_rate = (u_cmd[k] - u_act[k]) / tau
        limited_rate = np.clip(desired_rate, -rate_limit, rate_limit)
        u_act[k + 1] = np.clip(u_act[k] + dt * limited_rate, lower, upper)
    return u_act


def gust_signal(t: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray:
    gust = np.zeros_like(t)
    tau = 2.0
    sigma = 0.22
    decay = np.exp(-dt / tau)
    noise_scale = sigma * np.sqrt(1.0 - decay**2)
    for k in range(len(t) - 1):
        gust[k + 1] = decay * gust[k] + noise_scale * rng.normal()
    return gust


def eom(x: np.ndarray, u: np.ndarray, theta: np.ndarray, aircraft: Aircraft, gust: float = 0.0) -> np.ndarray:
    v, alpha, gamma, q_rate = x
    thrust, elevator = u
    v = max(v, 3.0)
    cl0, cla, cd0, k_drag, cm0, cma, cmq, cme = theta
    qbar = 0.5 * aircraft.rho * v**2
    q_hat = aircraft.chord * q_rate / (2.0 * v)
    cl = cl0 + cla * alpha
    cd = cd0 + k_drag * cl**2
    cm = cm0 + cma * alpha + cmq * q_hat + cme * elevator
    lift = cl * qbar * aircraft.wing_area
    drag = cd * qbar * aircraft.wing_area
    moment = cm * qbar * aircraft.wing_area * aircraft.chord
    v_dot = (-drag + thrust * np.cos(alpha) - aircraft.mass * aircraft.gravity * np.sin(gamma)) / aircraft.mass + gust
    gamma_dot = (lift + thrust * np.sin(alpha) - aircraft.mass * aircraft.gravity * np.cos(gamma)) / (
        aircraft.mass * v
    )
    q_dot = moment / aircraft.jy
    alpha_dot = q_rate - gamma_dot
    return np.array([v_dot, alpha_dot, gamma_dot, q_dot])


def rk4_step(
    x: np.ndarray,
    u0: np.ndarray,
    u1: np.ndarray,
    theta: np.ndarray,
    aircraft: Aircraft,
    dt: float,
    gust0: float = 0.0,
    gust1: float = 0.0,
) -> np.ndarray:
    umid = 0.5 * (u0 + u1)
    gmid = 0.5 * (gust0 + gust1)
    k1 = eom(x, u0, theta, aircraft, gust0)
    k2 = eom(x + 0.5 * dt * k1, umid, theta, aircraft, gmid)
    k3 = eom(x + 0.5 * dt * k2, umid, theta, aircraft, gmid)
    k4 = eom(x + dt * k3, u1, theta, aircraft, gust1)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next[0] = max(x_next[0], 3.0)
    return x_next


def simulate(
    x0: np.ndarray,
    u: np.ndarray,
    theta: np.ndarray,
    aircraft: Aircraft,
    dt: float,
    gust: np.ndarray | None = None,
) -> np.ndarray:
    x = np.empty((len(u), 4))
    x[0] = x0
    if gust is None:
        gust = np.zeros(len(u))
    for k in range(len(u) - 1):
        x[k + 1] = rk4_step(x[k], u[k], u[k + 1], theta, aircraft, dt, gust[k], gust[k + 1])
        if not np.all(np.isfinite(x[k + 1])) or np.linalg.norm(x[k + 1]) > 1e4:
            x[k + 1 :] = np.nan
            break
    return x


def eom_batch(x: np.ndarray, u: np.ndarray, theta: np.ndarray, aircraft: Aircraft, gust: float = 0.0) -> np.ndarray:
    """Vectorized eom over a batch: x (B,4), theta (B,8), u (2,) shared."""
    v = np.maximum(x[:, 0], 3.0)
    alpha = x[:, 1]
    gamma = x[:, 2]
    q_rate = x[:, 3]
    thrust, elevator = u
    cl0, cla, cd0, k_drag, cm0, cma, cmq, cme = theta.T
    qbar = 0.5 * aircraft.rho * v**2
    q_hat = aircraft.chord * q_rate / (2.0 * v)
    cl = cl0 + cla * alpha
    cd = cd0 + k_drag * cl**2
    cm = cm0 + cma * alpha + cmq * q_hat + cme * elevator
    lift = cl * qbar * aircraft.wing_area
    drag = cd * qbar * aircraft.wing_area
    moment = cm * qbar * aircraft.wing_area * aircraft.chord
    v_dot = (-drag + thrust * np.cos(alpha) - aircraft.mass * aircraft.gravity * np.sin(gamma)) / aircraft.mass + gust
    gamma_dot = (lift + thrust * np.sin(alpha) - aircraft.mass * aircraft.gravity * np.cos(gamma)) / (
        aircraft.mass * v
    )
    q_dot = moment / aircraft.jy
    out = np.empty_like(x)
    out[:, 0] = v_dot
    out[:, 1] = q_rate - gamma_dot
    out[:, 2] = gamma_dot
    out[:, 3] = q_dot
    return out


def rk4_step_batch(
    x: np.ndarray,
    u0: np.ndarray,
    u1: np.ndarray,
    theta: np.ndarray,
    aircraft: Aircraft,
    dt: float,
    gust0: float = 0.0,
    gust1: float = 0.0,
) -> np.ndarray:
    umid = 0.5 * (u0 + u1)
    gmid = 0.5 * (gust0 + gust1)
    k1 = eom_batch(x, u0, theta, aircraft, gust0)
    k2 = eom_batch(x + 0.5 * dt * k1, umid, theta, aircraft, gmid)
    k3 = eom_batch(x + 0.5 * dt * k2, umid, theta, aircraft, gmid)
    k4 = eom_batch(x + dt * k3, u1, theta, aircraft, gust1)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next[:, 0] = np.maximum(x_next[:, 0], 3.0)
    return x_next


def simulate_batch(
    x0: np.ndarray,
    u: np.ndarray,
    theta: np.ndarray,
    aircraft: Aircraft,
    dt: float,
    gust: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a batch of (theta, x0) candidates over a shared input history.

    Used only inside the batched Jacobian, where a diverged candidate's whole
    residual row is replaced by the 1e6 penalty fill, so no per-step freeze is
    needed: divergence (non-finite values or the same 1e4 norm guard simulate()
    uses) is detected in a vectorized post-pass. Returns (trajectories, bad_row).
    """
    batch = x0.shape[0]
    steps = len(u)
    if gust is None:
        gust = np.zeros(steps)
    out = np.empty((batch, steps, 4))
    out[:, 0] = x0
    x = x0.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(steps - 1):
            x = rk4_step_batch(x, u[k], u[k + 1], theta, aircraft, dt, gust[k], gust[k + 1])
            out[:, k + 1] = x
        bad_row = ~np.all(np.isfinite(out), axis=(1, 2)) | np.any(
            np.linalg.norm(np.nan_to_num(out), axis=2) > 1e4, axis=1
        )
    return out, bad_row


def make_batched_jacobian(residual_batch):
    """Two-point forward-difference Jacobian evaluated in one batched rollout."""

    def jac(z: np.ndarray, *args: object) -> np.ndarray:
        step = np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(z))
        candidates = np.tile(z, (len(z) + 1, 1))
        candidates[np.arange(1, len(z) + 1), np.arange(len(z))] += step
        residuals = residual_batch(candidates)
        return (residuals[1:] - residuals[0]).T / step

    return jac


def make_cases(duration: float, dt: float, seed: int) -> list[TestCase]:
    aircraft = Aircraft()
    theta = true_theta()
    x0 = trim_state(theta, aircraft)
    t = make_time(duration, dt)
    u_trim = trim_controls(theta, aircraft, x0)
    u_cmd = excitation(t, u_trim)
    rng = np.random.default_rng(seed)

    noise_std = np.array([0.08, 0.0040, 0.0040, 0.0150])
    x_case_1 = simulate(x0, u_cmd, theta, aircraft, dt)
    y_case_1 = x_case_1 + rng.normal(scale=noise_std, size=x_case_1.shape)

    u_act = actuator_response(t, u_cmd, dt)
    gust = gust_signal(t, dt, rng)
    x_case_2 = simulate(x0, u_act, theta, aircraft, dt, gust)
    y_case_2 = x_case_2 + rng.normal(scale=noise_std, size=x_case_2.shape)

    return [
        TestCase("noise_only", t, u_cmd, u_cmd, x_case_1, y_case_1, noise_std),
        TestCase("lag_limit_gust_mismatch", t, u_act, u_cmd, x_case_2, y_case_2, noise_std),
    ]


def state_bounds_from_measurements(y: np.ndarray, bounds: Bounds) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array(bounds.state_lower)
    upper = np.array(bounds.state_upper)
    measured_lower = np.nanmin(y, axis=0) - np.array([2.0, 0.20, 0.20, 0.75])
    measured_upper = np.nanmax(y, axis=0) + np.array([2.0, 0.20, 0.20, 0.75])
    return np.maximum(lower, measured_lower), np.minimum(upper, measured_upper)


def pack_ss(theta: np.ndarray, x0: np.ndarray) -> np.ndarray:
    return np.concatenate((theta, x0))


def unpack_ss(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return z[:8], z[8:12]


def fit_single_shooting(case: TestCase, aircraft: Aircraft, dt: float, max_nfev: int) -> FitResult:
    bounds = Bounds()
    x0_lower, x0_upper = state_bounds_from_measurements(case.y_meas[:1], bounds)
    lower = np.concatenate((bounds.theta_lower, x0_lower))
    upper = np.concatenate((bounds.theta_upper, x0_upper))
    z0 = np.clip(pack_ss(initial_theta(), case.y_meas[0]), lower + 1e-8, upper - 1e-8)

    def residual_batch(candidates: np.ndarray) -> np.ndarray:
        y_hat, bad = simulate_batch(candidates[:, 8:12], case.u_id, candidates[:, :8], aircraft, dt)
        residuals = ((y_hat - case.y_meas) / case.noise_std).reshape(candidates.shape[0], -1)
        residuals[bad] = 1e6
        return residuals

    def residual(z: np.ndarray) -> np.ndarray:
        theta, x0 = unpack_ss(z)
        y_hat = simulate(x0, case.u_id, theta, aircraft, dt)
        if not np.all(np.isfinite(y_hat)):
            return np.full(case.y_meas.size, 1e6)
        return ((y_hat - case.y_meas) / case.noise_std).ravel()

    start = time.perf_counter()
    result = least_squares(
        residual,
        z0,
        jac=make_batched_jacobian(residual_batch),
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        loss="linear",
        max_nfev=max_nfev,
        verbose=0,
    )
    elapsed = time.perf_counter() - start
    theta, x0 = unpack_ss(result.x)
    trajectory = simulate(x0, case.u_id, theta, aircraft, dt)
    return FitResult(
        theta=theta,
        trajectory=trajectory,
        cost=float(result.cost),
        nfev=int(result.nfev),
        elapsed_s=elapsed,
        success=bool(result.success),
        message=str(result.message),
        decision_variables=len(result.x),
        residuals=((trajectory - case.y_meas) / case.noise_std).ravel(),
    )


def segment_bounds(n: int, segments: int) -> np.ndarray:
    return np.linspace(0, n - 1, segments + 1, dtype=int)


def pack_ms(theta: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    return np.concatenate((theta, nodes.ravel()))


def unpack_ms(z: np.ndarray, segments: int) -> tuple[np.ndarray, np.ndarray]:
    return z[:8], z[8:].reshape((segments, 4))


def simulate_segments(
    nodes: np.ndarray,
    indices: np.ndarray,
    u: np.ndarray,
    theta: np.ndarray,
    aircraft: Aircraft,
    dt: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    y_hat = np.empty((len(u), 4))
    continuity = []
    for i in range(len(nodes)):
        start, stop = indices[i], indices[i + 1]
        local_u = u[start : stop + 1]
        local = simulate(nodes[i], local_u, theta, aircraft, dt)
        y_hat[start : stop + 1] = local
        if i < len(nodes) - 1:
            continuity.append(local[-1] - nodes[i + 1])
    return y_hat, continuity


def fit_multiple_shooting(
    case: TestCase,
    aircraft: Aircraft,
    dt: float,
    segments: int,
    max_nfev: int,
) -> FitResult:
    bounds = Bounds()
    indices = segment_bounds(len(case.t), segments)
    node_guess = case.y_meas[indices[:-1]].copy()
    node_lower, node_upper = state_bounds_from_measurements(case.y_meas, bounds)
    lower = np.concatenate((bounds.theta_lower, np.tile(node_lower, segments)))
    upper = np.concatenate((bounds.theta_upper, np.tile(node_upper, segments)))
    z0 = np.clip(pack_ms(initial_theta(), node_guess), lower + 1e-8, upper - 1e-8)
    continuity_scale = np.maximum(0.5 * case.noise_std, np.array([0.02, 0.0015, 0.0015, 0.006]))

    def residual_batch(candidates: np.ndarray) -> np.ndarray:
        batch = candidates.shape[0]
        theta = candidates[:, :8]
        y_hat = np.empty((batch, len(case.t), 4))
        continuity = []
        bad = np.zeros(batch, dtype=bool)
        for i in range(segments):
            start_idx, stop_idx = indices[i], indices[i + 1]
            node = candidates[:, 8 + 4 * i : 8 + 4 * (i + 1)]
            local, local_bad = simulate_batch(node, case.u_id[start_idx : stop_idx + 1], theta, aircraft, dt)
            bad |= local_bad
            y_hat[:, start_idx : stop_idx + 1] = local
            if i < segments - 1:
                next_node = candidates[:, 8 + 4 * (i + 1) : 8 + 4 * (i + 2)]
                continuity.append((local[:, -1] - next_node) / continuity_scale)
        measurement_residual = ((y_hat - case.y_meas) / case.noise_std).reshape(batch, -1)
        continuity_residual = np.concatenate(continuity, axis=1)
        residuals = np.concatenate((measurement_residual, continuity_residual), axis=1)
        residuals[bad] = 1e6
        return residuals

    def residual(z: np.ndarray) -> np.ndarray:
        theta, nodes = unpack_ms(z, segments)
        y_hat, continuity = simulate_segments(nodes, indices, case.u_id, theta, aircraft, dt)
        if not np.all(np.isfinite(y_hat)):
            return np.full(case.y_meas.size + 4 * (segments - 1), 1e6)
        measurement_residual = ((y_hat - case.y_meas) / case.noise_std).ravel()
        continuity_residual = np.concatenate([gap / continuity_scale for gap in continuity])
        return np.concatenate((measurement_residual, continuity_residual))

    start = time.perf_counter()
    result = least_squares(
        residual,
        z0,
        jac=make_batched_jacobian(residual_batch),
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        loss="linear",
        max_nfev=max_nfev,
        verbose=0,
    )
    elapsed = time.perf_counter() - start
    theta, nodes = unpack_ms(result.x, segments)
    trajectory, _ = simulate_segments(nodes, indices, case.u_id, theta, aircraft, dt)
    return FitResult(
        theta=theta,
        trajectory=trajectory,
        cost=float(result.cost),
        nfev=int(result.nfev),
        elapsed_s=elapsed,
        success=bool(result.success),
        message=str(result.message),
        decision_variables=len(result.x),
        residuals=((trajectory - case.y_meas) / case.noise_std).ravel(),
    )


def rmse(y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_hat - y_true) ** 2, axis=0))


def percent_error(theta_hat: np.ndarray, theta_ref: np.ndarray) -> np.ndarray:
    return 100.0 * np.abs((theta_hat - theta_ref) / theta_ref)


def finite_difference_sensitivity(
    theta: np.ndarray,
    x0: np.ndarray,
    u: np.ndarray,
    aircraft: Aircraft,
    dt: float,
    noise_std: np.ndarray,
) -> np.ndarray:
    base = simulate(x0, u, theta, aircraft, dt)
    cols = []
    for i, value in enumerate(theta):
        step = 1e-4 * max(1.0, abs(value))
        theta_step = theta.copy()
        theta_step[i] += step
        perturbed = simulate(x0, u, theta_step, aircraft, dt)
        cols.append(((perturbed - base) / step / noise_std).ravel())
    return np.column_stack(cols)


def save_summary(cases: list[TestCase], fits: dict[str, dict[str, FitResult]], output: Path) -> None:
    theta_ref = true_theta()
    rows = []
    for case in cases:
        for method, fit in fits[case.name].items():
            theta_error = percent_error(fit.theta, theta_ref)
            state_rmse = rmse(fit.trajectory, case.x_true)
            rows.append(
                {
                    "case": case.name,
                    "method": method,
                    "nfev": fit.nfev,
                    "elapsed_s": fit.elapsed_s,
                    "cost": fit.cost,
                    "success": fit.success,
                    "decision_variables": fit.decision_variables,
                    **{f"rmse_{name}": value for name, value in zip(STATE_NAMES, state_rmse)},
                    **{f"theta_{name}": value for name, value in zip(PARAMETER_NAMES, fit.theta)},
                    **{f"errpct_{name}": value for name, value in zip(PARAMETER_NAMES, theta_error)},
                }
            )
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_trajectories(cases: list[TestCase], fits: dict[str, dict[str, FitResult]], path_base: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(8.2, 5.6), sharex=True)
    for col, case in enumerate(cases):
        for row, label in enumerate(STATE_LABELS):
            ax = axes[row, col]
            truth = case.x_true[:, row].copy()
            meas = case.y_meas[:, row].copy()
            ss = fits[case.name]["SS"].trajectory[:, row].copy()
            ms = fits[case.name]["MS"].trajectory[:, row].copy()
            if row in (1, 2, 3):
                truth = np.rad2deg(truth)
                meas = np.rad2deg(meas)
                ss = np.rad2deg(ss)
                ms = np.rad2deg(ms)
            ax.plot(case.t, truth, color="black", linewidth=1.5, label="Truth")
            ax.plot(case.t, meas, color="0.75", linewidth=0.5, alpha=0.75, label="Measured")
            ax.plot(case.t, ss, "--", color="#d62728", linewidth=1.1, label="SS OEM")
            ax.plot(case.t, ms, "-.", color="#1f77b4", linewidth=1.1, label="MS OEM")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
            if row == 0:
                title = "Test Case 1: noise only" if col == 0 else "Test Case 2: mismatch"
                ax.set_title(title)
            if row == 3:
                ax.set_xlabel("time [s]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, path_base)


def plot_parameter_errors(cases: list[TestCase], fits: dict[str, dict[str, FitResult]], path_base: Path) -> None:
    labels = [r"$C_{L0}$", r"$C_{L\alpha}$", r"$C_{D0}$", r"$k$", r"$C_{M0}$", r"$C_{M\alpha}$", r"$C_{MQ}$", r"$C_{M\delta_e}$"]
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8.3, 3.9))
    offsets = [-1.5, -0.5, 0.5, 1.5]
    colors = ["#d62728", "#1f77b4", "#ff9896", "#aec7e8"]
    series = []
    for case in cases:
        for method in ("SS", "MS"):
            series.append((case.name, method, percent_error(fits[case.name][method].theta, true_theta())))
    for offset, color, (case_name, method, values) in zip(offsets, colors, series):
        label = f"{method}, {'case 1' if case_name == 'noise_only' else 'case 2'}"
        ax.bar(x + offset * width, values, width, label=label, color=color)
    ax.set_ylabel("absolute parameter error [%]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    save_figure(fig, path_base)


def plot_rmse(cases: list[TestCase], fits: dict[str, dict[str, FitResult]], path_base: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=False)
    labels = [r"$V$", r"$\alpha$", r"$\gamma$", r"$Q$"]
    for ax, case in zip(axes, cases):
        values = np.vstack([rmse(fits[case.name][method].trajectory, case.x_true) for method in ("SS", "MS")])
        values[:, 1:] = np.rad2deg(values[:, 1:])
        x = np.arange(4)
        ax.bar(x - 0.18, values[0], 0.36, label="SS OEM", color="#d62728")
        ax.bar(x + 0.18, values[1], 0.36, label="MS OEM", color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title("Noise only" if case.name == "noise_only" else "Mismatch")
        ax.set_ylabel("trajectory RMSE [native units]")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, path_base)


def plot_identifiability(case: TestCase, fit: FitResult, path_base: Path, aircraft: Aircraft, dt: float) -> None:
    sensitivity = finite_difference_sensitivity(fit.theta, fit.trajectory[0], case.u_id, aircraft, dt, case.noise_std)
    fim = sensitivity.T @ sensitivity
    scale = np.sqrt(np.maximum(np.diag(fim), 1e-12))
    correlation = fim / np.outer(scale, scale)
    singular_values = np.linalg.svd(sensitivity, compute_uv=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    axes[0].semilogy(np.arange(1, len(singular_values) + 1), singular_values, "o-", color="#1f77b4")
    axes[0].set_xlabel("singular-value index")
    axes[0].set_ylabel("weighted sensitivity singular value")
    axes[0].grid(True, which="both", alpha=0.25)

    im = axes[1].imshow(correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axes[1].set_xticks(np.arange(len(PARAMETER_NAMES)))
    axes[1].set_yticks(np.arange(len(PARAMETER_NAMES)))
    short = [r"$C_{L0}$", r"$C_{L\alpha}$", r"$C_{D0}$", r"$k$", r"$C_{M0}$", r"$C_{M\alpha}$", r"$C_{MQ}$", r"$C_{M\delta_e}$"]
    axes[1].set_xticklabels(short, rotation=45, ha="right")
    axes[1].set_yticklabels(short)
    axes[1].set_title("local parameter correlation")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_figure(fig, path_base)


def plot_cost(cases: list[TestCase], fits: dict[str, dict[str, FitResult]], path_base: Path) -> None:
    labels = []
    times = []
    variables = []
    for case in cases:
        prefix = "C1" if case.name == "noise_only" else "C2"
        for method in ("SS", "MS"):
            labels.append(f"{prefix} {method}")
            times.append(fits[case.name][method].elapsed_s)
            variables.append(fits[case.name][method].decision_variables)
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(7.4, 3.4))
    ax2 = ax1.twinx()
    ax1.bar(x - 0.18, times, 0.36, color="#2ca02c", label="solve time")
    ax2.bar(x + 0.18, variables, 0.36, color="#9467bd", label="decision variables")
    ax1.set_ylabel("solve time [s]")
    ax2.set_ylabel("decision variables")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(True, axis="y", alpha=0.25)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, path_base)


def save_figure(fig: plt.Figure, path_base: Path) -> None:
    fig.savefig(path_base.with_suffix(".svg"))
    fig.savefig(path_base.with_suffix(".png"), dpi=250)
    plt.close(fig)


def write_metadata(args: argparse.Namespace, cases: list[TestCase], fits: dict[str, dict[str, FitResult]]) -> None:
    payload = {
        "args": vars(args),
        "aircraft": asdict(Aircraft()),
        "true_theta": dict(zip(PARAMETER_NAMES, true_theta())),
        "initial_theta": dict(zip(PARAMETER_NAMES, initial_theta())),
        "cases": {
            case.name: {
                "duration_s": float(case.t[-1]),
                "samples": int(len(case.t)),
                "noise_std": dict(zip(STATE_NAMES, case.noise_std)),
            }
            for case in cases
        },
        "fits": {
            case_name: {
                method: {
                    "theta": dict(zip(PARAMETER_NAMES, fit.theta)),
                    "cost": fit.cost,
                    "nfev": fit.nfev,
                    "elapsed_s": fit.elapsed_s,
                    "success": fit.success,
                    "message": fit.message,
                    "decision_variables": fit.decision_variables,
                }
                for method, fit in method_fits.items()
            }
            for case_name, method_fits in fits.items()
        },
    }
    (RESULTS_DIR / "metadata.json").write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30.0, help="simulation duration in seconds")
    parser.add_argument("--dt", type=float, default=0.02, help="fixed integration step in seconds")
    parser.add_argument("--segments", type=int, default=6, help="multiple-shooting segment count")
    parser.add_argument("--seed", type=int, default=7, help="random seed for measurement noise and gust")
    parser.add_argument("--max-nfev-ss", type=int, default=80, help="max single-shooting function evaluations")
    parser.add_argument("--max-nfev-ms", type=int, default=100, help="max multiple-shooting function evaluations")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    aircraft = Aircraft()
    cases = make_cases(args.duration, args.dt, args.seed)

    fits: dict[str, dict[str, FitResult]] = {}
    for case in cases:
        print(f"Fitting {case.name}: single shooting")
        ss = fit_single_shooting(case, aircraft, args.dt, args.max_nfev_ss)
        print(f"  SS: cost={ss.cost:.3g}, nfev={ss.nfev}, time={ss.elapsed_s:.2f}s")
        print(f"Fitting {case.name}: multiple shooting")
        ms = fit_multiple_shooting(case, aircraft, args.dt, args.segments, args.max_nfev_ms)
        print(f"  MS: cost={ms.cost:.3g}, nfev={ms.nfev}, time={ms.elapsed_s:.2f}s")
        fits[case.name] = {"SS": ss, "MS": ms}

    save_summary(cases, fits, RESULTS_DIR / "oem_fit_summary.csv")
    write_metadata(args, cases, fits)
    plot_trajectories(cases, fits, FIG_DIR / "oem_ss_ms_trajectories")
    plot_parameter_errors(cases, fits, FIG_DIR / "oem_parameter_error")
    plot_rmse(cases, fits, FIG_DIR / "oem_trajectory_rmse")
    plot_cost(cases, fits, FIG_DIR / "oem_computational_cost")
    plot_identifiability(cases[1], fits[cases[1].name]["MS"], FIG_DIR / "oem_identifiability", aircraft, args.dt)
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote results to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
