#!/usr/bin/env python3
"""Run a shared train/validation comparison across system-identification methods."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import coherence, csd, savgol_filter, welch, windows

import casadi as ca

METHODS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(METHODS_ROOT))

from common.dataset import DEFAULT_DATASET, SplitData, load_dataset
from common.metrics import aggregate_trajectory_score, finite_difference_derivative, rmse
from common.paths import FIG_DIR, RESULTS_DIR, TABLE_DIR
from common.plotting import save_figure
from models.aircraft3dof.longitudinal import Aircraft, NominalAero, mocap_from_state


STATE_NAMES = ["V", "alpha", "gamma", "Q"]
AERO_COEFFICIENT_NAMES = ["CL", "CD", "Cm"]
COEFF_NAMES = ["C_L", "C_D", "C_M"]
PARAMETER_NAMES = ["C_L0", "C_L_alpha", "C_D0", "k", "C_M0", "C_M_alpha", "C_M_Q", "C_M_delta_e"]
STATE_LABELS = [r"$V$ [m/s]", r"$\alpha$ [deg]", r"$\gamma$ [deg]", r"$Q$ [deg/s]"]
WEB_TRACE_MAX_POINTS = 100


def default_stitch_train_dataset(dataset: Path) -> Path:
    candidate = dataset.parent / "longitudinal_3dof_nonlinear_trim_grid"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Stitched local-model methods require the trim-grid training dataset at {candidate}. "
            "Generate it with './results.py simulate --dataset-modes trim_grid'."
        )
    return candidate
STATE_NOISE = np.array([0.08, 0.0035, 0.0035, 0.012])
THETA_LOWER = np.array([-0.5, 0.0, 0.0, 0.0, -1.0, -5.0, -50.0, -3.0])
THETA_UPPER = np.array([0.5, 6.0, 0.2, 0.5, 1.0, 3.0, 0.0, 3.0])
# Outer state box for shooting nodes/initial states; matches the simulator's
# MAX_ENVELOPE_* limits so feasible trajectories are never clipped.
STATE_BOX_LOWER = np.array([3.0, -0.8, -1.25, -3.5])
STATE_BOX_UPPER = np.array([40.0, 0.8, 1.25, 3.5])


@dataclass
class MethodResult:
    name: str
    description: str
    train_elapsed_s: float
    rollout_elapsed_s: float
    validation_trajectories: np.ndarray
    validation_coeff_residual: np.ndarray | None
    validation_outputs: np.ndarray | None
    decision_variables: int
    train_samples: int
    notes: str
    train_loss_final: float = np.nan
    train_cpu_s: float = np.nan
    train_gpu_s: float = np.nan
    gpu_memory_mb: float = np.nan
    backend: str = "numpy"
    implementation_status: str = "implemented"
    evaluation_mode: str = "open_loop"
    diverged: bool = False


class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int):
        super().__init__()
        layers: list[torch.nn.Module] = [torch.nn.Linear(in_dim, width), torch.nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([torch.nn.Linear(width, width), torch.nn.Tanh()])
        layers.append(torch.nn.Linear(width, out_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def nominal_theta() -> np.ndarray:
    aero = NominalAero()
    return aero.as_array()


def theta_coefficients(x: np.ndarray, u: np.ndarray, theta: np.ndarray, aircraft: Aircraft = Aircraft()) -> np.ndarray:
    v, alpha, _, q_rate = x
    _, elevator = u
    cl0, cla, cd0, k_drag, cm0, cma, cmq, cme = theta
    q_hat = aircraft.chord * q_rate / (2.0 * max(v, 3.0))
    c_l = cl0 + cla * alpha
    c_d = cd0 + k_drag * c_l**2
    c_m = cm0 + cma * alpha + cmq * q_hat + cme * elevator
    return np.array([c_l, c_d, c_m])


def dynamics_from_coefficients(x: np.ndarray, u: np.ndarray, coeff: np.ndarray, aircraft: Aircraft) -> np.ndarray:
    v, alpha, gamma, q_rate = x
    thrust, _ = u
    v_safe = max(v, 3.0)
    qbar = 0.5 * aircraft.rho * v_safe**2
    lift, drag, moment = coeff * qbar * aircraft.wing_area
    moment *= aircraft.chord
    v_dot = (-drag + thrust * np.cos(alpha) - aircraft.mass * aircraft.gravity * np.sin(gamma)) / aircraft.mass
    gamma_dot = (lift + thrust * np.sin(alpha) - aircraft.mass * aircraft.gravity * np.cos(gamma)) / (
        aircraft.mass * v_safe
    )
    q_dot = moment / aircraft.jy
    alpha_dot = q_rate - gamma_dot
    return np.array([v_dot, alpha_dot, gamma_dot, q_dot])


def casadi_dynamics_from_coefficients(x, u: np.ndarray, coeff, aircraft: Aircraft):
    v = ca.fmax(x[0], 3.0)
    alpha = x[1]
    gamma = x[2]
    thrust = u[0]
    qbar = 0.5 * aircraft.rho * v**2
    lift = coeff[0] * qbar * aircraft.wing_area
    drag = coeff[1] * qbar * aircraft.wing_area
    moment = coeff[2] * qbar * aircraft.wing_area * aircraft.chord
    v_dot = (-drag + thrust * ca.cos(alpha) - aircraft.mass * aircraft.gravity * ca.sin(gamma)) / aircraft.mass
    gamma_dot = (lift + thrust * ca.sin(alpha) - aircraft.mass * aircraft.gravity * ca.cos(gamma)) / (
        aircraft.mass * v
    )
    q_dot = moment / aircraft.jy
    alpha_dot = x[3] - gamma_dot
    return ca.vertcat(v_dot, alpha_dot, gamma_dot, q_dot)


def theta_dynamics(x: np.ndarray, u: np.ndarray, theta: np.ndarray, aircraft: Aircraft) -> np.ndarray:
    return dynamics_from_coefficients(x, u, theta_coefficients(x, u, theta), aircraft)


def casadi_theta_dynamics(x, u: np.ndarray, theta, aircraft: Aircraft):
    v = ca.fmax(x[0], 3.0)
    alpha = x[1]
    gamma = x[2]
    q_rate = x[3]
    thrust = u[0]
    elevator = u[1]
    cl = theta[0] + theta[1] * alpha
    cd = theta[2] + theta[3] * cl**2
    q_hat = aircraft.chord * q_rate / (2.0 * v)
    cm = theta[4] + theta[5] * alpha + theta[6] * q_hat + theta[7] * elevator
    qbar = 0.5 * aircraft.rho * v**2
    lift = cl * qbar * aircraft.wing_area
    drag = cd * qbar * aircraft.wing_area
    moment = cm * qbar * aircraft.wing_area * aircraft.chord
    v_dot = (-drag + thrust * ca.cos(alpha) - aircraft.mass * aircraft.gravity * ca.sin(gamma)) / aircraft.mass
    gamma_dot = (lift + thrust * ca.sin(alpha) - aircraft.mass * aircraft.gravity * ca.cos(gamma)) / (
        aircraft.mass * v
    )
    q_dot = moment / aircraft.jy
    alpha_dot = q_rate - gamma_dot
    return ca.vertcat(v_dot, alpha_dot, gamma_dot, q_dot)


def casadi_controller_command(x, pilot_u: np.ndarray, trim_u: np.ndarray, ctrl):
    theta = x[1] + x[2]
    throttle_norm = ca.fmin(ca.fmax(pilot_u[0] / 3.0, 0.0), 1.0)
    elevator_stick = ca.fmin(ca.fmax((pilot_u[1] - trim_u[1]) / 0.20, -1.0), 1.0)
    theta_cmd = ctrl[0] + ctrl[1] * (throttle_norm - 0.5) + ctrl[2] * elevator_stick
    q_cmd = ca.fmin(ca.fmax(ctrl[3] * (theta_cmd - theta), -1.3), 1.3)
    safe_elevator = trim_u[1] + ctrl[4] * (q_cmd - x[3])
    as3x_elevator = pilot_u[1] - ctrl[5] * x[3]
    alpha_gate = 1.0 / (1.0 + ca.exp(-((x[1] - ctrl[6]) / 0.012)))
    speed_gate = 1.0 / (1.0 + ca.exp(-((ctrl[7] - x[0]) / 0.55)))
    recovery = 1.0 - (1.0 - alpha_gate) * (1.0 - speed_gate)
    panic_theta_cmd = ctrl[0] + ctrl[8]
    panic_q_cmd = ca.fmin(ca.fmax(ctrl[9] * (panic_theta_cmd - theta), -1.3), 1.3)
    panic_elevator = trim_u[1] + ctrl[10] * (panic_q_cmd - x[3])
    elevator_no_panic = ctrl[11] * safe_elevator + (1.0 - ctrl[11]) * as3x_elevator
    elevator = ca.fmin(
        ca.fmax(
            (1.0 - recovery) * elevator_no_panic + recovery * panic_elevator,
            -0.35,
        ),
        0.35,
    )
    return ca.vertcat(ca.fmin(ca.fmax(pilot_u[0], 0.0), 3.0), elevator)


def controller_command_np(x: np.ndarray, pilot_u: np.ndarray, trim_u: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    theta = x[1] + x[2]
    throttle_norm = np.clip(pilot_u[0] / 3.0, 0.0, 1.0)
    elevator_stick = np.clip((pilot_u[1] - trim_u[1]) / 0.20, -1.0, 1.0)
    theta_cmd = ctrl[0] + ctrl[1] * (throttle_norm - 0.5) + ctrl[2] * elevator_stick
    q_cmd = np.clip(ctrl[3] * (theta_cmd - theta), -1.3, 1.3)
    safe_elevator = trim_u[1] + ctrl[4] * (q_cmd - x[3])
    as3x_elevator = pilot_u[1] - ctrl[5] * x[3]
    alpha_gate = 1.0 / (1.0 + np.exp(-np.clip((x[1] - ctrl[6]) / 0.012, -60.0, 60.0)))
    speed_gate = 1.0 / (1.0 + np.exp(-np.clip((ctrl[7] - x[0]) / 0.55, -60.0, 60.0)))
    recovery = 1.0 - (1.0 - alpha_gate) * (1.0 - speed_gate)
    panic_q_cmd = np.clip(ctrl[9] * (ctrl[0] + ctrl[8] - theta), -1.3, 1.3)
    panic_elevator = trim_u[1] + ctrl[10] * (panic_q_cmd - x[3])
    elevator_no_panic = ctrl[11] * safe_elevator + (1.0 - ctrl[11]) * as3x_elevator
    return np.array(
        [
            np.clip(pilot_u[0], 0.0, 3.0),
            np.clip((1.0 - recovery) * elevator_no_panic + recovery * panic_elevator, -0.35, 0.35),
        ]
    )


def theta_controller_dynamics(
    x: np.ndarray,
    pilot_u: np.ndarray,
    theta: np.ndarray,
    ctrl: np.ndarray,
    trim_u: np.ndarray,
    aircraft: Aircraft,
) -> np.ndarray:
    return theta_dynamics(x, controller_command_np(x, pilot_u, trim_u, ctrl), theta, aircraft)


def casadi_rk4_step(x, u0: np.ndarray, u1: np.ndarray, theta, aircraft: Aircraft, dt: float):
    umid = 0.5 * (u0 + u1)
    k1 = casadi_theta_dynamics(x, u0, theta, aircraft)
    k2 = casadi_theta_dynamics(x + 0.5 * dt * k1, umid, theta, aircraft)
    k3 = casadi_theta_dynamics(x + 0.5 * dt * k2, umid, theta, aircraft)
    k4 = casadi_theta_dynamics(x + dt * k3, u1, theta, aircraft)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def casadi_controller_rk4_step(x, u0: np.ndarray, u1: np.ndarray, trim_u: np.ndarray, theta, ctrl, aircraft: Aircraft, dt: float):
    umid = 0.5 * (u0 + u1)
    u0_internal = casadi_controller_command(x, u0, trim_u, ctrl)
    k1 = casadi_theta_dynamics(x, u0_internal, theta, aircraft)
    x2 = x + 0.5 * dt * k1
    k2 = casadi_theta_dynamics(x2, casadi_controller_command(x2, umid, trim_u, ctrl), theta, aircraft)
    x3 = x + 0.5 * dt * k2
    k3 = casadi_theta_dynamics(x3, casadi_controller_command(x3, umid, trim_u, ctrl), theta, aircraft)
    x4 = x + dt * k3
    k4 = casadi_theta_dynamics(x4, casadi_controller_command(x4, u1, trim_u, ctrl), theta, aircraft)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def make_casadi_rk4_step_jacobian(aircraft: Aircraft) -> ca.Function:
    x = ca.MX.sym("x", 4)
    u0 = ca.MX.sym("u0", 2)
    u1 = ca.MX.sym("u1", 2)
    theta = ca.MX.sym("theta", 8)
    dt = ca.MX.sym("dt")
    x_next = casadi_rk4_step(x, u0, u1, theta, aircraft, dt)
    transition_jacobian = ca.jacobian(x_next, x)
    return ca.Function("rk4_step_jacobian", [x, u0, u1, theta, dt], [x_next, transition_jacobian])


def make_casadi_rk4_step_parameter_jacobian(aircraft: Aircraft) -> ca.Function:
    x = ca.MX.sym("x", 4)
    u0 = ca.MX.sym("u0", 2)
    u1 = ca.MX.sym("u1", 2)
    theta = ca.MX.sym("theta", 8)
    dt = ca.MX.sym("dt")
    x_next = casadi_rk4_step(x, u0, u1, theta, aircraft, dt)
    transition_jacobian = ca.jacobian(x_next, x)
    parameter_jacobian = ca.jacobian(x_next, theta)
    return ca.Function(
        "rk4_step_parameter_jacobian",
        [x, u0, u1, theta, dt],
        [x_next, transition_jacobian, parameter_jacobian],
    )


def coefficient_residual_dynamics(
    x: np.ndarray,
    u: np.ndarray,
    theta: np.ndarray,
    aircraft: Aircraft,
    residual_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    coeff = theta_coefficients(x, u, theta) + residual_fn(x, u)
    return dynamics_from_coefficients(x, u, coeff, aircraft)


def input_corrected_dynamics(
    x: np.ndarray,
    u: np.ndarray,
    theta: np.ndarray,
    aircraft: Aircraft,
    correction_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    correction = correction_fn(x, u)
    u_eff = np.array([np.clip(u[0] + correction[0], 0.0, 3.0), np.clip(u[1] + correction[1], -0.35, 0.35)])
    return theta_dynamics(x, u_eff, theta, aircraft)


def rk4_step(
    rhs: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    u0: np.ndarray,
    u1: np.ndarray,
    dt: float,
) -> np.ndarray:
    umid = 0.5 * (u0 + u1)
    k1 = rhs(x, u0)
    k2 = rhs(x + 0.5 * dt * k1, umid)
    k3 = rhs(x + 0.5 * dt * k2, umid)
    k4 = rhs(x + dt * k3, u1)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next[0] = max(x_next[0], 3.0)
    return x_next


def finite_difference_jacobian(fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, step: np.ndarray) -> np.ndarray:
    y0 = fn(x)
    jac = np.empty((len(y0), len(x)))
    for idx in range(len(x)):
        dx = np.zeros_like(x)
        dx[idx] = step[idx]
        jac[:, idx] = (fn(x + dx) - fn(x - dx)) / (2.0 * step[idx])
    return jac


# Generous physical envelope for the 3DOF state [V, gamma, alpha, Q]; a model
# that leaves it has failed physically and is frozen + flagged, instead of an
# exponential riding to the old norm-1e4 threshold and posting alpha-RMSE
# values of hundreds of radians as a finite "score".
STATE_ENVELOPE_LOW = np.array([0.5, -2.0, -1.5, -50.0])
STATE_ENVELOPE_HIGH = np.array([100.0, 2.0, 1.5, 50.0])


def outside_envelope(x: np.ndarray) -> bool:
    return bool(np.any(x < STATE_ENVELOPE_LOW) or np.any(x > STATE_ENVELOPE_HIGH))


def simulate_trials(
    split: SplitData,
    rhs_factory: Callable[[int], Callable[[np.ndarray, np.ndarray], np.ndarray]],
) -> np.ndarray:
    trajectories = np.empty_like(split.x_true)
    for trial in range(split.n_trials):
        x = np.empty((split.n_time, 4))
        x[0] = split.y_meas[trial, 0]
        rhs = rhs_factory(trial)
        for k in range(split.n_time - 1):
            x[k + 1] = rk4_step(rhs, x[k], split.u_act[trial, k], split.u_act[trial, k + 1], split.dt)
            if not np.all(np.isfinite(x[k + 1])) or outside_envelope(x[k + 1]):
                # Keep one clipped out-of-envelope sample as evidence for the
                # diverged flag, then freeze the remainder at the last state.
                x[k + 1] = np.clip(np.nan_to_num(x[k + 1], nan=1.1 * STATE_ENVELOPE_HIGH[0]),
                                   1.1 * STATE_ENVELOPE_LOW, 1.1 * STATE_ENVELOPE_HIGH)
                x[k + 2 :] = x[k]
                break
        trajectories[trial] = x
    return trajectories


def smooth_trials(y: np.ndarray, dt: float, window: int, polyorder: int) -> tuple[np.ndarray, np.ndarray]:
    window = min(window, y.shape[1] - (1 - y.shape[1] % 2))
    window = max(window if window % 2 == 1 else window - 1, polyorder + 3)
    if window % 2 == 0:
        window += 1
    x_smooth = savgol_filter(y, window_length=window, polyorder=polyorder, axis=1, mode="interp")
    dxdt = np.gradient(x_smooth, dt, axis=1, edge_order=2)
    return x_smooth, dxdt


def flatten_samples(split: SplitData, x: np.ndarray, dxdt: np.ndarray, max_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xu = np.concatenate((x, split.u_act), axis=2).reshape(-1, 6)
    deriv = dxdt.reshape(-1, 4)
    coeff_res = split.coeff_residual.reshape(-1, 3)
    if max_samples > 0 and len(xu) > max_samples:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(xu), size=max_samples, replace=False)
        xu = xu[keep]
        deriv = deriv[keep]
        coeff_res = coeff_res[keep]
    return xu, deriv, coeff_res


def infer_coefficients(x: np.ndarray, u: np.ndarray, dxdt: np.ndarray, aircraft: Aircraft) -> np.ndarray:
    v = np.maximum(x[:, 0], 3.0)
    alpha = x[:, 1]
    gamma = x[:, 2]
    thrust = u[:, 0]
    qbar_s = 0.5 * aircraft.rho * v**2 * aircraft.wing_area
    drag = thrust * np.cos(alpha) - aircraft.mass * aircraft.gravity * np.sin(gamma) - aircraft.mass * dxdt[:, 0]
    lift = aircraft.mass * v * dxdt[:, 2] - thrust * np.sin(alpha) + aircraft.mass * aircraft.gravity * np.cos(gamma)
    moment = aircraft.jy * dxdt[:, 3]
    return np.column_stack((lift / qbar_s, drag / qbar_s, moment / (qbar_s * aircraft.chord)))


def infer_input_correction(xu: np.ndarray, dxdt: np.ndarray, theta: np.ndarray, aircraft: Aircraft) -> np.ndarray:
    x = xu[:, :4]
    u = xu[:, 4:]
    v = np.maximum(x[:, 0], 3.0)
    alpha = x[:, 1]
    gamma = x[:, 2]
    qbar_s = 0.5 * aircraft.rho * v**2 * aircraft.wing_area
    coeff_nom = np.vstack([theta_coefficients(xk, uk, theta) for xk, uk in zip(x, u)])
    drag_nom = coeff_nom[:, 1] * qbar_s
    thrust_eff = (aircraft.mass * dxdt[:, 0] + drag_nom + aircraft.mass * aircraft.gravity * np.sin(gamma)) / np.maximum(
        np.cos(alpha),
        0.25,
    )
    cm_eff = aircraft.jy * dxdt[:, 3] / (qbar_s * aircraft.chord)
    q_hat = aircraft.chord * x[:, 3] / (2.0 * v)
    elevator_eff = (cm_eff - theta[4] - theta[5] * alpha - theta[6] * q_hat) / max(abs(theta[7]), 1e-6)
    correction = np.column_stack((thrust_eff - u[:, 0], elevator_eff - u[:, 1]))
    correction[:, 0] = np.clip(correction[:, 0], -0.45, 0.45)
    correction[:, 1] = np.clip(correction[:, 1], -0.22, 0.22)
    return correction


def fit_equation_error(xu: np.ndarray, dxdt: np.ndarray) -> np.ndarray:
    aircraft = Aircraft()
    x = xu[:, :4]
    u = xu[:, 4:]
    coeff = infer_coefficients(x, u, dxdt, aircraft)
    alpha = x[:, 1]
    q_hat = aircraft.chord * x[:, 3] / (2.0 * np.maximum(x[:, 0], 3.0))
    elevator = u[:, 1]
    cl_theta = np.linalg.lstsq(np.column_stack((np.ones_like(alpha), alpha)), coeff[:, 0], rcond=None)[0]
    cd_theta = np.linalg.lstsq(np.column_stack((np.ones_like(alpha), coeff[:, 0] ** 2)), coeff[:, 1], rcond=None)[0]
    cm_theta = np.linalg.lstsq(np.column_stack((np.ones_like(alpha), alpha, q_hat, elevator)), coeff[:, 2], rcond=None)[0]
    theta = np.array([cl_theta[0], cl_theta[1], cd_theta[0], cd_theta[1], *cm_theta])
    return np.clip(theta, THETA_LOWER, THETA_UPPER)


def sindy_library(xu: np.ndarray) -> tuple[np.ndarray, list[str]]:
    v, alpha, gamma, q_rate, thrust, elevator = xu.T
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
        q_rate * elevator,
        thrust * alpha,
        alpha**2,
        gamma**2,
        q_rate**2,
        elevator**2,
        np.sin(alpha),
        np.sin(gamma),
        np.cos(alpha),
        np.cos(gamma),
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
        "Q delta_e",
        "T alpha",
        "alpha^2",
        "gamma^2",
        "Q^2",
        "delta_e^2",
        "sin(alpha)",
        "sin(gamma)",
        "cos(alpha)",
        "cos(gamma)",
    ]
    return np.column_stack(columns), names


def structured_sindy_feature_blocks(xu: np.ndarray) -> list[tuple[np.ndarray, list[str], np.ndarray]]:
    v, alpha, _, q_rate, _, elevator = xu.T
    aircraft = Aircraft()
    # base pitch-rate feature matches the parametric model's non-dimensional q_hat
    q_hat = aircraft.chord * q_rate / (2.0 * np.maximum(v, 3.0))
    residual_library, residual_names = sindy_library(xu)
    blocks = [
        (
            np.column_stack((np.ones_like(alpha), alpha, residual_library[:, [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19]])),
            ["base:1", "base:alpha", *[f"res:{residual_names[idx]}" for idx in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19]]],
            np.array([True, True, *([False] * 17)]),
        ),
        (
            np.column_stack((np.ones_like(alpha), alpha, alpha**2, residual_library[:, [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19]])),
            ["base:1", "base:alpha", "base:alpha^2", *[f"res:{residual_names[idx]}" for idx in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19]]],
            np.array([True, True, True, *([False] * 16)]),
        ),
        (
            np.column_stack((np.ones_like(alpha), alpha, q_hat, elevator, residual_library[:, [1, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19]])),
            ["base:1", "base:alpha", "base:Q_hat", "base:delta_e", *[f"res:{residual_names[idx]}" for idx in [1, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19]]],
            np.array([True, True, True, True, *([False] * 14)]),
        ),
    ]
    return blocks


def fit_protected_sparse_regression(
    features: np.ndarray,
    target: np.ndarray,
    protected: np.ndarray,
    threshold: float,
    ridge: float,
) -> tuple[np.ndarray, float]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    constant = scale < 1e-12
    mean[constant] = 0.0
    scale[constant] = 1.0
    scale[scale < 1e-12] = 1.0
    features_s = (features - mean) / scale
    coeff_s = np.linalg.solve(features_s.T @ features_s + ridge * np.eye(features_s.shape[1]), features_s.T @ target)
    for _ in range(8):
        keep = protected | (np.abs(coeff_s) >= threshold)
        if not np.any(keep):
            keep[np.argmax(np.abs(coeff_s))] = True
        lhs = features_s[:, keep].T @ features_s[:, keep] + ridge * np.eye(np.count_nonzero(keep))
        rhs = features_s[:, keep].T @ target
        coeff_s[:] = 0.0
        coeff_s[keep] = np.linalg.solve(lhs, rhs)
    coeff = coeff_s / scale
    intercept = -np.sum((mean / scale) * coeff_s)
    constant_indices = np.flatnonzero(constant)
    if len(constant_indices):
        coeff[constant_indices[0]] += intercept
    train_mse = float(np.mean((features @ coeff - target) ** 2))
    return coeff, train_mse


def fit_sindy(xu: np.ndarray, target: np.ndarray, threshold: float, ridge: float) -> tuple[list[np.ndarray], list[list[str]], float]:
    coeffs: list[np.ndarray] = []
    names: list[list[str]] = []
    losses: list[float] = []
    for output_idx, (features, feature_names, protected) in enumerate(structured_sindy_feature_blocks(xu)):
        coeff, loss = fit_protected_sparse_regression(features, target[:, output_idx], protected, threshold, ridge)
        coeffs.append(coeff)
        names.append(feature_names)
        losses.append(loss)
    return coeffs, names, float(np.mean(losses))


def structured_sindy_coefficients_np(x: np.ndarray, u: np.ndarray, coeffs: list[np.ndarray]) -> np.ndarray:
    blocks = structured_sindy_feature_blocks(np.concatenate((x, u))[None, :])
    return np.array([blocks[idx][0] @ coeffs[idx] for idx in range(3)]).ravel()


def structured_sindy_coefficients_casadi(x, u, coeffs: list) -> ca.MX:
    v = x[0]
    alpha = x[1]
    gamma = x[2]
    q_rate = x[3]
    thrust = u[0]
    elevator = u[1]
    q_hat = Aircraft().chord * q_rate / (2.0 * ca.fmax(v, 3.0))
    cl_features = ca.vertcat(
        1,
        alpha,
        v,
        gamma,
        q_rate,
        thrust,
        elevator,
        v * alpha,
        v * gamma,
        alpha * q_rate,
        q_rate * elevator,
        thrust * alpha,
        alpha**2,
        gamma**2,
        q_rate**2,
        elevator**2,
        ca.sin(gamma),
        ca.cos(alpha),
        ca.cos(gamma),
    )
    cd_features = ca.vertcat(
        1,
        alpha,
        alpha**2,
        v,
        gamma,
        q_rate,
        thrust,
        elevator,
        v * alpha,
        v * gamma,
        alpha * q_rate,
        q_rate * elevator,
        thrust * alpha,
        gamma**2,
        q_rate**2,
        elevator**2,
        ca.sin(gamma),
        ca.cos(alpha),
        ca.cos(gamma),
    )
    cm_features = ca.vertcat(
        1,
        alpha,
        q_hat,
        elevator,
        v,
        gamma,
        v * alpha,
        v * gamma,
        alpha * q_rate,
        q_rate * elevator,
        thrust * alpha,
        alpha**2,
        gamma**2,
        q_rate**2,
        elevator**2,
        ca.sin(gamma),
        ca.cos(alpha),
        ca.cos(gamma),
    )
    return ca.vertcat(ca.dot(cl_features, coeffs[0]), ca.dot(cd_features, coeffs[1]), ca.dot(cm_features, coeffs[2]))


def fit_koopman_edmd(
    train_x: np.ndarray,
    train_u: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    xk = train_x[:, :-1, :].reshape(-1, 4)
    uk = train_u[:, :-1, :].reshape(-1, 2)
    xkp1 = train_x[:, 1:, :].reshape(-1, 4)
    library, names = sindy_library(np.column_stack((xk, uk)))
    mean = library.mean(axis=0)
    scale = library.std(axis=0)
    mean[0] = 0.0
    scale[0] = 1.0
    scale[scale < 1e-12] = 1.0
    library_s = (library - mean) / scale
    gram = library_s.T @ library_s + ridge * np.eye(library_s.shape[1])
    coeff = np.linalg.solve(gram, library_s.T @ xkp1)
    return coeff, mean, scale, names


def run_koopman_edmd(
    train_x: np.ndarray,
    train_u: np.ndarray,
    validation: SplitData,
    args: argparse.Namespace,
) -> MethodResult:
    start = time.perf_counter()
    coeff, mean, scale, names = fit_koopman_edmd(train_x, train_u, args.koopman_ridge)
    elapsed = time.perf_counter() - start

    def step(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        library, _ = sindy_library(np.concatenate((x, u))[None, :])
        x_next = ((library - mean) / scale @ coeff).ravel()
        x_next[0] = max(x_next[0], 3.0)
        x_next[1:] = np.clip(x_next[1:], [-0.75, -0.75, -3.0], [0.75, 0.75, 3.0])
        return x_next

    start_rollout = time.perf_counter()
    trajectories = np.empty_like(validation.x_true)
    for trial in range(validation.n_trials):
        x = np.empty((validation.n_time, 4))
        x[0] = validation.y_meas[trial, 0]
        for k in range(validation.n_time - 1):
            x[k + 1] = step(x[k], validation.u_act[trial, k])
            if not np.all(np.isfinite(x[k + 1])) or np.linalg.norm(x[k + 1]) > 1e4:
                x[k + 1 :] = x[k]
                break
        trajectories[trial] = x
    rollout_elapsed = time.perf_counter() - start_rollout
    return MethodResult(
        name="Koopman-EDMD",
        description="Lifted linear discrete-time predictor fitted by extended dynamic mode decomposition.",
        train_elapsed_s=elapsed,
        rollout_elapsed_s=rollout_elapsed,
        validation_trajectories=trajectories,
        validation_coeff_residual=None,
        validation_outputs=None,
        decision_variables=int(coeff.size + mean.size + scale.size),
        train_samples=int(train_x.shape[0] * (train_x.shape[1] - 1)),
        notes=(
            "Uses the same polynomial/trigonometric library as SINDy, but fits a one-step lifted map "
            "rather than sparse continuous-time derivatives; validation is open-loop from x0 and the "
            "supplied pilot-command history."
        ),
        backend="NumPy EDMD",
    )


def fit_rbf_surrogate(
    xu: np.ndarray,
    target: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    rng = np.random.default_rng(args.seed)
    max_centers = min(int(args.gp_centers), len(xu))
    center_idx = rng.choice(len(xu), size=max_centers, replace=False)
    x_mean = xu.mean(axis=0)
    x_scale = xu.std(axis=0)
    x_scale[x_scale < 1e-12] = 1.0
    z = (xu - x_mean) / x_scale
    centers = z[center_idx]
    y_lower = np.quantile(target, 0.002, axis=0)
    y_upper = np.quantile(target, 0.998, axis=0)
    target = np.clip(target, y_lower, y_upper)
    y_mean = target.mean(axis=0)
    y_scale = target.std(axis=0)
    y_scale[y_scale < 1e-12] = 1.0
    y = (target - y_mean) / y_scale
    if args.gp_length_scale > 0.0:
        length_scale = float(args.gp_length_scale)
    else:
        subset = centers[: min(len(centers), 256)]
        diff = subset[:, None, :] - subset[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        length_scale = float(np.median(dist[dist > 0.0])) if np.any(dist > 0.0) else 1.0
    diff = z[:, None, :] - centers[None, :, :]
    phi = np.exp(-0.5 * np.sum(diff**2, axis=2) / max(length_scale**2, 1e-12))
    phi = np.column_stack((np.ones(len(phi)), phi))
    lhs = phi.T @ phi + args.gp_ridge * np.eye(phi.shape[1])
    weights = np.linalg.solve(lhs, phi.T @ y)
    train_mse = float(np.mean((phi @ weights - y) ** 2))
    return centers, weights, x_mean, x_scale, y_mean, y_scale, y_lower, y_upper, length_scale, train_mse


def make_rbf_predictor(
    centers: np.ndarray,
    weights: np.ndarray,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    length_scale: float,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def predict(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        z = (np.concatenate((x, u)) - x_mean) / x_scale
        diff = centers - z
        phi = np.exp(-0.5 * np.sum(diff**2, axis=1) / max(length_scale**2, 1e-12))
        phi = np.concatenate(([1.0], phi))
        return np.clip(phi @ weights * y_scale + y_mean, y_lower, y_upper)

    return predict


def run_gp_coeff_closure(
    xu: np.ndarray,
    dxdt: np.ndarray,
    validation: SplitData,
    args: argparse.Namespace,
) -> MethodResult:
    aircraft = Aircraft()
    theta = nominal_theta()
    coeff_inferred = infer_coefficients(xu[:, :4], xu[:, 4:], dxdt, aircraft)
    coeff_nominal = np.vstack([theta_coefficients(x, u, theta) for x, u in zip(xu[:, :4], xu[:, 4:])])
    target = coeff_inferred - coeff_nominal
    start = time.perf_counter()
    centers, weights, x_mean, x_scale, y_mean, y_scale, y_lower, y_upper, length_scale, train_mse = fit_rbf_surrogate(
        xu,
        target,
        args,
    )
    elapsed = time.perf_counter() - start
    coeff_residual_fn = make_rbf_predictor(centers, weights, x_mean, x_scale, y_mean, y_scale, y_lower, y_upper, length_scale)
    result = evaluate_method(
        "GP-CoeffClosure",
        "Sparse RBF/Gaussian-process-style surrogate for aerodynamic coefficient residuals.",
        elapsed,
        len(xu),
        int(weights.size + centers.size),
        validation,
        lambda _trial: lambda x, u: coefficient_residual_dynamics(
            x,
            u,
            theta,
            aircraft,
            lambda x_local, u_local: args.pinn_gain * coeff_residual_fn(x_local, u_local),
        ),
        lambda _trial: lambda x, u: args.pinn_gain * coeff_residual_fn(x, u),
        (
            "Finite-basis RBF approximation to a GP coefficient closure trained from EOM-inferred coefficient residuals; "
            f"centers={len(centers)}, length_scale={length_scale:.3g}, residual gain={args.pinn_gain:g}."
        ),
    )
    result.backend = "NumPy RBF/KRR"
    result.train_loss_final = train_mse
    return result


def train_mlp(
    xu: np.ndarray,
    target: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    out_dim: int,
) -> tuple[MLP, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, float]], float, float, float, float]:
    y_lower = np.quantile(target, 0.002, axis=0)
    y_upper = np.quantile(target, 0.998, axis=0)
    target = np.clip(target, y_lower, y_upper)
    x_mean = xu.mean(axis=0)
    x_scale = xu.std(axis=0)
    x_scale[x_scale < 1e-12] = 1.0
    y_mean = target.mean(axis=0)
    y_scale = target.std(axis=0)
    y_scale[y_scale < 1e-12] = 1.0
    x_t = torch.tensor((xu - x_mean) / x_scale, dtype=torch.float32, device=device)
    y_t = torch.tensor((target - y_mean) / y_scale, dtype=torch.float32, device=device)
    net = MLP(6, out_dim, args.width, args.depth).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[tuple[int, float]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record()
    else:
        gpu_start = None
        gpu_end = None
    start = time.perf_counter()
    start_cpu = time.process_time()
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        pred = net(x_t)
        loss = torch.mean((pred - y_t) ** 2)
        loss.backward()
        optimizer.step()
        if epoch % max(args.log_every, 1) == 0 or epoch == args.epochs - 1:
            history.append((epoch, float(loss.detach().cpu())))
    if device.type == "cuda" and gpu_start is not None:
        gpu_end.record()
        torch.cuda.synchronize(device)
        gpu_elapsed = float(gpu_start.elapsed_time(gpu_end) / 1000.0)
        gpu_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
    else:
        gpu_elapsed = np.nan
        gpu_memory_mb = np.nan
    elapsed = time.perf_counter() - start
    cpu_elapsed = time.process_time() - start_cpu
    return net, x_mean, x_scale, y_mean, y_scale, y_lower, y_upper, history, elapsed, cpu_elapsed, gpu_elapsed, gpu_memory_mb


def make_mlp_predictor(
    net: MLP,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    device: torch.device,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for module in net.net:
        if isinstance(module, torch.nn.Linear):
            weights.append(module.weight.detach().cpu().numpy())
            biases.append(module.bias.detach().cpu().numpy())

    def predict(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        z = (np.concatenate((x, u)) - x_mean) / x_scale
        for layer_idx, (weight, bias) in enumerate(zip(weights, biases)):
            z = weight @ z + bias
            if layer_idx < len(weights) - 1:
                z = np.tanh(z)
        return np.clip(z * y_scale + y_mean, y_lower, y_upper)

    return predict


def evaluate_method(
    name: str,
    description: str,
    train_elapsed_s: float,
    train_samples: int,
    decision_variables: int,
    validation: SplitData,
    rhs_factory: Callable[[int], Callable[[np.ndarray, np.ndarray], np.ndarray]],
    coeff_residual_factory: Callable[[int], Callable[[np.ndarray, np.ndarray], np.ndarray]] | None,
    notes: str,
) -> MethodResult:
    start = time.perf_counter()
    trajectories = simulate_trials(validation, rhs_factory)
    rollout_elapsed = time.perf_counter() - start
    # The rollout freezes a trial once it leaves the physical state envelope;
    # flag such trials so divergence is reported instead of masquerading as a
    # huge-but-finite score.
    diverged = bool(
        not np.all(np.isfinite(trajectories))
        or np.any(trajectories < STATE_ENVELOPE_LOW)
        or np.any(trajectories > STATE_ENVELOPE_HIGH)
    )
    coeff_pred = None
    if coeff_residual_factory is not None:
        coeff_pred = np.empty_like(validation.coeff_residual)
        for trial in range(validation.n_trials):
            fn = coeff_residual_factory(trial)
            for k in range(validation.n_time):
                coeff_pred[trial, k] = fn(trajectories[trial, k], validation.u_act[trial, k])
    return MethodResult(
        name=name,
        description=description,
        train_elapsed_s=train_elapsed_s,
        rollout_elapsed_s=rollout_elapsed,
        validation_trajectories=trajectories,
        validation_coeff_residual=coeff_pred,
        validation_outputs=None,
        decision_variables=decision_variables,
        train_samples=train_samples,
        notes=notes,
        diverged=diverged,
    )


def run_nominal(validation: SplitData) -> MethodResult:
    theta = nominal_theta()
    aircraft = Aircraft()
    return evaluate_method(
        "Nominal-GreyBox",
        "Known nominal rigid-body and aerodynamic model, no fitted correction.",
        0.0,
        0,
        len(theta),
        validation,
        lambda _trial: lambda x, u: theta_dynamics(x, u, theta, aircraft),
        lambda _trial: lambda _x, _u: np.zeros(3),
        "Baseline; expected to miss hidden nonlinear aerodynamic terms.",
    )


def run_equation_error(xu: np.ndarray, dxdt: np.ndarray, validation: SplitData) -> MethodResult:
    start = time.perf_counter()
    theta = fit_equation_error(xu, dxdt)
    elapsed = time.perf_counter() - start
    aircraft = Aircraft()
    return evaluate_method(
        "EquationError-LS",
        "Derivative-based least squares on inferred lift, drag, and moment coefficients.",
        elapsed,
        len(xu),
        len(theta),
        validation,
        lambda _trial: lambda x, u: theta_dynamics(x, u, theta, aircraft),
        None,
        "Fast and interpretable, but differentiating noisy measurements biases the fit.",
    )


def run_filter_error_ekf(
    train: SplitData,
    xu: np.ndarray,
    dxdt: np.ndarray,
    validation: SplitData,
    args: argparse.Namespace,
) -> MethodResult:
    start = time.perf_counter()
    theta = fit_equation_error(xu, dxdt)
    aircraft = Aircraft()
    step_jacobian = make_casadi_rk4_step_parameter_jacobian(aircraft)
    theta_lower = THETA_LOWER
    theta_upper = THETA_UPPER
    process_cov = np.diag(np.square(np.asarray(args.ekf_process_std, dtype=float)))
    theta_process_cov = np.diag(np.square(np.asarray(args.ekf_theta_process_std, dtype=float)))
    measurement_cov = np.diag(np.square(np.asarray(args.ekf_measurement_std, dtype=float)))
    state_initial_cov = np.diag(np.square(np.asarray(args.ekf_initial_std, dtype=float)))
    theta_cov = np.diag(np.square(np.asarray(args.ekf_theta_initial_std, dtype=float)))
    h_mat = np.zeros((4, 12))
    h_mat[:, :4] = np.eye(4)
    eye_aug = np.eye(12)
    trial_count = min(args.ekf_param_trials, train.n_trials)
    trial_ids = np.linspace(0, train.n_trials - 1, trial_count, dtype=int)
    stride = max(1, int(args.ekf_param_stride))

    for trial in trial_ids:
        x = train.y_meas[trial, 0].copy()
        p_cov = np.zeros((12, 12))
        p_cov[:4, :4] = state_initial_cov
        p_cov[4:, 4:] = theta_cov
        for k in range(0, train.n_time - stride, stride):
            u0 = train.u_act[trial, k]
            u1 = train.u_act[trial, k + stride]
            dt = float(train.t[k + stride] - train.t[k])
            x_pred_dm, f_x_dm, f_theta_dm = step_jacobian(x, u0, u1, theta, dt)
            x_pred = np.asarray(x_pred_dm).ravel()
            f_x = np.asarray(f_x_dm)
            f_theta = np.asarray(f_theta_dm)
            transition = np.eye(12)
            transition[:4, :4] = f_x
            transition[:4, 4:] = f_theta
            q_aug = np.zeros((12, 12))
            q_aug[:4, :4] = process_cov
            q_aug[4:, 4:] = theta_process_cov
            p_pred = transition @ p_cov @ transition.T + q_aug
            innovation = train.y_meas[trial, k + stride] - x_pred
            s_cov = h_mat @ p_pred @ h_mat.T + measurement_cov
            kalman_gain = np.linalg.solve(s_cov, h_mat @ p_pred).T
            correction = kalman_gain @ innovation
            x = x_pred + correction[:4]
            x[0] = max(x[0], 3.0)
            theta = np.clip(theta + correction[4:], theta_lower, theta_upper)
            p_cov = (eye_aug - kalman_gain @ h_mat) @ p_pred @ (eye_aug - kalman_gain @ h_mat).T + kalman_gain @ measurement_cov @ kalman_gain.T
        theta_cov = p_cov[4:, 4:]
    elapsed = time.perf_counter() - start

    result = evaluate_method(
        "EKF-ParamID",
        "Extended Kalman filter parameter-identification baseline; parameters are learned from training measurements then frozen for open-loop validation.",
        elapsed,
        int(trial_count * ((train.n_time - 1) // stride)),
        8 + 4 + 12 * 12,
        validation,
        lambda _trial: lambda x, u: theta_dynamics(x, u, theta, aircraft),
        None,
        (
            "Training-only augmented-state EKF estimates the aerodynamic parameter vector using "
            f"{trial_count} trials at stride {stride}; validation is a pilot-command-only open-loop rollout."
        ),
    )
    result.backend = "CasADi AD EKF"
    result.train_loss_final = float(np.trace(theta_cov))
    return result


def run_oem_casadi(train: SplitData, validation: SplitData, args: argparse.Namespace) -> MethodResult:
    theta0 = nominal_theta()
    theta_lower = THETA_LOWER
    theta_upper = THETA_UPPER
    state_lower = STATE_BOX_LOWER
    state_upper = STATE_BOX_UPPER
    aircraft = Aircraft()
    trial_count = min(args.max_oem_trials, train.n_trials)
    trial_ids = np.linspace(0, train.n_trials - 1, trial_count, dtype=int)
    stride = max(1, int(args.oem_stride))
    t_fit = train.t[::stride]
    noise = np.array([0.08, 0.0035, 0.0035, 0.012])
    # Multiple shooting: single shooting diverges on the aggressive post-stall
    # trajectories, leaving theta at (or returning exactly) the initial guess.
    segments = max(1, int(getattr(args, "oem_segments", 6)))
    segments = min(segments, len(t_fit) - 1)
    continuity_scale = np.maximum(0.5 * noise, np.array([0.02, 0.0015, 0.0015, 0.006]))
    indices = np.linspace(0, len(t_fit) - 1, segments + 1, dtype=int)

    theta = ca.MX.sym("theta", 8)
    nodes = ca.MX.sym("nodes", 4, trial_count * segments)
    residuals = []
    for local_idx, trial in enumerate(trial_ids):
        u = train.u_act[trial, ::stride]
        y = train.y_meas[trial, ::stride]
        for seg in range(segments):
            x = nodes[:, local_idx * segments + seg]
            for k in range(indices[seg], indices[seg + 1]):
                for state_idx in range(4):
                    residuals.append((x[state_idx] - y[k, state_idx]) / noise[state_idx])
                dt = float(t_fit[k + 1] - t_fit[k])
                x = casadi_rk4_step(x, u[k], u[k + 1], theta, aircraft, dt)
            if seg < segments - 1:
                next_node = nodes[:, local_idx * segments + seg + 1]
                for state_idx in range(4):
                    residuals.append((x[state_idx] - next_node[state_idx]) / continuity_scale[state_idx])
            else:
                k_last = int(indices[segments])
                for state_idx in range(4):
                    residuals.append((x[state_idx] - y[k_last, state_idx]) / noise[state_idx])

    z = ca.vertcat(theta, ca.reshape(nodes, 4 * trial_count * segments, 1))
    objective = ca.sumsqr(ca.vertcat(*residuals))
    solver = ca.nlpsol(
        "oem_state",
        "ipopt",
        {"x": z, "f": objective},
        {
            "ipopt.print_level": 0,
            "ipopt.max_iter": args.max_oem_nfev,
            "ipopt.tol": 1e-5,
            "print_time": False,
        },
    )
    node_guess = np.vstack([np.clip(train.y_meas[trial, ::stride][indices[:-1]], state_lower, state_upper) for trial in trial_ids])
    z0 = np.concatenate((theta0, node_guess.ravel(order="C")))
    lower = np.concatenate((theta_lower, np.tile(state_lower, trial_count * segments)))
    upper = np.concatenate((theta_upper, np.tile(state_upper, trial_count * segments)))
    start = time.perf_counter()
    solution = solver(x0=z0, lbx=lower, ubx=upper)
    elapsed = time.perf_counter() - start
    stats = solver.stats()
    return_status = str(stats.get("return_status", "unknown"))
    converged = return_status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
    if not converged:
        print(f"  WARNING: OEM-SS IPOPT did not converge ({return_status}); treat the OEM row with caution", flush=True)
    theta_hat = np.asarray(solution["x"]).ravel()[:8]
    result = evaluate_method(
        "OEM-SS",
        "Multiple-shooting output-error fit of the nominal aerodynamic parameter vector.",
        elapsed,
        int(trial_count * len(t_fit)),
        len(z0),
        validation,
        lambda _trial: lambda x, u: theta_dynamics(x, u, theta_hat, aircraft),
        None,
        (
            f"CasADi/IPOPT multiple shooting ({segments} segments) on {trial_count} trials with stride {stride}; "
            f"nonlinear residuals are unmodeled. Solver status: {return_status}."
        ),
    )
    result.backend = "CasADi/IPOPT"
    result.train_loss_final = float(solution["f"])
    return result


def run_oem(train: SplitData, validation: SplitData, args: argparse.Namespace) -> MethodResult:
    return run_oem_casadi(train, validation, args)


def run_oem_hidden_controller(train: SplitData, validation: SplitData, args: argparse.Namespace) -> MethodResult:
    theta0 = nominal_theta()
    ctrl0 = np.array([0.068, np.deg2rad(13.0), np.deg2rad(10.0), 3.0, 0.105, 0.018, 0.110, 13.0, np.deg2rad(2.0), 5.0, 0.125, 0.85])
    theta_lower = THETA_LOWER
    theta_upper = THETA_UPPER
    ctrl_lower = np.array([0.00, 0.0, 0.0, 0.5, 0.01, 0.0, 0.07, 10.0, -0.08, 1.0, 0.02, 0.0])
    ctrl_upper = np.array([0.18, 0.45, 0.45, 8.0, 0.25, 0.08, 0.18, 16.0, 0.12, 9.0, 0.25, 1.0])
    state_lower = STATE_BOX_LOWER
    state_upper = STATE_BOX_UPPER
    aircraft = Aircraft()
    trim_u = np.nanmean(train.trim_controls.reshape(-1, 2), axis=0)
    trial_count = min(args.max_oem_trials, train.n_trials)
    trial_ids = np.linspace(0, train.n_trials - 1, trial_count, dtype=int)
    stride = max(1, int(args.oem_stride))
    t_fit = train.t[::stride]
    noise = np.array([0.08, 0.0035, 0.0035, 0.012])

    theta = ca.MX.sym("theta", 8)
    ctrl = ca.MX.sym("ctrl", 12)
    x0_nodes = ca.MX.sym("x0", 4, trial_count)
    variables = [theta, ctrl, ca.reshape(x0_nodes, 4 * trial_count, 1)]
    residuals = []
    for local_idx, trial in enumerate(trial_ids):
        x = x0_nodes[:, local_idx]
        u = train.u_act[trial, ::stride]
        y = train.y_meas[trial, ::stride]
        for k in range(len(t_fit)):
            for state_idx in range(4):
                residuals.append((x[state_idx] - y[k, state_idx]) / noise[state_idx])
            if k == len(t_fit) - 1:
                continue
            dt = float(t_fit[k + 1] - t_fit[k])
            x = casadi_controller_rk4_step(x, u[k], u[k + 1], trim_u, theta, ctrl, aircraft, dt)

    z = ca.vertcat(*variables)
    objective = ca.sumsqr(ca.vertcat(*residuals)) + 1e-2 * ca.sumsqr(ctrl - ctrl0)
    solver = ca.nlpsol(
        "oem_hidden_controller",
        "ipopt",
        {"x": z, "f": objective},
        {
            "ipopt.print_level": 0,
            "ipopt.max_iter": args.max_oem_nfev,
            "ipopt.tol": 1e-5,
            "print_time": False,
        },
    )
    x0_guess = np.vstack([train.y_meas[trial, 0] for trial in trial_ids])
    z0 = np.concatenate((theta0, ctrl0, np.clip(x0_guess, state_lower, state_upper).ravel(order="F")))
    lower = np.concatenate((theta_lower, ctrl_lower, np.tile(state_lower, trial_count)))
    upper = np.concatenate((theta_upper, ctrl_upper, np.tile(state_upper, trial_count)))
    start = time.perf_counter()
    solution = solver(x0=z0, lbx=lower, ubx=upper)
    elapsed = time.perf_counter() - start
    z_hat = np.asarray(solution["x"]).ravel()
    theta_hat = z_hat[:8]
    ctrl_hat = z_hat[8:20]
    result = evaluate_method(
        "OEM-HiddenController",
        "Single-shooting OEM with an inferred smooth stall/recovery controller between pilot command and plant input.",
        elapsed,
        int(trial_count * len(t_fit)),
        len(z0),
        validation,
        lambda _trial: lambda x, u: theta_controller_dynamics(x, u, theta_hat, ctrl_hat, trim_u, aircraft),
        None,
        (
            f"CasADi/IPOPT fitted aerodynamic and controller parameters using {args.input_channel}; "
            "controller parameters describe throttle-to-pitch scheduling, elevator-to-pitch mapping, SAFE pitch gain, AS3X q damping, stall/recovery gates, panic pitch target, panic pitch gain, and SAFE/direct blending."
        ),
    )
    result.backend = "CasADi/IPOPT"
    result.train_loss_final = float(solution["f"])
    return result


def simulate_single_trajectory(
    x0: np.ndarray,
    u: np.ndarray,
    t: np.ndarray,
    rhs: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    x = np.empty((len(t), 4))
    x[0] = x0
    for k in range(len(t) - 1):
        x[k + 1] = rk4_step(rhs, x[k], u[k], u[k + 1], float(t[k + 1] - t[k]))
        if not np.all(np.isfinite(x[k + 1])) or np.linalg.norm(x[k + 1]) > 1e4:
            x[k + 1 :] = x[k]
            break
    return x


def run_oem_mocap_output_casadi(train: SplitData, validation: SplitData, args: argparse.Namespace) -> MethodResult:
    theta0 = nominal_theta()
    theta_lower = THETA_LOWER
    theta_upper = THETA_UPPER
    state_lower = STATE_BOX_LOWER
    state_upper = STATE_BOX_UPPER
    aircraft = Aircraft()
    trial_count = min(args.max_oem_trials, train.n_trials)
    trial_ids = np.linspace(0, train.n_trials - 1, trial_count, dtype=int)
    stride = max(1, int(args.oem_stride))
    t_fit = train.t[::stride]
    noise = np.array([args.mocap_position_noise, args.mocap_position_noise, args.mocap_attitude_noise])

    theta = ca.MX.sym("theta", 8)
    x0_nodes = ca.MX.sym("x0", 4, trial_count)
    variables = [theta, ca.reshape(x0_nodes, 4 * trial_count, 1)]
    residuals = []
    for local_idx, trial in enumerate(trial_ids):
        x = x0_nodes[:, local_idx]
        px = ca.MX(0.0)
        pz = ca.MX(0.0)
        u = train.u_act[trial, ::stride]
        y = train.mocap_meas[trial, ::stride]
        for k in range(len(t_fit)):
            residuals.extend([(px - y[k, 0]) / noise[0], (pz - y[k, 1]) / noise[1], (x[1] + x[2] - y[k, 2]) / noise[2]])
            if k == len(t_fit) - 1:
                continue
            dt = float(t_fit[k + 1] - t_fit[k])
            x_next = casadi_rk4_step(x, u[k], u[k + 1], theta, aircraft, dt)
            px = px + 0.5 * dt * (x[0] * ca.cos(x[2]) + x_next[0] * ca.cos(x_next[2]))
            pz = pz + 0.5 * dt * (x[0] * ca.sin(x[2]) + x_next[0] * ca.sin(x_next[2]))
            x = x_next

    z = ca.vertcat(*variables)
    objective = ca.sumsqr(ca.vertcat(*residuals))
    nlp = {"x": z, "f": objective}
    options = {
        "ipopt.print_level": 0,
        "ipopt.max_iter": args.max_oem_nfev,
        "ipopt.tol": 1e-5,
        "print_time": False,
    }
    solver = ca.nlpsol("oem_mocap", "ipopt", nlp, options)
    x0_guess = np.vstack([train.mocap_derived_state[trial, 0] for trial in trial_ids])
    z0 = np.concatenate((theta0, np.clip(x0_guess, state_lower, state_upper).ravel(order="F")))
    lower = np.concatenate((theta_lower, np.tile(state_lower, trial_count)))
    upper = np.concatenate((theta_upper, np.tile(state_upper, trial_count)))

    start = time.perf_counter()
    solution = solver(x0=z0, lbx=lower, ubx=upper)
    elapsed = time.perf_counter() - start
    theta_hat = np.asarray(solution["x"]).ravel()[:8]

    start_rollout = time.perf_counter()
    trajectories = np.empty_like(validation.x_true)
    outputs = np.empty_like(validation.mocap_meas)
    for trial in range(validation.n_trials):
        x0 = np.clip(validation.mocap_derived_state[trial, 0], state_lower, state_upper)
        x = simulate_single_trajectory(x0, validation.u_act[trial], validation.t, lambda xk, uk: theta_dynamics(xk, uk, theta_hat, aircraft))
        trajectories[trial] = x
        outputs[trial] = mocap_from_state(validation.t, x)
    rollout_elapsed = time.perf_counter() - start_rollout
    return MethodResult(
        name="OEM-MocapOutput",
        description="Output-error fit with latent flight states and mocap position/attitude as the measured output.",
        train_elapsed_s=elapsed,
        rollout_elapsed_s=rollout_elapsed,
        validation_trajectories=trajectories,
        validation_coeff_residual=None,
        validation_outputs=outputs,
        decision_variables=len(z0),
        train_samples=int(trial_count * len(t_fit)),
        notes="CasADi/IPOPT backend; preferred mocap formulation because it avoids differentiating position/attitude into noisy state targets.",
        backend="CasADi/IPOPT",
    )


def run_variational_mocap_output_casadi(train: SplitData, validation: SplitData, args: argparse.Namespace) -> MethodResult:
    """Multiple-shooting MAP/variational-style fit with latent states and mocap outputs.

    This is not a full VI implementation with a learned posterior covariance. It is the
    deterministic sparse-NLP analogue used here for the benchmark: latent state nodes are
    optimized directly while process residuals regularize them toward the flight-dynamics
    model. That places it between single-shooting OEM and filter/variational methods.
    """

    theta0 = nominal_theta()
    theta_lower = THETA_LOWER
    theta_upper = THETA_UPPER
    state_lower = STATE_BOX_LOWER
    state_upper = STATE_BOX_UPPER
    aircraft = Aircraft()
    trial_count = min(args.max_vi_trials, train.n_trials)
    trial_ids = np.linspace(0, train.n_trials - 1, trial_count, dtype=int)
    stride = max(1, int(args.vi_stride))
    t_fit = train.t[::stride]
    n_nodes = len(t_fit)
    meas_noise = np.array([args.mocap_position_noise, args.mocap_position_noise, args.mocap_attitude_noise])
    process_noise = np.array(args.vi_process_noise, dtype=float)
    pos_process_noise = float(args.vi_position_process_noise)

    theta = ca.MX.sym("theta", 8)
    variables = [theta]
    residuals = []
    z0_parts = [theta0]
    lower_parts = [theta_lower]
    upper_parts = [theta_upper]

    for trial in trial_ids:
        x_nodes = ca.MX.sym(f"x_{trial}", 4, n_nodes)
        p_nodes = ca.MX.sym(f"p_{trial}", 2, n_nodes)
        variables.extend([ca.reshape(x_nodes, 4 * n_nodes, 1), ca.reshape(p_nodes, 2 * n_nodes, 1)])

        x_guess = np.clip(train.mocap_derived_state[trial, ::stride], state_lower, state_upper)
        p_guess = train.mocap_meas[trial, ::stride, :2]
        z0_parts.extend([x_guess.T.ravel(order="F"), p_guess.T.ravel(order="F")])
        lower_parts.extend([np.tile(state_lower, n_nodes), np.tile([-np.inf, -np.inf], n_nodes)])
        upper_parts.extend([np.tile(state_upper, n_nodes), np.tile([np.inf, np.inf], n_nodes)])

        u = train.u_act[trial, ::stride]
        y = train.mocap_meas[trial, ::stride]
        for k in range(n_nodes):
            residuals.extend(
                [
                    (p_nodes[0, k] - y[k, 0]) / meas_noise[0],
                    (p_nodes[1, k] - y[k, 1]) / meas_noise[1],
                    (x_nodes[1, k] + x_nodes[2, k] - y[k, 2]) / meas_noise[2],
                ]
            )
            if k == n_nodes - 1:
                continue
            dt = float(t_fit[k + 1] - t_fit[k])
            x_pred = casadi_rk4_step(x_nodes[:, k], u[k], u[k + 1], theta, aircraft, dt)
            residuals.extend([(x_nodes[state_idx, k + 1] - x_pred[state_idx]) / process_noise[state_idx] for state_idx in range(4)])
            px_pred = p_nodes[0, k] + 0.5 * dt * (
                x_nodes[0, k] * ca.cos(x_nodes[2, k]) + x_nodes[0, k + 1] * ca.cos(x_nodes[2, k + 1])
            )
            pz_pred = p_nodes[1, k] + 0.5 * dt * (
                x_nodes[0, k] * ca.sin(x_nodes[2, k]) + x_nodes[0, k + 1] * ca.sin(x_nodes[2, k + 1])
            )
            residuals.extend([(p_nodes[0, k + 1] - px_pred) / pos_process_noise, (p_nodes[1, k + 1] - pz_pred) / pos_process_noise])

    z = ca.vertcat(*variables)
    objective = ca.sumsqr(ca.vertcat(*residuals))
    solver = ca.nlpsol(
        "variational_mocap",
        "ipopt",
        {"x": z, "f": objective},
        {
            "ipopt.print_level": 0,
            "ipopt.max_iter": args.max_vi_nfev,
            "ipopt.tol": 1e-5,
            "print_time": False,
        },
    )
    z0 = np.concatenate(z0_parts)
    lower = np.concatenate(lower_parts)
    upper = np.concatenate(upper_parts)
    start = time.perf_counter()
    solution = solver(x0=z0, lbx=lower, ubx=upper)
    elapsed = time.perf_counter() - start
    theta_hat = np.asarray(solution["x"]).ravel()[:8]

    start_rollout = time.perf_counter()
    trajectories = np.empty_like(validation.x_true)
    outputs = np.empty_like(validation.mocap_meas)
    for trial in range(validation.n_trials):
        x0 = np.clip(validation.mocap_derived_state[trial, 0], state_lower, state_upper)
        x = simulate_single_trajectory(x0, validation.u_act[trial], validation.t, lambda xk, uk: theta_dynamics(xk, uk, theta_hat, aircraft))
        trajectories[trial] = x
        outputs[trial] = mocap_from_state(validation.t, x)
    rollout_elapsed = time.perf_counter() - start_rollout
    return MethodResult(
        name="Variational-Mocap",
        description="Sparse multiple-shooting MAP fit with latent states, mocap outputs, and process residuals.",
        train_elapsed_s=elapsed,
        rollout_elapsed_s=rollout_elapsed,
        validation_trajectories=trajectories,
        validation_coeff_residual=None,
        validation_outputs=outputs,
        decision_variables=len(z0),
        train_samples=int(trial_count * n_nodes),
        notes=(
            "CasADi/IPOPT sparse latent-state estimator; approximates the variational/filter-error role "
            "without posterior covariance propagation."
        ),
        backend="CasADi/IPOPT",
    )


def run_variational_mocap_output(train: SplitData, validation: SplitData, args: argparse.Namespace) -> MethodResult:
    return run_variational_mocap_output_casadi(train, validation, args)


def run_oem_mocap_output(train: SplitData, validation: SplitData, args: argparse.Namespace) -> MethodResult:
    return run_oem_mocap_output_casadi(train, validation, args)


def run_sindy(xu: np.ndarray, dxdt: np.ndarray, validation: SplitData, args: argparse.Namespace) -> tuple[MethodResult, list[str], np.ndarray]:
    aircraft = Aircraft()
    coeff_inferred = infer_coefficients(xu[:, :4], xu[:, 4:], dxdt, aircraft)
    coeff_lower = np.quantile(coeff_inferred, 0.002, axis=0)
    coeff_upper = np.quantile(coeff_inferred, 0.998, axis=0)
    target = np.clip(coeff_inferred, coeff_lower, coeff_upper)
    start = time.perf_counter()
    coeffs, names_by_output, train_mse = fit_sindy(xu, target, args.sindy_threshold, args.sindy_ridge)
    elapsed = time.perf_counter() - start

    def coefficient_fn(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        xu_local = np.concatenate((x, u))[None, :]
        blocks = structured_sindy_feature_blocks(xu_local)
        coeff = np.array([blocks[idx][0] @ coeffs[idx] for idx in range(3)]).ravel()
        return np.clip(coeff, coeff_lower, coeff_upper)

    csv_names: list[str] = []
    csv_coeff_rows: list[np.ndarray] = []
    for output_idx, coefficient_name in enumerate(AERO_COEFFICIENT_NAMES):
        for feature_name, value in zip(names_by_output[output_idx], coeffs[output_idx]):
            if abs(value) <= 0.0:
                continue
            row = np.zeros(3)
            row[output_idx] = value
            csv_names.append(f"{coefficient_name}:{feature_name}")
            csv_coeff_rows.append(row)
    csv_coeff = np.vstack(csv_coeff_rows) if csv_coeff_rows else np.zeros((0, 3))

    result = evaluate_method(
        "SINDy",
        "One-shot structured sparse aerodynamic coefficient model.",
        elapsed,
        len(xu),
        int(sum(np.count_nonzero(coeff) for coeff in coeffs)),
        validation,
        lambda _trial: lambda x, u: dynamics_from_coefficients(x, u, coefficient_fn(x, u), aircraft),
        None,
        (
            "Fits protected low-order aircraft coefficient terms and sparse residual-library terms in one regression; "
            "protected terms are not thresholded, residual terms are selected by sequential thresholded least squares."
        ),
    )
    result.train_loss_final = train_mse
    return result, csv_names, csv_coeff


def fit_symbolic_stepwise(xu: np.ndarray, dxdt: np.ndarray, max_terms: int, penalty: float, ridge: float) -> tuple[np.ndarray, list[str]]:
    theta_raw, names = sindy_library(xu)
    mean = theta_raw.mean(axis=0)
    scale = theta_raw.std(axis=0)
    mean[0] = 0.0
    scale[0] = 1.0
    scale[scale < 1e-12] = 1.0
    theta = (theta_raw - mean) / scale
    coeff_s = np.zeros((theta.shape[1], dxdt.shape[1]))
    n_samples = len(theta)
    for state_idx in range(dxdt.shape[1]):
        active = [0]
        remaining = list(range(1, theta.shape[1]))
        best_score = np.inf
        for _ in range(max_terms):
            candidate_best = None
            for candidate in remaining:
                cols = active + [candidate]
                lhs = theta[:, cols].T @ theta[:, cols] + ridge * np.eye(len(cols))
                rhs = theta[:, cols].T @ dxdt[:, state_idx]
                local_coeff = np.linalg.solve(lhs, rhs)
                residual = dxdt[:, state_idx] - theta[:, cols] @ local_coeff
                mse = float(np.mean(residual**2))
                score = np.log(mse + 1e-18) + penalty * len(cols) * np.log(n_samples) / n_samples
                if candidate_best is None or score < candidate_best[0]:
                    candidate_best = (score, candidate, local_coeff)
            if candidate_best is None or candidate_best[0] >= best_score:
                break
            best_score, candidate, local_coeff = candidate_best
            active.append(candidate)
            remaining.remove(candidate)
            coeff_s[:, state_idx] = 0.0
            coeff_s[active, state_idx] = local_coeff
    coeff = coeff_s / scale[:, None]
    coeff[0] += -(mean / scale) @ coeff_s
    return coeff, names


def run_symbolic_regression(
    xu: np.ndarray,
    dxdt: np.ndarray,
    validation: SplitData,
    args: argparse.Namespace,
) -> tuple[MethodResult, list[str], np.ndarray]:
    start = time.perf_counter()
    coeff, names = fit_symbolic_stepwise(xu, dxdt, args.symbolic_max_terms, args.symbolic_penalty, args.symbolic_ridge)
    elapsed = time.perf_counter() - start

    def rhs(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        theta, _ = sindy_library(np.concatenate((x, u))[None, :])
        return (theta @ coeff).ravel()

    result = evaluate_method(
        "Symbolic-Stepwise",
        "Forward stepwise symbolic regression over the shared nonlinear candidate library.",
        elapsed,
        len(xu),
        int(np.count_nonzero(coeff)),
        validation,
        lambda _trial: rhs,
        None,
        f"Selects at most {args.symbolic_max_terms} nonconstant terms per state with a BIC-style complexity penalty.",
    )
    result.backend = "NumPy stepwise"
    return result, names, coeff


def run_linear_state_space(train_x: np.ndarray, train_u: np.ndarray, validation: SplitData) -> MethodResult:
    start = time.perf_counter()
    xk = train_x[:, :-1, :].reshape(-1, 4)
    uk = train_u[:, :-1, :].reshape(-1, 2)
    xkp1 = train_x[:, 1:, :].reshape(-1, 4)
    features = np.column_stack((np.ones(len(xk)), xk, uk))
    coeff = np.linalg.lstsq(features, xkp1, rcond=None)[0]
    elapsed = time.perf_counter() - start

    start_rollout = time.perf_counter()
    trajectories = np.empty_like(validation.x_true)
    for trial in range(validation.n_trials):
        x = np.empty((validation.n_time, 4))
        x[0] = validation.y_meas[trial, 0]
        for k in range(validation.n_time - 1):
            z = np.concatenate(([1.0], x[k], validation.u_act[trial, k]))
            x[k + 1] = z @ coeff
            x[k + 1, 0] = max(x[k + 1, 0], 3.0)
            if not np.all(np.isfinite(x[k + 1])) or np.linalg.norm(x[k + 1]) > 1e4:
                x[k + 1 :] = x[k]
                break
        trajectories[trial] = x
    rollout_elapsed = time.perf_counter() - start_rollout
    return MethodResult(
        name="Linear-SS",
        description="Discrete local linear state-space baseline fitted by one-step least squares.",
        train_elapsed_s=elapsed,
        rollout_elapsed_s=rollout_elapsed,
        validation_trajectories=trajectories,
        validation_coeff_residual=None,
        validation_outputs=None,
        decision_variables=int(coeff.size),
        train_samples=len(xk),
        notes=(
            "Validation is an open-loop rollout from x0 using the supplied pilot-command history only; "
            "good scores indicate in-distribution discrete prediction, not aerodynamic interpretability."
        ),
    )


def kmeans_labels(samples: np.ndarray, n_clusters: int, seed: int, max_iter: int = 40) -> tuple[np.ndarray, np.ndarray]:
    n_clusters = max(1, min(int(n_clusters), len(samples)))
    mean = samples.mean(axis=0)
    scale = samples.std(axis=0)
    scale[scale < 1e-9] = 1.0
    z = (samples - mean) / scale
    rng = np.random.default_rng(seed)
    centers = z[rng.choice(len(z), size=n_clusters, replace=False)].copy()
    labels = np.zeros(len(z), dtype=int)
    for _ in range(max_iter):
        dist2 = np.sum((z[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dist2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(n_clusters):
            cluster_samples = z[labels == cluster]
            if len(cluster_samples):
                centers[cluster] = cluster_samples.mean(axis=0)
            else:
                centers[cluster] = z[rng.integers(len(z))]
    return labels, centers * scale + mean


def fit_discrete_linear_model(xk: np.ndarray, uk: np.ndarray, xkp1: np.ndarray, ridge: float) -> np.ndarray:
    features = np.column_stack((np.ones(len(xk)), xk, uk))
    gram = features.T @ features + ridge * np.eye(features.shape[1])
    rhs = features.T @ xkp1
    return np.linalg.solve(gram, rhs)


def run_model_stitching(
    train_x: np.ndarray,
    train_u: np.ndarray,
    train: SplitData,
    validation: SplitData,
    args: argparse.Namespace,
) -> MethodResult:
    start = time.perf_counter()
    global_xk = train_x[:, :-1, :].reshape(-1, 4)
    global_uk = train_u[:, :-1, :].reshape(-1, 2)
    global_xkp1 = train_x[:, 1:, :].reshape(-1, 4)
    global_x_ref = np.mean(train.trim_state, axis=0)
    global_u_ref = np.mean(train.trim_controls, axis=0)
    trial_operating = np.column_stack((train.trim_state.reshape(train.n_trials, -1), train.trim_controls.reshape(train.n_trials, -1)))
    if np.max(np.std(trial_operating, axis=0)) < 1e-9:
        trial_operating = np.column_stack((train_x[:, 0, :], train_u[:, 0, :]))
    n_models = max(1, min(int(args.stitch_models), train.n_trials))
    labels, _ = kmeans_labels(trial_operating, n_models, args.seed)
    global_coeff = fit_discrete_linear_model(
        global_xk - global_x_ref,
        global_uk - global_u_ref,
        global_xkp1 - global_x_ref,
        args.stitch_ridge,
    )
    coeffs = []
    centers = []
    local_x_refs = []
    local_u_refs = []
    one_step_errors = []
    min_samples = max(8, 8 * (1 + train_x.shape[2] + train_u.shape[2]))
    for model_idx in range(n_models):
        trial_ids = np.flatnonzero(labels == model_idx)
        if len(trial_ids):
            x_ref = np.mean(train.trim_state[trial_ids], axis=0)
            u_ref = np.mean(train.trim_controls[trial_ids], axis=0)
            xk = train_x[trial_ids, :-1, :].reshape(-1, 4)
            uk = train_u[trial_ids, :-1, :].reshape(-1, 2)
            xkp1 = train_x[trial_ids, 1:, :].reshape(-1, 4)
        else:
            x_ref = np.mean(train.trim_state, axis=0)
            u_ref = np.mean(train.trim_controls, axis=0)
            xk = np.empty((0, 4))
            uk = np.empty((0, 2))
            xkp1 = np.empty((0, 4))
        if len(xk) < min_samples:
            coeff = global_coeff.copy()
            x_ref = global_x_ref
            u_ref = global_u_ref
            xk = global_xk
            uk = global_uk
            xkp1 = global_xkp1
        else:
            coeff = fit_discrete_linear_model(xk - x_ref, uk - u_ref, xkp1 - x_ref, args.stitch_ridge)
        coeffs.append(coeff)
        centers.append(np.concatenate((x_ref, u_ref)))
        local_x_refs.append(x_ref)
        local_u_refs.append(u_ref)
        features = np.column_stack((np.ones(len(xk)), xk - x_ref, uk - u_ref))
        one_step_errors.append(np.mean((x_ref + features @ coeff - xkp1) ** 2))
    coeffs_arr = np.asarray(coeffs)
    centers_arr = np.asarray(centers)
    local_x_refs_arr = np.asarray(local_x_refs)
    local_u_refs_arr = np.asarray(local_u_refs)
    schedule_all = np.column_stack((global_xk, global_uk))
    schedule_scale = schedule_all.std(axis=0)
    schedule_scale[schedule_scale < 1e-9] = 1.0
    if args.stitch_length_scale > 0.0:
        length_scale = float(args.stitch_length_scale)
    elif len(centers_arr) > 1:
        center_z = (centers_arr - schedule_all.mean(axis=0)) / schedule_scale
        dist = np.sqrt(np.sum((center_z[:, None, :] - center_z[None, :, :]) ** 2, axis=2))
        median_center_distance = float(np.median(dist[dist > 0.0])) if np.any(dist > 0.0) else 1.0
        length_scale = 0.35 * median_center_distance
    else:
        length_scale = 1.0
    length_scale = max(length_scale, 1e-3)
    train_mse = float(np.mean(one_step_errors))
    elapsed = time.perf_counter() - start

    def step(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        dx = x[None, :] - local_x_refs_arr
        du = u[None, :] - local_u_refs_arr
        features = np.column_stack((np.ones(len(coeffs_arr)), dx, du))
        predictions = local_x_refs_arr + np.einsum("kj,kjl->kl", features, coeffs_arr)
        schedule = np.concatenate((x, u))
        delta = (centers_arr - schedule) / schedule_scale
        weights = np.exp(-0.5 * np.sum(delta**2, axis=1) / (length_scale**2))
        weight_sum = np.sum(weights)
        if not np.isfinite(weight_sum) or weight_sum <= 1e-12:
            weights = np.zeros(len(centers_arr))
            weights[int(np.argmin(np.sum(delta**2, axis=1)))] = 1.0
        else:
            weights = weights / weight_sum
        x_next = weights @ predictions
        x_next[0] = max(x_next[0], 3.0)
        x_next[1:] = np.clip(x_next[1:], [-0.75, -1.2, -3.5], [0.75, 1.2, 3.5])
        return x_next

    start_rollout = time.perf_counter()
    trajectories = np.empty_like(validation.x_true)
    for trial in range(validation.n_trials):
        x = np.empty((validation.n_time, 4))
        x[0] = validation.y_meas[trial, 0]
        for k in range(validation.n_time - 1):
            x[k + 1] = step(x[k], validation.u_act[trial, k])
            if not np.all(np.isfinite(x[k + 1])) or np.linalg.norm(x[k + 1]) > 1e4:
                x[k + 1 :] = x[k]
                break
        trajectories[trial] = x
    rollout_elapsed = time.perf_counter() - start_rollout
    return MethodResult(
        name="Model-Stitching",
        description="Initial-condition clustered local linear state-space models blended by scheduling weights.",
        train_elapsed_s=elapsed,
        rollout_elapsed_s=rollout_elapsed,
        validation_trajectories=trajectories,
        validation_coeff_residual=None,
        validation_outputs=None,
        decision_variables=int(coeffs_arr.size + centers_arr.size + schedule_scale.size + 1),
        train_samples=int(global_xk.shape[0]),
        notes=(
            f"Fits {n_models} local discrete deviation models from whole local-excitation trajectory groups; "
            f"rollout blends local predictions with RBF weights in state-input space, length_scale={length_scale:.3g}. "
            "Validation uses only x0 and the supplied pilot-command history."
        ),
        train_loss_final=train_mse,
        backend="NumPy local models",
    )


def build_lagged_features(y: np.ndarray, u: np.ndarray, past: int) -> tuple[np.ndarray, np.ndarray]:
    features = []
    index = []
    for trial in range(y.shape[0]):
        for k in range(past - 1, y.shape[1]):
            features.append(np.concatenate((y[trial, k - past + 1 : k + 1].ravel(), u[trial, k - past + 1 : k + 1].ravel())))
            index.append((trial, k))
    return np.asarray(features), np.asarray(index, dtype=int)


def run_subspace_hankel(train_x: np.ndarray, train_u: np.ndarray, validation: SplitData, args: argparse.Namespace) -> MethodResult:
    start = time.perf_counter()
    past = max(2, int(args.subspace_past))
    order = max(1, int(args.subspace_order))
    features, index = build_lagged_features(train_x, train_u, past)
    mean = features.mean(axis=0)
    centered = features - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:order].T
    latent = centered @ basis
    usable = index[:, 1] < train_x.shape[1] - 1
    latent_k = latent[usable]
    trial_k = index[usable, 0]
    time_k = index[usable, 1]
    latent_next_features = []
    for trial, k in zip(trial_k, time_k):
        next_feature = np.concatenate((train_x[trial, k - past + 2 : k + 2].ravel(), train_u[trial, k - past + 2 : k + 2].ravel()))
        latent_next_features.append((next_feature - mean) @ basis)
    latent_next = np.asarray(latent_next_features)
    u_k = train_u[trial_k, time_k]
    y_k = train_x[trial_k, time_k]
    dyn_features = np.column_stack((np.ones(len(latent_k)), latent_k, u_k))
    dyn_coeff = np.linalg.lstsq(dyn_features, latent_next, rcond=None)[0]
    out_coeff = np.linalg.lstsq(dyn_features, y_k, rcond=None)[0]
    elapsed = time.perf_counter() - start

    start_rollout = time.perf_counter()
    trajectories = np.empty_like(validation.x_true)
    for trial in range(validation.n_trials):
        y_hat = np.empty((validation.n_time, 4))
        x0 = validation.y_meas[trial, 0]
        initial_state_history = np.repeat(x0[None, :], past, axis=0)
        y_hat[:past] = initial_state_history
        init_feature = np.concatenate((initial_state_history.ravel(), validation.u_act[trial, :past].ravel()))
        z = (init_feature - mean) @ basis
        for k in range(past - 1, validation.n_time - 1):
            dyn = np.concatenate(([1.0], z, validation.u_act[trial, k]))
            z = dyn @ dyn_coeff
            out = np.concatenate(([1.0], z, validation.u_act[trial, k + 1])) @ out_coeff
            out[0] = max(out[0], 3.0)
            if not np.all(np.isfinite(out)) or np.linalg.norm(out) > 1e4:
                out = y_hat[k]
            y_hat[k + 1] = out
        trajectories[trial] = y_hat
    rollout_elapsed = time.perf_counter() - start_rollout
    return MethodResult(
        name="Subspace-Hankel",
        description="Hankel/SVD latent linear state-space baseline in the spirit of N4SID/MOESP.",
        train_elapsed_s=elapsed,
        rollout_elapsed_s=rollout_elapsed,
        validation_trajectories=trajectories,
        validation_coeff_residual=None,
        validation_outputs=None,
        decision_variables=int(basis.size + dyn_coeff.size + out_coeff.size),
        train_samples=len(latent_k),
        notes=(
            f"Past window={past}, latent order={order}; validation lag history is initialized "
            "from repeated x0 and pilot inputs, with no validation measurement feedback."
        ),
        backend="NumPy SVD",
    )


def project_stable_continuous(a_mat: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """Reflect unstable eigenvalues of a continuous-time A into the left half-plane.

    An identified LTI model that will be simulated open-loop for many seconds
    cannot carry spuriously unstable poles from noise/leakage bias; reflecting
    the real part (standard practice when simulating identified models) keeps
    the frequency content while bounding the rollout.
    """
    eigvals, eigvecs = np.linalg.eig(a_mat)
    projected = np.where(eigvals.real > -margin, -np.maximum(np.abs(eigvals.real), margin) + 1j * eigvals.imag, eigvals)
    if np.allclose(projected, eigvals):
        return a_mat
    try:
        return np.real(eigvecs @ np.diag(projected) @ np.linalg.inv(eigvecs))
    except np.linalg.LinAlgError:
        return a_mat


def fit_frequency_linear_model(
    train_x: np.ndarray,
    train_u: np.ndarray,
    dxdt: np.ndarray,
    dt: float,
    max_hz: float,
    ridge: float,
    nperseg: int,
    min_coherence: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit dx/dt = c + A x + B u from averaged spectra with input-power weighting.

    The per-bin weight is an input-power fraction heuristic, not a coherence
    estimate; bins dominated by state response relative to input content are
    down-weighted or skipped.
    """

    fs = 1.0 / dt
    z_mean = np.concatenate((train_x.reshape(-1, 4).mean(axis=0), train_u.reshape(-1, 2).mean(axis=0)))
    dx_mean = dxdt.reshape(-1, 4).mean(axis=0)
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[float] = []
    segment_count = 0
    nperseg = min(max(64, int(nperseg)), train_x.shape[1])
    if nperseg % 2:
        nperseg -= 1
    step = max(nperseg // 2, 1)
    window = windows.hann(nperseg, sym=False)
    omega = 2.0 * np.pi * np.fft.rfftfreq(nperseg, dt)
    freq = omega / (2.0 * np.pi)
    freq_keep = (freq >= 0.04) & (freq <= max_hz)
    freq_keep[0] = False

    for trial in range(train_x.shape[0]):
        z = np.column_stack((train_x[trial], train_u[trial]))
        z = z - np.mean(z, axis=0)
        for start_idx in range(0, z.shape[0] - nperseg + 1, step):
            segment = z[start_idx : start_idx + nperseg] * window[:, None]
            spectrum = np.fft.rfft(segment, axis=0)
            for idx in np.flatnonzero(freq_keep):
                row = spectrum[idx]
                if not np.all(np.isfinite(row)):
                    continue
                spectral_power = np.abs(row) ** 2
                state_power = float(np.sum(spectral_power[:4]))
                input_power = float(np.sum(spectral_power[4:]))
                if state_power <= 1e-12 or input_power <= 1e-12:
                    continue
                input_power_fraction = input_power / (input_power + 0.15 * state_power + 1e-18)
                if input_power_fraction < min_coherence:
                    continue
                rows.append(row)
                targets.append(1j * omega[idx] * row[:4])
                weights.append(float(np.sqrt(input_power_fraction * input_power / (np.sum(spectral_power) + 1e-18))))
                segment_count += 1

    if not rows:
        return fit_time_domain_linear_model(train_x, train_u, dxdt, ridge)
    z_freq = np.vstack(rows)
    y_freq = np.vstack(targets)
    w = np.asarray(weights)[:, None]
    z_weighted = z_freq * w
    y_weighted = y_freq * w
    gram = z_weighted.conj().T @ z_weighted + ridge * np.eye(z_weighted.shape[1])
    coeff = np.linalg.solve(gram, z_weighted.conj().T @ y_weighted).real
    coeff[:4, :] = project_stable_continuous(coeff[:4, :].T).T
    intercept = dx_mean - z_mean @ coeff
    return intercept, coeff, segment_count


def fit_time_domain_linear_model(
    train_x: np.ndarray,
    train_u: np.ndarray,
    dxdt: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    z = np.column_stack((train_x.reshape(-1, 4), train_u.reshape(-1, 2)))
    y = dxdt.reshape(-1, 4)
    z_mean = z.mean(axis=0)
    y_mean = y.mean(axis=0)
    zc = z - z_mean
    yc = y - y_mean
    coeff = np.linalg.solve(zc.T @ zc + ridge * np.eye(zc.shape[1]), zc.T @ yc)
    intercept = y_mean - z_mean @ coeff
    return intercept, coeff, len(z)


def run_frequency_linear(
    train_x: np.ndarray,
    train_u: np.ndarray,
    dxdt: np.ndarray,
    validation: SplitData,
    args: argparse.Namespace,
) -> MethodResult:
    start = time.perf_counter()
    intercept, coeff, freq_count = fit_frequency_linear_model(
        train_x,
        train_u,
        dxdt,
        validation.dt,
        args.frequency_max_hz,
        args.frequency_ridge,
        args.frequency_nperseg,
        args.frequency_min_coherence,
    )
    elapsed = time.perf_counter() - start

    def rhs(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return intercept + np.concatenate((x, u)) @ coeff

    result = evaluate_method(
        "Frequency-Welch",
        "Input-power-weighted continuous-time linear model fitted from averaged Welch state/input spectra.",
        elapsed,
        freq_count,
        int(intercept.size + coeff.size),
        validation,
        lambda _trial: rhs,
        None,
        f"Welch/ETFE-inspired spectral fit using bins up to {args.frequency_max_hz:g} Hz and input-power-fraction threshold {args.frequency_min_coherence:g}; this is not CIFER and should be interpreted only as a compact local linear frequency-domain baseline.",
    )
    result.backend = "NumPy Welch/CSD"
    return result


def run_frequency_stitching(
    train_x: np.ndarray,
    train_u: np.ndarray,
    dxdt: np.ndarray,
    train: SplitData,
    validation: SplitData,
    args: argparse.Namespace,
) -> MethodResult:
    start = time.perf_counter()
    global_x = train_x.reshape(-1, 4)
    global_u = train_u.reshape(-1, 2)
    global_dxdt = dxdt.reshape(-1, 4)
    trial_operating = np.column_stack((train.trim_state.reshape(train.n_trials, -1), train.trim_controls.reshape(train.n_trials, -1)))
    if np.max(np.std(trial_operating, axis=0)) < 1e-9:
        trial_operating = np.column_stack((train_x[:, 0, :], train_u[:, 0, :]))
    n_models = max(1, min(int(args.stitch_models), train.n_trials))
    labels, _ = kmeans_labels(trial_operating, n_models, args.seed)
    global_intercept, global_coeff, global_bins = fit_frequency_linear_model(
        train_x,
        train_u,
        dxdt,
        train.dt,
        args.frequency_max_hz,
        args.frequency_ridge,
        args.frequency_nperseg,
        args.frequency_min_coherence,
    )
    intercepts = []
    coeffs = []
    centers = []
    local_losses = []
    bin_counts = []
    min_trials = 2
    for model_idx in range(n_models):
        trial_ids = np.flatnonzero(labels == model_idx)
        if len(trial_ids) >= min_trials:
            x_local = train_x[trial_ids]
            u_local = train_u[trial_ids]
            dxdt_local = dxdt[trial_ids]
            intercept, coeff, bins = fit_frequency_linear_model(
                x_local,
                u_local,
                dxdt_local,
                train.dt,
                args.frequency_max_hz,
                args.frequency_ridge,
                args.frequency_nperseg,
                args.frequency_min_coherence,
            )
            x_ref = np.mean(train.trim_state[trial_ids], axis=0)
            u_ref = np.mean(train.trim_controls[trial_ids], axis=0)
            x_flat = x_local.reshape(-1, 4)
            u_flat = u_local.reshape(-1, 2)
            dx_flat = dxdt_local.reshape(-1, 4)
        else:
            intercept = global_intercept.copy()
            coeff = global_coeff.copy()
            bins = global_bins
            x_ref = np.mean(train.trim_state, axis=0)
            u_ref = np.mean(train.trim_controls, axis=0)
            x_flat = global_x
            u_flat = global_u
            dx_flat = global_dxdt
        z_flat = np.column_stack((x_flat, u_flat))
        local_losses.append(float(np.mean((intercept + z_flat @ coeff - dx_flat) ** 2)))
        intercepts.append(intercept)
        coeffs.append(coeff)
        centers.append(np.concatenate((x_ref, u_ref)))
        bin_counts.append(int(bins))
    intercepts_arr = np.asarray(intercepts)
    coeffs_arr = np.asarray(coeffs)
    centers_arr = np.asarray(centers)
    schedule_all = np.column_stack((global_x, global_u))
    schedule_scale = schedule_all.std(axis=0)
    schedule_scale[schedule_scale < 1e-9] = 1.0
    if args.stitch_length_scale > 0.0:
        length_scale = float(args.stitch_length_scale)
    elif len(centers_arr) > 1:
        center_z = (centers_arr - schedule_all.mean(axis=0)) / schedule_scale
        dist = np.sqrt(np.sum((center_z[:, None, :] - center_z[None, :, :]) ** 2, axis=2))
        median_center_distance = float(np.median(dist[dist > 0.0])) if np.any(dist > 0.0) else 1.0
        length_scale = 0.35 * median_center_distance
    else:
        length_scale = 1.0
    length_scale = max(length_scale, 1e-3)
    elapsed = time.perf_counter() - start

    def rhs(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        schedule = np.concatenate((x, u))
        delta = (centers_arr - schedule) / schedule_scale
        weights = np.exp(-0.5 * np.sum(delta**2, axis=1) / (length_scale**2))
        weight_sum = np.sum(weights)
        if not np.isfinite(weight_sum) or weight_sum <= 1e-12:
            weights = np.zeros(len(centers_arr))
            weights[int(np.argmin(np.sum(delta**2, axis=1)))] = 1.0
        else:
            weights = weights / weight_sum
        z = np.concatenate((x, u))
        local_rhs = intercepts_arr + np.einsum("j,kjl->kl", z, coeffs_arr)
        return weights @ local_rhs

    result = evaluate_method(
        "Frequency-Stitching",
        "Trim-grid collection of local Welch continuous-time linear models blended by scheduling weights.",
        elapsed,
        int(np.sum(bin_counts)),
        int(intercepts_arr.size + coeffs_arr.size + centers_arr.size + schedule_scale.size + 1),
        validation,
        lambda _trial: rhs,
        None,
        (
            f"Fits {n_models} local Welch/ETFE-inspired continuous-time models on trim-grid groups; "
            f"rollout blends local derivatives with RBF weights in state-input space, length_scale={length_scale:.3g}. "
            "This tests local frequency-domain stitching, not CIFER."
        ),
    )
    result.backend = "NumPy local Welch/CSD"
    result.train_loss_final = float(np.mean(local_losses))
    return result


def run_ude(xu: np.ndarray, dxdt: np.ndarray, validation: SplitData, args: argparse.Namespace, device: torch.device) -> tuple[MethodResult, list[tuple[int, float]]]:
    aircraft = Aircraft()
    theta = nominal_theta()
    nominal = np.vstack([theta_dynamics(x, u, theta, aircraft) for x, u in zip(xu[:, :4], xu[:, 4:])])
    target = dxdt - nominal
    net, x_mean, x_scale, y_mean, y_scale, y_lower, y_upper, history, elapsed, cpu_elapsed, gpu_elapsed, gpu_memory_mb = train_mlp(xu, target, args, device, 4)
    residual_fn = make_mlp_predictor(net, x_mean, x_scale, y_mean, y_scale, y_lower, y_upper, device)
    result = evaluate_method(
        "UDE-Residual",
        "Universal differential equation: nominal dynamics plus a learned state-derivative residual.",
        elapsed,
        len(xu),
        sum(p.numel() for p in net.parameters()),
        validation,
        lambda _trial: lambda x, u: theta_dynamics(x, u, theta, aircraft) + args.ude_gain * residual_fn(x, u),
        None,
        f"Good for model-form error; residual dynamics are less directly interpretable as aerodynamic coefficients. Residual gain={args.ude_gain:g}.",
    )
    result.backend = f"PyTorch/{device}"
    result.train_loss_final = history[-1][1] if history else np.nan
    result.train_cpu_s = cpu_elapsed
    result.train_gpu_s = gpu_elapsed
    result.gpu_memory_mb = gpu_memory_mb
    return result, history


def run_pinn_closure(xu: np.ndarray, dxdt: np.ndarray, validation: SplitData, args: argparse.Namespace, device: torch.device) -> tuple[MethodResult, list[tuple[int, float]]]:
    aircraft = Aircraft()
    theta = nominal_theta()
    coeff_inferred = infer_coefficients(xu[:, :4], xu[:, 4:], dxdt, aircraft)
    coeff_nominal = np.vstack([theta_coefficients(x, u, theta) for x, u in zip(xu[:, :4], xu[:, 4:])])
    target = coeff_inferred - coeff_nominal
    net, x_mean, x_scale, y_mean, y_scale, y_lower, y_upper, history, elapsed, cpu_elapsed, gpu_elapsed, gpu_memory_mb = train_mlp(xu, target, args, device, 3)
    coeff_residual_fn = make_mlp_predictor(net, x_mean, x_scale, y_mean, y_scale, y_lower, y_upper, device)
    result = evaluate_method(
        "PINN-CoeffClosure",
        "Modified Non-Determinant PINN-style closure: EOM residual identifies learned nonlinear aerodynamic coefficients.",
        elapsed,
        len(xu),
        sum(p.numel() for p in net.parameters()),
        validation,
        lambda _trial: lambda x, u: coefficient_residual_dynamics(
            x,
            u,
            theta,
            aircraft,
            lambda x_local, u_local: args.pinn_gain * coeff_residual_fn(x_local, u_local),
        ),
        lambda _trial: lambda x, u: args.pinn_gain * coeff_residual_fn(x, u),
        f"Flight-dynamics-level PINN surrogate; does not enforce CFD/Navier-Stokes physics. Residual gain={args.pinn_gain:g}.",
    )
    result.backend = f"PyTorch/{device}"
    result.train_loss_final = history[-1][1] if history else np.nan
    result.train_cpu_s = cpu_elapsed
    result.train_gpu_s = gpu_elapsed
    result.gpu_memory_mb = gpu_memory_mb
    return result, history


def run_ude_hidden_control(
    xu: np.ndarray,
    dxdt: np.ndarray,
    validation: SplitData,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[MethodResult, list[tuple[int, float]]]:
    aircraft = Aircraft()
    theta = nominal_theta()
    target = infer_input_correction(xu, dxdt, theta, aircraft)
    net, x_mean, x_scale, y_mean, y_scale, y_lower, y_upper, history, elapsed, cpu_elapsed, gpu_elapsed, gpu_memory_mb = train_mlp(xu, target, args, device, 2)
    correction_fn = make_mlp_predictor(net, x_mean, x_scale, y_mean, y_scale, y_lower, y_upper, device)
    result = evaluate_method(
        "UDE-HiddenControl",
        "Universal-differential-equation style hidden-input model: nominal plant driven by pilot command plus learned input correction.",
        elapsed,
        len(xu),
        sum(p.numel() for p in net.parameters()),
        validation,
        lambda _trial: lambda x, u: input_corrected_dynamics(x, u, theta, aircraft, correction_fn),
        None,
        "Learns an effective throttle/elevator correction from EOM residuals; intended for hidden actuator/autopilot input mismatch.",
    )
    result.backend = f"PyTorch/{device}"
    result.train_loss_final = history[-1][1] if history else np.nan
    result.train_cpu_s = cpu_elapsed
    result.train_gpu_s = gpu_elapsed
    result.gpu_memory_mb = gpu_memory_mb
    return result, history


def run_pinn_hidden_control(
    xu: np.ndarray,
    dxdt: np.ndarray,
    validation: SplitData,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[MethodResult, list[tuple[int, float]]]:
    aircraft = Aircraft()
    theta = nominal_theta()
    target = infer_input_correction(xu, dxdt, theta, aircraft)
    # Longitudinal SAFE/AS3X mainly enters through elevator.  This PINN-style
    # version constrains throttle correction to zero and learns only elevator
    # reshaping from the pitch-moment residual.
    elevator_target = target[:, 1:2]
    net, x_mean, x_scale, y_mean, y_scale, y_lower, y_upper, history, elapsed, cpu_elapsed, gpu_elapsed, gpu_memory_mb = train_mlp(
        xu,
        elevator_target,
        args,
        device,
        1,
    )
    elevator_fn = make_mlp_predictor(net, x_mean, x_scale, y_mean, y_scale, y_lower, y_upper, device)

    def correction_fn(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([0.0, float(elevator_fn(x, u)[0])])

    result = evaluate_method(
        "PINN-HiddenElevator",
        "PINN-style hidden-controller model: EOM pitch residual identifies a learned elevator command reshaping law.",
        elapsed,
        len(xu),
        sum(p.numel() for p in net.parameters()),
        validation,
        lambda _trial: lambda x, u: input_corrected_dynamics(x, u, theta, aircraft, correction_fn),
        None,
        "Uses the pitch-moment equation as the physics residual, so it targets SAFE/AS3X elevator behavior rather than aerodynamic residuals.",
    )
    result.backend = f"PyTorch/{device}"
    result.train_loss_final = history[-1][1] if history else np.nan
    result.train_cpu_s = cpu_elapsed
    result.train_gpu_s = gpu_elapsed
    result.gpu_memory_mb = gpu_memory_mb
    return result, history


def run_supervised_coeff_surrogate(
    xu: np.ndarray,
    coeff_residual: np.ndarray,
    validation: SplitData,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[MethodResult, list[tuple[int, float]]]:
    aircraft = Aircraft()
    theta = nominal_theta()
    net, x_mean, x_scale, y_mean, y_scale, y_lower, y_upper, history, elapsed, cpu_elapsed, gpu_elapsed, gpu_memory_mb = train_mlp(
        xu,
        coeff_residual,
        args,
        device,
        3,
    )
    coeff_residual_fn = make_mlp_predictor(net, x_mean, x_scale, y_mean, y_scale, y_lower, y_upper, device)
    result = evaluate_method(
        "NN-CoeffSurrogate",
        "Supervised neural surrogate for hidden aerodynamic coefficient residuals.",
        elapsed,
        len(xu),
        sum(p.numel() for p in net.parameters()),
        validation,
        lambda _trial: lambda x, u: coefficient_residual_dynamics(x, u, theta, aircraft, coeff_residual_fn),
        lambda _trial: coeff_residual_fn,
        "Uses synthetic residual labels, so treat as an upper-bound surrogate rather than a flight-data-only identifier.",
    )
    result.backend = f"PyTorch/{device}"
    result.train_loss_final = history[-1][1] if history else np.nan
    result.train_cpu_s = cpu_elapsed
    result.train_gpu_s = gpu_elapsed
    result.gpu_memory_mb = gpu_memory_mb
    return result, history


def summarize_results(results: list[MethodResult], validation: SplitData) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        y_hat = result.validation_trajectories
        state_rmse = rmse(y_hat.reshape(-1, 4), validation.x_true.reshape(-1, 4))
        score = aggregate_trajectory_score(y_hat.reshape(-1, 4), validation.x_true.reshape(-1, 4))
        row: dict[str, object] = {
            "method": result.name,
            "description": result.description,
            "implementation_status": result.implementation_status,
            "backend": result.backend,
            "validation_score": score,
            "train_elapsed_s": result.train_elapsed_s,
            "train_cpu_s": result.train_cpu_s,
            "train_gpu_s": result.train_gpu_s,
            "gpu_memory_mb": result.gpu_memory_mb,
            "rollout_elapsed_s": result.rollout_elapsed_s,
            "total_elapsed_s": result.train_elapsed_s + result.rollout_elapsed_s,
            "train_loss_final": result.train_loss_final,
            "decision_variables": result.decision_variables,
            "train_samples": result.train_samples,
            "notes": result.notes,
            "evaluation_mode": result.evaluation_mode,
            "diverged": result.diverged,
        }
        row.update({f"rmse_{name}": value for name, value in zip(STATE_NAMES, state_rmse)})
        if result.validation_outputs is not None:
            output_rmse = rmse(result.validation_outputs.reshape(-1, 3), validation.mocap_meas.reshape(-1, 3))
            row.update(
                {
                    "mocap_rmse_x_pos": output_rmse[0],
                    "mocap_rmse_z_pos": output_rmse[1],
                    "mocap_rmse_theta": output_rmse[2],
                }
            )
        else:
            row.update({"mocap_rmse_x_pos": np.nan, "mocap_rmse_z_pos": np.nan, "mocap_rmse_theta": np.nan})
        if result.validation_coeff_residual is not None:
            coeff_rmse = rmse(
                result.validation_coeff_residual.reshape(-1, 3),
                validation.coeff_residual.reshape(-1, 3),
            )
            row.update({f"coeff_residual_rmse_{name}": value for name, value in zip(COEFF_NAMES, coeff_rmse)})
        else:
            row.update({f"coeff_residual_rmse_{name}": np.nan for name in COEFF_NAMES})
        rows.append(row)
    return rows


def dataset_scenario_id(dataset: Path) -> str:
    metadata = dataset / "metadata.json"
    if metadata.exists():
        try:
            payload = json.loads(metadata.read_text())
            mode = payload.get("dataset_mode") or payload.get("config", {}).get("dataset_mode")
            if mode:
                return str(mode)
        except json.JSONDecodeError:
            pass
    name = dataset.name
    if name == "longitudinal_3dof_nonlinear":
        return "aggressive"
    for candidate in (
        "open_loop_safe",
        "sine_sweep_safe",
        "aggressive_safe",
        "safe_loop",
        "proprietary_autopilot",
        "open_loop",
        "sine_sweep",
        "trim_grid",
        "aggressive",
    ):
        if candidate in name:
            return candidate
    return "aggressive"


def quat_wxyz_from_rotation(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return quat / max(float(np.linalg.norm(quat)), 1e-12)


def planar_body_to_enu_quat(theta: np.ndarray) -> np.ndarray:
    quat = np.empty((len(theta), 4), dtype=float)
    for index, pitch in enumerate(theta):
        forward = np.array([np.cos(pitch), 0.0, np.sin(pitch)])
        right = np.array([0.0, -1.0, 0.0])
        down = np.cross(forward, right)
        quat[index] = quat_wxyz_from_rotation(np.column_stack([forward, right, down]))
    return quat


def trace_segment_3dof(t: np.ndarray, x: np.ndarray, u: np.ndarray, name: str) -> dict[str, object]:
    stride = max(1, int(np.ceil(len(t) / WEB_TRACE_MAX_POINTS)))
    t = t[::stride]
    x = x[::stride]
    u = u[::stride]
    mocap = mocap_from_state(t, x)
    position = np.column_stack([mocap[:, 0], np.zeros(len(mocap)), mocap[:, 1]])
    control = np.column_stack([u[:, 0], np.zeros(len(u)), u[:, 1], np.zeros(len(u))])
    return {
        "name": name,
        "time_s": np.round(t - t[0], 4).tolist(),
        "position_enu_m": np.round(position - position[0], 5).tolist(),
        "quaternion_wxyz": np.round(planar_body_to_enu_quat(mocap[:, 2]), 7).tolist(),
        "control_meas": np.round(control, 5).tolist(),
        "direct_state_meas": np.round(x, 6).tolist(),
    }


def write_method_traces(results: list[MethodResult], validation: SplitData, scenario: str, source: str) -> None:
    traces: list[dict[str, object]] = []
    for result in results:
        segments = [
            trace_segment_3dof(validation.t, result.validation_trajectories[0], validation.u_act[0], "validation_trial_1")
        ]
        traces.append(
            {
                "method": result.name,
                "model_family": "aircraft3dof",
                "scenario": scenario,
                "state_source": source,
                "segments": segments,
            }
        )
    (RESULTS_DIR / "shared_method_traces.json").write_text(json.dumps({"traces": traces}, indent=2, sort_keys=True) + "\n")


def write_rows(rows: list[dict[str, object]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    for output in [RESULTS_DIR / "shared_method_comparison.csv", TABLE_DIR / "shared_method_comparison.csv"]:
        with output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def write_sindy_coefficients(names: list[str], coeff: np.ndarray) -> None:
    with (RESULTS_DIR / "shared_sindy_coefficients.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["feature", *AERO_COEFFICIENT_NAMES])
        for name, row in zip(names, coeff):
            writer.writerow([name, *row])


def write_symbolic_coefficients(names: list[str], coeff: np.ndarray) -> None:
    with (RESULTS_DIR / "shared_symbolic_coefficients.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["feature", *[f"d{name}/dt" for name in STATE_NAMES]])
        for name, row in zip(names, coeff):
            writer.writerow([name, *row])


def write_histories(histories: dict[str, list[tuple[int, float]]]) -> None:
    for name, history in histories.items():
        with (RESULTS_DIR / f"{name.lower().replace('-', '_')}_loss.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["epoch", "loss"])
            writer.writerows(history)


def fisher_uq_diagnostics(
    train: SplitData,
    theta: np.ndarray,
    source: str,
    method: str,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    aircraft = Aircraft()
    trial_count = min(args.uq_trials, train.n_trials)
    trial_ids = np.linspace(0, train.n_trials - 1, trial_count, dtype=int)
    stride = max(1, int(args.uq_stride))
    theta_step = np.maximum(np.abs(theta) * 1e-4, 1e-5)

    def rollout(theta_local: np.ndarray) -> np.ndarray:
        predictions = []
        for trial in trial_ids:
            t_local = train.t[::stride]
            u_local = train.u_act[trial, ::stride]
            x0 = train.y_meas[trial, 0]
            x = simulate_single_trajectory(
                x0,
                u_local,
                t_local,
                lambda xk, uk: theta_dynamics(xk, uk, theta_local, aircraft),
            )
            predictions.append(x)
        return np.asarray(predictions)

    base = rollout(theta)
    row_scale = np.tile(STATE_NOISE, base.shape[0] * base.shape[1])
    jac = np.empty((base.size, len(theta)))
    for param_idx in range(len(theta)):
        plus = theta.copy()
        minus = theta.copy()
        plus[param_idx] += theta_step[param_idx]
        minus[param_idx] -= theta_step[param_idx]
        jac[:, param_idx] = ((rollout(plus) - rollout(minus)) / (2.0 * theta_step[param_idx])).ravel()
    jac = jac / row_scale[:, None]
    fisher = jac.T @ jac + args.uq_ridge * np.eye(len(theta))
    cov = np.linalg.pinv(fisher)
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    corr = cov / np.maximum(np.outer(std, std), 1e-18)
    rows = []
    for idx, name in enumerate(PARAMETER_NAMES):
        offdiag = np.delete(np.abs(corr[idx]), idx)
        rows.append(
            {
                "method": method,
                "state_source": source,
                "parameter": name,
                "estimate": theta[idx],
                "crlb_std": std[idx],
                "relative_std": std[idx] / max(abs(theta[idx]), 1e-12),
                "max_abs_correlation": float(np.max(offdiag)) if len(offdiag) else np.nan,
                "samples": int(base.shape[0] * base.shape[1]),
            }
        )
    return rows


def write_uq_diagnostics(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    for output in [RESULTS_DIR / "shared_uq_diagnostics.csv", TABLE_DIR / "shared_uq_diagnostics.csv"]:
        with output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def plot_comparison(rows: list[dict[str, object]]) -> None:
    ordered = sorted(rows, key=lambda row: float(row["validation_score"]))
    methods = [str(row["method"]) for row in ordered]
    scores = [float(row["validation_score"]) for row in ordered]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.bar(np.arange(len(methods)), scores, color="#4c78a8")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("validation trajectory score")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "shared_validation_score_comparison")


def plot_validation_trajectory(results: list[MethodResult], validation: SplitData, max_methods: int = 7) -> None:
    trial = 0
    fig, axes = plt.subplots(4, 1, figsize=(8.2, 7.2), sharex=True)
    colors = ["#000000", "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#ff9da6"]
    for row, label in enumerate(STATE_LABELS):
        truth = validation.x_true[trial, :, row].copy()
        meas = validation.y_meas[trial, :, row].copy()
        if row > 0:
            truth = np.rad2deg(truth)
            meas = np.rad2deg(meas)
        axes[row].plot(validation.t, truth, color=colors[0], linewidth=1.4, label="Truth")
        axes[row].plot(validation.t, meas, color="0.78", linewidth=0.45, alpha=0.7, label="Measured")
        y_ref = np.concatenate((truth, meas))
        y_min, y_max = np.nanpercentile(y_ref, [0.5, 99.5])
        margin = max(0.05 * (y_max - y_min), 1e-3)
        for idx, result in enumerate(results[:max_methods], start=1):
            pred = result.validation_trajectories[trial, :, row].copy()
            if row > 0:
                pred = np.rad2deg(pred)
            axes[row].plot(validation.t, pred, linewidth=0.95, color=colors[idx % len(colors)], label=result.name)
        axes[row].set_ylim(y_min - 2.0 * margin, y_max + 2.0 * margin)
        axes[row].set_ylabel(label)
        axes[row].grid(True, alpha=0.25)
    axes[-1].set_xlabel("time [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, FIG_DIR / "shared_validation_trajectory_overlay")


def plot_coeff_residual(result: MethodResult, validation: SplitData) -> None:
    if result.validation_coeff_residual is None:
        return
    trial = 0
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 5.8), sharex=True)
    for idx, name in enumerate(COEFF_NAMES):
        axes[idx].plot(validation.t, validation.coeff_residual[trial, :, idx], color="black", linewidth=1.4, label="True residual")
        axes[idx].plot(validation.t, result.validation_coeff_residual[trial, :, idx], color="#e45756", linewidth=1.1, label=result.name)
        axes[idx].set_ylabel(name)
        axes[idx].grid(True, alpha=0.25)
    axes[-1].set_xlabel("time [s]")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "shared_pinn_coeff_residual_validation")


def plot_frequency_diagnostic(validation: SplitData, args: argparse.Namespace) -> list[dict[str, object]]:
    trial = 0
    fs = 1.0 / validation.dt
    nperseg = min(args.nperseg, validation.n_time // 2)
    elevator = validation.u_act[trial, :, 1] - np.mean(validation.u_act[trial, :, 1])
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.2), sharex=True)
    for output_idx, label, color in [(3, "Q", "#4c78a8"), (2, "gamma", "#f58518")]:
        output = validation.y_meas[trial, :, output_idx] - np.mean(validation.y_meas[trial, :, output_idx])
        # H1 estimate: scipy csd(x, y) averages conj(X) * Y, so the input must be
        # the first argument or the returned phase is conjugated.
        freq, pxy = csd(elevator, output, fs=fs, nperseg=nperseg, detrend="constant")
        _, pxx = welch(elevator, fs=fs, nperseg=nperseg, detrend="constant")
        _, coh = coherence(elevator, output, fs=fs, nperseg=nperseg, detrend="constant")
        h = pxy / pxx
        valid = freq > 0
        axes[0].semilogx(freq[valid], 20.0 * np.log10(np.abs(h[valid])), color=color, label=label)
        axes[1].semilogx(freq[valid], np.rad2deg(np.unwrap(np.angle(h[valid]))), color=color)
        axes[2].semilogx(freq[valid], coh[valid], color=color)
        rows.append(
            {
                "method": "Frequency-Welch",
                "output": label,
                "mean_coherence": float(np.mean(coh[valid])),
                "peak_frequency_hz": float(freq[valid][np.argmax(np.abs(h[valid]))]),
                "peak_magnitude_db": float(20.0 * np.log10(np.max(np.abs(h[valid])))),
            }
        )
    axes[0].set_ylabel("magnitude [dB]")
    axes[1].set_ylabel("phase [deg]")
    axes[2].set_ylabel("coherence")
    axes[2].set_xlabel("frequency [Hz]")
    axes[2].set_ylim(0.0, 1.05)
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "shared_frequency_validation_diagnostic")
    with (RESULTS_DIR / "shared_frequency_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_logged(label: str, fn: Callable[[], object]) -> object:
    print(f"  {label} ...", flush=True)
    start = time.perf_counter()
    result = fn()
    print(f"  {label} done in {time.perf_counter() - start:.2f}s", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="directory containing train.npz and validation.npz")
    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=None,
        help="dataset directory whose train.npz is used for fitting; validation still comes from --dataset",
    )
    parser.add_argument(
        "--input-channel",
        choices=["u_act", "u_cmd"],
        default="u_act",
        help="input channel supplied to identification methods; u_cmd represents practical external pilot commands",
    )
    parser.add_argument(
        "--state-source",
        choices=["direct", "mocap", "both"],
        default="direct",
        help="direct uses noisy V/alpha/gamma/Q; mocap uses states derived from position/attitude mocap",
    )
    parser.add_argument("--max-samples", type=int, default=50000, help="maximum derivative samples for regression/neural fits")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smooth-window", type=int, default=21)
    parser.add_argument("--polyorder", type=int, default=3)
    parser.add_argument("--subspace-order", type=int, default=6)
    parser.add_argument("--subspace-past", type=int, default=12)
    parser.add_argument("--stitch-models", type=int, default=6, help="number of local linear models for model stitching")
    parser.add_argument("--stitch-ridge", type=float, default=1e-7, help="ridge regularization for each stitched local model")
    parser.add_argument(
        "--stitch-length-scale",
        type=float,
        default=0.0,
        help="RBF scheduling length scale in normalized state-control space; 0 uses 0.35 times the median local-center distance",
    )
    parser.add_argument("--ekf-process-std", type=float, nargs=4, default=[0.05, 0.002, 0.002, 0.01])
    parser.add_argument("--ekf-measurement-std", type=float, nargs=4, default=[0.12, 0.006, 0.006, 0.02])
    parser.add_argument("--ekf-initial-std", type=float, nargs=4, default=[0.3, 0.02, 0.02, 0.05])
    parser.add_argument("--ekf-jac-stride", type=int, default=1)
    parser.add_argument("--ekf-param-trials", type=int, default=8)
    parser.add_argument("--ekf-param-stride", type=int, default=10)
    parser.add_argument("--ekf-theta-initial-std", type=float, nargs=8, default=[0.08, 0.6, 0.02, 0.08, 0.2, 1.0, 8.0, 0.5])
    parser.add_argument("--ekf-theta-process-std", type=float, nargs=8, default=[1e-5, 2e-4, 1e-5, 2e-5, 5e-5, 3e-4, 1.5e-3, 1e-4])
    parser.add_argument("--skip-oem", action="store_true", help="skip the slower output-error fit")
    parser.add_argument("--max-oem-trials", type=int, default=4)
    parser.add_argument("--oem-stride", type=int, default=4)
    parser.add_argument("--max-oem-nfev", type=int, default=200)
    parser.add_argument("--oem-segments", type=int, default=6, help="multiple-shooting segments per trial for OEM-SS")
    parser.add_argument("--max-vi-trials", type=int, default=2)
    parser.add_argument("--vi-stride", type=int, default=20)
    parser.add_argument("--max-vi-nfev", type=int, default=45)
    parser.add_argument("--vi-process-noise", type=float, nargs=4, default=[0.25, 0.015, 0.015, 0.06])
    parser.add_argument("--vi-position-process-noise", type=float, default=0.02)
    parser.add_argument("--sindy-threshold", type=float, default=0.04)
    parser.add_argument("--sindy-ridge", type=float, default=1e-6)
    parser.add_argument("--symbolic-max-terms", type=int, default=5)
    parser.add_argument("--symbolic-penalty", type=float, default=1.0)
    parser.add_argument("--symbolic-ridge", type=float, default=1e-6)
    parser.add_argument("--koopman-ridge", type=float, default=1e-6)
    parser.add_argument("--gp-centers", type=int, default=384)
    parser.add_argument("--gp-length-scale", type=float, default=0.0, help="RBF length scale for the sparse GP-style surrogate; 0 uses median center distance")
    parser.add_argument("--gp-ridge", type=float, default=1e-5)
    parser.add_argument("--frequency-max-hz", type=float, default=12.0)
    parser.add_argument("--frequency-ridge", type=float, default=1e-5)
    parser.add_argument("--frequency-nperseg", type=int, default=1024)
    parser.add_argument("--frequency-min-coherence", type=float, default=0.08)
    parser.add_argument("--uq-trials", type=int, default=2)
    parser.add_argument("--uq-stride", type=int, default=40)
    parser.add_argument("--uq-ridge", type=float, default=1e-8)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--ude-gain",
        type=float,
        default=float("nan"),
        help="rollout gain applied to learned UDE residual dynamics; default is source-aware",
    )
    parser.add_argument(
        "--pinn-gain",
        type=float,
        default=float("nan"),
        help="rollout gain applied to learned PINN coefficient residuals; default is source-aware",
    )
    parser.add_argument("--mocap-position-noise", type=float, default=0.002, help="mocap position noise scale used for output-error weighting")
    parser.add_argument("--mocap-attitude-noise", type=float, default=0.0015, help="mocap attitude noise scale used for output-error weighting")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--nperseg", type=int, default=512)
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--table-dir", type=Path, default=TABLE_DIR)
    parser.add_argument("--include-methods", nargs="*", default=["all"], help="method rows to run; default runs all")
    return parser.parse_args()


def main() -> int:
    global FIG_DIR, RESULTS_DIR, TABLE_DIR
    args = parse_args()
    FIG_DIR = args.fig_dir
    RESULTS_DIR = args.results_dir
    TABLE_DIR = args.table_dir
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    include_methods = set(args.include_methods or ["all"])
    stitch_requested = "all" in include_methods or bool({"Model-Stitching", "Frequency-Stitching"} & include_methods)
    train_dataset = args.train_dataset or args.dataset
    raw_train, _unused_train_validation = load_dataset(train_dataset)
    _unused_validation_train, raw_validation = load_dataset(args.dataset)
    stitch_raw_train: SplitData | None = None
    stitch_dataset: Path | None = None
    if stitch_requested:
        stitch_dataset = default_stitch_train_dataset(args.dataset)
        stitch_raw_train, _ = load_dataset(stitch_dataset)
    if args.input_channel == "u_cmd":
        raw_train = replace(raw_train, u_act=raw_train.u_cmd)
        raw_validation = replace(raw_validation, u_act=raw_validation.u_cmd)
        if stitch_raw_train is not None:
            stitch_raw_train = replace(stitch_raw_train, u_act=stitch_raw_train.u_cmd)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"dataset: {args.dataset}")
    print(f"input channel: {args.input_channel}")
    print(f"state source: {args.state_source}")
    print(f"train dataset: {train_dataset}")
    print(f"train: {raw_train.n_trials} trials x {raw_train.n_time} samples, validation: {raw_validation.n_trials} trials")
    if stitch_raw_train is not None and stitch_dataset is not None:
        print(f"model stitching train: {stitch_dataset} ({stitch_raw_train.n_trials} trim-grid trials)")
    print(f"torch device: {device}")
    dataset_text = str(args.dataset).lower()
    has_hidden_controller = (
        "proprietary" in dataset_text
        or "autopilot" in dataset_text
        or "safe_loop" in dataset_text
        or dataset_text.endswith("_safe")
        or "_safe" in dataset_text
    )

    source_names = ["direct", "mocap"] if args.state_source == "both" else [args.state_source]
    all_rows: list[dict[str, object]] = []
    plot_results: list[MethodResult] | None = None
    plot_validation: SplitData | None = None
    coeff_plot_result: MethodResult | None = None
    histories: dict[str, list[tuple[int, float]]] = {}
    all_uq_rows: list[dict[str, object]] = []
    last_sindy_names: list[str] | None = None
    last_sindy_coeff: np.ndarray | None = None
    last_symbolic_names: list[str] | None = None
    last_symbolic_coeff: np.ndarray | None = None
    trace_rows: list[dict[str, object]] = []
    def wants(method_name: str) -> bool:
        return "all" in include_methods or method_name in include_methods

    for source in source_names:
        train = raw_train
        validation = raw_validation
        if source == "mocap":
            train = replace(train, y_meas=train.mocap_derived_state)
            validation = replace(validation, y_meas=validation.mocap_derived_state)
        stitch_train = stitch_raw_train
        if source == "mocap" and stitch_train is not None:
            stitch_train = replace(stitch_train, y_meas=stitch_train.mocap_derived_state)
        local_args = argparse.Namespace(**vars(args))
        if not np.isfinite(local_args.ude_gain):
            local_args.ude_gain = 1.0 if source == "direct" else 0.1
        if not np.isfinite(local_args.pinn_gain):
            local_args.pinn_gain = 1.0 if source == "direct" else 0.1
        print(f"\n--- running {source} state-source benchmark ---")
        print(f"  residual gains: UDE={local_args.ude_gain:g}, PINN={local_args.pinn_gain:g}", flush=True)

        x_smooth, dxdt = smooth_trials(train.y_meas, train.dt, local_args.smooth_window, local_args.polyorder)
        xu, deriv, coeff_res = flatten_samples(train, x_smooth, dxdt, local_args.max_samples, local_args.seed)

        results: list[MethodResult] = []
        pinn_result: MethodResult | None = None
        if wants("Nominal-GreyBox"):
            results.append(run_logged("Nominal-GreyBox", lambda: run_nominal(validation)))
        if wants("Linear-SS"):
            results.append(run_logged("Linear-SS", lambda: run_linear_state_space(x_smooth, train.u_act, validation)))
        if wants("Model-Stitching"):
            if stitch_train is None:
                raise RuntimeError("Model-Stitching requested without a loaded trim-grid training dataset")
            stitch_x_smooth, _stitch_dxdt = smooth_trials(
                stitch_train.y_meas,
                stitch_train.dt,
                local_args.smooth_window,
                local_args.polyorder,
            )
            results.append(
                run_logged(
                    "Model-Stitching",
                    lambda: run_model_stitching(stitch_x_smooth, stitch_train.u_act, stitch_train, validation, local_args),
                )
            )
        if wants("Koopman-EDMD"):
            results.append(run_logged("Koopman-EDMD", lambda: run_koopman_edmd(x_smooth, train.u_act, validation, local_args)))
        if wants("Subspace-Hankel"):
            results.append(run_logged("Subspace-Hankel", lambda: run_subspace_hankel(x_smooth, train.u_act, validation, local_args)))
        if wants("Frequency-Welch"):
            results.append(run_logged("Frequency-Welch", lambda: run_frequency_linear(x_smooth, train.u_act, dxdt, validation, local_args)))
        if wants("Frequency-Stitching"):
            if stitch_train is None:
                raise RuntimeError("Frequency-Stitching requested without a loaded trim-grid training dataset")
            stitch_x_smooth, stitch_dxdt = smooth_trials(
                stitch_train.y_meas,
                stitch_train.dt,
                local_args.smooth_window,
                local_args.polyorder,
            )
            results.append(
                run_logged(
                    "Frequency-Stitching",
                    lambda: run_frequency_stitching(
                        stitch_x_smooth,
                        stitch_train.u_act,
                        stitch_dxdt,
                        stitch_train,
                        validation,
                        local_args,
                    ),
                )
            )
        if wants("EquationError-LS"):
            results.append(run_logged("EquationError-LS", lambda: run_equation_error(xu, deriv, validation)))
        if wants("EKF-ParamID"):
            results.append(run_logged("EKF-ParamID", lambda: run_filter_error_ekf(train, xu, deriv, validation, local_args)))
        if wants("Fisher-UQ"):
            theta_uq = fit_equation_error(xu, deriv)
            all_uq_rows.extend(run_logged("Fisher-UQ", lambda: fisher_uq_diagnostics(train, theta_uq, source, "EquationError-LS", local_args)))
        if not local_args.skip_oem:
            if wants("OEM-SS"):
                results.append(run_logged("OEM-SS", lambda: run_oem(train, validation, local_args)))
            if wants("OEM-MocapOutput"):
                results.append(run_logged("OEM-MocapOutput", lambda: run_oem_mocap_output(train, validation, local_args)))
            if wants("Variational-Mocap"):
                results.append(run_logged("Variational-Mocap", lambda: run_variational_mocap_output(train, validation, local_args)))
            if wants("OEM-HiddenController") and local_args.input_channel == "u_cmd" and has_hidden_controller:
                results.append(run_logged("OEM-HiddenController", lambda: run_oem_hidden_controller(train, validation, local_args)))
        if wants("SINDy"):
            sindy_result, sindy_names, sindy_coeff = run_logged("SINDy", lambda: run_sindy(xu, deriv, validation, local_args))
            results.append(sindy_result)
            last_sindy_names = sindy_names
            last_sindy_coeff = sindy_coeff
        if wants("Symbolic-Stepwise"):
            symbolic_result, symbolic_names, symbolic_coeff = run_logged(
                "Symbolic-Stepwise",
                lambda: run_symbolic_regression(xu, deriv, validation, local_args),
            )
            results.append(symbolic_result)
            last_symbolic_names = symbolic_names
            last_symbolic_coeff = symbolic_coeff
        if wants("GP-CoeffClosure"):
            results.append(run_logged("GP-CoeffClosure", lambda: run_gp_coeff_closure(xu, deriv, validation, local_args)))
        if wants("UDE-Residual"):
            ude_result, ude_history = run_logged("UDE-Residual", lambda: run_ude(xu, deriv, validation, local_args, device))
            results.append(ude_result)
            histories[f"{source}_ude_residual"] = ude_history
        if wants("PINN-CoeffClosure"):
            pinn_result, pinn_history = run_logged("PINN-CoeffClosure", lambda: run_pinn_closure(xu, deriv, validation, local_args, device))
            results.append(pinn_result)
            histories[f"{source}_pinn_coeff_closure"] = pinn_history
        if wants("UDE-HiddenControl"):
            hidden_ude_result, hidden_ude_history = run_logged(
                "UDE-HiddenControl",
                lambda: run_ude_hidden_control(xu, deriv, validation, local_args, device),
            )
            results.append(hidden_ude_result)
            histories[f"{source}_ude_hidden_control"] = hidden_ude_history
        if wants("PINN-HiddenElevator"):
            hidden_pinn_result, hidden_pinn_history = run_logged(
                "PINN-HiddenElevator",
                lambda: run_pinn_hidden_control(xu, deriv, validation, local_args, device),
            )
            results.append(hidden_pinn_result)
            histories[f"{source}_pinn_hidden_elevator"] = hidden_pinn_history
        if wants("NN-CoeffSurrogate"):
            surrogate_result, surrogate_history = run_logged("NN-CoeffSurrogate", lambda: run_supervised_coeff_surrogate(xu, coeff_res, validation, local_args, device))
            results.append(surrogate_result)
            histories[f"{source}_nn_coeff_surrogate"] = surrogate_history

        rows = summarize_results(results, validation)
        write_method_traces(results, validation, dataset_scenario_id(args.dataset), source)
        trace_payload = json.loads((RESULTS_DIR / "shared_method_traces.json").read_text())
        trace_rows.extend(trace_payload.get("traces", []))
        for row in rows:
            row["input_channel"] = args.input_channel
        if args.state_source == "both" or source != "direct":
            for row in rows:
                row["method"] = f"{row['method']} ({source})"
                row["state_source"] = source
        else:
            for row in rows:
                row["state_source"] = source
        all_rows.extend(rows)
        if results:
            plot_results = results
            plot_validation = validation
        if pinn_result is not None:
            coeff_plot_result = pinn_result

    if all_rows:
        write_rows(all_rows)
    if trace_rows:
        (RESULTS_DIR / "shared_method_traces.json").write_text(json.dumps({"traces": trace_rows}, indent=2, sort_keys=True) + "\n")
    if last_sindy_names is not None and last_sindy_coeff is not None:
        write_sindy_coefficients(last_sindy_names, last_sindy_coeff)
    if last_symbolic_names is not None and last_symbolic_coeff is not None:
        write_symbolic_coefficients(last_symbolic_names, last_symbolic_coeff)
    write_uq_diagnostics(all_uq_rows)
    write_histories(histories)
    if all_rows:
        plot_comparison(all_rows)
    if plot_results is not None and plot_validation is not None:
        ordered_results = sorted(
            plot_results,
            key=lambda r: aggregate_trajectory_score(r.validation_trajectories.reshape(-1, 4), plot_validation.x_true.reshape(-1, 4)),
        )
        plot_validation_trajectory(ordered_results, plot_validation)
        if coeff_plot_result is not None:
            plot_coeff_residual(coeff_plot_result, plot_validation)
        plot_frequency_diagnostic(plot_validation, args)

    for row in sorted(all_rows, key=lambda item: float(item["validation_score"])):
        print(
            f"{row['method']}: validation_score={float(row['validation_score']):.4g}, "
            f"train_time={float(row['train_elapsed_s']):.2f}s, backend={row['backend']}"
        )
    print(f"wrote {RESULTS_DIR / 'shared_method_comparison.csv'}")
    print(f"wrote {TABLE_DIR / 'shared_method_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
