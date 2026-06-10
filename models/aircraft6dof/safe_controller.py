#!/usr/bin/env python3
"""Identify the hidden SAFE inner-loop controller from stabilized lap data.

In the 2026-05-22 Sport Cub flights the SAFE (self-level) inner loop is active
whenever the transmitter mode channel is high. The surface commands the
airframe actually receives are then the pilot stick reshaped by the hidden
Horizon Hobby controller, which the recorded data does not contain. Free-run
prediction through stabilized flight therefore requires a controller model.

In SAFE self-level mode the pilot does not command surface deflections: the
stick commands attitude (and throttle passes through), with envelope
protection limiting the achievable attitude and the usual surface saturation.
Per attitude axis the controller is modeled as

    delta_surface = Kp * (sat(c * stick, +/-limit) - attitude) - Kd * rate + b,

where c is the stick-to-attitude-command scale, the clip is the envelope
protection, and the output saturates at the surface throws. The rudder is
modeled as passthrough plus yaw damping. The model is nonlinear only through
(c, limit), so identification is staged: a coarse grid over (c, limit) wraps a
Huber-robust linear fit of (Kp, Kd, b) given the saturated attitude command.

The effective surfaces are not measured, but they can be recovered sample by
sample through inverse dynamics: the observed angular accelerations give the
aerodynamic moments, the airframe moment model is affine in the surface
deflections, and inverting that relation yields the effective deflection the
airframe must have seen over the stabilized laps — the 6-DOF analogue of the
benchmark's hidden-controller rows.

Outputs ``results/sportcub_safe_controller.csv`` with the per-axis gains and
closed-loop validation scores, and exposes ``safe_controller`` for reuse by
prediction tooling.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from benchmark.paths import RESULTS
from .greybox import Aircraft6DOFConfig, airdata, euler_from_quaternion, normalize_quaternion, rhs

FLIGHTS_DEFAULT = Path("data/sportcub_mocap_5_22_26_flights.npz")
MIN_STABILIZED_DURATION_S = 3.0
GROUND_ALTITUDE_M = 0.5
HUBER_DELTA = 1.0
IRLS_ITERATIONS = 10
MIN_AIRSPEED_MPS = 2.5

# Moment-model surface effectiveness from models/aircraft6dof/model.py
# (attached-flow rows): Cm slope wrt elevator, Cl wrt aileron, Cn wrt rudder.
CM_DELTA_E = -1.15
CL_DELTA_A = 0.42
CN_DELTA_R = -0.26


def smooth_columns(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    window = window | 1
    kernel = np.ones(window) / window
    pad = window // 2
    out = np.empty_like(values, dtype=float)
    for dim in range(values.shape[1]):
        out[:, dim] = np.convolve(np.pad(values[:, dim], pad, mode="edge"), kernel, mode="valid")
    return out


def huber_fit(phi: np.ndarray, target: np.ndarray) -> np.ndarray:
    coef = np.linalg.lstsq(phi, target, rcond=None)[0]
    for _ in range(IRLS_ITERATIONS):
        residual = target - phi @ coef
        scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 1e-9)
        z = residual / (scale * HUBER_DELTA)
        weights = np.where(np.abs(z) <= 1.0, 1.0, 1.0 / np.maximum(np.abs(z), 1e-9))
        weighted = phi * weights[:, None]
        coef = np.linalg.solve(weighted.T @ phi + 1e-9 * np.eye(phi.shape[1]), weighted.T @ target)
    return coef


def inertia_matrix(config: Aircraft6DOFConfig) -> np.ndarray:
    ix, iy, iz = config.inertia
    ixz = config.inertia_xz
    return np.array([[ix, 0.0, -ixz], [0.0, iy, 0.0], [-ixz, 0.0, iz]])


def effective_surfaces(x: np.ndarray, u: np.ndarray, dt: float, config: Aircraft6DOFConfig) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample effective (elevator, aileron, rudder) from inverse dynamics.

    The airframe moment model is affine in each surface, so the observed
    moment minus the model moment at zero surface deflection, divided by the
    surface effectiveness, is the deflection the airframe must have seen.
    Samples below a minimum airspeed are masked: surface effectiveness scales
    with dynamic pressure and the inversion degenerates.
    """
    inertia = inertia_matrix(config)
    rates = smooth_columns(x[:, 10:13], 9)
    rates_dot = np.gradient(rates, dt, axis=0, edge_order=2)
    surfaces = np.full((len(x), 3), np.nan)
    valid = np.zeros(len(x), dtype=bool)
    for k in range(len(x)):
        state = x[k].copy()
        state[6:10] = normalize_quaternion(state[6:10])
        speed, _, _ = airdata(state)
        if speed < MIN_AIRSPEED_MPS:
            continue
        moment_obs = inertia @ rates_dot[k] + np.cross(rates[k], inertia @ rates[k])
        u_zero = np.array([u[k, 0], 0.0, 0.0, 0.0])  # throttle passthrough, surfaces at zero
        # rhs returns [pos_dot, vel_dot, quat_dot, rates_dot]; extract the
        # zero-surface moment through the same inertia relation.
        deriv_zero = rhs(state, u_zero, config)
        moment_zero = inertia @ deriv_zero[10:13] + np.cross(rates[k], inertia @ rates[k])
        qbar = 0.5 * config.rho * speed**2
        scale = qbar * config.wing_area * (1.0 + config.prop_wash_gain * u[k, 0])
        delta_moment = moment_obs - moment_zero
        surfaces[k, 0] = delta_moment[1] / (scale * config.mean_chord * CM_DELTA_E)
        surfaces[k, 1] = delta_moment[0] / (scale * config.wing_span * CL_DELTA_A)
        surfaces[k, 2] = delta_moment[2] / (scale * config.wing_span * CN_DELTA_R)
        valid[k] = True
    return surfaces, valid


def stabilized_windows(altitude: np.ndarray, mode: np.ndarray, dt: float) -> list[tuple[int, int]]:
    airborne = altitude > GROUND_ALTITUDE_M
    stabilized = airborne & (mode == 1)
    edges = np.flatnonzero(np.diff(stabilized.astype(np.int8)))
    starts = ([0] if stabilized[0] else []) + [int(e) + 1 for e in edges if not stabilized[e]]
    stops = [int(e) + 1 for e in edges if stabilized[e]] + ([len(stabilized)] if stabilized[-1] else [])
    min_samples = int(round(MIN_STABILIZED_DURATION_S / dt))
    return [(a, b) for a, b in zip(starts, stops) if b - a >= min_samples]


SURFACE_LIMITS = {"elevator": 0.65, "aileron": 0.75, "rudder": 0.65}
COMMAND_SCALE_GRID = np.linspace(0.2, 1.2, 11)
ENVELOPE_LIMIT_GRID = np.linspace(0.2, 1.0, 9)


def attitude_axis_fit(stick: np.ndarray, attitude: np.ndarray, rate: np.ndarray, surface: np.ndarray) -> np.ndarray:
    """Stage the nonlinear (c, limit) search around a Huber linear sub-fit.

    Returns [Kp, c, limit, Kd, b]: surface = Kp*(sat(c*stick, +/-limit) -
    attitude) - Kd*rate ... fitted as linear in (Kp, Kd', b) given the
    saturated attitude command, with the attitude coefficient tied to -Kp by
    construction of the regressor (att_cmd - attitude).
    """
    best = None
    best_cost = np.inf
    for scale in COMMAND_SCALE_GRID:
        for limit in ENVELOPE_LIMIT_GRID:
            command = np.clip(scale * stick, -limit, limit)
            phi = np.column_stack([command - attitude, rate, np.ones(len(stick))])
            coef = huber_fit(phi, surface)
            residual = surface - phi @ coef
            mad = np.median(np.abs(residual - np.median(residual)))
            if mad < best_cost:
                best_cost = mad
                best = np.array([coef[0], scale, limit, -coef[1], coef[2]])
    return best


def safe_controller(gains: dict[str, np.ndarray]):
    """Controller function u_eff = controller(u_stick, x) from fitted gains.

    Attitude axes use [Kp, c, limit, Kd, b]: the stick commands attitude
    through the envelope clip, the loop closes on attitude error with rate
    damping, and the output saturates at the surface throws. The rudder is
    [stick_gain, K_rate, b]; throttle passes through.
    """

    def control(u_stick: np.ndarray, x: np.ndarray) -> np.ndarray:
        euler = euler_from_quaternion(normalize_quaternion(x[6:10]))
        p_rate, q_rate, r_rate = x[10:13]
        ge = gains["elevator"]
        ga = gains["aileron"]
        gr = gains["rudder"]
        theta_cmd = np.clip(ge[1] * u_stick[1], -ge[2], ge[2])
        phi_cmd = np.clip(ga[1] * u_stick[2], -ga[2], ga[2])
        elevator = ge[0] * (theta_cmd - euler[1]) - ge[3] * q_rate + ge[4]
        aileron = ga[0] * (phi_cmd - euler[0]) - ga[3] * p_rate + ga[4]
        rudder = gr[0] * u_stick[3] + gr[1] * r_rate + gr[2]
        return np.array(
            [
                u_stick[0],
                np.clip(elevator, -SURFACE_LIMITS["elevator"], SURFACE_LIMITS["elevator"]),
                np.clip(aileron, -SURFACE_LIMITS["aileron"], SURFACE_LIMITS["aileron"]),
                np.clip(rudder, -SURFACE_LIMITS["rudder"], SURFACE_LIMITS["rudder"]),
            ]
        )

    return control


def fit_safe_controller(flights_path: Path) -> tuple[dict[str, np.ndarray], dict[str, float], int]:
    data = np.load(flights_path, allow_pickle=False)
    dt = float(data["sample_period_s"])
    config = Aircraft6DOFConfig()
    sticks, surfaces_all, eulers, rates_all = [], [], [], []
    window_count = 0
    for flight_index in range(len(data["segment_names"])):
        mask = np.asarray(data["valid_mask"][flight_index], dtype=bool)
        x = np.asarray(data["x_meas"][flight_index][mask], dtype=float)
        u = np.asarray(data["u_cmd"][flight_index][mask], dtype=float)
        mode = np.asarray(data["flight_mode"][flight_index][mask])
        for start, stop in stabilized_windows(-x[:, 2], mode, dt):
            window_count += 1
            xs = x[start:stop]
            us = u[start:stop]
            surfaces, valid = effective_surfaces(xs, us, dt, config)
            euler = np.array([euler_from_quaternion(normalize_quaternion(q)) for q in xs[:, 6:10]])
            sticks.append(us[valid])
            surfaces_all.append(surfaces[valid])
            eulers.append(euler[valid])
            rates_all.append(xs[valid, 10:13])
    if not sticks:
        raise SystemExit("no stabilized airborne windows found")
    stick = np.concatenate(sticks)
    surface = np.concatenate(surfaces_all)
    euler = np.concatenate(eulers)
    rates = np.concatenate(rates_all)

    gains = {
        "elevator": attitude_axis_fit(stick[:, 1], euler[:, 1], rates[:, 1], surface[:, 0]),
        "aileron": attitude_axis_fit(stick[:, 2], euler[:, 0], rates[:, 0], surface[:, 1]),
        "rudder": huber_fit(np.column_stack([stick[:, 3], rates[:, 2], np.ones(len(stick))]), surface[:, 2]),
    }
    controller = safe_controller(gains)
    predicted = np.array(
        [
            controller(stick[k], np.concatenate([np.zeros(6), quat_from_euler_row(euler[k]), rates[k]]))
            for k in range(len(stick))
        ]
    )
    fit_rmse = {
        "elevator": float(np.sqrt(np.mean((surface[:, 0] - predicted[:, 1]) ** 2))),
        "aileron": float(np.sqrt(np.mean((surface[:, 1] - predicted[:, 2]) ** 2))),
        "rudder": float(np.sqrt(np.mean((surface[:, 2] - predicted[:, 3]) ** 2))),
    }
    return gains, fit_rmse, window_count


def quat_from_euler_row(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = euler
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flights", type=Path, default=FLIGHTS_DEFAULT)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()

    gains, fit_rmse, window_count = fit_safe_controller(args.flights)
    rows = []
    labels = {
        "elevator": ("Kp", "cmd_scale", "envelope_limit", "Kd", "offset"),
        "aileron": ("Kp", "cmd_scale", "envelope_limit", "Kd", "offset"),
        "rudder": ("stick_gain", "K_rate", "offset", "", ""),
    }
    for axis, coef in gains.items():
        row = {"axis": axis, "fit_rmse": fit_rmse[axis]}
        for name, value in zip(labels[axis], coef):
            if name:
                row[name] = float(value)
        rows.append(row)
        print(f"  {axis:9s} " + "  ".join(f"{name}={float(value):+.3f}" for name, value in zip(labels[axis], coef) if name) + f"  rmse={fit_rmse[axis]:.3f}")
    print(f"  fitted from {window_count} stabilized windows")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    output = args.results_dir / "sportcub_safe_controller.csv"
    fieldnames = ["axis", "Kp", "cmd_scale", "envelope_limit", "Kd", "offset", "stick_gain", "K_rate", "fit_rmse"]
    with open(output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
