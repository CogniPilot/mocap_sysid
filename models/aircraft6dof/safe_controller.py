#!/usr/bin/env python3
"""Identify the hidden SAFE inner-loop controller against the fitted grey-box.

In the 2026-05-22 Sport Cub flights the SAFE (self-level) inner loop is active
whenever the transmitter mode channel is high: the stick commands attitude
(throttle passes through) and the hidden Horizon Hobby controller reshapes it
into surface deflections the recording does not contain. Per attitude axis
the controller is modeled as

    delta = Kp * (sat(c * stick, +/-limit) - attitude) - Kd * rate + b,

followed by a first-order surface lag and the physical throws; the rudder is
stick passthrough plus yaw-rate damping.

Identification is staged against the *fitted* grey-box airframe (never the
nominal model -- its control-effectiveness rows are wrong for this airframe):

1. Effective surfaces by inverse dynamics through the fitted moment model.
   The pitch inversion is scalar in KMe; the lateral inversion is the coupled
   2x2 solve in (KLda, KLdr; KNda, KNdr). The fitted airframe carries most
   roll authority in the dihedral/rudder path (KLdr >> KLda), which makes the
   recovered aileron channel weakly identified -- the reason this stage only
   initializes the fit.
2. Staged Huber fit of the per-axis law on the recovered surfaces: a grid
   over the nonlinear (c, limit) wraps a robust linear solve of (Kp, Kd, b).
3. Joint closed-loop simulation-error refinement at the validation horizon:
   the controller parameters, per-surface lags, and airframe corrections
   (regularized by the manual fit's Cramer-Rao standard deviations) are
   optimized through 5 s rollouts of the composed system over the train
   stabilized windows.

The held-out metric is the same 5 s free-run position error the website's
direct closed-loop linear fit reports, so the composed grey-box is directly
comparable. ``analyze_implied_law`` additionally extracts the control law
implied by the closed-loop fit through the grey-box control effectiveness
(B K = A_cl - A on the rate rows) as a structure diagnostic.

Outputs ``results/sportcub_safe_controller.json`` via the CLI and exposes
``fit_safe_controller``/``safe_controller`` for the website exporter.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from benchmark.paths import RESULTS
from .segmentation import (
    euler_from_quat_array,
    is_autonomous,
    sample_labels,
    split_stabilized_windows,
    stabilized_windows,
)

FLIGHTS_DEFAULT = Path("data/sportcub_mocap_5_22_26_flights.npz")
EXPLORER_JSON_DEFAULT = Path("site/public/data/flight_explorer.json")
MIN_AIRSPEED_MPS = 2.5
HUBER_DELTA = 1.0
IRLS_ITERATIONS = 10
SURFACE_LIMITS = np.array([0.65, 0.75, 0.65])  # elev, ail, rud command units
VALIDATION_HORIZON_S = 5.0
N_CONTROLLER = 13
N_LAGS = 3
PRIOR_WEIGHT = 1.0


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


class GreyboxAirframe:
    """Vectorized Euler-state grey-box dynamics for a parameter dict."""

    def __init__(self, greybox: dict):
        self.names = list(greybox["parameter_names"])
        self.theta = np.asarray(greybox["parameters"], dtype=float)
        self.cr_std = np.asarray(greybox.get("cr_std", np.ones_like(self.theta)), dtype=float)
        self.fixed = dict(greybox["fixed_parameters"])
        self.throw = {k: np.deg2rad(v) for k, v in greybox["max_deflection_deg"].items()}
        self.idx = {n: i for i, n in enumerate(self.names)}

    def rhs(self, x: np.ndarray, u_cmd: np.ndarray, theta: np.ndarray | None = None) -> np.ndarray:
        """Batched rhs: x (n,12) = [pos3, uvw3, euler3, pqr3], u_cmd (n,4) in
        u_cmd order (thr, elev, ail, rud). Matches build_casadi_dynamics."""
        P = self.theta if theta is None else theta
        I = self.idx
        F = self.fixed
        u_b, v_b, w_b = x[:, 3], x[:, 4], x[:, 5]
        phi, theta_e, psi = x[:, 6], x[:, 7], x[:, 8]
        p_r, q_r, r_r = x[:, 9], x[:, 10], x[:, 11]
        thr = np.maximum(u_cmd[:, 0], 0.0)
        de = self.throw["elevator"] * u_cmd[:, 1]
        da = self.throw["aileron"] * u_cmd[:, 2]
        dr = self.throw["rudder"] * u_cmd[:, 3]
        speed = np.maximum(np.sqrt(u_b**2 + v_b**2 + w_b**2 + 1e-9), 1e-3)
        alpha = np.arctan2(w_b, u_b)
        beta = np.arcsin(np.clip(v_b / speed, -0.99, 0.99))
        qbar = 0.5 * F["rho"] * speed**2
        CL = P[I["CL0"]] + P[I["CLa"]] * alpha
        CD = P[I["CD0"]] + P[I["CDCLS"]] * CL**2
        CY = P[I["CYb"]] * beta
        lift, drag, side = qbar * F["S"] * CL, qbar * F["S"] * CD, qbar * F["S"] * CY
        thrust = P[I["KT"]] * F["m"] * thr
        ca_, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        fx = -drag * ca_ * cb - side * ca_ * sb + lift * sa + thrust
        fy = -drag * sb + side * cb
        fz = -drag * sa * cb - side * sa * sb - lift * ca_
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta_e), np.sin(theta_e)
        cpsi, spsi = np.cos(psi), np.sin(psi)
        g, m = F["g"], F["m"]
        ud = fx / m - g * sth + r_r * v_b - q_r * w_b
        vd = fy / m + g * sphi * cth + p_r * w_b - r_r * u_b
        wd = fz / m + g * cphi * cth + q_r * u_b - p_r * v_b
        bV = F["b"] / (2 * speed)
        cV = F["cbar"] / (2 * speed)
        roll_acc = qbar * (P[I["KL0"]] + P[I["KLb"]] * beta + P[I["KLp"]] * bV * p_r + P[I["KLr"]] * bV * r_r + P[I["KLda"]] * da + P[I["KLdr"]] * dr)
        pitch_acc = qbar * (P[I["KM0"]] + P[I["KMa"]] * alpha + P[I["KMq"]] * cV * q_r + P[I["KMe"]] * de)
        yaw_acc = qbar * (P[I["KN0"]] + P[I["KNb"]] * beta + P[I["KNp"]] * bV * p_r + P[I["KNr"]] * bV * r_r + P[I["KNda"]] * da + P[I["KNdr"]] * dr)
        Ixx, Iyy, Izz, Ixz = F["Ixx"], F["Iyy"], F["Izz"], F["Ixz"]
        pd_ = roll_acc + ((Iyy - Izz) / Ixx) * q_r * r_r + (Ixz / Ixx) * p_r * q_r
        qd = pitch_acc + ((Izz - Ixx) / Iyy) * p_r * r_r + (Ixz / Iyy) * (r_r**2 - p_r**2)
        rd = yaw_acc + ((Ixx - Iyy) / Izz) * p_r * q_r + (Ixz / Izz) * q_r * r_r
        cth_safe = np.where(cth >= 0, np.maximum(cth, 1e-3), np.minimum(cth, -1e-3))
        common = q_r * sphi + r_r * cphi
        phid = p_r + (sth / cth_safe) * common
        thetad = q_r * cphi - r_r * sphi
        psid = common / cth_safe
        pn = cth * cpsi * u_b + (sphi * sth * cpsi - cphi * spsi) * v_b + (cphi * sth * cpsi + sphi * spsi) * w_b
        pe = cth * spsi * u_b + (sphi * sth * spsi + cphi * cpsi) * v_b + (cphi * sth * spsi - sphi * cpsi) * w_b
        pdn = -sth * u_b + sphi * cth * v_b + cphi * cth * w_b
        return np.stack([pn, pe, pdn, ud, vd, wd, phid, thetad, psid, pd_, qd, rd], axis=1)

    def rk4(self, x: np.ndarray, u_cmd: np.ndarray, dt: float, theta: np.ndarray | None = None) -> np.ndarray:
        k1 = self.rhs(x, u_cmd, theta)
        k2 = self.rhs(x + 0.5 * dt * k1, u_cmd, theta)
        k3 = self.rhs(x + 0.5 * dt * k2, u_cmd, theta)
        k4 = self.rhs(x + dt * k3, u_cmd, theta)
        return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def effective_surfaces(airframe: GreyboxAirframe, x: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample effective (elev, ail, rud) commands by inverse dynamics.

    The observed angular accelerations minus inertia coupling give the
    aerodynamic moments; subtracting the fitted surface-free model and
    inverting the surface-effectiveness block yields the deflections the
    airframe must have seen. Pitch is scalar in KMe; roll/yaw is the coupled
    2x2 solve. Low-airspeed samples are masked (the inversion degenerates
    with dynamic pressure).
    """
    P, I, F = airframe.theta, airframe.idx, airframe.fixed
    rates = smooth_columns(x[:, 10:13], 9)
    rates_dot = np.gradient(rates, dt, axis=0, edge_order=2)
    u_b, v_b, w_b = x[:, 3], x[:, 4], x[:, 5]
    speed = np.sqrt(u_b**2 + v_b**2 + w_b**2)
    valid = speed >= MIN_AIRSPEED_MPS
    speed_s = np.maximum(speed, 1e-3)
    alpha = np.arctan2(w_b, u_b)
    beta = np.arcsin(np.clip(v_b / speed_s, -0.99, 0.99))
    qbar = 0.5 * F["rho"] * speed_s**2
    bV = F["b"] / (2 * speed_s)
    cV = F["cbar"] / (2 * speed_s)
    p_r, q_r, r_r = rates.T
    Ixx, Iyy, Izz, Ixz = F["Ixx"], F["Iyy"], F["Izz"], F["Ixz"]
    pd_a = rates_dot[:, 0] - ((Iyy - Izz) / Ixx) * q_r * r_r - (Ixz / Ixx) * p_r * q_r
    qd_a = rates_dot[:, 1] - ((Izz - Ixx) / Iyy) * p_r * r_r - (Ixz / Iyy) * (r_r**2 - p_r**2)
    rd_a = rates_dot[:, 2] - ((Ixx - Iyy) / Izz) * p_r * q_r - (Ixz / Izz) * q_r * r_r
    l0 = qbar * (P[I["KL0"]] + P[I["KLb"]] * beta + P[I["KLp"]] * bV * p_r + P[I["KLr"]] * bV * r_r)
    m0 = qbar * (P[I["KM0"]] + P[I["KMa"]] * alpha + P[I["KMq"]] * cV * q_r)
    n0 = qbar * (P[I["KN0"]] + P[I["KNb"]] * beta + P[I["KNp"]] * bV * p_r + P[I["KNr"]] * bV * r_r)
    de = (qd_a - m0) / (qbar * P[I["KMe"]])
    lat_m = np.array([[P[I["KLda"]], P[I["KLdr"]]], [P[I["KNda"]], P[I["KNdr"]]]])
    lat_inv = np.linalg.inv(lat_m)
    lat = (lat_inv @ np.stack([(pd_a - l0) / qbar, (rd_a - n0) / qbar])).T
    out = np.column_stack([
        de / airframe.throw["elevator"],
        lat[:, 0] / airframe.throw["aileron"],
        lat[:, 1] / airframe.throw["rudder"],
    ])
    return out, valid


def pd_command(theta_c: np.ndarray, u_stick: np.ndarray, s12: np.ndarray) -> np.ndarray:
    """Batched static law: elev/ail [Kp, c, limit, Kd, b], rud [gs, Kr, b]."""
    ge, ga, gr = theta_c[0:5], theta_c[5:10], theta_c[10:13]
    th_cmd = np.clip(ge[1] * u_stick[:, 1], -abs(ge[2]), abs(ge[2]))
    ph_cmd = np.clip(ga[1] * u_stick[:, 2], -abs(ga[2]), abs(ga[2]))
    de = ge[0] * (th_cmd - s12[:, 7]) - ge[3] * s12[:, 10] + ge[4]
    da = ga[0] * (ph_cmd - s12[:, 6]) - ga[3] * s12[:, 9] + ga[4]
    dr = gr[0] * u_stick[:, 3] + gr[1] * s12[:, 11] + gr[2]
    return np.stack([de, da, dr], axis=1)


def safe_controller(gains: dict[str, np.ndarray]):
    """Static controller u_eff = controller(u_stick, x_quat13) from gains.

    Single-sample convenience used by prediction tooling; the surface lags
    are omitted (they matter for the closed-loop fit, not for reshaping one
    stick sample).
    """

    theta_c = np.concatenate([gains["elevator"], gains["aileron"], gains["rudder"]])

    def control(u_stick: np.ndarray, x: np.ndarray) -> np.ndarray:
        eul = euler_from_quat_array(x[None, 6:10])[0]
        s12 = np.concatenate([x[0:3], x[3:6], eul, x[10:13]])[None, :]
        surf = np.clip(pd_command(theta_c, u_stick[None, :], s12)[0], -SURFACE_LIMITS, SURFACE_LIMITS)
        return np.array([u_stick[0], surf[0], surf[1], surf[2]])

    return control


def _rollout(airframe, theta_c, taus, theta_air, x0_batch, sticks_batch, n, dt, every):
    s = x0_batch.copy()
    surf = np.clip(pd_command(theta_c, sticks_batch[:, 0], s), -SURFACE_LIMITS, SURFACE_LIMITS)
    alphas = dt / np.maximum(np.abs(taus), dt)
    outs = []
    u = np.empty((len(s), 4))
    for k in range(n):
        cmd = np.clip(pd_command(theta_c, sticks_batch[:, k], s), -SURFACE_LIMITS, SURFACE_LIMITS)
        surf = surf + alphas * (cmd - surf)
        u[:, 0] = sticks_batch[:, k, 0]
        u[:, 1:4] = surf
        s = airframe.rk4(s, u, dt, theta_air)
        if (k + 1) % every == 0:
            outs.append(s.copy())
    return np.stack(outs, axis=1), s


def _to_state12(row: np.ndarray) -> np.ndarray:
    eul = euler_from_quat_array(row[None, 6:10])[0]
    return np.concatenate([row[0:3], row[3:6], eul, row[10:13]])


def _meas12(rows: np.ndarray) -> np.ndarray:
    return np.column_stack([rows[:, 0:3], rows[:, 3:6], euler_from_quat_array(rows[:, 6:10]), rows[:, 10:13]])


def collect_flights(data) -> tuple[list[dict], float]:
    dt = float(data["sample_period_s"])
    flights = []
    for fi, name in enumerate(str(s) for s in data["segment_names"]):
        mask = np.asarray(data["valid_mask"][fi], dtype=bool)
        x = np.asarray(data["x_meas"][fi][mask], dtype=float)
        sticks = np.asarray(data["u_cmd"][fi][mask], dtype=float)
        mode = np.asarray(data["flight_mode"][fi][mask])
        tracked = (
            np.asarray(data["mocap_tracked"][fi][mask]) != 0
            if "mocap_tracked" in data.files
            else np.ones(len(x), dtype=bool)
        )
        if is_autonomous(name):
            continue
        labels = sample_labels(x, mode)
        windows = split_stabilized_windows(stabilized_windows(labels, tracked, dt))
        flights.append({"name": name, "x": x, "sticks": sticks, "windows": windows})
    return flights, dt


def analyze_implied_law(airframe: GreyboxAirframe, closed_loop_weights: np.ndarray, flights: list[dict], dt: float) -> dict:
    """Control law implied by the closed-loop fit through the grey-box.

    On the rate rows the 3x3 surface effectiveness is invertible, so
    B K = A_cl - A attributes the closed-loop change exactly. The u,v,w
    columns of the demanded feedback are reported as airframe regime
    mismatch: the SAFE unit has no airspeed sensor and cannot feed them back.
    """
    inv_sum, stick_sum, count = np.zeros(8), np.zeros(4), 0
    for fl in flights:
        for a, b, split in fl["windows"]:
            eul = euler_from_quat_array(fl["x"][a:b, 6:10])
            inv = np.column_stack([fl["x"][a:b, 3:6], eul[:, 0:2], fl["x"][a:b, 10:13]])
            inv_sum += inv.sum(axis=0)
            stick_sum += fl["sticks"][a:b].sum(axis=0)
            count += b - a
    inv_trim, stick_trim = inv_sum / count, stick_sum / count

    def rhs_inv(inv, stick):
        x = np.zeros((1, 12))
        x[0, 3:6], x[0, 6:8], x[0, 9:12] = inv[0:3], inv[3:5], inv[5:8]
        xd = airframe.rhs(x, stick[None, :])[0]
        return np.concatenate([xd[3:6], xd[6:8], xd[9:12]])

    eps = 1e-4
    A = np.zeros((8, 8))
    for j in range(8):
        d = np.zeros(8)
        d[j] = eps
        A[:, j] = (rhs_inv(inv_trim + d, stick_trim) - rhs_inv(inv_trim - d, stick_trim)) / (2 * eps)
    B = np.zeros((8, 4))
    for j in range(4):
        d = np.zeros(4)
        d[j] = eps
        B[:, j] = (rhs_inv(inv_trim, stick_trim + d) - rhs_inv(inv_trim, stick_trim - d)) / (2 * eps)
    W = np.asarray(closed_loop_weights)
    A_cl = (W[0:8, 0:8].T - np.eye(8)) / dt
    B_cl = W[8:12, 0:8].T / dt
    B_rate = B[5:8, 1:4]
    K_x = np.linalg.solve(B_rate, (A_cl - A)[5:8, :])
    K_s = np.linalg.solve(B_rate, B_cl[5:8, :])
    K_sensed = K_x.copy()
    K_sensed[:, 0:3] = 0.0
    resid = B_rate @ K_sensed - (A_cl - A)[5:8, :]
    explained = 1 - np.linalg.norm(resid) / max(np.linalg.norm((A_cl - A)[5:8, :]), 1e-9)
    return {
        "state_names": ["u", "v", "w", "phi", "theta", "p", "q", "r"],
        "K_x": np.round(K_x, 3).tolist(),
        "K_stick": np.round(K_s, 3).tolist(),
        "sensed_feedback_explained": round(float(explained), 3),
        "trim_invariant": np.round(inv_trim, 3).tolist(),
        "rate_row_effectiveness": np.round(B_rate, 3).tolist(),
    }


def fit_safe_controller(
    flights_path: Path,
    greybox: dict,
    closed_loop_weights: np.ndarray | None = None,
    max_nfev: int = 120,
    verbose: bool = False,
) -> dict:
    """Full staged identification; returns gains, lags, corrections, scores."""
    airframe = GreyboxAirframe(greybox)
    data = np.load(flights_path, allow_pickle=False)
    flights, dt = collect_flights(data)
    if not flights:
        raise SystemExit("no piloted flights with stabilized windows found")

    # Stage 1: effective surfaces on train windows.
    sticks_l, surf_l, eul_l, rate_l = [], [], [], []
    for fl in flights:
        surf, valid = effective_surfaces(airframe, fl["x"], dt)
        eul = euler_from_quat_array(fl["x"][:, 6:10])
        for a, b, split in fl["windows"]:
            if split != "train":
                continue
            v = valid[a:b]
            sticks_l.append(fl["sticks"][a:b][v])
            surf_l.append(surf[a:b][v])
            eul_l.append(eul[a:b][v])
            rate_l.append(fl["x"][a:b, 10:13][v])
    stick = np.concatenate(sticks_l)
    surf = np.concatenate(surf_l)
    eul = np.concatenate(eul_l)
    rate = np.concatenate(rate_l)

    # Stage 2: staged Huber fit per axis.
    def staged_axis(stick1, att, rate1, surf1):
        best, best_cost = None, np.inf
        for c in np.linspace(0.2, 1.2, 11):
            for lim in np.linspace(0.2, 1.0, 9):
                cmd = np.clip(c * stick1, -lim, lim)
                phi_m = np.column_stack([cmd - att, rate1, np.ones(len(stick1))])
                coef = huber_fit(phi_m, surf1)
                r = surf1 - phi_m @ coef
                mad = np.median(np.abs(r - np.median(r)))
                if mad < best_cost:
                    best_cost, best = mad, np.array([coef[0], c, lim, -coef[1], coef[2]])
        return best

    g_elev = staged_axis(stick[:, 1], eul[:, 1], rate[:, 1], surf[:, 0])
    g_ail = staged_axis(stick[:, 2], eul[:, 0], rate[:, 0], surf[:, 1])
    g_rud = huber_fit(np.column_stack([stick[:, 3], rate[:, 2], np.ones(len(stick))]), surf[:, 2])
    theta_c0 = np.concatenate([g_elev, g_ail, g_rud])

    # Stage 3: joint simulation-error refinement at the validation horizon.
    horizon = int(round(VALIDATION_HORIZON_S / dt))
    stride = horizon // 2
    every = int(round(0.25 / dt))
    shot_x0, shot_sticks, shot_meas = [], [], []
    val_x0, val_sticks, val_target = [], [], []
    for fl in flights:
        for a, b, split in fl["windows"]:
            if split == "train":
                for s0 in range(a, b - horizon - 1, stride):
                    shot_x0.append(_to_state12(fl["x"][s0]))
                    shot_sticks.append(fl["sticks"][s0 : s0 + horizon])
                    meas = fl["x"][s0 + every : s0 + 1 + horizon : every][: horizon // every]
                    shot_meas.append(_meas12(meas))
            elif b - a >= horizon + 1:
                window = fl["x"][max(0, a - 12) : a + 1]
                x0 = fl["x"][a].copy() if len(window) < 3 else None
                # local linear fit initial state, matching the website metric
                lo = max(0, a - 12)
                samples = fl["x"][lo : a + 1]
                t_rel = (np.arange(lo, a + 1) - a) * dt
                phi_m = np.column_stack([np.ones(len(samples)), t_rel])
                coef = np.linalg.lstsq(phi_m, samples, rcond=None)[0]
                x0 = coef[0]
                x0[6:10] /= max(np.linalg.norm(x0[6:10]), 1e-12)
                val_x0.append(_to_state12(x0))
                val_sticks.append(fl["sticks"][a : a + horizon])
                val_target.append(fl["x"][a + horizon, 0:3])
    shot_x0 = np.array(shot_x0)
    shot_sticks = np.array(shot_sticks)
    shot_meas = np.array(shot_meas)
    val_x0 = np.array(val_x0)
    val_sticks = np.array(val_sticks)
    val_target = np.array(val_target)

    w_state = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 2.0, 2.0, 0.0, 0.05, 0.05, 0.05])
    n_air = len(airframe.theta)

    def split_z(z):
        return z[:N_CONTROLLER], np.exp(z[N_CONTROLLER : N_CONTROLLER + N_LAGS]), airframe.theta + z[N_CONTROLLER + N_LAGS :] * airframe.cr_std

    def residuals(z):
        theta_c, taus, theta_air = split_z(z)
        pred, _ = _rollout(airframe, theta_c, taus, theta_air, shot_x0, shot_sticks, horizon, dt, every)
        err = (pred - shot_meas) * w_state
        return np.concatenate([err.ravel() / np.sqrt(err.size) * 30.0, PRIOR_WEIGHT * z[N_CONTROLLER + N_LAGS :] / np.sqrt(n_air)])

    def validate(z):
        theta_c, taus, theta_air = split_z(z)
        _, final = _rollout(airframe, theta_c, taus, theta_air, val_x0, val_sticks, horizon, dt, every=horizon)
        ok = np.all(np.isfinite(final), axis=1)
        return np.linalg.norm(final[ok, 0:3] - val_target[ok], axis=1)

    z0 = np.concatenate([theta_c0, np.log([0.1, 0.1, 0.1]), np.zeros(n_air)])
    init_err = float(np.mean(validate(z0)))
    start = time.time()
    fit = least_squares(residuals, z0, method="trf", diff_step=1e-2, max_nfev=max_nfev, verbose=2 if verbose else 0)
    theta_c, taus, theta_air = split_z(fit.x)
    errors = validate(fit.x)
    elapsed = time.time() - start

    corrections = {
        name: {
            "manual": round(float(airframe.theta[i]), 5),
            "refined": round(float(theta_air[i]), 5),
            "sigma": round(float(fit.x[N_CONTROLLER + N_LAGS + i]), 2),
        }
        for i, name in enumerate(airframe.names)
        if abs(fit.x[N_CONTROLLER + N_LAGS + i]) > 0.5
    }
    result = {
        "gains": {
            "elevator": np.round(theta_c[0:5], 5).tolist(),
            "aileron": np.round(theta_c[5:10], 5).tolist(),
            "rudder": np.round(theta_c[10:13], 5).tolist(),
        },
        "surface_lag_s": {k: round(float(t), 4) for k, t in zip(("elevator", "aileron", "rudder"), taus)},
        "airframe_corrections": corrections,
        "scores": {
            "composed_pos_err_5s_m": round(float(np.mean(errors)), 3),
            "staged_init_pos_err_5s_m": round(init_err, 3),
            "validation_windows": int(len(errors)),
            "train_shooting_windows": int(len(shot_x0)),
            "fit_seconds": round(elapsed, 1),
        },
        "stage1_surface_std": np.round(surf.std(axis=0), 3).tolist(),
        "staged_gains": {
            "elevator": np.round(g_elev, 4).tolist(),
            "aileron": np.round(g_ail, 4).tolist(),
            "rudder": np.round(g_rud, 4).tolist(),
        },
    }
    if closed_loop_weights is not None:
        result["implied_law"] = analyze_implied_law(airframe, np.asarray(closed_loop_weights), flights, dt)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flights", type=Path, default=FLIGHTS_DEFAULT)
    parser.add_argument("--explorer-json", type=Path, default=EXPLORER_JSON_DEFAULT, help="source of the fitted grey-box and closed-loop weights")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--max-nfev", type=int, default=120)
    args = parser.parse_args()

    explorer = json.loads(args.explorer_json.read_text())
    greybox = explorer["models"]["greybox"]
    weights = explorer["models"].get("safe_invariant_weights")
    result = fit_safe_controller(args.flights, greybox, weights, max_nfev=args.max_nfev, verbose=True)

    print("gains:")
    for axis, coef in result["gains"].items():
        print(f"  {axis:9s}", np.round(coef, 3))
    print("surface lags:", result["surface_lag_s"])
    print("composed held-out 5 s position error:", result["scores"]["composed_pos_err_5s_m"], "m")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    output = args.results_dir / "sportcub_safe_controller.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
