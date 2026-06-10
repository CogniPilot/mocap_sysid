"""3-DOF longitudinal aircraft simulator with hidden nonlinear aerodynamics.

The nominal model matches the four-state longitudinal benchmark used elsewhere
in this repository.  The true simulator keeps the nominal terms near trim but
adds smooth stall-like coefficient nonlinearities: lift-curve rollover, drag
rise, and a nose-down pitching-moment break.  This keeps the benchmark close to
nominal for local maneuvers while making aggressive trajectories expose
model-form error for OEM, SINDy, PINN, and UDE comparisons.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


STATE_NAMES = np.array(["V", "alpha", "gamma", "Q"])
MOCAP_NAMES = np.array(["x_pos", "z_pos", "theta"])
POSE_NAMES = np.array(["x_e", "z_u", "theta"])
INPUT_NAMES = np.array(["T", "delta_e"])
CONTROL_NAMES = np.array(["thrust", "aileron", "elevator", "rudder"])
DIRECT_STATE_NAMES = np.array(["V", "alpha", "gamma", "q"])
COEFFICIENT_NAMES = np.array(["C_L", "C_D", "C_M"])
LOAD_NAMES = np.array(["L", "D", "M"])
STATE_LABELS = [r"$V$ [m/s]", r"$\alpha$ [deg]", r"$\gamma$ [deg]", r"$Q$ [deg/s]"]
MOCAP_RATE_HZ = 100.0
MOCAP_DT = 1.0 / MOCAP_RATE_HZ
FORMAT_VERSION = "sysid.timeseries.ragged.v1"
STALL_ALPHA_RAD = np.deg2rad(12.0)
STALL_WIDTH_RAD = np.deg2rad(2.0)
STALL_CL_MAX = 1.05
MIN_ENVELOPE_SPEED = 4.0
MAX_ENVELOPE_SPEED = 34.0
MAX_ENVELOPE_ALPHA = 0.75
MAX_ENVELOPE_GAMMA = 1.20
MAX_ENVELOPE_Q = 3.50


@dataclass(frozen=True)
class Aircraft:
    mass: float = 1.0
    jy: float = 0.15
    wing_area: float = 0.25
    chord: float = 0.18
    rho: float = 1.225
    gravity: float = 9.81


@dataclass(frozen=True)
class NominalAero:
    """Standard non-dimensional aerodynamic coefficients.

    Cm = Cm0 + Cma a + Cmq (cbar q / 2V) + Cmde de with M = Cm qbar S cbar.
    The (cbar q / 2V) rate scaling keeps Cmq dimensionless and makes pitch
    damping speed-dependent.
    """

    c_l0: float = 0.10
    c_l_alpha: float = 3.00
    c_d0: float = 0.030
    k: float = 0.10
    c_m0: float = 0.05
    c_m_alpha: float = -0.60
    c_m_q: float = -12.0
    c_m_delta_e: float = 0.60

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.c_l0,
                self.c_l_alpha,
                self.c_d0,
                self.k,
                self.c_m0,
                self.c_m_alpha,
                self.c_m_q,
                self.c_m_delta_e,
            ]
        )


@dataclass(frozen=True)
class SimulationConfig:
    duration: float = 40.0
    dt: float = 0.01
    train_trials: int = 64
    validation_trials: int = 16
    seed: int = 7
    dataset_mode: Literal[
        "open_loop",
        "sine_sweep",
        "aggressive",
        "trim_grid",
        "safe_loop",
        "open_loop_safe",
        "sine_sweep_safe",
        "aggressive_safe",
        "proprietary_autopilot",
    ] = "open_loop"
    measurement_noise: tuple[float, float, float, float] = (0.08, 0.0035, 0.0035, 0.012)
    mocap_position_noise: float = 0.002
    mocap_attitude_noise: float = 0.0015
    mocap_smoothing_window: int = 21
    process_disturbance: bool = False
    aero_variation: float = 0.0
    max_resample_attempts: int = 80

    def __post_init__(self) -> None:
        if not np.isclose(self.dt, MOCAP_DT):
            raise ValueError(f"mocap observation rate is locked to {MOCAP_RATE_HZ:g} Hz, so dt must be {MOCAP_DT:g} s")


@dataclass
class Trial:
    t: np.ndarray
    x_true: np.ndarray
    y_meas: np.ndarray
    mocap_true: np.ndarray
    mocap_meas: np.ndarray
    mocap_derived_state: np.ndarray
    u_cmd: np.ndarray
    u_act: np.ndarray
    autopilot_correction: np.ndarray
    coeff_nominal: np.ndarray
    coeff_true: np.ndarray
    coeff_residual: np.ndarray
    loads_nominal: np.ndarray
    loads_true: np.ndarray
    residual_dynamics: np.ndarray
    disturbance: np.ndarray
    x0: np.ndarray
    trim_state: np.ndarray
    trim_controls: np.ndarray
    aero_scale: float


def make_time(duration: float, dt: float) -> np.ndarray:
    return np.arange(0.0, duration + 0.5 * dt, dt)


def trim_state(aircraft: Aircraft, aero: NominalAero, speed: float = 15.0) -> np.ndarray:
    qbar = 0.5 * aircraft.rho * speed**2
    c_l_trim = aircraft.mass * aircraft.gravity / (qbar * aircraft.wing_area)
    alpha_trim = (c_l_trim - aero.c_l0) / aero.c_l_alpha
    return np.array([speed, alpha_trim, 0.0, 0.0])


def trim_controls(aircraft: Aircraft, aero: NominalAero, x_trim: np.ndarray) -> np.ndarray:
    v, alpha, _, q_rate = x_trim
    coeff = nominal_coefficients(x_trim, np.array([0.0, 0.0]), aero, aircraft)
    qbar = dynamic_pressure(v, aircraft)
    drag = coeff[1] * qbar * aircraft.wing_area
    thrust = drag / max(np.cos(alpha), 0.25)
    q_hat = pitch_rate_hat(v, q_rate, aircraft)
    elevator = -(aero.c_m0 + aero.c_m_alpha * alpha + aero.c_m_q * q_hat) / aero.c_m_delta_e
    return np.array([thrust, np.clip(elevator, -0.30, 0.30)])


def dynamic_pressure(v: float, aircraft: Aircraft) -> float:
    return 0.5 * aircraft.rho * max(v, 3.0) ** 2


def pitch_rate_hat(v: float, q_rate: float, aircraft: Aircraft) -> float:
    return aircraft.chord * q_rate / (2.0 * max(v, 3.0))


def nominal_coefficients(x: np.ndarray, u: np.ndarray, aero: NominalAero, aircraft: Aircraft = Aircraft()) -> np.ndarray:
    v, alpha, _, q_rate = x
    _, elevator = u
    q_hat = pitch_rate_hat(v, q_rate, aircraft)
    c_l = aero.c_l0 + aero.c_l_alpha * alpha
    c_d = aero.c_d0 + aero.k * c_l**2
    c_m = aero.c_m0 + aero.c_m_alpha * alpha + aero.c_m_q * q_hat + aero.c_m_delta_e * elevator
    return np.array([c_l, c_d, c_m])


def nonlinear_residual_coefficients(
    x: np.ndarray,
    u: np.ndarray,
    scale: float = 1.0,
    aero: NominalAero | None = None,
) -> np.ndarray:
    v, alpha, gamma, q_rate = x
    _, elevator = u
    v_ref = (v - 15.0) / 15.0
    q_ref = np.clip(q_rate / 0.8, -2.0, 2.0)
    a_ref = alpha / 0.20
    e_ref = elevator / 0.25

    alpha_sign = np.tanh(alpha / 0.015)
    abs_alpha = abs(alpha)
    stall_gate = 1.0 / (1.0 + np.exp(-np.clip((abs_alpha - STALL_ALPHA_RAD) / STALL_WIDTH_RAD, -60.0, 60.0)))
    alpha_excess = STALL_WIDTH_RAD * np.log1p(np.exp(np.clip((abs_alpha - STALL_ALPHA_RAD) / STALL_WIDTH_RAD, -60.0, 60.0)))

    aero = aero or NominalAero()
    nominal = nominal_coefficients(x, u, aero)
    c_l_attached = nominal[0]

    # Smooth small-RC stall model.  The attached-flow lift curve is retained
    # near trim.  Past the stall onset the true lift rolls toward a finite
    # separated-flow plateau instead of growing linearly without bound.
    c_l_post = alpha_sign * (STALL_CL_MAX - 1.35 * alpha_excess)
    c_l_post += 0.035 * np.tanh(1.5 * e_ref) + 0.018 * np.tanh(q_ref)
    c_l_true = (1.0 - stall_gate) * c_l_attached + stall_gate * c_l_post
    c_l_true += (1.0 - 0.65 * stall_gate) * (0.010 * np.sin(1.8 * e_ref) + 0.010 * a_ref * q_ref)

    # Drag rises strongly in separated flow.  This creates the main practical
    # penalty of post-stall RC maneuvers: speed bleeds off quickly during hard
    # pull-ups even when the lift curve has already saturated.
    d_c_d = 0.0035 * e_ref**2 + 0.0025 * a_ref**4
    d_c_d += stall_gate * (0.090 + 0.95 * alpha_excess + 4.50 * alpha_excess**2)
    d_c_d += 0.016 * stall_gate * max(-v_ref, 0.0)

    # The pitching moment has a mild pre-stall residual and a post-stall
    # nose-down break.  For positive alpha this is negative, matching the
    # recovery tendency of a conventional trainer wing/tail combination.
    # Values are in the standard Cm normalization (M = Cm qbar S cbar); the
    # post-stall break must dominate the attached Cma term so held high-alpha
    # probes recover instead of departing.
    d_c_m = -0.033 * a_ref**3 + 0.033 * a_ref * e_ref + 0.022 * np.tanh(q_ref) * abs(a_ref)
    d_c_m += 0.014 * np.sin(2.0 * gamma)
    d_c_m += -stall_gate * (0.42 + 3.0 * alpha_excess) * alpha_sign
    d_c_m += -0.10 * stall_gate * np.tanh(e_ref)

    d_c_l = c_l_true - c_l_attached
    return scale * np.array([d_c_l, d_c_d, d_c_m])


def true_coefficients(x: np.ndarray, u: np.ndarray, aero: NominalAero, scale: float = 1.0) -> np.ndarray:
    return nominal_coefficients(x, u, aero) + nonlinear_residual_coefficients(x, u, scale, aero)


def loads_from_coefficients(x: np.ndarray, coeff: np.ndarray, aircraft: Aircraft) -> np.ndarray:
    qbar = dynamic_pressure(x[0], aircraft)
    loads = coeff * qbar * aircraft.wing_area
    loads[..., 2] *= aircraft.chord
    return loads


def dynamics(
    x: np.ndarray,
    u: np.ndarray,
    aircraft: Aircraft,
    aero: NominalAero,
    *,
    true_aero: bool,
    aero_scale: float = 1.0,
    disturbance: np.ndarray | None = None,
) -> np.ndarray:
    v, alpha, gamma, q_rate = x
    thrust, _ = u
    coeff = true_coefficients(x, u, aero, aero_scale) if true_aero else nominal_coefficients(x, u, aero)
    lift, drag, moment = loads_from_coefficients(x, coeff, aircraft)
    v_safe = max(v, 3.0)
    x_dot = np.array(
        [
            (-drag + thrust * np.cos(alpha) - aircraft.mass * aircraft.gravity * np.sin(gamma)) / aircraft.mass,
            0.0,
            (lift + thrust * np.sin(alpha) - aircraft.mass * aircraft.gravity * np.cos(gamma))
            / (aircraft.mass * v_safe),
            moment / aircraft.jy,
        ]
    )
    x_dot[1] = q_rate - x_dot[2]
    if disturbance is not None:
        x_dot = x_dot + disturbance
    return x_dot


def rk4_step(
    x: np.ndarray,
    u0: np.ndarray,
    u1: np.ndarray,
    d0: np.ndarray,
    d1: np.ndarray,
    aircraft: Aircraft,
    aero: NominalAero,
    dt: float,
    aero_scale: float,
) -> np.ndarray:
    umid = 0.5 * (u0 + u1)
    dmid = 0.5 * (d0 + d1)
    k1 = dynamics(x, u0, aircraft, aero, true_aero=True, aero_scale=aero_scale, disturbance=d0)
    k2 = dynamics(x + 0.5 * dt * k1, umid, aircraft, aero, true_aero=True, aero_scale=aero_scale, disturbance=dmid)
    k3 = dynamics(x + 0.5 * dt * k2, umid, aircraft, aero, true_aero=True, aero_scale=aero_scale, disturbance=dmid)
    k4 = dynamics(x + dt * k3, u1, aircraft, aero, true_aero=True, aero_scale=aero_scale, disturbance=d1)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next[0] = max(x_next[0], 3.0)
    return x_next


def actuator_response(u_cmd: np.ndarray, dt: float) -> np.ndarray:
    u_act = np.empty_like(u_cmd)
    u_act[0] = u_cmd[0]
    tau = np.array([0.12, 0.055])
    rate_limit = np.array([7.0, 2.4])
    lower = np.array([0.0, -0.35])
    upper = np.array([3.0, 0.35])
    for k in range(len(u_cmd) - 1):
        desired_rate = (u_cmd[k] - u_act[k]) / tau
        limited_rate = np.clip(desired_rate, -rate_limit, rate_limit)
        u_act[k + 1] = np.clip(u_act[k] + dt * limited_rate, lower, upper)
    return u_act


def mocap_from_state(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    velocity_x = x[:, 0] * np.cos(x[:, 2])
    velocity_z = x[:, 0] * np.sin(x[:, 2])
    x_pos = np.zeros(len(t))
    z_pos = np.zeros(len(t))
    dt = np.diff(t)
    x_pos[1:] = np.cumsum(0.5 * dt * (velocity_x[:-1] + velocity_x[1:]))
    z_pos[1:] = np.cumsum(0.5 * dt * (velocity_z[:-1] + velocity_z[1:]))
    theta = x[:, 1] + x[:, 2]
    return np.column_stack((x_pos, z_pos, theta))


def pose_from_mocap(mocap: np.ndarray) -> np.ndarray:
    return np.column_stack((mocap[:, 0], mocap[:, 1], mocap[:, 2]))


def control_from_command(u_cmd: np.ndarray) -> np.ndarray:
    return np.column_stack((u_cmd[:, 0], np.zeros(len(u_cmd)), u_cmd[:, 1], np.zeros(len(u_cmd))))


def smooth_signal(y: np.ndarray, window: int) -> np.ndarray:
    window = min(max(1, window), len(y))
    if window <= 1:
        return y.copy()
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return y.copy()
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(y, ((pad, pad), (0, 0)), mode="edge")
    return np.column_stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(y.shape[1])])


def derive_state_from_mocap(t: np.ndarray, mocap: np.ndarray, smoothing_window: int) -> np.ndarray:
    smoothed = smooth_signal(mocap, smoothing_window)
    x_dot = np.gradient(smoothed[:, 0], t, edge_order=2)
    z_dot = np.gradient(smoothed[:, 1], t, edge_order=2)
    theta_dot = np.gradient(smoothed[:, 2], t, edge_order=2)
    speed = np.hypot(x_dot, z_dot)
    gamma = np.arctan2(z_dot, x_dot)
    alpha = smoothed[:, 2] - gamma
    return np.column_stack((speed, alpha, gamma, theta_dot))


def random_command(
    t: np.ndarray,
    u_trim: np.ndarray,
    rng: np.random.Generator,
    *,
    split: Literal["train", "validation"],
) -> np.ndarray:
    duration = t[-1] - t[0]
    thrust = np.full_like(t, u_trim[0])
    elevator = np.full_like(t, u_trim[1])
    freq_scale = 1.0 if split == "train" else 1.18

    # Thrust content avoids the lightly damped phugoid band (~0.67 rad/s):
    # one slow component below it plus faster components above it, so 40 s
    # open-loop trials do not resonantly pump the phugoid out of the envelope.
    freq = freq_scale * rng.uniform(0.08, 0.25)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    thrust += rng.uniform(0.025, 0.080) * np.sin(freq * t + phase)
    for _ in range(3):
        freq = freq_scale * rng.uniform(1.30, 2.60)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        thrust += rng.uniform(0.025, 0.090) * np.sin(freq * t + phase)

    for _ in range(5):
        freq = freq_scale * rng.uniform(0.25, 2.10)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        elevator += rng.uniform(0.008, 0.035) * np.sin(freq * t + phase)

    for _ in range(3):
        start = rng.uniform(0.10 * duration, 0.80 * duration)
        width = rng.uniform(1.0, 4.0)
        smooth_box = 0.5 * (np.tanh(4.0 * (t - start)) - np.tanh(4.0 * (t - start - width)))
        elevator += rng.uniform(-0.040, 0.040) * smooth_box
        thrust += rng.uniform(-0.060, 0.060) * smooth_box

    return np.column_stack((np.clip(thrust, 0.0, 3.0), np.clip(elevator, -0.35, 0.35)))


def open_loop_command(
    t: np.ndarray,
    u_trim: np.ndarray,
    rng: np.random.Generator,
    *,
    split: Literal["train", "validation"],
) -> np.ndarray:
    command = random_command(t, u_trim, rng, split=split)
    duration = t[-1] - t[0]
    elevator = command[:, 1].copy()
    thrust = command[:, 0].copy()
    for _ in range(5):
        center = rng.uniform(0.08 * duration, 0.92 * duration)
        width = rng.uniform(0.12, 0.45)
        sign = rng.choice([-1.0, 1.0])
        doublet = 0.5 * (
            np.tanh(24.0 * (t - center))
            - 2.0 * np.tanh(24.0 * (t - center - width))
            + np.tanh(24.0 * (t - center - 2.0 * width))
        )
        elevator += sign * rng.uniform(0.012, 0.032) * doublet
    for _ in range(3):
        center = rng.uniform(0.12 * duration, 0.88 * duration)
        width = rng.uniform(0.8, 2.8)
        pulse = 0.5 * (np.tanh(6.0 * (t - center)) - np.tanh(6.0 * (t - center - width)))
        thrust += rng.uniform(-0.045, 0.045) * pulse
    return np.column_stack((np.clip(thrust, 0.0, 3.0), np.clip(elevator, -0.35, 0.35)))


def sine_sweep_command(
    t: np.ndarray,
    u_trim: np.ndarray,
    rng: np.random.Generator,
    *,
    split: Literal["train", "validation"],
) -> np.ndarray:
    duration = max(t[-1] - t[0], 1.0)
    command = open_loop_command(t, u_trim, rng, split=split)
    thrust = command[:, 0].copy()
    elevator = command[:, 1].copy()
    # start frequency adapts to the record length so short flight-test
    # records still complete their lowest sweep cycle
    f0 = rng.uniform(max(0.04, 1.0 / duration), max(0.08, 2.0 / duration))
    f1 = rng.uniform(1.80, 3.20) if split == "train" else rng.uniform(2.10, 3.60)
    phase0 = rng.uniform(0.0, 2.0 * np.pi)
    sweep_rate = (f1 - f0) / duration
    phase = 2.0 * np.pi * (f0 * t + 0.5 * sweep_rate * t**2) + phase0
    amp = rng.uniform(0.110, 0.170) if split == "train" else rng.uniform(0.130, 0.200)
    envelope = np.sin(np.pi * np.clip(t / duration, 0.0, 1.0)) ** 0.35
    elevator += amp * envelope * np.sin(phase)
    elevator += 0.018 * envelope * np.sin(2.0 * phase + rng.uniform(0.0, 2.0 * np.pi))

    # Add deterministic high-alpha probe segments so the frequency-rich dataset
    # still samples separated-flow behavior instead of remaining a small-signal
    # linear frequency-response test. Brief pull-and-release probes sized for
    # the realistic pitch dynamics (long held deflections would loop out of
    # the gamma envelope).
    centers = np.linspace(0.16 * duration, 0.82 * duration, 6 if split == "train" else 5)
    centers += rng.uniform(-0.025 * duration, 0.025 * duration, size=len(centers))
    for center in centers:
        width = rng.uniform(0.8, 1.8)
        pulse = 0.5 * (np.tanh(4.2 * (t - center)) - np.tanh(4.2 * (t - center - width)))
        elevator += rng.uniform(0.200, 0.300) * pulse
        thrust -= rng.uniform(0.100, 0.300) * pulse

    # Slow speed variation below the lightly damped phugoid (~0.107 Hz) to
    # avoid resonant pumping during the open-loop sweep.
    thrust_phase = rng.uniform(0.0, 2.0 * np.pi)
    thrust += rng.uniform(0.05, 0.12) * np.sin(2.0 * np.pi * rng.uniform(0.015, 0.05) * t + thrust_phase)
    return np.column_stack((np.clip(thrust, 0.08, 2.80), np.clip(elevator, -0.35, 0.35)))


def recovery_probe_command(
    t: np.ndarray,
    u_trim: np.ndarray,
    rng: np.random.Generator,
    *,
    split: Literal["train", "validation"],
) -> np.ndarray:
    command = aggressive_command(t, u_trim, rng, split=split)
    duration = t[-1] - t[0]
    thrust = command[:, 0].copy()
    elevator = command[:, 1].copy()
    pulse_count = 6 if split == "train" else 5
    centers = np.linspace(0.12 * duration, 0.86 * duration, pulse_count)
    centers += rng.uniform(-0.035 * duration, 0.035 * duration, size=pulse_count)
    for center in centers:
        width = rng.uniform(1.2, 2.8)
        pulse = 0.5 * (np.tanh(5.0 * (t - center)) - np.tanh(5.0 * (t - center - width)))
        elevator += rng.uniform(0.120, 0.240) * pulse
        thrust -= rng.uniform(0.060, 0.260) * pulse
    return np.column_stack((np.clip(thrust, 0.20, 2.80), np.clip(elevator, -0.35, 0.35)))


def aggressive_command(
    t: np.ndarray,
    u_trim: np.ndarray,
    rng: np.random.Generator,
    *,
    split: Literal["train", "validation"],
) -> np.ndarray:
    """High-excitation longitudinal maneuver for nonlinear aerodynamic stress tests."""

    command = open_loop_command(t, u_trim, rng, split=split)
    duration = t[-1] - t[0]
    thrust = command[:, 0].copy()
    elevator = command[:, 1].copy()

    # Fast elevator doublets excite pitch-rate and alpha-dependent residuals.
    doublet_count = 8 if split == "train" else 7
    for _ in range(doublet_count):
        center = rng.uniform(0.06 * duration, 0.90 * duration)
        width = rng.uniform(0.18, 0.70)
        sign = rng.choice([-1.0, 1.0])
        doublet = 0.5 * (
            np.tanh(34.0 * (t - center))
            - 2.0 * np.tanh(34.0 * (t - center - width))
            + np.tanh(34.0 * (t - center - 2.0 * width))
        )
        elevator += sign * rng.uniform(0.080, 0.150) * doublet

    # Longer pull-up and pushover pulses move the trajectory away from trim so
    # the residual coefficient model is no longer well approximated by a local
    # Taylor expansion. Amplitudes are sized for the realistic pitch dynamics:
    # a held elevator now produces sustained pitch rate, so long large pulses
    # would loop the aircraft out of the gamma envelope.
    pulse_count = 5 if split == "train" else 4
    centers = np.linspace(0.12 * duration, 0.84 * duration, pulse_count)
    centers += rng.uniform(-0.035 * duration, 0.035 * duration, size=pulse_count)
    for center in centers:
        width = rng.uniform(0.50, 1.20)
        sign = rng.choice([-1.0, 1.0])
        pulse = 0.5 * (np.tanh(6.0 * (t - center)) - np.tanh(6.0 * (t - center - width)))
        elevator += sign * rng.uniform(0.070, 0.130) * pulse
        thrust += -sign * rng.uniform(0.080, 0.200) * pulse

    # Dedicated high-alpha probes deliberately enter the smooth stall region:
    # a brief pull to past stall onset followed by release, like a real
    # stall-probe maneuver, rather than the multi-second held deflections the
    # old quasi-static pitch model tolerated.
    stall_probe_count = 7 if split == "train" else 6
    centers = np.linspace(0.14 * duration, 0.82 * duration, stall_probe_count)
    centers += rng.uniform(-0.030 * duration, 0.030 * duration, size=stall_probe_count)
    for center in centers:
        width = rng.uniform(0.8, 1.6)
        pulse = 0.5 * (np.tanh(4.5 * (t - center)) - np.tanh(4.5 * (t - center - width)))
        elevator += rng.uniform(0.200, 0.320) * pulse
        thrust -= rng.uniform(0.100, 0.300) * pulse

    # Slow throttle excursions create speed variation, which changes dynamic
    # pressure and makes coefficient errors translate into different load
    # errors. Kept well below the lightly damped phugoid (~0.107 Hz) so the
    # open-loop trial does not resonantly pump it out of the envelope.
    for _ in range(3):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        freq = rng.uniform(0.015, 0.045)
        thrust += rng.uniform(0.06, 0.15) * np.sin(2.0 * np.pi * freq * t + phase)

    return np.column_stack((np.clip(thrust, 0.08, 2.80), np.clip(elevator, -0.35, 0.35)))


def local_trim_grid_command(
    t: np.ndarray,
    u_trim: np.ndarray,
    rng: np.random.Generator,
    *,
    split: Literal["train", "validation"],
) -> np.ndarray:
    """Small-signal local excitation around a selected operating point."""

    duration = max(t[-1] - t[0], 1.0)
    thrust = np.full_like(t, u_trim[0])
    elevator = np.full_like(t, u_trim[1])
    envelope = np.sin(np.pi * np.clip(t / duration, 0.0, 1.0)) ** 0.5
    freq_scale = 1.0 if split == "train" else 1.12

    # Independent low-amplitude multisines keep each trial close to its local
    # operating point while providing enough bandwidth to fit local linear maps.
    for _ in range(5):
        freq = freq_scale * rng.uniform(0.05, 0.65)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        thrust += rng.uniform(0.025, 0.085) * envelope * np.sin(2.0 * np.pi * freq * t + phase)
    for _ in range(7):
        freq = freq_scale * rng.uniform(0.08, 2.40)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        elevator += rng.uniform(0.006, 0.024) * envelope * np.sin(2.0 * np.pi * freq * t + phase)

    for _ in range(3):
        center = rng.uniform(0.12 * duration, 0.86 * duration)
        width = rng.uniform(0.20, 0.65)
        sign = rng.choice([-1.0, 1.0])
        doublet = 0.5 * (
            np.tanh(28.0 * (t - center))
            - 2.0 * np.tanh(28.0 * (t - center - width))
            + np.tanh(28.0 * (t - center - 2.0 * width))
        )
        elevator += sign * rng.uniform(0.006, 0.020) * doublet

    return np.column_stack((np.clip(thrust, 0.0, 3.0), np.clip(elevator, -0.35, 0.35)))


def actuator_step(u_prev: np.ndarray, u_cmd: np.ndarray, dt: float) -> np.ndarray:
    tau = np.array([0.12, 0.055])
    rate_limit = np.array([7.0, 2.4])
    lower = np.array([0.0, -0.35])
    upper = np.array([3.0, 0.35])
    desired_rate = (u_cmd - u_prev) / tau
    limited_rate = np.clip(desired_rate, -rate_limit, rate_limit)
    return np.clip(u_prev + dt * limited_rate, lower, upper)


def proprietary_autopilot_command(x: np.ndarray, pilot_cmd: np.ndarray, u_trim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v, alpha, _, q_rate = x
    gamma = x[2]
    theta = alpha + gamma
    throttle_norm = np.clip(pilot_cmd[0] / 3.0, 0.0, 1.0)
    elevator_stick = np.clip((pilot_cmd[1] - u_trim[1]) / 0.20, -1.0, 1.0)

    theta_throttle = 0.5 * np.interp(throttle_norm, [0.0, 0.5, 1.0], np.deg2rad([-5.0, 0.0, 8.0]))
    theta_cmd = 0.068 + theta_throttle + np.deg2rad(10.0) * elevator_stick
    q_cmd = np.clip(3.0 * (theta_cmd - theta), -1.10, 1.10)
    safe_elevator = u_trim[1] + 0.105 * (q_cmd - q_rate)
    as3x_elevator = pilot_cmd[1] - 0.018 * q_rate

    alpha_gate = 1.0 / (1.0 + np.exp(-np.clip((alpha - 0.110) / 0.012, -60.0, 60.0)))
    speed_gate = 1.0 / (1.0 + np.exp(-np.clip((13.0 - v) / 0.55, -60.0, 60.0)))
    recovery = 1.0 - (1.0 - alpha_gate) * (1.0 - speed_gate)
    panic_theta_cmd = 0.068 + np.deg2rad(2.0)
    panic_q_cmd = np.clip(5.0 * (panic_theta_cmd - theta), -1.30, 1.30)
    panic_elevator = u_trim[1] + 0.125 * (panic_q_cmd - q_rate)

    safe_blend = 0.85
    elevator_no_panic = safe_blend * safe_elevator + (1.0 - safe_blend) * as3x_elevator
    internal_elevator = (1.0 - recovery) * elevator_no_panic + recovery * panic_elevator
    internal_cmd = np.array(
        [
            np.clip(pilot_cmd[0], 0.0, 3.0),
            np.clip(internal_elevator, -0.35, 0.35),
        ]
    )
    return internal_cmd, internal_cmd - pilot_cmd


def colored_disturbance(t: np.ndarray, dt: float, rng: np.random.Generator, enabled: bool) -> np.ndarray:
    disturbance = np.zeros((len(t), 4))
    if not enabled:
        return disturbance
    tau = np.array([2.0, 1.4, 1.6, 1.0])
    sigma = np.array([0.035, 0.0010, 0.0010, 0.0030])
    decay = np.exp(-dt / tau)
    noise_scale = sigma * np.sqrt(1.0 - decay**2)
    for k in range(len(t) - 1):
        disturbance[k + 1] = decay * disturbance[k] + noise_scale * rng.normal(size=4)
    return disturbance


def sample_initial_state(
    x_trim: np.ndarray,
    rng: np.random.Generator,
    dataset_mode: str,
    split: Literal["train", "validation"],
) -> np.ndarray:
    if dataset_mode in {"aggressive", "sine_sweep"}:
        # Initial regimes sized for the realistic pitch dynamics: large initial
        # gamma/Q now integrates into a departure before the open-loop command
        # can counteract it, so the off-trim starts are kept recoverable.
        validation_shift = split == "validation"
        regimes = [
            ((12.5, 18.0), (-3.0, 8.0), (-8.0, 8.0), (-30.0, 30.0)),
            ((8.5, 12.0), (10.0, 18.0) if validation_shift else (8.0, 16.0), (-8.0, 10.0), (-25.0, 40.0)),
            ((10.0, 15.0), (5.0, 13.0) if validation_shift else (4.0, 12.0), (5.0, 18.0), (10.0, 50.0)),
            ((9.0, 16.0), (-2.0, 8.0), (-20.0, -5.0), (-55.0, -10.0)),
            ((18.0, 26.0), (-4.0, 4.0), (-15.0, 4.0), (-35.0, 20.0)),
        ]
    elif dataset_mode in {"safe_loop", "sine_sweep_safe", "aggressive_safe", "proprietary_autopilot"}:
        regimes = [
            ((12.0, 18.0), (0.0, 8.0), (-10.0, 10.0), (-45.0, 45.0)),
            ((8.5, 12.5), (7.0, 13.0), (-8.0, 16.0), (-50.0, 70.0)),
            ((9.0, 17.0), (0.0, 10.0), (-25.0, -8.0), (-90.0, -20.0)),
            ((19.0, 27.0), (-3.0, 4.0), (-20.0, 3.0), (-45.0, 25.0)),
        ]
    else:
        spread = np.array([0.8, 0.025, 0.025, 0.040])
        return x_trim + rng.uniform(-spread, spread)

    v_range, alpha_deg_range, gamma_deg_range, q_deg_range = regimes[int(rng.integers(0, len(regimes)))]
    return np.array(
        [
            rng.uniform(*v_range),
            np.deg2rad(rng.uniform(*alpha_deg_range)),
            np.deg2rad(rng.uniform(*gamma_deg_range)),
            np.deg2rad(rng.uniform(*q_deg_range)),
        ]
    )


def sample_trim_grid_operating_point(
    aircraft: Aircraft,
    aero: NominalAero,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose one local operating point for trim-grid small-signal excitation."""

    speeds = np.array([9.5, 11.0, 13.0, 15.0, 18.0, 22.0])
    speed = float(speeds[int(rng.integers(0, len(speeds)))])
    x_ref = trim_state(aircraft, aero, speed=speed)
    x_ref = x_ref + np.array(
        [
            rng.uniform(-0.20, 0.20),
            np.deg2rad(rng.uniform(-0.45, 0.45)),
            np.deg2rad(rng.uniform(-0.80, 0.80)),
            np.deg2rad(rng.uniform(-2.0, 2.0)),
        ]
    )
    u_ref = trim_controls(aircraft, aero, x_ref)
    return x_ref, u_ref


def simulate_trial(
    *,
    split: Literal["train", "validation"],
    seed: int,
    config: SimulationConfig,
    aircraft: Aircraft | None = None,
    aero: NominalAero | None = None,
) -> Trial:
    aircraft = aircraft or Aircraft()
    aero = aero or NominalAero()
    rng = np.random.default_rng(seed)
    t = make_time(config.duration, config.dt)
    if config.dataset_mode == "trim_grid":
        x_trim, u_trim = sample_trim_grid_operating_point(aircraft, aero, rng)
        x0 = x_trim + np.array(
            [
                rng.uniform(-0.18, 0.18),
                np.deg2rad(rng.uniform(-0.70, 0.70)),
                np.deg2rad(rng.uniform(-1.00, 1.00)),
                np.deg2rad(rng.uniform(-3.0, 3.0)),
            ]
        )
    else:
        x_trim = trim_state(aircraft, aero)
        u_trim = trim_controls(aircraft, aero, x_trim)
        x0 = sample_initial_state(x_trim, rng, config.dataset_mode, split)
    if config.dataset_mode == "trim_grid":
        u_cmd = local_trim_grid_command(t, u_trim, rng, split=split)
    elif config.dataset_mode in {"sine_sweep", "sine_sweep_safe"}:
        u_cmd = sine_sweep_command(t, u_trim, rng, split=split)
    elif config.dataset_mode in {"aggressive", "aggressive_safe"}:
        u_cmd = aggressive_command(t, u_trim, rng, split=split)
    elif config.dataset_mode in {"safe_loop", "proprietary_autopilot"}:
        u_cmd = recovery_probe_command(t, u_trim, rng, split=split)
    else:
        u_cmd = open_loop_command(t, u_trim, rng, split=split)
    u_internal = u_cmd.copy()
    autopilot_correction = np.zeros_like(u_cmd)
    u_act = np.empty_like(u_cmd)
    u_act[0] = u_cmd[0]
    disturbance = colored_disturbance(t, config.dt, rng, config.process_disturbance)
    aero_scale = max(0.0, 1.0 + config.aero_variation * rng.normal())

    x = np.empty((len(t), 4))
    x[0] = x0
    for k in range(len(t) - 1):
        if config.dataset_mode in {"safe_loop", "open_loop_safe", "sine_sweep_safe", "aggressive_safe", "proprietary_autopilot"}:
            u_internal[k], autopilot_correction[k] = proprietary_autopilot_command(x[k], u_cmd[k], u_trim)
        u_act[k + 1] = actuator_step(u_act[k], u_internal[k], config.dt)
        x[k + 1] = rk4_step(x[k], u_act[k], u_act[k + 1], disturbance[k], disturbance[k + 1], aircraft, aero, config.dt, aero_scale)
        if not is_reasonable_state(x[k + 1]):
            raise FloatingPointError("simulation left the intended flight envelope")
    if config.dataset_mode in {"safe_loop", "open_loop_safe", "sine_sweep_safe", "aggressive_safe", "proprietary_autopilot"}:
        u_internal[-1], autopilot_correction[-1] = proprietary_autopilot_command(x[-1], u_cmd[-1], u_trim)

    noise_std = np.asarray(config.measurement_noise)
    y_meas = x + rng.normal(scale=noise_std, size=x.shape)
    mocap_true = mocap_from_state(t, x)
    mocap_noise = rng.normal(
        scale=np.array([config.mocap_position_noise, config.mocap_position_noise, config.mocap_attitude_noise]),
        size=mocap_true.shape,
    )
    mocap_meas = mocap_true + mocap_noise
    mocap_derived_state = derive_state_from_mocap(t, mocap_meas, config.mocap_smoothing_window)
    coeff_nom, coeff_true, loads_nom, loads_true, residual_dyn = evaluate_auxiliary_arrays(
        x, u_act, disturbance, aircraft, aero, aero_scale
    )
    return Trial(
        t=t,
        x_true=x,
        y_meas=y_meas,
        mocap_true=mocap_true,
        mocap_meas=mocap_meas,
        mocap_derived_state=mocap_derived_state,
        u_cmd=u_cmd,
        u_act=u_act,
        autopilot_correction=autopilot_correction,
        coeff_nominal=coeff_nom,
        coeff_true=coeff_true,
        coeff_residual=coeff_true - coeff_nom,
        loads_nominal=loads_nom,
        loads_true=loads_true,
        residual_dynamics=residual_dyn,
        disturbance=disturbance,
        x0=x0,
        trim_state=x_trim,
        trim_controls=u_trim,
        aero_scale=aero_scale,
    )


def is_reasonable_state(x: np.ndarray) -> bool:
    if not np.all(np.isfinite(x)):
        return False
    v, alpha, gamma, q_rate = x
    return (
        MIN_ENVELOPE_SPEED <= v <= MAX_ENVELOPE_SPEED
        and abs(alpha) <= MAX_ENVELOPE_ALPHA
        and abs(gamma) <= MAX_ENVELOPE_GAMMA
        and abs(q_rate) <= MAX_ENVELOPE_Q
    )


def evaluate_auxiliary_arrays(
    x: np.ndarray,
    u: np.ndarray,
    disturbance: np.ndarray,
    aircraft: Aircraft,
    aero: NominalAero,
    aero_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    coeff_nom = np.empty((n, 3))
    coeff_true = np.empty((n, 3))
    loads_nom = np.empty((n, 3))
    loads_true = np.empty((n, 3))
    residual_dyn = np.empty((n, 4))
    for k in range(n):
        coeff_nom[k] = nominal_coefficients(x[k], u[k], aero)
        coeff_true[k] = true_coefficients(x[k], u[k], aero, aero_scale)
        loads_nom[k] = loads_from_coefficients(x[k], coeff_nom[k], aircraft)
        loads_true[k] = loads_from_coefficients(x[k], coeff_true[k], aircraft)
        true_rhs = dynamics(x[k], u[k], aircraft, aero, true_aero=True, aero_scale=aero_scale, disturbance=disturbance[k])
        nominal_rhs = dynamics(x[k], u[k], aircraft, aero, true_aero=False, disturbance=np.zeros(4))
        residual_dyn[k] = true_rhs - nominal_rhs
    return coeff_nom, coeff_true, loads_nom, loads_true, residual_dyn


def generate_split(
    split: Literal["train", "validation"],
    n_trials: int,
    config: SimulationConfig,
    seed: int,
) -> list[Trial]:
    trials: list[Trial] = []
    rng = np.random.default_rng(seed)
    attempts = 0
    while len(trials) < n_trials:
        if attempts > config.max_resample_attempts * max(n_trials, 1):
            raise RuntimeError(f"could not generate stable {split} trials after {attempts} attempts")
        attempts += 1
        trial_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        try:
            trials.append(simulate_trial(split=split, seed=trial_seed, config=config))
        except FloatingPointError:
            continue
    return trials


def stack_trials(trials: list[Trial]) -> dict[str, np.ndarray]:
    fields = [
        "x_true",
        "y_meas",
        "mocap_true",
        "mocap_meas",
        "mocap_derived_state",
        "u_cmd",
        "u_act",
        "autopilot_correction",
        "coeff_nominal",
        "coeff_true",
        "coeff_residual",
        "loads_nominal",
        "loads_true",
        "residual_dynamics",
        "disturbance",
        "x0",
        "trim_state",
        "trim_controls",
        "aero_scale",
    ]
    data = {"t": trials[0].t}
    for field in fields:
        data[field] = np.asarray([getattr(trial, field) for trial in trials])
    data["pose_meas"] = np.asarray([pose_from_mocap(trial.mocap_meas) for trial in trials])
    data["control_meas"] = np.asarray([control_from_command(trial.u_cmd) for trial in trials])
    data["direct_state_meas"] = data["y_meas"]
    data["time_s"] = np.broadcast_to(trials[0].t, data["x_true"].shape[:2]).copy()
    data["valid_mask"] = np.ones(data["x_true"].shape[:2], dtype=bool)
    data["segment_names"] = np.asarray([f"trial_{index:03d}" for index in range(len(trials))])
    data["state_names"] = STATE_NAMES
    data["mocap_names"] = MOCAP_NAMES
    data["pose_names"] = POSE_NAMES
    data["input_names"] = INPUT_NAMES
    data["control_names"] = CONTROL_NAMES
    data["direct_state_names"] = DIRECT_STATE_NAMES
    data["coefficient_names"] = COEFFICIENT_NAMES
    data["load_names"] = LOAD_NAMES
    data["truth_available"] = np.asarray(True)
    data["system_dof"] = np.asarray(3)
    data["format_version"] = np.asarray(FORMAT_VERSION)
    return data


def write_dataset(output_dir: Path, config: SimulationConfig, make_plot: bool = True) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = generate_split("train", config.train_trials, config, config.seed)
    validation = generate_split("validation", config.validation_trials, config, config.seed + 10_000)
    train_data = stack_trials(train)
    validation_data = stack_trials(validation)
    for split_data in (train_data, validation_data):
        split_data["dataset_id"] = np.asarray(output_dir.name)
        split_data["sample_period_s"] = np.asarray(config.dt)
    np.savez_compressed(output_dir / "train.npz", **train_data)
    np.savez_compressed(output_dir / "validation.npz", **validation_data)

    metadata = {
        "description": "Longitudinal 3-DOF aircraft trials with nominal-plus-hidden-nonlinear aerodynamics and mocap-style measurements.",
        "format_version": FORMAT_VERSION,
        "sample_period_s": config.dt,
        "config": asdict(config),
        "aircraft": asdict(Aircraft()),
        "nominal_aero": asdict(NominalAero()),
        "state_names": STATE_NAMES.tolist(),
        "mocap_names": MOCAP_NAMES.tolist(),
        "pose_names": POSE_NAMES.tolist(),
        "input_names": INPUT_NAMES.tolist(),
        "control_names": CONTROL_NAMES.tolist(),
        "direct_state_names": DIRECT_STATE_NAMES.tolist(),
        "coefficient_names": COEFFICIENT_NAMES.tolist(),
        "load_names": LOAD_NAMES.tolist(),
        "truth_available": True,
        "system_dof": 3,
        "files": {"train": "train.npz", "validation": "validation.npz"},
        "notes": [
            f"dataset_mode={config.dataset_mode}. safe_loop/open_loop_safe/sine_sweep_safe/aggressive_safe/proprietary_autopilot use a hidden SAFE/AS3X pitch input modifier; trim_grid uses local small-signal training and validation around multiple trim-like operating points; open_loop, sine_sweep, and aggressive use pilot commands with actuator lag only.",
            "u_cmd is the commanded input; u_act is the actuator-realized input used by the simulator.",
            "autopilot_correction is written for diagnostic use only; practical identification should treat it as hidden.",
            "mocap_meas is the primary experimental measurement channel: inertial position and pitch attitude.",
            "pose_meas is the canonical planar pose channel ordered as x_e, z_u, theta.",
            "control_meas is the canonical control channel ordered as thrust, aileron, elevator, rudder; lateral controls are zero for this longitudinal benchmark.",
            "direct_state_meas is the noisy direct-state channel ordered as V, alpha, gamma, q.",
            "mocap_derived_state estimates V, alpha, gamma, and Q from mocap_meas for methods that need state histories.",
            "y_meas is retained as an idealized direct-state measurement channel for oracle/debug comparisons.",
            "coeff_residual is coeff_true - coeff_nominal and is the hidden nonlinear aerodynamic term.",
            "residual_dynamics is true continuous-time RHS minus nominal RHS at the same state and input.",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_summary(output_dir / "summary.csv", train, validation)
    if make_plot:
        write_preview_plot(output_dir / "preview_trials.png", train, validation)
        write_preview_plot(output_dir / "preview_trials.svg", train, validation)
    return metadata


def write_summary(path: Path, train: list[Trial], validation: list[Trial]) -> None:
    rows = []
    for split, trials in (("train", train), ("validation", validation)):
        data = stack_trials(trials)
        for name, array_name in (
            ("V", "x_true"),
            ("alpha", "x_true"),
            ("gamma", "x_true"),
            ("Q", "x_true"),
            ("x_pos", "mocap_true"),
            ("z_pos", "mocap_true"),
            ("theta", "mocap_true"),
            ("C_L_residual", "coeff_residual"),
            ("C_D_residual", "coeff_residual"),
            ("C_M_residual", "coeff_residual"),
        ):
            index = {
                "V": 0,
                "alpha": 1,
                "gamma": 2,
                "Q": 3,
                "x_pos": 0,
                "z_pos": 1,
                "theta": 2,
                "C_L_residual": 0,
                "C_D_residual": 1,
                "C_M_residual": 2,
            }[name]
            values = data[array_name][..., index]
            rows.append(
                {
                    "split": split,
                    "signal": name,
                    "min": float(np.min(values)),
                    "mean": float(np.mean(values)),
                    "max": float(np.max(values)),
                    "std": float(np.std(values)),
                }
            )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "signal", "min", "mean", "max", "std"])
        writer.writeheader()
        writer.writerows(rows)


def write_preview_plot(path: Path, train: list[Trial], validation: list[Trial]) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(10.0, 8.0), sharex="col", constrained_layout=True)
    samples = [("train", train[: min(8, len(train))]), ("validation", validation[: min(4, len(validation))])]
    for col, (split, trials) in enumerate(samples):
        for trial in trials:
            t = trial.t
            for row in range(4):
                y = np.rad2deg(trial.x_true[:, row]) if row > 0 else trial.x_true[:, row]
                axes[row, col].plot(t, y, linewidth=0.8, alpha=0.75)
        axes[0, col].set_title(split)
        axes[-1, col].set_xlabel("Time [s]")
    for row, label in enumerate(STATE_LABELS):
        axes[row, 0].set_ylabel(label)
        for col in range(2):
            axes[row, col].grid(True, alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)
