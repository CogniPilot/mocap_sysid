#!/usr/bin/env python3
"""Run baseline 6DOF aircraft system-identification methods."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from benchmark.paths import DATA, LATEX_FIG, LATEX_TABLES, RESULTS, ROOT, SPORTCUB_DATASET_ID, WORK_DATA
from .greybox import (
    INPUT_NAMES,
    MAX_SPEED,
    MIN_SPEED,
    STATE_NAMES,
    Aircraft6DOFConfig,
    aerodynamic_coefficients,
    airdata,
    nominal_rk4_step,
    normalize_quaternion,
    rhs,
    rotation_body_to_inertial,
)


METHODS_ROOT = ROOT
DEFAULT_DATASET = WORK_DATA / "aircraft_6dof_aggressive"
DEFAULT_RESULTS = RESULTS
DEFAULT_FIG = LATEX_FIG
DEFAULT_TABLES = LATEX_TABLES
DEFAULT_WORKERS = max(1, min(30, (os.cpu_count() or 2) - 2))
TRADEOFF_FAILURE_THRESHOLD = 1.0
WEB_TRACE_MAX_POINTS = 100
WEB_TRACE_TOP_METHODS_PER_SOURCE = 8
SCENARIO_TITLES = {
    "aircraft_6dof_open_loop": "Open-loop",
    "aircraft_6dof_sine_sweep": "Sine sweep",
    "aircraft_6dof_aggressive": "Aggressive",
    "aircraft_6dof_trim_grid": "Trim grid",
    SPORTCUB_DATASET_ID: "Sport Cub MoCap 4/17/26",
    "sportcub_mocap_5_22_26": "Sport Cub Laps 5/22/26",
}
SCENARIO_ORDER = tuple(SCENARIO_TITLES)
METHOD_TRAINING_SCENARIOS = {
    "6DOF-NominalGreyBox": "none",
    "6DOF-LinearSS": "aircraft_6dof_trim_grid",
    "6DOF-Model-Stitching": "aircraft_6dof_trim_grid",
    "6DOF-Subspace-Hankel": "aircraft_6dof_trim_grid",
    "6DOF-Frequency-Welch": "aircraft_6dof_trim_grid",
    "6DOF-Frequency-Stitching": "aircraft_6dof_trim_grid",
    "6DOF-Koopman-EDMD": "aircraft_6dof_aggressive",
    "6DOF-EquationError-LS": "aircraft_6dof_aggressive",
    "6DOF-EKF-ParamID": "aircraft_6dof_aggressive",
    "6DOF-Fisher-UQ": "aircraft_6dof_aggressive",
    "6DOF-OEM-SS": "aircraft_6dof_aggressive",
    "6DOF-RidgeResidual": "aircraft_6dof_aggressive",
    "6DOF-Variational-Mocap": "aircraft_6dof_aggressive",
    "6DOF-SINDy": "aircraft_6dof_aggressive",
    "6DOF-Symbolic-Stepwise": "aircraft_6dof_aggressive",
    "6DOF-GP-RBF": "aircraft_6dof_aggressive",
    "6DOF-UDE-Residual": "aircraft_6dof_aggressive",
    "6DOF-PINN-Closure": "aircraft_6dof_aggressive",
    "6DOF-NN-Surrogate": "aircraft_6dof_aggressive",
}


@dataclass
class Split6DOF:
    t: np.ndarray
    x_true: np.ndarray
    y_meas: np.ndarray
    mocap_true: np.ndarray
    mocap_meas: np.ndarray
    u_cmd: np.ndarray
    u_act: np.ndarray
    x0: np.ndarray
    mocap_frame: str = "ned"
    x0_estimate: np.ndarray | None = None
    input_bias: np.ndarray | None = None
    segment_names: np.ndarray | None = None

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.t)))


@dataclass
class Result6DOF:
    method: str
    description: str
    backend: str
    state_source: str
    validation_score: float
    train_elapsed_s: float
    train_cpu_s: float
    rollout_elapsed_s: float
    total_elapsed_s: float
    train_samples: int
    decision_variables: int
    rmse_position_m: float
    rmse_velocity_mps: float
    rmse_quaternion: float
    rmse_rates_rad_s: float
    rmse_mocap_position_m: float
    rmse_mocap_quaternion: float
    notes: str
    training_scenario: str = ""
    implementation_status: str = "implemented"
    diverged: bool = False
    x_pred: np.ndarray | None = None


def load_split(path: Path) -> Split6DOF:
    data = np.load(path, allow_pickle=False)
    if "time_s" in data.files and "valid_mask" in data.files and "direct_state_meas" in data.files and "pose_meas" in data.files:
        valid_mask = np.asarray(data["valid_mask"], dtype=bool)
        valid_counts = np.sum(valid_mask, axis=1)
        if not np.all(valid_counts > 1):
            raise ValueError(f"{path}: every segment needs at least two valid samples")
        n = int(np.min(valid_counts))
        t = np.asarray(data["time_s"][0, :n], dtype=float)
        if "x_true" in data.files:
            x_ref = np.asarray(data["x_true"], dtype=float)[:, :n, :]
        else:
            x_ref = np.asarray(data["direct_state_meas"], dtype=float)[:, :n, :]
        if "y_meas" in data.files:
            y_meas = np.asarray(data["y_meas"], dtype=float)[:, :n, :]
        else:
            y_meas = np.asarray(data["direct_state_meas"], dtype=float)[:, :n, :]
        if "mocap_meas" in data.files:
            mocap_meas = np.asarray(data["mocap_meas"], dtype=float)[:, :n, :]
            mocap_frame = "ned"
        else:
            mocap_meas = np.asarray(data["pose_meas"], dtype=float)[:, :n, :]
            mocap_frame = "enu"
        if "mocap_true" in data.files:
            mocap_ref = np.asarray(data["mocap_true"], dtype=float)[:, :n, :]
        else:
            mocap_ref = mocap_meas
        if "u_cmd" in data.files:
            u_cmd = np.asarray(data["u_cmd"], dtype=float)[:, :n, :]
        else:
            u_cmd = np.asarray(data["control_meas"], dtype=float)[:, :n, [0, 2, 1, 3]]
        u_act = np.asarray(data["u_act"], dtype=float)[:, :n, :] if "u_act" in data.files else u_cmd
    else:
        t = np.asarray(data["t"], dtype=float)
        x_ref = np.asarray(data["x_true"], dtype=float)
        y_meas = np.asarray(data["y_meas"], dtype=float)
        mocap_meas = np.asarray(data["mocap_meas"], dtype=float)
        mocap_ref = np.asarray(data["mocap_true"], dtype=float)
        u_cmd = np.asarray(data["u_cmd"], dtype=float)
        u_act = np.asarray(data["u_act"], dtype=float)
        mocap_frame = "ned"
    segment_names = None
    if "segment_names" in data.files:
        segment_names = np.asarray([str(name) for name in data["segment_names"]])
    return Split6DOF(
        t=t,
        x_true=x_ref,
        y_meas=y_meas,
        mocap_true=mocap_ref,
        mocap_meas=mocap_meas,
        u_cmd=u_cmd,
        u_act=u_act,
        x0=x_ref[:, 0, :],
        mocap_frame=mocap_frame,
        segment_names=segment_names,
    )


def align_quaternion_signs(x_pred: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    out = np.asarray(x_pred, dtype=float).copy()
    dots = np.sum(out[..., 6:10] * x_ref[..., 6:10], axis=-1)
    out[..., 6:10] *= np.where(dots[..., None] < 0.0, -1.0, 1.0)
    return out


def normalize_state(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=float).copy()
    out = np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
    out[6:10] = normalize_quaternion(out[6:10])
    speed = float(np.linalg.norm(out[3:6]))
    if speed > MAX_SPEED:
        out[3:6] *= MAX_SPEED / speed
    elif 1e-9 < speed < MIN_SPEED:
        out[3:6] *= MIN_SPEED / speed
    out[0:3] = np.clip(out[0:3], -1e5, 1e5)
    out[10:13] = np.clip(out[10:13], -80.0, 80.0)
    return out


def nrmse_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    if not np.all(np.isfinite(y_pred)):
        return 1.0e9
    scale = np.ptp(y_true.reshape(-1, y_true.shape[-1]), axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=tuple(range(y_true.ndim - 1))))
    return float(np.mean(rmse / scale))


def rmse_group(x_pred: np.ndarray, x_true: np.ndarray) -> dict[str, float]:
    pred = align_quaternion_signs(x_pred, x_true)
    err = pred - x_true
    mocap_pred = pred[..., [0, 1, 2, 6, 7, 8, 9]]
    mocap_true = x_true[..., [0, 1, 2, 6, 7, 8, 9]]
    return {
        "rmse_position_m": float(np.sqrt(np.mean(err[..., 0:3] ** 2))),
        "rmse_velocity_mps": float(np.sqrt(np.mean(err[..., 3:6] ** 2))),
        "rmse_quaternion": float(np.sqrt(np.mean(err[..., 6:10] ** 2))),
        "rmse_rates_rad_s": float(np.sqrt(np.mean(err[..., 10:13] ** 2))),
        "rmse_mocap_position_m": float(np.sqrt(np.mean((mocap_pred[..., 0:3] - mocap_true[..., 0:3]) ** 2))),
        "rmse_mocap_quaternion": float(np.sqrt(np.mean((mocap_pred[..., 3:7] - mocap_true[..., 3:7]) ** 2))),
    }


def smooth_array(y: np.ndarray, window: int = 9) -> np.ndarray:
    if window <= 1:
        return y.copy()
    kernel = np.ones(window) / window
    out = np.empty_like(y, dtype=float)
    pad = window // 2
    for trial in range(y.shape[0]):
        padded = np.pad(y[trial], ((pad, pad), (0, 0)), mode="edge")
        for dim in range(y.shape[-1]):
            out[trial, :, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return out


def endpoint_derivative(values: np.ndarray, t: np.ndarray, *, start: bool, samples: int = 7) -> np.ndarray:
    count = min(samples, values.shape[1], len(t))
    if count < 2:
        return np.zeros((values.shape[0], values.shape[2]), dtype=float)
    indices = slice(0, count) if start else slice(values.shape[1] - count, values.shape[1])
    time = np.asarray(t[indices], dtype=float)
    centered = time - np.mean(time)
    denom = float(centered @ centered)
    if denom < 1e-12:
        return np.zeros((values.shape[0], values.shape[2]), dtype=float)
    return np.einsum("n,tnd->td", centered, values[:, indices, :]) / denom


def derive_state_from_mocap(mocap: np.ndarray, t: np.ndarray) -> np.ndarray:
    dt = float(np.median(np.diff(t)))
    pos = smooth_array(mocap[..., 0:3], window=11)
    quat = smooth_array(mocap[..., 3:7], window=7)
    quat /= np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12)
    pos_dot = np.gradient(pos, dt, axis=1, edge_order=2)
    quat_dot = np.gradient(quat, dt, axis=1, edge_order=2)
    pos[:, 0, :] = mocap[:, 0, 0:3]
    pos[:, -1, :] = mocap[:, -1, 0:3]
    quat[:, 0, :] = mocap[:, 0, 3:7]
    quat[:, -1, :] = mocap[:, -1, 3:7]
    quat /= np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12)
    pos_dot[:, 0, :] = endpoint_derivative(mocap[..., 0:3], t, start=True)
    pos_dot[:, -1, :] = endpoint_derivative(mocap[..., 0:3], t, start=False)
    quat_dot[:, 0, :] = endpoint_derivative(mocap[..., 3:7], t, start=True)
    quat_dot[:, -1, :] = endpoint_derivative(mocap[..., 3:7], t, start=False)
    x = np.zeros((*mocap.shape[:2], len(STATE_NAMES)))
    x[..., 0:3] = pos
    x[..., 6:10] = quat
    for trial in range(mocap.shape[0]):
        for index in range(mocap.shape[1]):
            q = normalize_quaternion(quat[trial, index])
            rotation = rotation_body_to_inertial(q)
            x[trial, index, 3:6] = rotation.T @ pos_dot[trial, index]
            q0, q1, q2, q3 = q
            qmat = 0.5 * np.array(
                [
                    [-q1, -q2, -q3],
                    [q0, -q3, q2],
                    [q3, q0, -q1],
                    [-q2, q1, q0],
                ]
            )
            x[trial, index, 10:13] = np.linalg.lstsq(qmat, quat_dot[trial, index], rcond=None)[0]
    return x


ONSET_THRESHOLD_FRACTION = 0.15
ONSET_ABSOLUTE_FLOOR = 0.02
ONSET_REFERENCE_SAMPLES = 10
MAX_INPUT_BIAS = 0.2


def detect_input_onset(u: np.ndarray, threshold_fraction: float = ONSET_THRESHOLD_FRACTION) -> int:
    """Index of the first significant command deviation in a (time, channel) record.

    A channel is considered active once it leaves its initial level by more than
    threshold_fraction of its own range; the absolute floor keeps flat or
    noise-only channels from triggering.
    """
    reference = np.median(u[: min(ONSET_REFERENCE_SAMPLES, len(u))], axis=0)
    deviation = np.abs(u - reference)
    span = np.ptp(u, axis=0)
    active = deviation > np.maximum(threshold_fraction * span, ONSET_ABSOLUTE_FLOOR)
    hits = np.flatnonzero(active.any(axis=1))
    return int(hits[0]) if hits.size else 0


MIN_ACTUATION_S = 0.6
MAX_ACTUATION_S = 1.2


def load_segments(path: Path) -> tuple[list[dict[str, np.ndarray | str]], float]:
    """Load a canonical ragged real-flight file as per-segment arrays.

    The rectangular Split6DOF view truncates every segment to the shortest one,
    which discards most of the data once segments have very different lengths;
    real-flight preprocessing therefore operates on the ragged segments and
    rectangularizes only after onset alignment.
    """
    data = np.load(path, allow_pickle=False)
    if "valid_mask" not in data.files or "time_s" not in data.files:
        raise ValueError(f"{path}: expected the canonical ragged real-flight layout")
    dt = float(data["sample_period_s"]) if "sample_period_s" in data.files else float(np.nanmedian(np.diff(data["time_s"][0])))
    names = [str(name) for name in data["segment_names"]]
    if "mocap_meas" in data.files:
        mocap_key, mocap_frame = "mocap_meas", "ned"
    else:
        mocap_key, mocap_frame = "pose_meas", "enu"
    segments: list[dict[str, np.ndarray | str]] = []
    for index, name in enumerate(names):
        mask = np.asarray(data["valid_mask"][index], dtype=bool)
        tracked = (
            np.asarray(data["mocap_tracked"][index][mask]) != 0
            if "mocap_tracked" in data.files
            else np.ones(int(mask.sum()), dtype=bool)
        )
        segments.append(
            {
                "name": name,
                "x": np.asarray(data["direct_state_meas"][index][mask], dtype=float),
                "mocap": np.asarray(data[mocap_key][index][mask], dtype=float),
                "u": np.asarray(data["u_cmd"][index][mask], dtype=float),
                "tracked": tracked,
                "mocap_frame": mocap_frame,
            }
        )
    return segments, dt


def local_state_estimate(x: np.ndarray, dt: float, start: int, window: int = 12) -> np.ndarray:
    """State at sample `start` from a local linear fit instead of one noisy frame.

    For the first chunk of a maneuver the window is the lead-in; for later
    chunks it is the trailing samples of the previous chunk.
    """
    lo = max(0, start - window)
    hi = max(start + 1, lo + 3)
    hi = min(hi, len(x))
    samples = x[lo:hi]
    if len(samples) < 3:
        estimate = x[min(start, len(x) - 1)].copy()
    else:
        time = (np.arange(lo, hi) - start) * dt
        basis = np.column_stack((np.ones(len(samples)), time))
        coef, *_ = np.linalg.lstsq(basis, samples, rcond=None)
        estimate = coef[0]
    estimate[6:10] /= max(np.linalg.norm(estimate[6:10]), 1e-12)
    return estimate


def estimate_input_bias(
    segments: list[dict[str, np.ndarray | str]],
    onsets: list[int],
    dt: float,
    config: Aircraft6DOFConfig,
) -> np.ndarray:
    """Shared constant input offsets from the quasi-steady lead-in segments.

    Real transmitter trims are rarely zero, so recorded stick values carry an
    unknown offset relative to the model's neutral surfaces. This is the
    classical data-compatibility bias estimate: solve J_u b = f(x_bar, u_bar) -
    x_dot_obs over the input-free lead-in, where the kinematic rows drop out
    because they do not depend on the inputs. The trim setting is one physical
    quantity per flight, so the normal equations are pooled across segments
    (weighted by lead-in length) instead of fitted per segment — per-segment
    estimates absorb local model error into the offsets.
    """
    n_channels = segments[0]["u"].shape[1]
    lhs = 1e-3 * np.eye(n_channels)
    rhs_acc = np.zeros(n_channels)
    pooled = 0
    for segment, onset in zip(segments, onsets):
        dwell = int(onset)
        if dwell < 5:
            continue
        tracked = segment.get("tracked")
        if tracked is not None and not bool(np.all(tracked[:dwell])):
            continue  # mocap dropout inside the lead-in: interpolated, not measured
        x = segment["x"]
        u = segment["u"]
        x_bar = x[:dwell].mean(axis=0)
        x_bar[6:10] = normalize_quaternion(x_bar[6:10])
        u_bar = u[:dwell].mean(axis=0)
        elapsed = (dwell - 1) * dt
        if elapsed <= 1e-9:
            continue
        x_dot_obs = (x[dwell - 1] - x[0]) / elapsed
        f0 = rhs(x_bar, u_bar, config)
        jac = np.zeros((len(f0), n_channels))
        eps = 1e-4
        for channel in range(n_channels):
            du = np.zeros(n_channels)
            du[channel] = eps
            jac[:, channel] = (rhs(x_bar, u_bar + du, config) - rhs(x_bar, u_bar - du, config)) / (2.0 * eps)
        weight = float(dwell)
        lhs += weight * (jac.T @ jac)
        rhs_acc += weight * (jac.T @ (f0 - x_dot_obs))
        pooled += 1
    if pooled == 0:
        return np.zeros(n_channels)
    return np.clip(np.linalg.solve(lhs, rhs_acc), -MAX_INPUT_BIAS, MAX_INPUT_BIAS)


def trim_to_input_onset(
    segments: list[dict[str, np.ndarray | str]],
    dt: float,
    *,
    estimate_bias: bool = True,
    label: str = "",
) -> Split6DOF:
    """Split each real flight record into lead-in and control-actuation segments.

    On an open-loop unstable airframe, input-free dwell at the start of a record
    measures initial-condition sensitivity rather than model quality. The
    lead-in length is detected per segment from the command record itself: the
    lead-in segment calibrates the trim (initial state and input offsets), and
    only the control-actuation segment is used for dynamics — fitting on the
    training split, open-loop scoring on the validation split.

    Actuation windows are rectangularized by chunking: a common window length is
    chosen from the shortest usable record (clamped to a benchmark range) and
    longer maneuvers contribute several consecutive windows, so no flight data
    is discarded to the shortest segment. Initial states for follow-on chunks
    come from a local fit over the trailing samples of the previous chunk.
    """
    onsets = [detect_input_onset(np.asarray(segment["u"])) for segment in segments]

    input_bias = {}
    flights = [str(segment["name"]).split("__")[0] for segment in segments]
    if estimate_bias:
        # The trim setting is one physical quantity per flight, so pool the
        # bias normal equations per flight (segment-name prefix before "__").
        for flight in sorted(set(flights)):
            members = [index for index, name in enumerate(flights) if name == flight]
            input_bias[flight] = estimate_input_bias(
                [segments[index] for index in members],
                [onsets[index] for index in members],
                dt,
                Aircraft6DOFConfig(),
            )

    min_actuation = int(round(MIN_ACTUATION_S / dt))
    max_actuation = int(round(MAX_ACTUATION_S / dt))
    usable = [len(np.asarray(segment["u"])) - onset for segment, onset in zip(segments, onsets)]
    keep = [index for index, count in enumerate(usable) if count >= min_actuation]
    dropped = [segments[index]["name"] for index in range(len(segments)) if index not in keep]
    if not keep:
        raise ValueError(f"{label}: no segment retains {MIN_ACTUATION_S} s of actuation after onset trimming")
    window = int(np.clip(min(usable[index] for index in keep), min_actuation, max_actuation))

    chunks_x, chunks_mocap, chunks_u, chunk_x0, chunk_names, chunk_bias = [], [], [], [], [], []
    for index in keep:
        segment = segments[index]
        x = np.asarray(segment["x"])
        mocap = np.asarray(segment["mocap"])
        u = np.asarray(segment["u"]).copy()
        bias = input_bias.get(flights[index])
        if bias is not None:
            u = u - bias
        onset = onsets[index]
        tracked = segments[index].get("tracked")
        n_chunks = (len(x) - onset) // window
        for chunk in range(n_chunks):
            start = onset + chunk * window
            if tracked is not None and not bool(np.all(tracked[start : start + window])):
                continue  # window touches a mocap dropout: never train or score on it
            chunks_x.append(x[start : start + window])
            chunks_mocap.append(mocap[start : start + window])
            chunks_u.append(u[start : start + window])
            fit_window = max(int(round(0.12 / dt)), 3)
            chunk_x0.append(local_state_estimate(x, dt, start, window=fit_window if chunk else max(fit_window, onset)))
            chunk_names.append(f"{segment['name']}_w{chunk}")
            chunk_bias.append(bias if bias is not None else np.zeros(u.shape[1]))

    t = np.arange(window) * dt
    x_arr = np.stack(chunks_x)
    mocap_arr = np.stack(chunks_mocap)
    u_arr = np.stack(chunks_u)
    x0_estimate = np.stack(chunk_x0)
    if label:
        onset_desc = ", ".join(f"{onset * dt:.2f}" for onset in onsets)
        bias_desc = "off"
        if estimate_bias and input_bias:
            bias_desc = "; ".join(
                f"{flight.split('_')[0]}: " + ",".join(f"{value:+.2f}" for value in bias)
                for flight, bias in sorted(input_bias.items())
            )
        print(
            f"[onset-trim] {label}: {len(chunks_x)} windows of {window} samples from {len(keep)} maneuvers "
            f"(dropped {dropped if dropped else 'none'}), lead-in (s): {onset_desc}, input bias {bias_desc}",
            flush=True,
        )
    return Split6DOF(
        t=t,
        x_true=x_arr,
        y_meas=x_arr,
        mocap_true=mocap_arr,
        mocap_meas=mocap_arr,
        u_cmd=u_arr,
        u_act=u_arr,
        x0=x0_estimate,
        mocap_frame=str(segments[0]["mocap_frame"]),
        x0_estimate=x0_estimate,
        input_bias=np.stack(chunk_bias),
        segment_names=np.asarray(chunk_names),
    )


def design_matrix(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.concatenate((x, u, np.ones((*x.shape[:-1], 1))), axis=-1)


def ridge_fit(phi: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    phi2 = phi.reshape(-1, phi.shape[-1])
    target2 = target.reshape(-1, target.shape[-1])
    lhs = phi2.T @ phi2 + ridge * np.eye(phi2.shape[1])
    rhs = phi2.T @ target2
    return np.linalg.solve(lhs, rhs)


def standardize_fit(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(phi, axis=0)
    scale = np.std(phi, axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return (phi - mean) / scale, mean, scale


# Dynamic rows of the 13-state: body velocities and body rates. The kinematic
# rows (position, quaternion) are exactly known and are never fitted; rollouts
# integrate them from the predicted dynamic states (Klein/Morelli equation
# error and SINDy practice fit only the force/moment equations).
DYNAMIC_ROWS = [3, 4, 5, 10, 11, 12]
# Constant + 9 invariant state features + inputs are protected from pruning.
PROTECTED_FEATURES = 1 + 9 + len(INPUT_NAMES)


def gravity_direction(x: np.ndarray) -> np.ndarray:
    """Body-frame gravity direction (third row of R(q)^T), vectorized.

    Attitude enters the rigid-body dynamics only through this vector: using
    it instead of raw quaternion components makes the features heading-
    invariant and lets even linear features represent the gravity term.
    """
    q = x[..., 6:10]
    norm = np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    q0, q1, q2, q3 = (q / norm)[..., 0], (q / norm)[..., 1], (q / norm)[..., 2], (q / norm)[..., 3]
    return np.stack([2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2)], axis=-1)


def invariant_state(x: np.ndarray) -> np.ndarray:
    """[u, v, w, gx, gy, gz, p, q, r]: heading/position-invariant features."""
    return np.concatenate((x[..., 3:6], gravity_direction(x), x[..., 10:13]), axis=-1)


def savgol_states(x: np.ndarray, dt: float, window_s: float = 0.05) -> np.ndarray:
    """Savitzky-Golay smoothed states for forming increment targets."""
    from scipy.signal import savgol_filter

    window = max(5, int(round(window_s / dt)) | 1)
    if x.shape[1] <= window:
        return x.copy()
    return savgol_filter(x, window_length=window, polyorder=2, axis=1)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ])


def linear_features(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    # Heading/position-invariant features: see invariant_state.
    return np.concatenate((invariant_state(x), u, np.ones((*x.shape[:-1], 1))), axis=-1)


def poly_features(x: np.ndarray, u: np.ndarray, *, degree: int = 2) -> np.ndarray:
    z = np.concatenate((invariant_state(x), u), axis=-1)
    parts = [np.ones((*z.shape[:-1], 1)), z]
    if degree >= 2:
        quad = []
        for i in range(z.shape[-1]):
            for j in range(i, z.shape[-1]):
                quad.append((z[..., i] * z[..., j])[..., None])
        parts.append(np.concatenate(quad, axis=-1))
    return np.concatenate(parts, axis=-1)


def fit_standardized_ridge(phi: np.ndarray, target: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi2 = phi.reshape(-1, phi.shape[-1])
    target2 = target.reshape(-1, target.shape[-1])
    phi_s, mean, scale = standardize_fit(phi2)
    lhs = phi_s.T @ phi_s + ridge * np.eye(phi_s.shape[1])
    weights = np.linalg.solve(lhs, phi_s.T @ target2)
    return weights, mean, scale


def apply_standardized(phi: np.ndarray, weights: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((phi - mean) / scale) @ weights


def sparsify_weights(weights: np.ndarray, fraction: float = 0.08, protected: int = 18) -> np.ndarray:
    sparse = weights.copy()
    for col in range(sparse.shape[1]):
        values = np.abs(sparse[protected:, col])
        if values.size == 0:
            continue
        threshold = np.quantile(values, 1.0 - fraction)
        sparse[protected:, col] *= values >= threshold
    return sparse


def stlsq_fit(
    phi: np.ndarray,
    target: np.ndarray,
    ridge: float,
    *,
    fraction: float = 0.06,
    protected: int = 18,
    iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sequentially thresholded least squares (the actual SINDy estimator).

    Alternates magnitude thresholding with a ridge refit restricted to each
    output's surviving features, so retained coefficients are re-estimated
    without the pruned columns; a single post-hoc prune leaves the dense
    fit's coefficient values in place.
    """
    phi2 = phi.reshape(-1, phi.shape[-1])
    target2 = target.reshape(-1, target.shape[-1])
    phi_s, mean, scale = standardize_fit(phi2)
    gram = phi_s.T @ phi_s
    rhs_all = phi_s.T @ target2
    n_features = phi_s.shape[1]
    weights = np.linalg.solve(gram + ridge * np.eye(n_features), rhs_all)
    for _ in range(iterations):
        pruned = sparsify_weights(weights, fraction=fraction, protected=protected)
        changed = False
        for col in range(weights.shape[1]):
            active = np.flatnonzero(pruned[:, col])
            if active.size == 0:
                weights[:, col] = 0.0
                continue
            if active.size == np.count_nonzero(weights[:, col]):
                continue
            sub = gram[np.ix_(active, active)] + ridge * np.eye(active.size)
            weights[:, col] = 0.0
            weights[active, col] = np.linalg.solve(sub, rhs_all[active, col])
            changed = True
        if not changed:
            break
    return weights, mean, scale


def savgol_derivative(x: np.ndarray, dt: float, window_s: float = 0.05) -> np.ndarray:
    """Savitzky-Golay smoothed derivative of (trials, samples, states) data.

    Raw forward differences amplify measurement noise by sqrt(2)/dt (~340x at
    240 Hz); local quadratic smoothing with a documented window is the
    standard equation-error practice.
    """
    from scipy.signal import savgol_filter

    window = max(5, int(round(window_s / dt)) | 1)
    if x.shape[1] <= window:
        return np.gradient(x, dt, axis=1)
    return savgol_filter(x, window_length=window, polyorder=2, deriv=1, delta=dt, axis=1)


def kinematic_step(state: np.ndarray, dyn_next: np.ndarray, dt: float) -> np.ndarray:
    """Advance position and attitude exactly; surrogates predict dynamics only.

    ``dyn_next`` is the six predicted dynamic states [u, v, w, p, q, r].
    Position integrates the rotated body velocity and the quaternion takes an
    exact axis-angle step from the current body rates, so a learned map can
    neither teleport nor break the attitude kinematics it never needed to fit.
    """
    quat = normalize_quaternion(state[6:10])
    pos = state[0:3] + rotation_body_to_inertial(quat) @ state[3:6] * dt
    omega = state[10:13]
    angle = float(np.linalg.norm(omega)) * dt
    if angle > 1e-12:
        axis = omega / np.linalg.norm(omega)
        dq = np.concatenate(([np.cos(0.5 * angle)], np.sin(0.5 * angle) * axis))
        quat = quat_multiply(quat, dq)
    return normalize_state(np.concatenate((pos, dyn_next[0:3], quat, dyn_next[3:6])))


def derivative_rollout(
    initial: np.ndarray,
    u: np.ndarray,
    t: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    degree: int,
) -> np.ndarray:
    pred = np.zeros((u.shape[0], len(t), len(STATE_NAMES)))
    pred[:, 0, :] = initial
    dt = float(np.median(np.diff(t)))
    for trial in range(u.shape[0]):
        pred[trial, 0] = normalize_state(pred[trial, 0])
        for index in range(len(t) - 1):
            phi = poly_features(pred[trial, index][None, :], u[trial, index][None, :], degree=degree)[0]
            delta = apply_standardized(phi, weights, mean, scale)
            pred[trial, index + 1] = kinematic_step(pred[trial, index], pred[trial, index, DYNAMIC_ROWS] + dt * delta, dt)
    return pred


def one_step_rollout(
    initial: np.ndarray,
    u: np.ndarray,
    t: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    *,
    degree: int,
) -> np.ndarray:
    pred = np.zeros((u.shape[0], len(t), len(STATE_NAMES)))
    pred[:, 0, :] = initial
    dt = float(np.median(np.diff(t)))
    for trial in range(u.shape[0]):
        pred[trial, 0] = normalize_state(pred[trial, 0])
        for index in range(len(t) - 1):
            phi = poly_features(pred[trial, index][None, :], u[trial, index][None, :], degree=degree)[0]
            delta = apply_standardized(phi, weights, mean, scale)
            pred[trial, index + 1] = kinematic_step(pred[trial, index], pred[trial, index, DYNAMIC_ROWS] + delta, dt)
    return pred


def residual_feature_rollout(
    initial: np.ndarray,
    u: np.ndarray,
    t: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    config: Aircraft6DOFConfig,
    *,
    degree: int,
) -> np.ndarray:
    pred = np.zeros((u.shape[0], len(t), len(STATE_NAMES)))
    pred[:, 0, :] = initial
    dt = float(np.median(np.diff(t)))
    cfg = Aircraft6DOFConfig(duration=float(t[-1] - t[0]), dt=dt, wing_speed=config.wing_speed)
    for trial in range(u.shape[0]):
        pred[trial, 0] = normalize_state(pred[trial, 0])
        for index in range(len(t) - 1):
            base = nominal_rk4_step(pred[trial, index], u[trial, index], dt, cfg)
            phi = poly_features(pred[trial, index][None, :], u[trial, index][None, :], degree=degree)[0]
            pred[trial, index + 1] = kinematic_step(pred[trial, index], base[DYNAMIC_ROWS] + apply_standardized(phi, weights, mean, scale), dt)
    return pred


def greybox_residual_rollout(
    initial: np.ndarray,
    u: np.ndarray,
    t: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    theta_full: np.ndarray,
    *,
    degree: int,
) -> np.ndarray:
    from .greybox_oem_fit import make_greybox_quat_step

    pred = np.zeros((u.shape[0], len(t), len(STATE_NAMES)))
    pred[:, 0, :] = initial
    dt = float(np.median(np.diff(t)))
    step = make_greybox_quat_step(theta_full, dt)
    for trial in range(u.shape[0]):
        pred[trial, 0] = normalize_state(pred[trial, 0])
        for index in range(len(t) - 1):
            base = step(pred[trial, index], u[trial, index])
            phi = poly_features(pred[trial, index][None, :], u[trial, index][None, :], degree=degree)[0]
            pred[trial, index + 1] = kinematic_step(pred[trial, index], base[DYNAMIC_ROWS] + apply_standardized(phi, weights, mean, scale), dt)
    return pred


def rbf_features(z: np.ndarray, centers: np.ndarray, length_scale: np.ndarray) -> np.ndarray:
    diff = (z[:, None, :] - centers[None, :, :]) / length_scale[None, None, :]
    return np.exp(-0.5 * np.sum(diff * diff, axis=-1))


def sample_indices(count: int, max_count: int, seed: int) -> np.ndarray:
    if count <= max_count:
        return np.arange(count)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=max_count, replace=False))


def airdata_features(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(-1, x.shape[-1])
    values = np.zeros((flat.shape[0], 3))
    for idx, state in enumerate(flat):
        values[idx] = airdata(state)
    return values.reshape(*x.shape[:-1], 3)


def make_local_centers(train_x: np.ndarray) -> np.ndarray:
    features = airdata_features(train_x[:, :-1, :]).reshape(-1, 3)
    speed_levels = np.quantile(features[:, 0], [0.18, 0.50, 0.82])
    alpha_levels = np.quantile(features[:, 1], [0.25, 0.75])
    beta_level = np.array([0.0])
    centers = np.array([[speed, alpha, beta] for speed in speed_levels for alpha in alpha_levels for beta in beta_level])
    return centers


def fit_local_linear_models(
    train_x: np.ndarray,
    train_u: np.ndarray,
    target: np.ndarray,
    centers: np.ndarray,
    ridge: float,
) -> list[np.ndarray]:
    xk_local = train_x[:, :-1, :]
    uk_local = train_u[:, :-1, :]
    flat_x = xk_local.reshape(-1, xk_local.shape[-1])
    flat_u = uk_local.reshape(-1, uk_local.shape[-1])
    flat_target = target.reshape(-1, target.shape[-1])
    flat_features = airdata_features(xk_local).reshape(-1, 3)
    feature_scale = np.std(flat_features, axis=0)
    feature_scale = np.where(feature_scale > 1e-6, feature_scale, 1.0)
    distances = np.sum(((flat_features[:, None, :] - centers[None, :, :]) / feature_scale[None, None, :]) ** 2, axis=2)
    assignment = np.argmin(distances, axis=1)
    global_weights = ridge_fit(linear_features(flat_x, flat_u), flat_target, ridge)
    weights: list[np.ndarray] = []
    for center_index in range(len(centers)):
        mask = assignment == center_index
        if int(np.count_nonzero(mask)) < linear_features(flat_x[:1], flat_u[:1]).shape[-1] + 4:
            weights.append(global_weights)
        else:
            weights.append(ridge_fit(linear_features(flat_x[mask], flat_u[mask]), flat_target[mask], ridge))
    return weights


def local_linear_rollout(
    initial: np.ndarray,
    u: np.ndarray,
    t: np.ndarray,
    centers: np.ndarray,
    weights: list[np.ndarray],
    *,
    residual: bool,
    config: Aircraft6DOFConfig,
) -> np.ndarray:
    pred = np.zeros((u.shape[0], len(t), len(STATE_NAMES)))
    pred[:, 0, :] = initial
    feature_scale = np.std(centers, axis=0)
    feature_scale = np.where(feature_scale > 1e-6, feature_scale, 1.0)
    dt = float(np.median(np.diff(t)))
    cfg = Aircraft6DOFConfig(duration=float(t[-1] - t[0]), dt=dt, wing_speed=config.wing_speed)
    for trial in range(u.shape[0]):
        pred[trial, 0] = normalize_state(pred[trial, 0])
        for index in range(len(t) - 1):
            feature = np.asarray(airdata(pred[trial, index]))
            center_index = int(np.argmin(np.sum(((centers - feature) / feature_scale) ** 2, axis=1)))
            phi = linear_features(pred[trial, index][None, :], u[trial, index][None, :])[0]
            update = phi @ weights[center_index]
            if residual:
                base = nominal_rk4_step(pred[trial, index], u[trial, index], dt, cfg)
                pred[trial, index + 1] = normalize_state(base + update)
            else:
                pred[trial, index + 1] = normalize_state(update)
    return pred


def rbf_residual_rollout(
    initial: np.ndarray,
    u: np.ndarray,
    t: np.ndarray,
    weights: np.ndarray,
    centers: np.ndarray,
    length_scale: np.ndarray,
    config: Aircraft6DOFConfig,
) -> np.ndarray:
    pred = np.zeros((u.shape[0], len(t), len(STATE_NAMES)))
    pred[:, 0, :] = initial
    dt = float(np.median(np.diff(t)))
    cfg = Aircraft6DOFConfig(duration=float(t[-1] - t[0]), dt=dt, wing_speed=config.wing_speed)
    for trial in range(u.shape[0]):
        pred[trial, 0] = normalize_state(pred[trial, 0])
        for index in range(len(t) - 1):
            base = nominal_rk4_step(pred[trial, index], u[trial, index], dt, cfg)
            z = np.concatenate((invariant_state(pred[trial, index][None, :])[0], u[trial, index]))[None, :]
            phi = np.concatenate((rbf_features(z, centers, length_scale), np.ones((1, 1))), axis=1)[0]
            pred[trial, index + 1] = kinematic_step(pred[trial, index], base[DYNAMIC_ROWS] + phi @ weights, dt)
    return pred


def nn_residual_rollout(
    initial: np.ndarray,
    u: np.ndarray,
    t: np.ndarray,
    weights: np.ndarray,
    centers: np.ndarray,
    length_scale: np.ndarray,
    config: Aircraft6DOFConfig,
) -> np.ndarray:
    pred = np.zeros((u.shape[0], len(t), len(STATE_NAMES)))
    pred[:, 0, :] = initial
    dt = float(np.median(np.diff(t)))
    cfg = Aircraft6DOFConfig(duration=float(t[-1] - t[0]), dt=dt, wing_speed=config.wing_speed)
    for trial in range(u.shape[0]):
        pred[trial, 0] = normalize_state(pred[trial, 0])
        for index in range(len(t) - 1):
            base = nominal_rk4_step(pred[trial, index], u[trial, index], dt, cfg)
            z_now = np.concatenate((invariant_state(pred[trial, index][None, :])[0], u[trial, index]))[None, :]
            phi_now = np.concatenate(
                (
                    rbf_features(z_now, centers, length_scale),
                    linear_features(pred[trial, index][None, :], u[trial, index][None, :]),
                ),
                axis=1,
            )[0]
            pred[trial, index + 1] = kinematic_step(pred[trial, index], base[DYNAMIC_ROWS] + phi_now @ weights, dt)
    return pred


def lagged_rollout(initial: np.ndarray, u: np.ndarray, t: np.ndarray, weights: np.ndarray, lag: int = 3) -> np.ndarray:
    pred = np.zeros((u.shape[0], len(t), len(STATE_NAMES)))
    pred[:, :lag, :] = initial[:, None, :]
    for trial in range(u.shape[0]):
        for index in range(lag - 1, len(t) - 1):
            history = invariant_state(pred[trial, index - lag + 1 : index + 1]).reshape(-1)
            phi = np.concatenate((history, u[trial, index], [1.0]))
            dt = float(np.median(np.diff(t)))
            pred[trial, index + 1] = kinematic_step(pred[trial, index], pred[trial, index, DYNAMIC_ROWS] + phi @ weights, dt)
    return pred


def parallel_rollout(
    function_name: str,
    workers: int,
    initial: np.ndarray,
    u: np.ndarray,
    t: np.ndarray,
    *args: object,
    **kwargs: object,
) -> np.ndarray:
    worker_count = min(max(1, workers), int(initial.shape[0]))
    if worker_count <= 1:
        return globals()[function_name](initial, u, t, *args, **kwargs)
    initial_chunks = np.array_split(initial, worker_count, axis=0)
    u_chunks = np.array_split(u, worker_count, axis=0)
    chunks = [(function_name, x0, u_chunk, t, args, kwargs) for x0, u_chunk in zip(initial_chunks, u_chunks) if x0.size]
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        parts = list(executor.map(_rollout_chunk, chunks))
    return np.concatenate(parts, axis=0)


def _rollout_chunk(payload: tuple[str, np.ndarray, np.ndarray, np.ndarray, tuple[object, ...], dict[str, object]]) -> np.ndarray:
    function_name, initial, u, t, args, kwargs = payload
    return globals()[function_name](initial, u, t, *args, **kwargs)


def nominal_next_grid(train_x: np.ndarray, train_u: np.ndarray, dt: float, config: Aircraft6DOFConfig) -> np.ndarray:
    nominal_next = np.zeros_like(train_x[:, 1:, :])
    for trial in range(train_x.shape[0]):
        for index in range(train_x.shape[1] - 1):
            nominal_next[trial, index] = nominal_rk4_step(train_x[trial, index], train_u[trial, index], dt, config)
    return nominal_next


def parallel_nominal_next(train_x: np.ndarray, train_u: np.ndarray, dt: float, config: Aircraft6DOFConfig, workers: int) -> np.ndarray:
    worker_count = min(max(1, workers), int(train_x.shape[0]))
    if worker_count <= 1:
        return nominal_next_grid(train_x, train_u, dt, config)
    x_chunks = np.array_split(train_x, worker_count, axis=0)
    u_chunks = np.array_split(train_u, worker_count, axis=0)
    chunks = [(x_chunk, u_chunk, dt, config) for x_chunk, u_chunk in zip(x_chunks, u_chunks) if x_chunk.size]
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        parts = list(executor.map(_nominal_next_chunk, chunks))
    return np.concatenate(parts, axis=0)


def _nominal_next_chunk(payload: tuple[np.ndarray, np.ndarray, float, Aircraft6DOFConfig]) -> np.ndarray:
    train_x, train_u, dt, config = payload
    return nominal_next_grid(train_x, train_u, dt, config)


def nominal_rollout(initial: np.ndarray, u: np.ndarray, t: np.ndarray, config: Aircraft6DOFConfig) -> np.ndarray:
    pred = np.zeros((u.shape[0], len(t), len(STATE_NAMES)))
    pred[:, 0, :] = initial
    dt = float(np.median(np.diff(t)))
    cfg = Aircraft6DOFConfig(duration=float(t[-1] - t[0]), dt=dt, wing_speed=config.wing_speed)
    for trial in range(u.shape[0]):
        pred[trial, 0] = normalize_state(pred[trial, 0])
        for index in range(len(t) - 1):
            pred[trial, index + 1] = nominal_rk4_step(pred[trial, index], u[trial, index], dt, cfg)
    return pred


def linear_rollout(initial: np.ndarray, u: np.ndarray, t: np.ndarray, weights: np.ndarray) -> np.ndarray:
    pred = np.zeros((u.shape[0], len(t), len(STATE_NAMES)))
    pred[:, 0, :] = initial
    for trial in range(u.shape[0]):
        pred[trial, 0] = normalize_state(pred[trial, 0])
        for index in range(len(t) - 1):
            phi = np.concatenate((pred[trial, index], u[trial, index], [1.0]))
            pred[trial, index + 1] = normalize_state(phi @ weights)
    return pred


def residual_rollout(initial: np.ndarray, u: np.ndarray, t: np.ndarray, weights: np.ndarray, config: Aircraft6DOFConfig) -> np.ndarray:
    pred = np.zeros((u.shape[0], len(t), len(STATE_NAMES)))
    pred[:, 0, :] = initial
    dt = float(np.median(np.diff(t)))
    cfg = Aircraft6DOFConfig(duration=float(t[-1] - t[0]), dt=dt, wing_speed=config.wing_speed)
    for trial in range(u.shape[0]):
        pred[trial, 0] = normalize_state(pred[trial, 0])
        for index in range(len(t) - 1):
            base = nominal_rk4_step(pred[trial, index], u[trial, index], dt, cfg)
            phi = np.concatenate((pred[trial, index], u[trial, index], [1.0]))
            pred[trial, index + 1] = normalize_state(base + phi @ weights)
    return pred


def mocap_output_rollout(initial: np.ndarray, u: np.ndarray, t: np.ndarray, weights: np.ndarray) -> np.ndarray:
    pred = np.zeros((u.shape[0], len(t), 7))
    pred[:, 0, :] = initial
    for trial in range(u.shape[0]):
        pred[trial, 0, 3:7] = normalize_quaternion(pred[trial, 0, 3:7])
        for index in range(len(t) - 1):
            phi = np.concatenate((pred[trial, index], u[trial, index], [1.0]))
            pred[trial, index + 1] = phi @ weights
            pred[trial, index + 1, 3:7] = normalize_quaternion(pred[trial, index + 1, 3:7])
    return pred


def score_state_method(
    method: str,
    description: str,
    backend: str,
    state_source: str,
    train_elapsed: float,
    train_cpu: float,
    rollout_elapsed: float,
    train_samples: int,
    decision_variables: int,
    pred: np.ndarray,
    validation: Split6DOF,
    notes: str,
    implementation_status: str = "implemented",
) -> Result6DOF:
    pred_aligned = align_quaternion_signs(pred, validation.x_true)
    score = nrmse_score(pred_aligned, validation.x_true)
    metrics = rmse_group(pred_aligned, validation.x_true)
    diverged = bool(not np.all(np.isfinite(pred_aligned)) or np.nanmax(np.abs(pred_aligned[..., 0:3])) > 1.0e3)
    return Result6DOF(
        method=method,
        description=description,
        backend=backend,
        state_source=state_source,
        validation_score=score,
        train_elapsed_s=train_elapsed,
        train_cpu_s=train_cpu,
        rollout_elapsed_s=rollout_elapsed,
        total_elapsed_s=train_elapsed + rollout_elapsed,
        train_samples=train_samples,
        decision_variables=decision_variables,
        notes=notes,
        implementation_status=implementation_status,
        diverged=diverged,
        x_pred=pred_aligned,
        **metrics,
    )


def run_methods(
    train_splits: dict[str, Split6DOF],
    validation: Split6DOF,
    state_source: str,
    ridge: float,
    workers: int,
    training_scenario_override: str | None = None,
) -> list[Result6DOF]:
    config = Aircraft6DOFConfig(duration=float(validation.t[-1] - validation.t[0]), dt=validation.dt)
    if state_source == "direct":
        validation_x0 = validation.x0_estimate if validation.x0_estimate is not None else validation.y_meas[:, 0, :]
    elif state_source == "mocap":
        x0_window = min(max(int(round(0.2 / max(validation.dt, 1e-6))), 5), len(validation.t))
        validation_x0 = derive_state_from_mocap(validation.mocap_meas[:, :x0_window, :], validation.t[:x0_window])[:, 0, :]
    else:
        raise ValueError(f"unsupported state source: {state_source}")

    training_cache: dict[str, tuple[Split6DOF, np.ndarray]] = {}

    def training_context(method: str) -> tuple[str, Split6DOF, np.ndarray, int]:
        scenario = training_scenario_override or METHOD_TRAINING_SCENARIOS.get(method, "aircraft_6dof_aggressive")
        if scenario == "none":
            fallback = train_splits.get("aircraft_6dof_open_loop") or next(iter(train_splits.values()))
            return scenario, fallback, fallback.y_meas, 0
        if scenario not in training_cache:
            split = train_splits[scenario]
            if state_source == "direct":
                split_x = split.y_meas
            else:
                split_x = derive_state_from_mocap(split.mocap_meas, split.t)
            training_cache[scenario] = (split, split_x)
        split, split_x = training_cache[scenario]
        train_samples = int(np.prod(split_x[:, :-1, :].shape[:2]))
        return scenario, split, split_x, train_samples

    def add_result(result: Result6DOF) -> None:
        if training_scenario_override is not None and result.method != "6DOF-NominalGreyBox":
            result.training_scenario = training_scenario_override
        else:
            result.training_scenario = METHOD_TRAINING_SCENARIOS.get(result.method, "aircraft_6dof_aggressive")
        results.append(result)

    results: list[Result6DOF] = []

    if training_scenario_override is None:
        # The attached-flow nominal row is the truth-minus-residual baseline of
        # the synthetic benchmark; on real flights it is just a wrong model
        # with no baseline meaning, so the row is omitted there.
        start = time.perf_counter()
        pred = parallel_rollout("nominal_rollout", workers, validation_x0, validation.u_cmd, validation.t, config)
        rollout_elapsed = time.perf_counter() - start
        add_result(
            score_state_method(
                "6DOF-NominalGreyBox",
                "Attached-flow nominal 6DOF rollout using pilot commands and no fitted stall correction.",
                "numpy-rk4",
                state_source,
                0.0,
                0.0,
                rollout_elapsed,
                0,
                0,
                pred,
                validation,
                "No-fit baseline; mismatch includes actuator lag, hidden stall/nonlinear aerodynamics, and mocap-derived initialization error.",
            )
        )

    start = time.perf_counter()
    cpu_start = time.process_time()
    _scenario, train, train_x, train_samples = training_context("6DOF-LinearSS")
    weights = ridge_fit(design_matrix(train_x[:, :-1, :], train.u_cmd[:, :-1, :]), train_x[:, 1:, :], ridge)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("linear_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights)
    rollout_elapsed = time.perf_counter() - rollout_start
    add_result(
        score_state_method(
            "6DOF-LinearSS",
            "Global affine discrete state-space fit x[k+1]=A x[k]+B u_cmd[k]+c.",
            "numpy-ridge",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            train_samples,
            int(weights.size),
            pred,
            validation,
            "Open-loop rollout; validation measurements are not assimilated after initialization.",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    _scenario, train, train_x, train_samples = training_context("6DOF-RidgeResidual")
    print(f"  {state_source}: nominal residual targets using {workers} workers", flush=True)
    nominal_next = parallel_nominal_next(train_x, train.u_cmd, train.dt, config, workers)
    residual = train_x[:, 1:, :] - nominal_next
    weights = ridge_fit(design_matrix(train_x[:, :-1, :], train.u_cmd[:, :-1, :]), residual, ridge)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("residual_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights, config)
    rollout_elapsed = time.perf_counter() - rollout_start
    add_result(
        score_state_method(
            "6DOF-RidgeResidual",
            "Nominal RK4 model plus ridge-fitted one-step residual correction.",
            "numpy-rk4-ridge",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            train_samples,
            int(weights.size),
            pred,
            validation,
            "Residual corrects actuator lag and hidden nonlinear stall/aerodynamic effects around the attached-flow model.",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    _scenario, train, train_x, train_samples = training_context("6DOF-Model-Stitching")
    centers = make_local_centers(train_x)
    local_weights = fit_local_linear_models(train_x, train.u_cmd, train_x[:, 1:, :], centers, ridge)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("local_linear_rollout", workers, validation_x0, validation.u_cmd, validation.t, centers, local_weights, residual=False, config=config)
    rollout_elapsed = time.perf_counter() - rollout_start
    add_result(
        score_state_method(
            "6DOF-Model-Stitching",
            "Airdata-scheduled family of local affine one-step state models.",
            "numpy-local-ridge",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            train_samples,
            int(sum(weight.size for weight in local_weights) + centers.size),
            pred,
            validation,
            "Local models are selected by speed, angle of attack, and sideslip during open-loop validation.",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    _scenario, train, train_x, train_samples = training_context("6DOF-Frequency-Welch")
    weights_freq = ridge_fit(design_matrix(train_x[:, :-1, :], train.u_cmd[:, :-1, :]), train_x[:, 1:, :], 25.0 * ridge)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("linear_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_freq)
    rollout_elapsed = time.perf_counter() - rollout_start
    add_result(
        score_state_method(
            "6DOF-Frequency-Welch",
            "Frequency-domain-inspired global linear baseline approximated by a regularized one-step realization.",
            "numpy-regularized-realization",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            train_samples,
            int(weights_freq.size),
            pred,
            validation,
            "Placeholder 6DOF frequency row: uses an identified realization rather than CIFER/SIDPAC tooling.",
            implementation_status="placeholder",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    _scenario, train, train_x, train_samples = training_context("6DOF-Frequency-Stitching")
    centers = make_local_centers(train_x)
    nominal_next_local = parallel_nominal_next(train_x, train.u_cmd, train.dt, config, workers)
    residual_local = train_x[:, 1:, :] - nominal_next_local
    local_residual_weights = fit_local_linear_models(train_x, train.u_cmd, residual_local, centers, 10.0 * ridge)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("local_linear_rollout", workers, validation_x0, validation.u_cmd, validation.t, centers, local_residual_weights, residual=True, config=config)
    rollout_elapsed = time.perf_counter() - rollout_start
    add_result(
        score_state_method(
            "6DOF-Frequency-Stitching",
            "Airdata-scheduled local realization residuals around the nominal 6DOF equations.",
            "numpy-local-realization",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            train_samples,
            int(sum(weight.size for weight in local_residual_weights) + centers.size),
            pred,
            validation,
            "6DOF counterpart to local frequency/model stitching; trained from the selected 6DOF dataset. "
            "Not a frequency-domain method: local affine residual fits scheduled on airdata.",
            implementation_status="placeholder",
        )
    )

    if training_scenario_override and "sportcub" in training_scenario_override:
        # Real filter-error EKF over the grey-box parameters (port of the
        # 3DOF implementation): training data only, frozen-theta validation.
        from .greybox_oem_fit import fit_greybox_ekf, greybox_rollout_quat as _gb_rollout

        _scenario, train, train_x, train_samples = training_context("6DOF-EKF-ParamID")
        start = time.perf_counter()
        cpu_start = time.process_time()
        ekf = fit_greybox_ekf(train_x, train.u_cmd, train.dt)
        train_elapsed = time.perf_counter() - start
        train_cpu = time.process_time() - cpu_start
        rollout_start = time.perf_counter()
        pred = _gb_rollout(ekf["spec"], ekf["theta_full"], validation_x0, validation.u_cmd, validation.dt)
        rollout_elapsed = time.perf_counter() - rollout_start
        add_result(
            score_state_method(
                "6DOF-EKF-ParamID",
                "Filter-error estimation: augmented-state EKF over the grey-box parameters with CasADi Jacobians.",
                "casadi-ad-ekf",
                state_source,
                train_elapsed,
                train_cpu,
                rollout_elapsed,
                int(ekf["updates"]),
                int(ekf["theta"].size),
                pred,
                validation,
                "Joseph-form augmented-state EKF runs over the manual training chunks only; "
                "validation is a frozen-theta open-loop rollout receiving pilot commands.",
            )
        )

    start = time.perf_counter()
    cpu_start = time.process_time()
    _scenario, train, train_x, train_samples = training_context("6DOF-EKF-ParamID")
    nominal_next = parallel_nominal_next(train_x, train.u_cmd, train.dt, config, workers)
    residual = train_x[:, 1:, :] - nominal_next
    weights_ekf = ridge_fit(design_matrix(train_x[:, :-1, :], train.u_cmd[:, :-1, :]), residual, 5.0 * ridge)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("residual_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_ekf, config)
    rollout_elapsed = time.perf_counter() - rollout_start
    if not (training_scenario_override and "sportcub" in training_scenario_override):
        # Placeholder rows retained for synthetic scenarios only; on the real
        # Sport Cub data the real filter-error EKF above replaces EKF-ParamID,
        # Fisher-UQ is reported from the grey-box Cramer-Rao analysis, and
        # GreyBoxOEM is the output-error row (no duplicate OEM-SS name).
        for placeholder_name, placeholder_desc, placeholder_backend, placeholder_note in (
            (
                "6DOF-EKF-ParamID",
                "Recursive-estimation analogue represented by a fitted affine residual parameter vector.",
                "numpy-ridge-paramid",
                "The validation phase is open loop and receives only pilot commands after initialization. "
                "No recursive filter is run: this row is a ridge-fitted affine residual model.",
            ),
            (
                "6DOF-Fisher-UQ",
                "Fisher-information wrapper around the fitted residual parameter model.",
                "numpy-ridge-uq",
                "Duplicate of the 6DOF-EKF-ParamID fit (same weights and prediction); no Fisher-information analysis is computed for 6DOF.",
            ),
            (
                "6DOF-OEM-SS",
                "Output-error state-space residual model using the same open-loop rollout structure as the fitted parameter model.",
                "numpy-rk4-ridge",
                "Duplicate of the 6DOF-EKF-ParamID fit (same weights and prediction); no output-error optimization is run for 6DOF.",
            ),
        ):
            add_result(
                score_state_method(
                    placeholder_name,
                    placeholder_desc,
                    placeholder_backend,
                    state_source,
                    train_elapsed,
                    train_cpu,
                    rollout_elapsed,
                    train_samples,
                    int(weights_ekf.size),
                    pred,
                    validation,
                    placeholder_note,
                    implementation_status="placeholder",
                )
            )


    _scenario, train, train_x, train_samples = training_context("6DOF-EquationError-LS")
    nominal_next = parallel_nominal_next(train_x, train.u_cmd, train.dt, config, workers)
    xk = train_x[:, :-1, :]
    uk = train.u_cmd[:, :-1, :]
    xkp1 = train_x[:, 1:, :]
    dxdt = savgol_derivative(train_x, train.dt)[:, :-1, :]
    flat_x = xk.reshape(-1, len(STATE_NAMES))
    flat_u = uk.reshape(-1, len(INPUT_NAMES))
    flat_xkp1 = xkp1.reshape(-1, len(STATE_NAMES))
    flat_dxdt = dxdt.reshape(-1, len(STATE_NAMES))
    fit_idx_poly = sample_indices(flat_x.shape[0], 90_000, 20_000 + (0 if state_source == "direct" else 1))
    fit_x = flat_x[fit_idx_poly]
    fit_u = flat_u[fit_idx_poly]
    fit_xkp1 = flat_xkp1[fit_idx_poly]
    fit_dxdt = flat_dxdt[fit_idx_poly]

    start = time.perf_counter()
    cpu_start = time.process_time()
    phi = linear_features(fit_x, fit_u)
    weights_deriv, mean_deriv, scale_deriv = fit_standardized_ridge(phi, fit_dxdt[:, DYNAMIC_ROWS], ridge)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("derivative_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_deriv, mean_deriv, scale_deriv, degree=1)
    rollout_elapsed = time.perf_counter() - rollout_start
    add_result(
        score_state_method(
            "6DOF-EquationError-LS",
            "Affine derivative regression rolled out open-loop with explicit integration.",
            "numpy-ridge-derivative",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            len(fit_idx_poly),
            int(weights_deriv.size + mean_deriv.size + scale_deriv.size),
            pred,
            validation,
            "Equation-error least squares on Savitzky-Golay smoothed derivatives (50 ms quadratic window).",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    smoothed_x = smooth_array(train_x, window=15)
    var_xk = smoothed_x[:, :-1, :].reshape(-1, len(STATE_NAMES))[fit_idx_poly]
    var_uk = train.u_cmd[:, :-1, :].reshape(-1, len(INPUT_NAMES))[fit_idx_poly]
    var_dxdt = ((smoothed_x[:, 1:, :] - smoothed_x[:, :-1, :]) / train.dt).reshape(-1, len(STATE_NAMES))[fit_idx_poly]
    weights_var, mean_var, scale_var = fit_standardized_ridge(linear_features(var_xk, var_uk), var_dxdt[:, DYNAMIC_ROWS], 10.0 * ridge)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("derivative_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_var, mean_var, scale_var, degree=1)
    rollout_elapsed = time.perf_counter() - rollout_start
    add_result(
        score_state_method(
            "6DOF-Variational-Mocap",
            "Smoothed weak-form derivative fit used as a lightweight variational baseline.",
            "numpy-smoothed-weak",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            len(fit_idx_poly),
            int(weights_var.size + mean_var.size + scale_var.size),
            pred,
            validation,
            "Approximates the variational idea by smoothing trajectories before derivative regression.",
            implementation_status="placeholder",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    phi_poly = poly_features(fit_x, fit_u, degree=2)
    # Increment targets from smoothed states: raw one-step differences of the
    # 240 Hz mocap states are noise-dominated (measured SNR 0.2-0.8).
    x_smooth_inc = savgol_states(train_x, train.dt)
    fit_inc = (x_smooth_inc[:, 1:, :] - x_smooth_inc[:, :-1, :]).reshape(-1, len(STATE_NAMES))[fit_idx_poly][:, DYNAMIC_ROWS]
    weights_sindy, mean_sindy, scale_sindy = stlsq_fit(
        phi_poly, fit_dxdt[:, DYNAMIC_ROWS], 10.0 * ridge, fraction=0.06, protected=PROTECTED_FEATURES
    )
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("derivative_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_sindy, mean_sindy, scale_sindy, degree=2)
    rollout_elapsed = time.perf_counter() - rollout_start
    add_result(
        score_state_method(
            "6DOF-SINDy",
            "Sparse quadratic-library derivative model fitted by iterated threshold-and-refit (STLSQ).",
            "numpy-stlsq",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            len(fit_idx_poly),
            int(np.count_nonzero(weights_sindy)),
            pred,
            validation,
            "Uses a generic polynomial library rather than aerodynamic-coefficient structure.",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    weights_symbolic, mean_symbolic, scale_symbolic = stlsq_fit(
        phi_poly, fit_inc, ridge, fraction=0.12, protected=PROTECTED_FEATURES
    )
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("one_step_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_symbolic, mean_symbolic, scale_symbolic, degree=2)
    rollout_elapsed = time.perf_counter() - rollout_start
    add_result(
        score_state_method(
            "6DOF-Symbolic-Stepwise",
            "Sparse stepwise quadratic one-step predictor.",
            "numpy-sparse-ridge",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            len(fit_idx_poly),
            int(np.count_nonzero(weights_symbolic)),
            pred,
            validation,
            "Sparse-ridge analogue: magnitude pruning, not statistical forward selection (no PSE/PRESS).",
            implementation_status="analogue",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    weights_edmd, mean_edmd, scale_edmd = fit_standardized_ridge(phi_poly, fit_inc, 100.0 * ridge)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("one_step_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_edmd, mean_edmd, scale_edmd, degree=2)
    rollout_elapsed = time.perf_counter() - rollout_start
    add_result(
        score_state_method(
            "6DOF-Koopman-EDMD",
            "Quadratic lifted one-step predictor rolled out in the original state coordinates.",
            "numpy-edmd",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            len(fit_idx_poly),
            int(weights_edmd.size + mean_edmd.size + scale_edmd.size),
            pred,
            validation,
            "EDMD analogue: quadratic one-step ridge predicting raw states; lifted observables are not propagated.",
            implementation_status="analogue",
        )
    )

    greybox = None
    if training_scenario_override and "sportcub" in training_scenario_override:
        from .greybox_oem_fit import fit_greybox, greybox_one_step

        print(f"  {state_source}: grey-box OEM fit (shared by UDE base and GreyBoxOEM row)", flush=True)
        greybox_fit_start = time.perf_counter()
        greybox_fit_cpu = time.process_time()
        greybox = fit_greybox(train_x, train.u_cmd, train.dt)
        greybox_fit_elapsed = time.perf_counter() - greybox_fit_start
        greybox_fit_cpu = time.process_time() - greybox_fit_cpu

    start = time.perf_counter()
    cpu_start = time.process_time()
    if greybox is not None:
        base_next = greybox_one_step(greybox["spec"], greybox["theta_full"], train_x[:, :-1, :], train.u_cmd[:, :-1, :], train.dt)
        ude_note = "Residual around the fitted grey-box airframe (UDE premise: best known physics + learned residual)."
        # A good base leaves a one-step residual far below measurement noise
        # at 240 Hz; without heavy shrinkage the residual fit is noise that
        # poisons the base on rollout. The standardized gram scales with the
        # sample count, so the penalty must too (lambda = alpha * n).
        ude_ridge = 1.0 * len(fit_idx_poly)
    else:
        base_next = nominal_next
        ude_note = "Residual around the attached-flow nominal model."
        ude_ridge = 10.0 * ridge
    residual = x_smooth_inc[:, 1:, :] - base_next
    fit_residual = residual.reshape(-1, len(STATE_NAMES))[fit_idx_poly]
    weights_ude, mean_ude, scale_ude = fit_standardized_ridge(phi_poly, fit_residual[:, DYNAMIC_ROWS], ude_ridge)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    if greybox is not None:
        pred = parallel_rollout("greybox_residual_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_ude, mean_ude, scale_ude, np.asarray(greybox["theta_full"]), degree=2)
    else:
        pred = parallel_rollout("residual_feature_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_ude, mean_ude, scale_ude, config, degree=2)
    rollout_elapsed = time.perf_counter() - rollout_start
    add_result(
        score_state_method(
            "6DOF-UDE-Residual",
            "Attached-flow nominal dynamics plus quadratic learned residual map.",
            "numpy-residual-ridge",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            len(fit_idx_poly),
            int(weights_ude.size + mean_ude.size + scale_ude.size),
            pred,
            validation,
            "Deterministic UDE analogue: ridge residual map, no neural network or ODE-in-the-loop training. " + ude_note,
            implementation_status="analogue",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    weights_pinn = sparsify_weights(weights_ude, fraction=0.08, protected=PROTECTED_FEATURES)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    if greybox is not None:
        pred = parallel_rollout("greybox_residual_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_pinn, mean_ude, scale_ude, np.asarray(greybox["theta_full"]), degree=2)
    else:
        pred = parallel_rollout("residual_feature_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_pinn, mean_ude, scale_ude, config, degree=2)
    rollout_elapsed = time.perf_counter() - rollout_start
    add_result(
        score_state_method(
            "6DOF-PINN-Closure",
            "Physics-structured residual closure constrained to the attached-flow 6DOF equations.",
            "numpy-sparse-closure",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            len(fit_idx_poly),
            int(np.count_nonzero(weights_pinn)),
            pred,
            validation,
            "Tractable PINN-style row: sparsified copy of the 6DOF-UDE-Residual ridge weights; no physics-informed training is run.",
            implementation_status="placeholder",
        )
    )

    _scenario, train, train_x, train_samples = training_context("6DOF-Subspace-Hankel")
    start = time.perf_counter()
    cpu_start = time.process_time()
    lag = 3
    x_smooth = savgol_states(train_x, train.dt)
    history = []
    targets = []
    for trial in range(train_x.shape[0]):
        for index in range(lag - 1, train_x.shape[1] - 1):
            history.append(np.concatenate((invariant_state(train_x[trial, index - lag + 1 : index + 1]).reshape(-1), train.u_cmd[trial, index], [1.0])))
            targets.append(x_smooth[trial, index + 1, DYNAMIC_ROWS] - x_smooth[trial, index, DYNAMIC_ROWS])
    weights_hankel = ridge_fit(np.asarray(history)[:, None, :], np.asarray(targets)[:, None, :], ridge)
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("lagged_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_hankel, lag=lag)
    rollout_elapsed = time.perf_counter() - rollout_start
    add_result(
        score_state_method(
            "6DOF-Subspace-Hankel",
            "Lagged ARX/Hankel linear predictor using a three-sample state history.",
            "numpy-hankel-ridge",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            max(0, train_samples - train_x.shape[0] * (lag - 1)),
            int(weights_hankel.size),
            pred,
            validation,
            "Lagged ARX analogue: no Hankel SVD or state realization (not N4SID/MOESP).",
            implementation_status="analogue",
        )
    )

    _scenario, train, train_x, train_samples = training_context("6DOF-GP-RBF")
    nominal_next = parallel_nominal_next(train_x, train.u_cmd, train.dt, config, workers)
    xk = train_x[:, :-1, :]
    uk = train.u_cmd[:, :-1, :]
    xkp1 = train_x[:, 1:, :]
    residual = xkp1 - nominal_next
    start = time.perf_counter()
    cpu_start = time.process_time()
    z = np.concatenate((invariant_state(xk.reshape(-1, len(STATE_NAMES))), uk.reshape(-1, len(INPUT_NAMES))), axis=1)
    target_res = residual.reshape(-1, len(STATE_NAMES))
    rng = np.random.default_rng(12_345 + (0 if state_source == "direct" else 1))
    fit_count = min(60_000, z.shape[0])
    fit_idx = rng.choice(z.shape[0], size=fit_count, replace=False)
    center_count = min(96, fit_count)
    center_idx = rng.choice(fit_idx, size=center_count, replace=False)
    centers = z[center_idx]
    length_scale = np.std(z[fit_idx], axis=0)
    length_scale = np.where(length_scale > 1e-6, length_scale, 1.0)
    phi_rbf = np.concatenate((rbf_features(z[fit_idx], centers, length_scale), np.ones((fit_count, 1))), axis=1)
    weights_rbf = np.linalg.solve(phi_rbf.T @ phi_rbf + 1e-4 * np.eye(phi_rbf.shape[1]), phi_rbf.T @ target_res[fit_idx][:, DYNAMIC_ROWS])
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start
    rollout_start = time.perf_counter()
    pred = parallel_rollout("rbf_residual_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_rbf, centers, length_scale, config)
    rollout_elapsed = time.perf_counter() - rollout_start
    add_result(
        score_state_method(
            "6DOF-GP-RBF",
            "Sparse RBF/Gaussian-process-style residual surrogate around attached-flow dynamics.",
            "numpy-rbf-ridge",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            fit_count,
            int(weights_rbf.size + centers.size + length_scale.size),
            pred,
            validation,
            "RBF residual closure trained on a deterministic subset to keep the 6DOF suite tractable.",
        )
    )

    start = time.perf_counter()
    cpu_start = time.process_time()
    nn_count = min(128, fit_count)
    nn_centers = centers[:nn_count]
    nn_phi = np.concatenate((rbf_features(z[fit_idx], nn_centers, length_scale), linear_features(xk.reshape(-1, len(STATE_NAMES))[fit_idx], uk.reshape(-1, len(INPUT_NAMES))[fit_idx])), axis=1)
    weights_nn = np.linalg.solve(nn_phi.T @ nn_phi + 1e-4 * np.eye(nn_phi.shape[1]), nn_phi.T @ target_res[fit_idx][:, DYNAMIC_ROWS])
    train_elapsed = time.perf_counter() - start
    train_cpu = time.process_time() - cpu_start

    rollout_start = time.perf_counter()
    pred = parallel_rollout("nn_residual_rollout", workers, validation_x0, validation.u_cmd, validation.t, weights_nn, nn_centers, length_scale, config)
    rollout_elapsed = time.perf_counter() - rollout_start
    add_result(
        score_state_method(
            "6DOF-NN-Surrogate",
            "Random-feature neural-surrogate analogue for residual dynamics.",
            "numpy-random-feature",
            state_source,
            train_elapsed,
            train_cpu,
            rollout_elapsed,
            fit_count,
            int(weights_nn.size + nn_centers.size + length_scale.size),
            pred,
            validation,
            "Closed-form random-feature surrogate used as a lightweight 6DOF neural baseline; no neural network is trained.",
            implementation_status="placeholder",
        )
    )

    if training_scenario_override and "sportcub" in training_scenario_override:
        # Physical grey-box OEM: the spec encodes the Sport Cub airframe, so
        # the method only applies to its real-flight scenarios.
        from .greybox_oem_fit import greybox_rollout_quat

        _scenario, train, train_x, train_samples = training_context("6DOF-GreyBoxOEM")
        train_elapsed = greybox_fit_elapsed
        train_cpu = greybox_fit_cpu
        rollout_start = time.perf_counter()
        pred = greybox_rollout_quat(greybox["spec"], greybox["theta_full"], validation_x0, validation.u_cmd, validation.dt)
        rollout_elapsed = time.perf_counter() - rollout_start
        add_result(
            score_state_method(
                "6DOF-GreyBoxOEM",
                "Physical lumped-parameter Sport Cub grey-box fitted by output-error multiple shooting over the training chunks.",
                "casadi-rk4-trf",
                state_source,
                train_elapsed,
                train_cpu,
                rollout_elapsed,
                train_samples,
                int(greybox["theta"].size),
                pred,
                validation,
                "22 aerodynamic coefficients fitted within physical bounds; the same parameters drive the browser free-runs.",
            )
        )
        try:
            import torch  # noqa: F401

            from .greybox_oem_fit import OEM_CONTROL_ORDER as _OEM
            from .greybox_oem_fit import quat_states_to_euler as _q2e
            from .ude_nn import rollout_ude_quat, train_greybox_ude

            start = time.perf_counter()
            cpu_start = time.process_time()
            ude_model = train_greybox_ude(
                _q2e(train_x), train.u_cmd[:, :, _OEM], np.asarray(greybox["theta_full"]), train.dt
            )
            ude_train_elapsed = time.perf_counter() - start
            ude_train_cpu = time.process_time() - cpu_start
            rollout_start = time.perf_counter()
            pred_nn = rollout_ude_quat(ude_model, validation_x0, validation.u_cmd[:, :, _OEM], validation.dt)
            ude_rollout_elapsed = time.perf_counter() - rollout_start
            add_result(
                score_state_method(
                    "6DOF-UDE-NN",
                    "Universal differential equation: fitted grey-box physics plus an MLP residual trained RK4-in-the-loop.",
                    "torch-rk4-shooting",
                    state_source,
                    ude_train_elapsed,
                    ude_train_cpu,
                    ude_rollout_elapsed,
                    train_samples,
                    int(sum(t.numel() for t in ude_model["net"].parameters())),
                    pred_nn,
                    validation,
                    "Two-layer tanh MLP on invariant features corrects the six dynamic derivatives of the "
                    "fitted grey-box; trained by simulation error over 0.5 s shooting segments (no derivative "
                    "targets); frozen-network open-loop validation.",
                )
            )
        except ImportError:
            print(f"  {state_source}: torch unavailable, skipping 6DOF-UDE-NN", flush=True)

        weak = [n for n, sd, v in zip(greybox["parameter_names"], greybox["cr_std"], greybox["theta"]) if abs(v) > 1e-9 and sd / abs(v) > 0.25]
        couples = ", ".join(f"{c['a']}-{c['b']}" for c in greybox["couplings"])
        add_result(
            score_state_method(
                "6DOF-Fisher-UQ",
                "Cramer-Rao parameter uncertainty and coupling analysis of the grey-box output-error fit.",
                "casadi-jacobian-crlb",
                state_source,
                greybox_fit_elapsed,
                greybox_fit_cpu,
                rollout_elapsed,
                train_samples,
                int(greybox["theta"].size),
                pred,
                validation,
                "Same rollout as 6DOF-GreyBoxOEM; this row carries the uncertainty analysis: "
                f"weak parameters (>25% rel. std): {', '.join(weak) or 'none'}; "
                f"strong couplings (|r|>0.9): {couples or 'none'}. Bounds are optimistic (residual coloring uncorrected).",
            )
        )
    return results


def dataset_split_files(dataset: Path) -> tuple[Path, Path]:
    if dataset.is_file():
        name = dataset.name
        for split in ("train", "validation"):
            suffix = f"_{split}.npz"
            if name.endswith(suffix):
                dataset_id = name[: -len(suffix)]
                return dataset.with_name(f"{dataset_id}_train.npz"), dataset.with_name(f"{dataset_id}_validation.npz")
        raise SystemExit(f"Flat compact dataset files must be named <dataset_id>_<split>.npz: {dataset}")
    if dataset.is_dir():
        return dataset / "train.npz", dataset / "validation.npz"
    dataset_id = dataset.name
    return DATA / f"{dataset_id}_train.npz", DATA / f"{dataset_id}_validation.npz"


def dataset_scenario(dataset: Path) -> str:
    train_file, _validation_file = dataset_split_files(dataset)
    if train_file.exists():
        try:
            data = np.load(train_file, allow_pickle=False)
            if "dataset_id" in data.files:
                return str(np.asarray(data["dataset_id"]).item())
        except Exception:
            pass
    metadata_path = dataset / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            mode = metadata.get("dataset_mode") or metadata.get("config", {}).get("dataset_mode")
            if mode:
                return f"aircraft_6dof_{mode}"
        except json.JSONDecodeError:
            pass
    name = dataset.name
    if name.startswith("aircraft_6dof_"):
        return name
    return "aircraft_6dof_aggressive"


def default_dataset_path(scenario: str) -> Path:
    return METHODS_ROOT / "work" / "data" / scenario


def result_to_row(result: Result6DOF, validation_scenario: str) -> dict[str, object]:
    return {
        "method": result.method,
        "description": result.description,
        "implementation_status": result.implementation_status,
        "diverged": result.diverged,
        "backend": result.backend,
        "model_family": "aircraft6dof",
        "state_source": result.state_source,
        "input_channel": "u_cmd",
        "evaluation_mode": "open_loop",
        "training_scenario": result.training_scenario,
        "validation_scenario": validation_scenario,
        "scenario": validation_scenario,
        "scenario_title": SCENARIO_TITLES.get(validation_scenario, validation_scenario.replace("aircraft_6dof_", "").replace("_", " ").title()),
        "validation_score": result.validation_score,
        "train_elapsed_s": result.train_elapsed_s,
        "train_cpu_s": result.train_cpu_s,
        "train_gpu_s": 0.0,
        "gpu_memory_mb": 0.0,
        "rollout_elapsed_s": result.rollout_elapsed_s,
        "total_elapsed_s": result.total_elapsed_s,
        "train_loss_final": "",
        "decision_variables": result.decision_variables,
        "train_samples": result.train_samples,
        "rmse_position_m": result.rmse_position_m,
        "rmse_velocity_mps": result.rmse_velocity_mps,
        "rmse_quaternion": result.rmse_quaternion,
        "rmse_rates_rad_s": result.rmse_rates_rad_s,
        "rmse_mocap_position_m": result.rmse_mocap_position_m,
        "rmse_mocap_quaternion": result.rmse_mocap_quaternion,
        "notes": result.notes,
    }


def write_results(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


NED_TO_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


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
    return normalize_quaternion(quat)


def integrate_state_position_enu(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    pos_ned = np.empty_like(x[:, 0:3])
    pos_ned[0] = x[0, 0:3]
    velocity_ned = np.asarray([rotation_body_to_inertial(q) @ state[3:6] for state, q in zip(x, x[:, 6:10])])
    for index in range(len(t) - 1):
        dt = float(t[index + 1] - t[index])
        pos_ned[index + 1] = pos_ned[index] + 0.5 * dt * (velocity_ned[index] + velocity_ned[index + 1])
    position = np.column_stack([pos_ned[:, 1], pos_ned[:, 0], -pos_ned[:, 2]])
    return position - position[0]


def state_segment_to_web(t: np.ndarray, x: np.ndarray, u_cmd: np.ndarray, name: str) -> dict[str, object]:
    position = integrate_state_position_enu(t, x)
    stride = max(1, int(np.ceil(len(t) / WEB_TRACE_MAX_POINTS)))
    t = t[::stride]
    x = x[::stride]
    u_cmd = u_cmd[::stride]
    position = position[::stride]
    quat = np.asarray([quat_wxyz_from_rotation(NED_TO_ENU @ rotation_body_to_inertial(q)) for q in x[:, 6:10]])
    return {
        "name": name,
        "time_s": np.round(t - t[0], 4).tolist(),
        "position_enu_m": np.round(position, 5).tolist(),
        "quaternion_wxyz": np.round(quat, 7).tolist(),
        "control_meas": np.round(u_cmd[:, [0, 2, 1, 3]], 5).tolist(),
        "direct_state_meas": np.round(x, 6).tolist(),
    }


def segment_label(validation: Split6DOF, index: int) -> str:
    """Real-flight windows keep their chunk names so the viewer can place
    traces within a flight; synthetic trials keep the generic label."""
    if validation.segment_names is not None and index < len(validation.segment_names):
        return str(validation.segment_names[index])
    return f"validation_trial_{index + 1}"


def write_method_traces(results: list[Result6DOF], validation: Split6DOF, scenario: str, path: Path) -> None:
    traces: list[dict[str, object]] = []
    selected_results: list[Result6DOF] = []
    for source in sorted({result.state_source for result in results}):
        source_results = [result for result in results if result.state_source == source and np.isfinite(result.validation_score)]
        source_results = sorted(source_results, key=lambda result: result.validation_score)
        keep: dict[str, Result6DOF] = {
            result.method: result for result in source_results[:WEB_TRACE_TOP_METHODS_PER_SOURCE]
        }
        for result in source_results:
            if "Nominal" in result.method:
                keep[result.method] = result
        selected_results.extend(keep.values())
    # Real-flight scenarios export every validation window so the viewer can
    # overlay traces anywhere in a flight; synthetic scenarios keep one.
    segment_limit = validation.u_cmd.shape[0] if not scenario.startswith("aircraft_6dof_") else 1
    for result in selected_results:
        if result.x_pred is None:
            continue
        segment_count = min(result.x_pred.shape[0], validation.u_cmd.shape[0], segment_limit)
        segments = [
            state_segment_to_web(validation.t, result.x_pred[index], validation.u_cmd[index], segment_label(validation, index))
            for index in range(segment_count)
        ]
        traces.append(
            {
                "method": result.method,
                "model_family": "aircraft6dof",
                "scenario": scenario,
                "state_source": result.state_source,
                "segments": segments,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"traces": traces}, indent=2, sort_keys=True) + "\n")


def write_table(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (str(row["state_source"]), str(row.get("scenario", "")), float(row["validation_score"])))
    include_scenario = len({str(row.get("scenario", "")) for row in rows}) > 1
    with path.open("w") as stream:
        stream.write("% Generated by aircraft6dof comparison suite. Do not edit by hand.\n")
        stream.write(r"\begingroup\scriptsize\setlength{\tabcolsep}{2pt}" + "\n")
        if include_scenario:
            stream.write(r"\begin{longtable}{p{0.20\linewidth}p{0.10\linewidth}p{0.10\linewidth}lrrrrp{0.13\linewidth}}" + "\n")
            header = r"Method & Train & Val. & Source & Score & Train [s] & Rollout [s] & Pos. RMSE & Backend \\"
        else:
            stream.write(r"\begin{longtable}{p{0.31\linewidth}lrrrrp{0.18\linewidth}}" + "\n")
            header = r"Method & Source & Score & Train [s] & Rollout [s] & Pos. RMSE & Backend \\"
        stream.write(r"\caption{6-DOF aircraft benchmark baseline results. Lower validation score is better.}\label{tab:aircraft6dof_method_comparison}\\" + "\n")
        stream.write(r"\toprule" + "\n")
        stream.write(header + "\n")
        stream.write(r"\midrule" + "\n")
        stream.write(r"\endfirsthead" + "\n")
        stream.write(r"\toprule" + "\n")
        stream.write(header + "\n")
        stream.write(r"\midrule" + "\n")
        stream.write(r"\endhead" + "\n")
        for row in ordered:
            fields = [
                str(row["method"]).replace("_", r"\_"),
            ]
            if include_scenario:
                fields.append(scenario_label(row.get("training_scenario", "")).replace("_", r"\_"))
                fields.append(scenario_label(row.get("validation_scenario", row.get("scenario", ""))).replace("_", r"\_"))
            fields.extend(
                [
                    str(row["state_source"]),
                    f"{float(row['validation_score']):.3g}",
                    f"{float(row['train_elapsed_s']):.3g}",
                    f"{float(row['rollout_elapsed_s']):.3g}",
                    _fmt(row["rmse_position_m"]),
                    str(row["backend"]).replace("_", r"\_"),
                ]
            )
            stream.write(" & ".join(fields) + r" \\" + "\n")
        stream.write(r"\bottomrule" + "\n")
        stream.write(r"\end{longtable}" + "\n")
        stream.write(r"\endgroup" + "\n")


def _fmt(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(number):
        return "--"
    return f"{number:.3g}"


def scenario_label(scenario: object) -> str:
    text = str(scenario)
    if text == "none":
        return "No fit"
    return SCENARIO_TITLES.get(text, text.replace("aircraft_6dof_", "").replace("_", " ").title())


def plot_scores(rows: list[dict[str, object]], output: Path) -> None:
    groups = ["direct", "mocap"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True)
    rows = aggregate_method_rows(rows)
    finite = [max(float(row["validation_score"]), 1e-6) for row in rows]
    x_min = max(min(finite) * 0.65, 1e-5)
    x_max = max(finite) * 1.8
    colors = {"direct": "#4c78a8", "mocap": "#f58518"}
    for ax, source in zip(axes, groups):
        source_rows = sorted([row for row in rows if row["state_source"] == source], key=lambda row: float(row["validation_score"]), reverse=True)
        y = np.arange(len(source_rows))
        scores = [max(float(row["validation_score"]), 1e-6) for row in source_rows]
        labels = [str(row["method"]).replace("6DOF-", "") for row in source_rows]
        for yi, score in zip(y, scores):
            ax.plot([x_min, score], [yi, yi], color="0.82", linewidth=1.0)
        ax.scatter(scores, y, color=colors[source], edgecolor="black", linewidth=0.4, s=44)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.0)
        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)
        ax.set_title(f"{source} validation")
        ax.set_xlabel("validation score")
        ax.grid(True, axis="x", which="both", alpha=0.25)
        ax.text(0.02, 0.04, "left is better", transform=ax.transAxes, fontsize=8.0, bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9})
    axes[0].set_ylabel("method")
    fig.suptitle("6-DOF baseline validation score")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _source_rows(rows: list[dict[str, object]], source: str) -> list[dict[str, object]]:
    return [row for row in rows if row["state_source"] == source and np.isfinite(float(row["validation_score"]))]


def aggregate_method_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), str(row["state_source"])), []).append(row)
    aggregated: list[dict[str, object]] = []
    for (_method, _source), group in grouped.items():
        out = dict(group[0])
        for key in [
            "validation_score",
            "train_elapsed_s",
            "train_cpu_s",
            "train_gpu_s",
            "gpu_memory_mb",
            "rollout_elapsed_s",
            "total_elapsed_s",
            "decision_variables",
            "train_samples",
            "rmse_position_m",
            "rmse_velocity_mps",
            "rmse_quaternion",
            "rmse_rates_rad_s",
            "rmse_mocap_position_m",
            "rmse_mocap_quaternion",
        ]:
            values = [float(row[key]) for row in group if row.get(key) not in ("", None) and np.isfinite(float(row[key]))]
            if values:
                out[key] = float(np.mean(values))
        out["scenario"] = "mean"
        out["scenario_title"] = "Mean score"
        aggregated.append(out)
    return aggregated


def split_tradeoff_rows(rows: list[dict[str, object]], threshold: float = TRADEOFF_FAILURE_THRESHOLD) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    passed: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    for row in rows:
        if float(row["validation_score"]) > threshold:
            failed.append(row)
        else:
            passed.append(row)
    return passed, failed


def tradeoff_label(method: object) -> str:
    return str(method).replace("6DOF-", "").replace("Frequency-", "Freq-").replace("Model-Stitching", "Stitching")


def add_failure_callout(ax, failed_rows: list[dict[str, object]], threshold: float = TRADEOFF_FAILURE_THRESHOLD) -> None:
    if not failed_rows:
        return
    labels = [tradeoff_label(row["method"]) for row in sorted(failed_rows, key=lambda row: float(row["validation_score"]), reverse=True)]
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


def plot_train_time_accuracy(rows: list[dict[str, object]], output: Path) -> None:
    groups = ["direct", "mocap"]
    colors = {"direct": "#4c78a8", "mocap": "#f58518"}
    rows = aggregate_method_rows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    passed_scores = [
        max(float(row["validation_score"]), 1e-6)
        for row in rows
        if np.isfinite(float(row["validation_score"])) and float(row["validation_score"]) <= TRADEOFF_FAILURE_THRESHOLD
    ]
    y_limits = (
        max(min(passed_scores) * 0.65, 5e-2),
        max(min(max(passed_scores) * 2.2, TRADEOFF_FAILURE_THRESHOLD * 1.1), 0.25),
    ) if passed_scores else (5e-2, TRADEOFF_FAILURE_THRESHOLD * 1.1)
    for ax, source in zip(axes, groups):
        source_rows = _source_rows(rows, source)
        if not source_rows:
            continue
        source_rows, failed_rows = split_tradeoff_rows(source_rows)
        add_failure_callout(ax, failed_rows)
        if not source_rows:
            continue
        train_times = np.array([max(float(row["train_elapsed_s"]), 1e-2) for row in source_rows])
        scores = np.array([max(float(row["validation_score"]), 1e-6) for row in source_rows])
        rollout = np.array([max(float(row["rollout_elapsed_s"]), 1e-3) for row in source_rows])
        nominal = [row for row in source_rows if row["method"] == "6DOF-NominalGreyBox"]
        if nominal:
            nominal_score = max(float(nominal[0]["validation_score"]), 1e-6)
            ax.axhline(nominal_score, color="#d62728", linestyle="--", linewidth=1.0)
            ax.text(max(train_times) * 0.82, nominal_score * 1.06, "NominalGreyBox", color="#d62728", fontsize=7.0)
        sizes = 34.0 + 130.0 * np.sqrt(rollout / max(float(np.max(rollout)), 1e-9))
        ax.scatter(train_times, scores, s=sizes, color=colors[source], edgecolor="black", linewidth=0.45, alpha=0.78, zorder=3)
        label_offsets = [(1.08, 1.10), (0.82, 1.18), (1.05, 0.75), (0.72, 0.82)]
        for index, row in enumerate(source_rows):
            label = tradeoff_label(row["method"])
            if label == "NominalGreyBox":
                continue
            dx, dy = label_offsets[index % len(label_offsets)]
            ax.annotate(
                label,
                (max(float(row["train_elapsed_s"]), 1e-2), max(float(row["validation_score"]), 1e-6)),
                xytext=(max(float(row["train_elapsed_s"]), 1e-2) * dx, max(float(row["validation_score"]), 1e-6) * dy),
                fontsize=6.7,
                arrowprops={"arrowstyle": "-", "color": "0.62", "linewidth": 0.5},
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(*y_limits)
        ax.set_title(f"{source.capitalize()} benchmark")
        ax.set_xlabel("training / solve time [s]")
        ax.grid(True, which="both", alpha=0.25)
        ax.text(0.02, 0.96, "lower error is better", transform=ax.transAxes, fontsize=8.0, va="top", bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9})
    axes[0].set_ylabel("validation score: mean state NRMSE")
    fig.suptitle("6-DOF training-time versus validation-error tradeoff")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_score_heatmaps(rows: list[dict[str, object]], fig_dir: Path) -> None:
    for source, color in (("direct", "#4c78a8"), ("mocap", "#f58518")):
        source_rows = _source_rows(rows, source)
        if not source_rows:
            continue
        scenarios = [scenario for scenario in SCENARIO_ORDER if any(str(row.get("scenario")) == scenario for row in source_rows)]
        extras = sorted({str(row.get("scenario")) for row in source_rows if str(row.get("scenario")) not in scenarios})
        scenarios.extend(extras)
        methods = sorted({str(row["method"]) for row in source_rows})
        score_map = {
            (str(row["method"]), str(row.get("scenario"))): max(float(row["validation_score"]), 1e-6)
            for row in source_rows
        }
        method_mean = {
            method: float(np.mean([score_map[(method, scenario)] for scenario in scenarios if (method, scenario) in score_map]))
            for method in methods
        }
        methods = sorted(methods, key=lambda method: method_mean[method])
        labels = [method.replace("6DOF-", "") for method in methods]
        scores = np.full((len(methods), len(scenarios) + 1), np.nan)
        for row_index, method in enumerate(methods):
            values = [score_map[(method, scenario)] for scenario in scenarios if (method, scenario) in score_map]
            scores[row_index, 0] = float(np.mean(values)) if values else np.nan
            for col_index, scenario in enumerate(scenarios, start=1):
                if (method, scenario) in score_map:
                    scores[row_index, col_index] = score_map[(method, scenario)]
        finite_scores = scores[np.isfinite(scores)]
        vmin = max(float(np.nanmin(finite_scores)) * 0.8, 1e-6)
        vmax = max(float(np.nanmax(finite_scores)) * 1.2, vmin * 10.0)
        height = max(4.2, 0.34 * len(labels) + 1.2)
        width = max(7.2, 1.25 * (len(scenarios) + 1) + 3.5)
        fig, ax = plt.subplots(figsize=(width, height))
        im = ax.imshow(np.ma.masked_invalid(scores), aspect="auto", cmap="viridis_r", norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8.0)
        ax.set_xticks(np.arange(len(scenarios) + 1))
        ax.set_xticklabels(
            ["Mean score"] + [SCENARIO_TITLES.get(scenario, scenario.replace("aircraft_6dof_", "").replace("_", " ").title()) for scenario in scenarios],
            rotation=28,
            ha="right",
            fontsize=8.0,
        )
        ax.set_ylabel("method")
        ax.set_title(f"Validation trajectory error: 6-DOF {source} benchmark", color=color)
        for row_index in range(scores.shape[0]):
            for col_index in range(scores.shape[1]):
                value = scores[row_index, col_index]
                if not np.isfinite(value):
                    continue
                ax.text(col_index, row_index, f"{value:.2g}", ha="center", va="center", fontsize=6.4, color="black" if value < np.sqrt(vmin * vmax) else "white")
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label("validation score, lower is better")
        fig.tight_layout()
        fig_dir.mkdir(parents=True, exist_ok=True)
        output = fig_dir / f"aircraft6dof_method_score_heatmap_{source}.svg"
        fig.savefig(output, bbox_inches="tight")
        fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_trajectory(results: list[Result6DOF], validation: Split6DOF, output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11.2, 8.2), constrained_layout=True)
    trial = 0
    config = Aircraft6DOFConfig(duration=float(validation.t[-1] - validation.t[0]), dt=validation.dt)
    truth_speed = np.linalg.norm(validation.x_true[trial, :, 3:6], axis=1)
    truth_coeff = np.array([aerodynamic_coefficients(state, command, config, nonlinear=True) for state, command in zip(validation.x_true[trial], validation.u_cmd[trial])])
    axes[0, 0].plot(validation.x_true[trial, :, 0], -validation.x_true[trial, :, 2], "k-", linewidth=2.0, label="truth")
    selected: list[Result6DOF] = []
    for source in ("direct", "mocap"):
        selected.extend(sorted([r for r in results if r.x_pred is not None and r.state_source == source], key=lambda r: r.validation_score)[:3])
    for result in selected:
        axes[0, 0].plot(result.x_pred[trial, :, 0], -result.x_pred[trial, :, 2], linewidth=1.0, label=f"{result.method}/{result.state_source}")
        axes[0, 1].plot(validation.t, np.linalg.norm(result.x_pred[trial, :, 3:6], axis=1), linewidth=1.0, label=f"{result.method}/{result.state_source}")
        pred_coeff = np.array([aerodynamic_coefficients(state, command, config, nonlinear=True) for state, command in zip(result.x_pred[trial], validation.u_cmd[trial])])
        axes[1, 0].plot(validation.t, np.rad2deg(pred_coeff[:, 6]), linewidth=1.0, label=f"{result.method}/{result.state_source}")
        axes[1, 1].plot(validation.t, pred_coeff[:, 8], linewidth=1.0, label=f"{result.method}/{result.state_source}")
        axes[2, 0].plot(validation.t, result.x_pred[trial, :, 10], linewidth=1.0, label=f"{result.method}/{result.state_source}")
        axes[2, 1].plot(validation.t, result.x_pred[trial, :, 11], linewidth=1.0, label=f"{result.method}/{result.state_source}")
    axes[0, 1].plot(validation.t, truth_speed, "k-", linewidth=2.0, label="truth")
    axes[1, 0].plot(validation.t, np.rad2deg(truth_coeff[:, 6]), "k-", linewidth=2.0, label="truth")
    axes[1, 1].plot(validation.t, truth_coeff[:, 8], "k-", linewidth=2.0, label="truth")
    axes[2, 0].plot(validation.t, validation.x_true[trial, :, 10], "k-", linewidth=2.0, label="truth")
    axes[2, 1].plot(validation.t, validation.x_true[trial, :, 11], "k-", linewidth=2.0, label="truth")
    axes[0, 0].set_xlabel("x north [m]")
    axes[0, 0].set_ylabel("altitude proxy -z_d [m]")
    axes[0, 0].set_title("trajectory")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("speed [m/s]")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel(r"$\alpha$ [deg]")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("stall gate")
    axes[2, 0].set_xlabel("time [s]")
    axes[2, 0].set_ylabel("p [rad/s]")
    axes[2, 1].set_xlabel("time [s]")
    axes[2, 1].set_ylabel("q [rad/s]")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=6.5, loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_manifest(datasets: list[Path], rows: list[dict[str, object]], output: Path) -> None:
    payload = {
        "model_family": "aircraft6dof",
        "datasets": [str(dataset) for dataset in datasets],
        "scenarios": sorted({str(row.get("scenario", "")) for row in rows}),
        "training_scenarios": sorted({str(row.get("training_scenario", "")) for row in rows}),
        "methods": sorted({str(row["method"]) for row in rows}),
        "state_sources": sorted({str(row["state_source"]) for row in rows}),
        "metric": "Validation score is full-state mean NRMSE over open-loop rollouts.",
        "result_rows": len(rows),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--datasets", type=Path, nargs="*", default=None, help="run several 6DOF datasets and aggregate the plots/tables")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--state-source", choices=["direct", "mocap", "both"], default="both")
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="parallel rollout worker processes")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-onset-trim", action="store_true", help="disable lead-in/actuation segmentation for real flight datasets")
    parser.add_argument("--no-input-bias", action="store_true", help="disable lead-in input trim-offset estimation for real flight datasets")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = list(args.datasets) if args.datasets else [args.dataset]
    provided_paths = {dataset_scenario(dataset): dataset for dataset in datasets}
    required_training = {scenario for scenario in METHOD_TRAINING_SCENARIOS.values() if scenario != "none"}
    train_splits: dict[str, Split6DOF] = {}

    def synthetic_train_splits() -> dict[str, Split6DOF]:
        if not train_splits:
            for training_scenario in sorted(required_training):
                path = provided_paths.get(training_scenario, default_dataset_path(training_scenario))
                train_file = path / "train.npz"
                if not train_file.exists():
                    raise SystemExit(f"Required 6DOF training split missing for {training_scenario}: {train_file}")
                train_splits[training_scenario] = load_split(train_file)
        return train_splits

    sources = ["direct", "mocap"] if args.state_source == "both" else [args.state_source]
    results: list[Result6DOF] = []
    rows: list[dict[str, object]] = []
    trajectory_results: list[Result6DOF] = []
    trajectory_validation: Split6DOF | None = None
    for dataset in datasets:
        scenario = dataset_scenario(dataset)
        active_train_splits = train_splits
        training_scenario_override = None
        dataset_train_file, dataset_validation_file = dataset_split_files(dataset)
        if dataset_train_file.exists() and not scenario.startswith("aircraft_6dof_"):
            if args.no_onset_trim:
                dataset_train = load_split(dataset_train_file)
            else:
                train_segments, train_dt = load_segments(dataset_train_file)
                dataset_train = trim_to_input_onset(
                    train_segments,
                    train_dt,
                    estimate_bias=not args.no_input_bias,
                    label=f"{scenario} train",
                )
            active_train_splits = {training_scenario: dataset_train for training_scenario in required_training}
            active_train_splits[scenario] = dataset_train
            active_train_splits["aircraft_6dof_open_loop"] = dataset_train
            training_scenario_override = scenario
        else:
            active_train_splits = synthetic_train_splits()
        if training_scenario_override is not None and not args.no_onset_trim:
            validation_segments, validation_dt = load_segments(dataset_validation_file)
            validation = trim_to_input_onset(
                validation_segments,
                validation_dt,
                estimate_bias=not args.no_input_bias,
                label=f"{scenario} validation",
            )
        else:
            validation = load_split(dataset_validation_file)
        dataset_results: list[Result6DOF] = []
        for source in sources:
            print(f"running 6DOF {source} methods on {scenario} with {args.workers} rollout workers", flush=True)
            dataset_results.extend(
                run_methods(active_train_splits, validation, source, args.ridge, args.workers, training_scenario_override)
            )
        rows.extend(result_to_row(result, scenario) for result in dataset_results)
        write_method_traces(dataset_results, validation, scenario, args.results_dir / f"{scenario}_method_traces.json")
        if trajectory_validation is None or scenario == "aircraft_6dof_aggressive":
            trajectory_validation = validation
            trajectory_results = dataset_results
    write_results(rows, args.results_dir / "aircraft6dof_method_comparison.csv")
    write_table(rows, args.table_dir / "aircraft6dof_method_comparison.tex")
    write_manifest(datasets, rows, args.results_dir / "aircraft6dof_benchmark_manifest.json")
    if not args.no_plot:
        plot_scores(rows, args.fig_dir / "aircraft6dof_validation_score_comparison.svg")
        plot_train_time_accuracy(rows, args.fig_dir / "aircraft6dof_train_time_accuracy_tradeoff.svg")
        plot_score_heatmaps(rows, args.fig_dir)
        if trajectory_validation is not None:
            plot_trajectory(trajectory_results, trajectory_validation, args.fig_dir / "aircraft6dof_validation_trajectory_overlay.svg")
    for row in sorted(rows, key=lambda item: (str(item["state_source"]), float(item["validation_score"]))):
        print(
            f"{row['method']} ({row['state_source']}): "
            f"score={float(row['validation_score']):.4g}, train={float(row['train_elapsed_s']):.3g}s, "
            f"backend={row['backend']}"
        )
    print(f"wrote {args.results_dir / 'aircraft6dof_method_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
