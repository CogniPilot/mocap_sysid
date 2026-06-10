#!/usr/bin/env python3
"""Export full-flight segmentation and prediction data for the website explorer.

For each 2026-05-22 Sport Cub flight this produces:

- the downsampled measured trajectory (ENU position, Euler attitude, sticks)
  with a per-sample segmentation label (ground / ground-effect / stabilized /
  manual);
- a segment table with each segment's class, train/validation membership, and
  per-method prediction score where the rollout starts from an initial
  condition estimated at the segment start (so the cost is not dominated by
  accumulated error from earlier flight);
- the fitted model parameters themselves (linear state-space weights,
  nominal-plus-residual weights, SAFE inner-loop controller gains, per-flight
  trim bias, and the nominal airframe constants), so the browser integrates
  free-run predictions on the fly from any clicked segment — through manual
  segments using the recorded sticks (re-referenced by the per-flight trim
  bias) and through stabilized segments by closing the loop with the
  identified SAFE controller model, since the bare airframe alone cannot
  represent stabilized flight. Computing rollouts client-side avoids
  exporting a separate trace per (segment, method) pair and lets the viewer
  set the initial condition anywhere.

Training mirrors the benchmark: methods are fitted on the train split's
maneuver windows with initial conditions estimated at each window start, so
identification cost functions never start from an already-diverged state.

Output: ``site/public/data/flight_explorer.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.paths import ROOT
from . import comparison_suite as suite
from .greybox import Aircraft6DOFConfig, euler_from_quaternion, nominal_rk4_step, normalize_quaternion
from .safe_controller import fit_safe_controller, safe_controller

FLIGHTS_DEFAULT = Path("data/sportcub_mocap_5_22_26_flights.npz")
TRAIN_DEFAULT = Path("data/sportcub_mocap_5_22_26_train.npz")
VALIDATION_DEFAULT = Path("data/sportcub_mocap_5_22_26_validation.npz")
OUTPUT_DEFAULT = ROOT / "site" / "public" / "data" / "flight_explorer.json"

GROUND_ALTITUDE_M = 0.5
GROUND_EFFECT_ALTITUDE_M = 0.65
DISPLAY_RATE_HZ = 10.0  # browser display rate; stride derived from the data rate
RIDGE = 1e-5
METHODS = ("6DOF-Nominal", "6DOF-LinearSS", "6DOF-RidgeResidual")
LABELS = ("ground", "ground_effect", "stabilized", "manual")


def sample_labels(x: np.ndarray, mode: np.ndarray) -> np.ndarray:
    """Per-sample segmentation label index into LABELS."""
    altitude = -x[:, 2]
    airborne = np.zeros(len(x), dtype=bool)
    state = altitude[0] > GROUND_ALTITUDE_M
    for k, alt in enumerate(altitude):
        if state and alt < GROUND_ALTITUDE_M - 0.15:
            state = False
        elif not state and alt > GROUND_ALTITUDE_M + 0.15:
            state = True
        airborne[k] = state
    labels = np.zeros(len(x), dtype=np.int8)  # ground
    labels[airborne & (altitude < GROUND_EFFECT_ALTITUDE_M)] = 1
    labels[airborne & (altitude >= GROUND_EFFECT_ALTITUDE_M) & (mode == 1)] = 2
    labels[airborne & (altitude >= GROUND_EFFECT_ALTITUDE_M) & (mode == 0)] = 3
    # Low-altitude samples keep their controller state distinction when manual.
    labels[airborne & (altitude < GROUND_EFFECT_ALTITUDE_M) & (mode == 0)] = 1
    return labels


def contiguous_segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    edges = np.flatnonzero(np.diff(labels)) + 1
    bounds = [0, *edges.tolist(), len(labels)]
    return [(bounds[i], bounds[i + 1], int(labels[bounds[i]])) for i in range(len(bounds) - 1)]


def euler_array(x: np.ndarray) -> np.ndarray:
    out = np.empty((len(x), 3))
    for k in range(len(x)):
        out[k] = euler_from_quaternion(normalize_quaternion(x[k, 6:10]))
    return out


def ned_to_enu_pos(pos_ned: np.ndarray) -> np.ndarray:
    return np.column_stack([pos_ned[:, 1], pos_ned[:, 0], -pos_ned[:, 2]])


def train_safe_closed_loop(data: np.lib.npyio.NpzFile) -> np.ndarray:
    """Identify the stabilized closed-loop dynamics directly.

    Free-running a bare-airframe model through SAFE segments requires the
    hidden-controller decomposition, whose inverse-dynamics identification is
    weakly conditioned. The closed loop itself (state + stick -> next state)
    is abundantly excited by the lap data and is exactly what stabilized
    segments replay, so it is fitted directly by ridge regression on every
    tracked stabilized sample.
    """
    dt = float(data["sample_period_s"])
    xs, us, nexts = [], [], []
    for flight_index in range(len(data["segment_names"])):
        mask = np.asarray(data["valid_mask"][flight_index], dtype=bool)
        x = np.asarray(data["x_meas"][flight_index][mask], dtype=float)
        sticks = np.asarray(data["u_cmd"][flight_index][mask], dtype=float)
        mode = np.asarray(data["flight_mode"][flight_index][mask])
        tracked = (
            np.asarray(data["mocap_tracked"][flight_index][mask]) != 0
            if "mocap_tracked" in data.files
            else np.ones(len(x), dtype=bool)
        )
        labels = sample_labels(x, mode)
        keep = (labels == 2) & tracked
        keep_next = keep[:-1] & keep[1:]
        xs.append(x[:-1][keep_next])
        us.append(sticks[:-1][keep_next])
        nexts.append(x[1:][keep_next])
    x_all = np.concatenate(xs)
    u_all = np.concatenate(us)
    next_all = np.concatenate(nexts)
    print(f"closed-loop SAFE fit: {len(x_all)} stabilized samples at dt={dt}")
    return suite.ridge_fit(suite.design_matrix(x_all, u_all), next_all, RIDGE)


def train_safe_closed_loop(data: np.lib.npyio.NpzFile) -> np.ndarray:
    """Identify the stabilized closed-loop dynamics directly.

    Free-running a bare-airframe model through SAFE segments requires the
    hidden-controller decomposition, whose inverse-dynamics identification is
    weakly conditioned. The closed loop itself (state + stick -> next state)
    is abundantly excited by the lap data and is exactly what stabilized
    segments replay, so it is fitted directly by ridge regression on every
    tracked stabilized sample.
    """
    dt = float(data["sample_period_s"])
    xs, us, nexts = [], [], []
    for flight_index in range(len(data["segment_names"])):
        mask = np.asarray(data["valid_mask"][flight_index], dtype=bool)
        x = np.asarray(data["x_meas"][flight_index][mask], dtype=float)
        sticks = np.asarray(data["u_cmd"][flight_index][mask], dtype=float)
        mode = np.asarray(data["flight_mode"][flight_index][mask])
        tracked = (
            np.asarray(data["mocap_tracked"][flight_index][mask]) != 0
            if "mocap_tracked" in data.files
            else np.ones(len(x), dtype=bool)
        )
        labels = sample_labels(x, mode)
        keep = (labels == 2) & tracked
        keep_next = keep[:-1] & keep[1:]
        xs.append(x[:-1][keep_next])
        us.append(sticks[:-1][keep_next])
        nexts.append(x[1:][keep_next])
    x_all = np.concatenate(xs)
    u_all = np.concatenate(us)
    next_all = np.concatenate(nexts)
    print(f"closed-loop SAFE fit: {len(x_all)} stabilized samples at dt={dt}")
    return suite.ridge_fit(suite.design_matrix(x_all, u_all), next_all, RIDGE)


def train_methods(train_path: Path) -> dict[str, np.ndarray | None]:
    segments, dt = suite.load_segments(train_path)
    split = suite.trim_to_input_onset(segments, dt, estimate_bias=True, label="explorer train")
    x = split.x_true
    u = split.u_cmd
    config = Aircraft6DOFConfig()
    linear = suite.ridge_fit(suite.design_matrix(x[:, :-1, :], u[:, :-1, :]), x[:, 1:, :], RIDGE)
    nominal_next = np.empty_like(x[:, :-1, :])
    for trial in range(x.shape[0]):
        for index in range(x.shape[1] - 1):
            nominal_next[trial, index] = nominal_rk4_step(x[trial, index], u[trial, index], split.dt, config)
    residual_target = x[:, 1:, :] - nominal_next
    residual = suite.ridge_fit(suite.design_matrix(x[:, :-1, :], u[:, :-1, :]), residual_target, RIDGE)
    return {"6DOF-Nominal": None, "6DOF-LinearSS": linear, "6DOF-RidgeResidual": residual}


def make_stepper(method: str, weights: np.ndarray | None, dt: float, config: Aircraft6DOFConfig):
    def step(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        if method == "6DOF-Nominal":
            return nominal_rk4_step(x, u, dt, config)
        phi = np.concatenate((x, u, [1.0]))
        if method == "6DOF-LinearSS":
            return suite.normalize_state(phi @ weights)
        return suite.normalize_state(nominal_rk4_step(x, u, dt, config) + phi @ weights)

    return step


def rollout(
    stepper,
    x0: np.ndarray,
    sticks: np.ndarray,
    labels: np.ndarray,
    bias: np.ndarray,
    safe_step,
) -> np.ndarray:
    """Integrate the bare airframe on manual segments and the directly
    identified closed-loop model on stabilized segments."""
    pred = np.empty((len(sticks), len(x0)))
    pred[0] = suite.normalize_state(x0)
    for k in range(len(sticks) - 1):
        if labels[k] == 2:
            pred[k + 1] = safe_step(pred[k], sticks[k])
        else:
            pred[k + 1] = stepper(pred[k], sticks[k] - bias)
    return pred


def manual_window_splits(paths: dict[str, Path]) -> dict[tuple[str, int], str]:
    membership: dict[tuple[str, int], str] = {}
    for split_name, path in paths.items():
        data = np.load(path, allow_pickle=False)
        for name in data["segment_names"]:
            flight, _, start = str(name).rpartition("_manual_")
            if flight:
                membership[(flight.rstrip("_"), int(start))] = split_name
    return membership


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flights", type=Path, default=FLIGHTS_DEFAULT)
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument("--validation", type=Path, default=VALIDATION_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()

    data = np.load(args.flights, allow_pickle=False)
    dt = float(data["sample_period_s"])
    downsample = max(1, int(round(1.0 / (DISPLAY_RATE_HZ * dt))))
    config = Aircraft6DOFConfig()
    weights = train_methods(args.train)
    safe_weights = train_safe_closed_loop(data)

    def safe_step(x, stick):
        phi = np.concatenate((x, stick, [1.0]))
        return suite.normalize_state(phi @ safe_weights)

    gains, _rmse, _count = fit_safe_controller(args.flights)
    membership = manual_window_splits({"train": args.train, "validation": args.validation})

    flights_payload = []
    for flight_index, name in enumerate(str(s) for s in data["segment_names"]):
        mask = np.asarray(data["valid_mask"][flight_index], dtype=bool)
        x = np.asarray(data["x_meas"][flight_index][mask], dtype=float)
        pose = np.asarray(data["pose_meas"][flight_index][mask], dtype=float)
        sticks = np.asarray(data["u_cmd"][flight_index][mask], dtype=float)
        mode = np.asarray(data["flight_mode"][flight_index][mask])
        tracked = (
            np.asarray(data["mocap_tracked"][flight_index][mask]) != 0
            if "mocap_tracked" in data.files
            else np.ones(len(x), dtype=bool)
        )
        labels = sample_labels(x, mode)
        segments = contiguous_segments(labels)

        # Per-flight trim bias from the manual windows' lead-ins, matching the
        # benchmark preprocessing.
        manual_segs = [
            {"name": name, "x": x[a:b], "u": sticks[a:b], "mocap_frame": "ned"}
            for a, b, label in segments
            if label == 3 and b - a >= 50
        ]
        if manual_segs:
            onsets = [suite.detect_input_onset(np.asarray(seg["u"])) for seg in manual_segs]
            bias = suite.estimate_input_bias(manual_segs, onsets, dt, config)
        else:
            bias = np.zeros(sticks.shape[1])

        steppers = {method: make_stepper(method, weights[method], dt, config) for method in METHODS}
        segment_rows = []
        freeruns: dict[str, dict[str, object]] = {}
        for seg_index, (start, stop, label) in enumerate(segments):
            row: dict[str, object] = {
                "kind": LABELS[label],
                "start_s": round(start * dt, 2),
                "stop_s": round(stop * dt, 2),
                "split": membership.get((name, start)),
            }
            if label in (2, 3) and stop - start >= 30:
                x0 = suite.local_state_estimate(x, dt, start, window=12)
                scores = {}
                keep = tracked[start:stop]
                for method in METHODS:
                    pred = rollout(steppers[method], x0, sticks[start:stop], labels[start:stop], bias, safe_step)
                    pred_aligned = suite.align_quaternion_signs(pred, x[start:stop])
                    # Score only on genuinely tracked samples: interpolated
                    # dropout spans are fabricated data.
                    scores[method] = round(float(suite.nrmse_score(pred_aligned[keep], x[start:stop][keep])), 4)
                row["scores"] = scores
            segment_rows.append(row)

        ds = slice(None, None, downsample)
        flights_payload.append(
            {
                "name": name,
                "dt": dt * downsample,
                "dt_full": dt,
                "time": np.round(np.arange(len(x))[ds] * dt, 2).tolist(),
                # 13-state truth at display rate: IC estimation and prediction
                # overlays both happen client-side.
                "state": np.round(x[ds], 4).tolist(),
                "pos": np.round(ned_to_enu_pos(x[ds, 0:3]), 3).tolist(),
                # Canonical ENU/body-FRD pose quaternion so full flights can be
                # animated as first-class playback tracks.
                "quat": np.round(pose[ds, 3:7], 5).tolist(),
                "euler": np.round(euler_array(x[ds]), 4).tolist(),
                # Full-rate sticks and labels drive the on-the-fly rollouts.
                "stick_full": np.round(sticks, 3).tolist(),
                "labels_full": labels.tolist(),
                "labels": labels[ds].tolist(),
                "mode": np.asarray(mode[ds], dtype=int).tolist(),
                "tracked": tracked[ds].astype(int).tolist(),
                "tracked_full": tracked.astype(int).tolist(),
                "bias": np.round(bias, 4).tolist(),
                "segments": segment_rows,
            }
        )
        print(f"{name}: {len(segment_rows)} segments", flush=True)

    payload = {
        "dataset": "sportcub_mocap_5_22_26",
        "labels": list(LABELS),
        "methods": list(METHODS),
        "models": {
            "linear_weights": np.round(weights["6DOF-LinearSS"], 6).tolist(),
            "residual_weights": np.round(weights["6DOF-RidgeResidual"], 6).tolist(),
            "safe_gains": {axis: np.round(coef, 5).tolist() for axis, coef in gains.items()},
            "safe_linear_weights": np.round(safe_weights, 6).tolist(),
            "safe_linear_weights": np.round(safe_weights, 6).tolist(),
            "config": {
                "mass": config.mass,
                "gravity": config.gravity,
                "inertia": list(config.inertia),
                "inertia_xz": config.inertia_xz,
                "rho": config.rho,
                "wing_area": config.wing_area,
                "wing_span": config.wing_span,
                "mean_chord": config.mean_chord,
                "prop_arm": config.prop_arm,
                "wing_speed": config.wing_speed,
                "max_thrust": config.max_thrust,
                "prop_wash_gain": config.prop_wash_gain,
            },
        },
        "flights": flights_payload,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload))
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
