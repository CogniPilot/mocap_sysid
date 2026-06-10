#!/usr/bin/env python3
"""Automatically segment the 2026-05-22 Sport Cub flights for identification.

Each recorded flight is a stabilized lap pattern (SAFE inner loop on) with the
inner loop switched off mid-facility for a piloted maneuver. The segmenter
classifies every sample of every flight into:

- ``ground``: motion-capture altitude below a threshold (with hysteresis), so
  taxi, handling, and crash samples never reach the airborne identification
  splits;
- ``stabilized``: airborne with the SAFE inner loop engaged (flight-mode
  channel high) — kept out of the bare-airframe splits because the hidden
  inner loop reshapes the surface commands;
- ``manual``: airborne with the inner loop disengaged — the bare-airframe
  dynamics windows used for identification.

Rolling ground windows are additionally exported to a separate ``ground``
file for landing-gear and rolling-friction identification. Ground data is
highly susceptible to floor disturbances (wheel debris, surface seams,
handling), so every sample carries a robust outlier flag: a modified z-score
(median/MAD, Iglewicz--Hoaglin) on the planar acceleration magnitude combined
with an on-ground attitude gate, dilated to cover disturbance transients.
Downstream friction analyses should drop flagged samples rather than trusting
any single window.

Manual windows are quality-gated (duration, mocap glitch limits) and exported
as canonical ``sysid.timeseries.ragged.v1`` train/validation splits, assigned
round-robin per flight so both splits span multiple flights and one model is
learned across runs. Bare-airframe trim is estimated per flight by pooling
quasi-steady manual samples (low angular rate, low stick rate), and stored per
segment so downstream consumers can re-reference inputs per flight.

The segmentation philosophy follows flight-test practice: window the records
around the excitation and treat trim/bias as per-flight calibration (Klein &
Morelli 2006; Jategaonkar 2006; Morelli's flight-condition data partitioning).
The controller-state channel recorded by the transmitter makes maneuver
recognition exact, so no statistical changepoint or learned maneuver detector
is required.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATASET_ID = "sportcub_mocap_5_22_26"
FLIGHTS_DEFAULT = Path("data") / f"{DATASET_ID}_flights.npz"
OUTPUT_DIR_DEFAULT = Path("data")

GROUND_ALTITUDE_M = 0.5
GROUND_HYSTERESIS_M = 0.15
MIN_MANUAL_DURATION_S = 1.0
MAX_SPEED_MPS = 12.0
MAX_RATE_RAD_S = 12.0
VALIDATION_FRACTION = 0.3
QUASI_STEADY_RATE_RAD_S = 1.2
QUASI_STEADY_STICK_RATE = 1.5  # normalized stick units per second
MIN_GROUND_DURATION_S = 1.0
MIN_ROLLING_SPEED_MPS = 0.3
GROUND_OUTLIER_Z = 3.5  # modified z-score threshold (Iglewicz-Hoaglin)
GROUND_OUTLIER_FLOOR_MPS = 0.15  # ignore sub-floor residuals: periodic wheel/floor texture, not events
GROUND_OUTLIER_DILATE_S = 0.08
GROUND_MAX_ATTITUDE_RAD = 0.5  # attitude deviation beyond this on the ground is handling/nose-over
GROUND_MAX_YAW_RATE_RAD_S = 1.5  # taildragger ground loop / spin-out
GROUND_EFFECT_ALTITUDE_M = 0.65  # about one wingspan; airborne below this is in ground effect
MIN_GROUND_EFFECT_DURATION_S = 0.5


@dataclass
class Window:
    flight: str
    flight_index: int
    start: int
    stop: int
    label: str


def hysteresis_airborne(altitude: np.ndarray) -> np.ndarray:
    """Ground/airborne classification from mocap altitude with hysteresis."""
    airborne = np.zeros(len(altitude), dtype=bool)
    state = altitude[0] > GROUND_ALTITUDE_M
    for k, alt in enumerate(altitude):
        if state and alt < GROUND_ALTITUDE_M - GROUND_HYSTERESIS_M:
            state = False
        elif not state and alt > GROUND_ALTITUDE_M + GROUND_HYSTERESIS_M:
            state = True
        airborne[k] = state
    return airborne


def contiguous_windows(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))
    starts = [0] if mask[0] else []
    starts += [int(e) + 1 for e in edges if not mask[e]]
    stops = [int(e) + 1 for e in edges if mask[e]]
    if mask[-1]:
        stops.append(len(mask))
    return list(zip(starts, stops))


def quality_ok(x: np.ndarray) -> bool:
    speed = np.linalg.norm(x[:, 3:6], axis=1)
    rates = np.abs(x[:, 10:13]).max()
    return bool(np.all(np.isfinite(x)) and speed.max() <= MAX_SPEED_MPS and rates <= MAX_RATE_RAD_S)


def find_manual_windows(flight: str, flight_index: int, x: np.ndarray, mode: np.ndarray, tracked: np.ndarray, dt: float) -> list[Window]:
    airborne = hysteresis_airborne(-x[:, 2])
    # Motion-capture dropout samples are interpolated, not measured; they end
    # any identification window rather than passing fabricated data through.
    manual = airborne & (mode == 0) & (tracked != 0)
    min_samples = int(round(MIN_MANUAL_DURATION_S / dt))
    windows = []
    for start, stop in contiguous_windows(manual):
        if stop - start < min_samples:
            continue
        if not quality_ok(x[start:stop]):
            print(f"  dropped {flight} [{start*dt:.1f},{stop*dt:.1f}]s: failed quality gate", flush=True)
            continue
        windows.append(Window(flight, flight_index, start, stop, "manual"))
    return windows


def ground_outlier_mask(x: np.ndarray, euler: np.ndarray, dt: float) -> np.ndarray:
    """Per-sample outlier flags for a rolling ground window.

    Floor disturbances (wheel debris, surface seams, handling contact) appear
    as spikes in planar acceleration or as attitude excursions impossible for
    a tricycle gear rolling on its wheels. Robust median/MAD statistics keep
    the detector from being tuned by the outliers themselves, and the flags
    are dilated so the transient shoulders are excluded along with the peak.
    """
    speed_xy = np.linalg.norm(x[:, 3:5], axis=1)
    # Debris strikes and seams knock the speed away from locally smooth
    # motion, so screen on the residual against a short moving-average fit.
    # Differentiation-based detectors (acceleration or jerk z-scores) fail
    # here: a takeoff roll legitimately ramps its acceleration, and segments
    # recorded at lower odometry rates interpolate to piecewise-linear speed
    # whose derivative spikes at the knots.
    window = max(3, int(round(0.25 / dt)) | 1)
    kernel = np.ones(window) / window
    pad = window // 2
    smooth = np.convolve(np.pad(speed_xy, pad, mode="edge"), kernel, mode="valid")
    residual = speed_xy - smooth
    median = np.median(residual)
    mad = np.median(np.abs(residual - median))
    # Rolling contact has a small-amplitude periodic texture (wheel and floor
    # seams) that a pure z-score flags wholesale once the MAD is tiny; a real
    # debris strike or handling event also clears an absolute magnitude floor.
    threshold = max(GROUND_OUTLIER_Z * mad / 0.6745, GROUND_OUTLIER_FLOOR_MPS)
    spike = np.abs(residual - median) > threshold
    # Attitude gate on deviation from the window median: a taildragger rests
    # nose-up, so absolute pitch is not an anomaly — departures from the
    # rolling attitude are.
    attitude_dev = np.abs(euler[:, :2] - np.median(euler[:, :2], axis=0))
    attitude = attitude_dev.max(axis=1) > GROUND_MAX_ATTITUDE_RAD
    # The taildragger gear tends to ground-loop: a spin-out is a yaw-rate burst
    # whose wheel-scrub forces invalidate the straight-rolling friction model.
    spinout = np.abs(x[:, 12]) > GROUND_MAX_YAW_RATE_RAD_S
    outlier = spike | attitude | spinout
    dilate = max(1, int(round(GROUND_OUTLIER_DILATE_S / dt)))
    padded = np.convolve(outlier.astype(float), np.ones(2 * dilate + 1), mode="same")
    return padded > 0.0


def find_ground_windows(flight: str, flight_index: int, x: np.ndarray, u: np.ndarray, tracked: np.ndarray, dt: float) -> list[Window]:
    """Rolling ground windows classified as takeoff roll, landing rollout, or taxi.

    The takeoff roll is the primary friction/thrust identification window: it
    starts where the pilot advances the throttle, the aircraft accelerates
    under a known input, and it ends at the instant of liftoff (the
    ground-to-airborne transition). Landing rollouts and taxi are kept as
    separate classes.
    """
    on_ground = ~hysteresis_airborne(-x[:, 2]) & (tracked != 0)
    min_samples = int(round(MIN_GROUND_DURATION_S / dt))
    windows = []
    for start, stop in contiguous_windows(on_ground):
        if stop - start < min_samples:
            continue
        speed_xy = np.linalg.norm(x[start:stop, 3:5], axis=1)
        if speed_xy.max() < MIN_ROLLING_SPEED_MPS:
            continue  # parked or being carried slowly; no friction information
        lifts_off = stop < len(x)  # followed by airborne samples
        lands = start > 0  # preceded by airborne samples
        throttle = u[start:stop, 0]
        if lifts_off and throttle[-min(20, stop - start):].mean() > 0.3:
            # Anchor the takeoff roll at throttle onset: the last quiet sample
            # before the throttle climbs past its pre-roll level.
            baseline = np.median(throttle[: max(5, min_samples // 2)])
            advanced = np.flatnonzero(throttle > baseline + 0.1)
            roll_start = start + (int(advanced[0]) if advanced.size else 0)
            if stop - roll_start >= min_samples // 2:
                windows.append(Window(flight, flight_index, roll_start, stop, "takeoff_roll"))
            continue
        label = "landing_rollout" if lands else "taxi"
        windows.append(Window(flight, flight_index, start, stop, label))
    return windows


def find_ground_effect_windows(flight: str, flight_index: int, x: np.ndarray, dt: float) -> list[Window]:
    """Airborne passes below about one wingspan: a (small) ground-effect set.

    Climb-out and approach are the only times the aircraft flies this low, so
    the yield is minimal, but tagging the windows now makes the data queryable
    for a future ground-effect lift/drag increment study.
    """
    airborne = hysteresis_airborne(-x[:, 2])
    low = airborne & (-x[:, 2] < GROUND_EFFECT_ALTITUDE_M)
    min_samples = int(round(MIN_GROUND_EFFECT_DURATION_S / dt))
    return [
        Window(flight, flight_index, start, stop, "ground_effect")
        for start, stop in contiguous_windows(low)
        if stop - start >= min_samples
    ]


def export_ground(
    data: np.lib.npyio.NpzFile,
    windows: list[Window],
    dt: float,
) -> dict[str, np.ndarray]:
    max_len = max(window.stop - window.start for window in windows)
    n = len(windows)

    def cut(key: str, width: int) -> np.ndarray:
        out = np.full((n, max_len, width), np.nan)
        for index, window in enumerate(windows):
            out[index, : window.stop - window.start] = data[key][window.flight_index, window.start : window.stop]
        return out

    run_offsets = data["session_offset_s"] if "session_offset_s" in data.files else np.zeros(len(data["segment_names"]))
    time_s = np.full((n, max_len), np.nan)
    valid_mask = np.zeros((n, max_len), dtype=bool)
    outlier_mask = np.zeros((n, max_len), dtype=bool)
    session_offset = np.zeros(n)
    names, kinds = [], []
    for index, window in enumerate(windows):
        length = window.stop - window.start
        time_s[index, :length] = np.arange(length) * dt
        session_offset[index] = float(run_offsets[window.flight_index]) + window.start * dt
        valid_mask[index, :length] = True
        x = data["x_meas"][window.flight_index, window.start : window.stop]
        euler = data["euler_meas"][window.flight_index, window.start : window.stop]
        if window.label == "ground_effect":
            # Airborne: the rolling-contact outlier model does not apply.
            outlier_mask[index, :length] = False
        else:
            outlier_mask[index, :length] = ground_outlier_mask(x, euler, dt)
        names.append(f"{window.flight}__{window.label}_{window.start}")
        kinds.append(window.label)

    x = cut("x_meas", 13)
    return {
        "time_s": time_s,
        "session_offset_s": session_offset,
        "valid_mask": valid_mask,
        "outlier_mask": outlier_mask,
        "window_kinds": np.asarray(kinds),
        "x_meas": x,
        "y_meas": x,
        "direct_state_meas": x,
        "mocap_meas": cut("mocap_meas", 7),
        "pose_meas": cut("pose_meas", 7),
        "u_cmd": cut("u_cmd", 4),
        "control_meas": cut("control_meas", 4),
        "control_pwm": cut("control_pwm", 5),
        "euler_meas": cut("euler_meas", 3),
        "segment_names": np.asarray(names),
        "state_names": data["state_names"],
        "direct_state_names": data["direct_state_names"],
        "input_names": data["input_names"],
        "control_names": data["control_names"],
        "control_pwm_names": data["control_pwm_names"],
        "pose_names": data["pose_names"],
        "mocap_names": data["mocap_names"],
        "euler_names": data["euler_names"],
        "truth_available": np.asarray(False),
        "system_dof": np.asarray(6),
        "dataset_id": np.asarray(DATASET_ID),
        "split_name": np.asarray("ground"),
        "sample_period_s": data["sample_period_s"],
        "format_version": np.asarray("sysid.timeseries.ragged.v1"),
    }


def per_flight_trim(x: np.ndarray, u: np.ndarray, windows: list[Window], dt: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Pool quasi-steady manual samples into a bare-airframe trim estimate.

    Stabilized samples cannot be used: the hidden inner loop adds surface
    corrections the stick channels do not show, so only manual samples with
    low angular rate and low stick rate represent the trimmed bare airframe.
    """
    states, inputs = [], []
    for window in windows:
        xs = x[window.start : window.stop]
        us = u[window.start : window.stop]
        rate_ok = np.linalg.norm(xs[:, 10:13], axis=1) < QUASI_STEADY_RATE_RAD_S
        stick_rate = np.linalg.norm(np.gradient(us, dt, axis=0), axis=1)
        steady = rate_ok & (stick_rate < QUASI_STEADY_STICK_RATE)
        states.append(xs[steady])
        inputs.append(us[steady])
    states = np.concatenate(states) if states else np.empty((0, x.shape[1]))
    inputs = np.concatenate(inputs) if inputs else np.empty((0, u.shape[1]))
    if len(states) == 0:
        return np.full(x.shape[1], np.nan), np.full(u.shape[1], np.nan), 0
    trim_state = np.median(states, axis=0)
    trim_state[6:10] /= max(np.linalg.norm(trim_state[6:10]), 1e-12)
    return trim_state, np.median(inputs, axis=0), int(len(states))


def assign_splits(windows: list[Window]) -> dict[str, list[Window]]:
    """Round-robin assignment per flight so both splits span multiple flights."""
    train: list[Window] = []
    validation: list[Window] = []
    by_flight: dict[str, list[Window]] = {}
    for window in windows:
        by_flight.setdefault(window.flight, []).append(window)
    for flight_windows in by_flight.values():
        period = max(2, int(round(1.0 / VALIDATION_FRACTION)))
        for index, window in enumerate(flight_windows):
            if len(flight_windows) >= 2 and index % period == period - 1:
                validation.append(window)
            else:
                train.append(window)
    return {"train": train, "validation": validation}


def export_split(
    data: np.lib.npyio.NpzFile,
    windows: list[Window],
    trims: dict[str, tuple[np.ndarray, np.ndarray, int]],
    split_name: str,
    dt: float,
) -> dict[str, np.ndarray]:
    max_len = max(window.stop - window.start for window in windows)
    n = len(windows)

    def cut(key: str, width: int) -> np.ndarray:
        out = np.full((n, max_len, width), np.nan)
        for index, window in enumerate(windows):
            out[index, : window.stop - window.start] = data[key][window.flight_index, window.start : window.stop]
        return out

    run_offsets = data["session_offset_s"] if "session_offset_s" in data.files else np.zeros(len(data["segment_names"]))

    def cut_tracked() -> np.ndarray:
        out = np.zeros((n, max_len), dtype=np.int8)
        for index, window in enumerate(windows):
            if "mocap_tracked" in data.files:
                out[index, : window.stop - window.start] = data["mocap_tracked"][window.flight_index, window.start : window.stop]
            else:
                out[index, : window.stop - window.start] = 1
        return out

    time_s = np.full((n, max_len), np.nan)
    valid_mask = np.zeros((n, max_len), dtype=bool)
    flight_mode = np.full((n, max_len), -1, dtype=np.int8)
    session_offset = np.zeros(n)
    names, trim_states, trim_inputs = [], [], []
    for index, window in enumerate(windows):
        length = window.stop - window.start
        time_s[index, :length] = np.arange(length) * dt
        valid_mask[index, :length] = True
        flight_mode[index, :length] = data["flight_mode"][window.flight_index, window.start : window.stop]
        session_offset[index] = float(run_offsets[window.flight_index]) + window.start * dt
        names.append(f"{window.flight}__manual_{window.start}")
        trim_state, trim_input, _count = trims[window.flight]
        trim_states.append(trim_state)
        trim_inputs.append(trim_input)

    x = cut("x_meas", 13)
    return {
        "time_s": time_s,
        "session_offset_s": session_offset,
        "valid_mask": valid_mask,
        "x_meas": x,
        "y_meas": x,
        "direct_state_meas": x,
        "mocap_meas": cut("mocap_meas", 7),
        "pose_meas": cut("pose_meas", 7),
        "u_cmd": cut("u_cmd", 4),
        "control_meas": cut("control_meas", 4),
        "control_pwm": cut("control_pwm", 5),
        "euler_meas": cut("euler_meas", 3),
        "flight_mode": flight_mode,
        "mocap_tracked": cut_tracked(),
        "trim_state": np.asarray(trim_states),
        "trim_controls": np.asarray(trim_inputs),
        "segment_names": np.asarray(names),
        "state_names": data["state_names"],
        "direct_state_names": data["direct_state_names"],
        "input_names": data["input_names"],
        "control_names": data["control_names"],
        "control_pwm_names": data["control_pwm_names"],
        "pose_names": data["pose_names"],
        "mocap_names": data["mocap_names"],
        "euler_names": data["euler_names"],
        "truth_available": np.asarray(False),
        "system_dof": np.asarray(6),
        "dataset_id": np.asarray(DATASET_ID),
        "split_name": np.asarray(split_name),
        "sample_period_s": data["sample_period_s"],
        "format_version": np.asarray("sysid.timeseries.ragged.v1"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flights", type=Path, default=FLIGHTS_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = np.load(args.flights, allow_pickle=False)
    dt = float(data["sample_period_s"])
    flights = [str(s) for s in data["segment_names"]]

    all_windows: list[Window] = []
    ground_windows: list[Window] = []
    trims: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for flight_index, flight in enumerate(flights):
        mask = data["valid_mask"][flight_index]
        x = data["x_meas"][flight_index][mask]
        u = data["u_cmd"][flight_index][mask]
        mode = data["flight_mode"][flight_index][mask]
        tracked = (
            data["mocap_tracked"][flight_index][mask]
            if "mocap_tracked" in data.files
            else np.ones(len(x), dtype=np.int8)
        )
        windows = find_manual_windows(flight, flight_index, x, mode, tracked, dt)
        rolling = find_ground_windows(flight, flight_index, x, u, tracked, dt)
        effect = find_ground_effect_windows(flight, flight_index, x, dt)
        ground_windows.extend(rolling)
        ground_windows.extend(effect)
        trim_state, trim_input, steady_count = per_flight_trim(x, u, windows, dt)
        trims[flight] = (trim_state, trim_input, steady_count)
        airborne = hysteresis_airborne(-x[:, 2])
        kinds = ", ".join(f"{w.label}@{w.start*dt:.0f}s" for w in rolling) or "none"
        print(
            f"{flight}: airborne {airborne.mean()*100:.0f}%, manual windows kept "
            f"{len(windows)} ({sum((w.stop-w.start) for w in windows)*dt:.1f} s), "
            f"ground [{kinds}], ground-effect windows {len(effect)}, "
            f"steady trim samples {steady_count}",
            flush=True,
        )
        if steady_count:
            v_trim = float(np.linalg.norm(trim_state[3:6]))
            print(
                f"  trim: V={v_trim:.2f} m/s, controls (thr,elev,ail,rud) = "
                + ", ".join(f"{value:+.3f}" for value in trim_input),
                flush=True,
            )
        all_windows.extend(windows)

    splits = assign_splits(all_windows)
    for split_name, windows in splits.items():
        if not windows:
            raise SystemExit(f"no {split_name} windows survived segmentation")
        payload = export_split(data, windows, trims, split_name, dt)
        output = args.output_dir / f"{DATASET_ID}_{split_name}.npz"
        np.savez_compressed(output, **payload)
        print(f"wrote {output} ({len(windows)} segments, {output.stat().st_size/1e6:.1f} MB)")
    if ground_windows:
        payload = export_ground(data, ground_windows, dt)
        output = args.output_dir / f"{DATASET_ID}_ground.npz"
        np.savez_compressed(output, **payload)
        flagged = float(payload["outlier_mask"][payload["valid_mask"]].mean()) * 100.0
        print(f"wrote {output} ({len(ground_windows)} rolling windows, {flagged:.1f}% samples outlier-flagged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
