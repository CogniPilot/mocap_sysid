"""Flight-record segmentation shared by the explorer export and SAFE tooling.

Owns the per-sample flight-phase labeling (ground / ground effect /
stabilized / manual), the stabilized fitting/validation window cuts, and the
autonomous-record marker, so the website exporter and the SAFE controller
identification agree exactly on which samples are stabilized flight.
"""

from __future__ import annotations

import numpy as np

GROUND_ALTITUDE_M = 0.5
GROUND_EFFECT_ALTITUDE_M = 0.65
LABELS = ("ground", "ground_effect", "stabilized", "manual")

STABILIZED_WINDOW_S = 10.0
STABILIZED_MIN_WINDOW_S = 5.0
STABILIZED_VALIDATION_PERIOD = 3  # every third window held out, like manual


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


def stabilized_windows(labels: np.ndarray, tracked: np.ndarray, dt: float) -> list[tuple[int, int]]:
    """Cut the tracked stabilized spans into fitting/validation windows."""
    keep = (labels == 2) & tracked
    edges = np.flatnonzero(np.diff(keep.astype(np.int8)))
    bounds = [0, *(edges + 1).tolist(), len(keep)]
    windows: list[tuple[int, int]] = []
    span = int(round(STABILIZED_WINDOW_S / dt))
    minimum = int(round(STABILIZED_MIN_WINDOW_S / dt))
    for left, right in zip(bounds[:-1], bounds[1:]):
        if not keep[left]:
            continue
        start = left
        while right - start >= minimum:
            stop = min(start + span, right)
            windows.append((start, stop))
            start = stop
    return windows


def split_stabilized_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
    """Round-robin train/validation assignment shared with the manual splits."""
    out = []
    for index, (a, b) in enumerate(windows):
        held = len(windows) >= 2 and index % STABILIZED_VALIDATION_PERIOD == STABILIZED_VALIDATION_PERIOD - 1
        out.append((a, b, "validation" if held else "train"))
    return out


def is_autonomous(name: str) -> bool:
    """Whether a flight record was flown by the offboard autopilot.

    The autopilot's lateral commands never reach the recorded transmitter
    channels (the aileron stick barely moves and the rudder is constant while
    the aircraft banks through laps), so stick-driven identification and
    prediction are unsound for these records. A stick-variance test cannot
    separate them from piloted elevator-only flights (the autopilot passes
    pitch commands through the elevator channel), so the recording name is
    the authoritative marker.
    """
    return name.startswith("auto")


def euler_from_quat_array(quat: np.ndarray) -> np.ndarray:
    q = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12)
    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
    pitch = np.arcsin(np.clip(2 * (q0 * q2 - q3 * q1), -1.0, 1.0))
    yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    return np.column_stack([roll, pitch, yaw])
