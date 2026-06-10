#!/usr/bin/env python3
"""Convert the 2026-05-22 Sport Cub rosbag (MCAP) flights to compressed NPZ.

The raw recordings under ``data/data_20260522`` are full piloted laps: the
aircraft flies a stabilized circuit (SAFE inner loop on), the pilot switches
the inner loop off mid-facility, performs a maneuver, and re-enables
stabilization. Each rosbag becomes one ragged segment in the canonical
``sysid.timeseries.ragged.v1`` layout used by the other repository datasets,
with two additions:

- ``flight_mode``: 1 where the SAFE inner-loop stabilization is enabled,
  0 where the pilot is flying the bare airframe (from transmitter channel 5).
- ``control_pwm``: the five raw transmitter PWM channels echoed over serial
  (throttle, elevator, aileron, rudder, mode), zero-order-held to the grid.

Commands are reconstructed from the 53 Hz ``/cub1/joy`` stick stream using the
fixed transmitter calibration (pwm = 1500 +/- 500 * axis), verified exact
(|r| > 0.999) against the 8 Hz serial echo on every manually flown bag. For
the autonomous bag the serial echo and the stick stream can disagree while the
autopilot has authority, so both channels are stored.

States are derived from the Qualisys odometry: position is converted from the
ENU mocap frame to NED, the rigid-body attitude is converted from the ROS
front-left-up body convention to the aircraft front-right-down convention,
and body velocity/angular rate are obtained by smoothed differentiation of
the 100 Hz resampled pose, mirroring the conventions of the existing
``sportcub_mocap_4_17_26`` pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from mcap_ros2.reader import read_ros2_messages
except ImportError as error:  # pragma: no cover
    raise SystemExit("convert.py requires the 'mcap' and 'mcap-ros2-support' packages") from error

DATASET_ID = "sportcub_mocap_5_22_26"
RAW_DEFAULT = Path("data/data_20260522")
OUTPUT_DEFAULT = Path("data") / f"{DATASET_ID}_flights.npz"
SAMPLE_RATE_HZ = 240.0  # native Qualisys odometry rate (the auto bag streams 100 Hz)

STATE_NAMES = ("x_n", "y_e", "z_d", "u", "v", "w", "q_w", "q_x", "q_y", "q_z", "p", "q", "r")
INPUT_NAMES = ("throttle", "elevator", "aileron", "rudder")
CONTROL_NAMES = ("thrust", "aileron", "elevator", "rudder")
CONTROL_PWM_NAMES = ("throttle_pwm", "elevator_pwm", "aileron_pwm", "rudder_pwm", "mode_pwm")
MOCAP_NAMES = ("x_n", "y_e", "z_d", "q_w", "q_x", "q_y", "q_z")
POSE_NAMES = ("x_e", "y_n", "z_u", "q_w", "q_x", "q_y", "q_z")
EULER_NAMES = ("phi", "theta", "psi")

# serial channel -> (joy axis, gain): pwm = 1500 + gain * axis
JOY_TO_PWM = {
    0: (1, +500.0),  # throttle
    1: (2, -500.0),  # elevator
    2: (0, -500.0),  # aileron
    3: (3, +500.0),  # rudder
    4: (4, +500.0),  # flight mode switch
}
MODE_STABILIZED_PWM = 1500.0  # > threshold means SAFE inner loop enabled

ENU_TO_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
FLU_TO_FRD = np.diag([1.0, -1.0, -1.0])


def quat_to_rotation(q: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = q / max(np.linalg.norm(q), 1e-12)
    return np.array(
        [
            [1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
            [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1**2 + q2**2)],
        ]
    )


def quat_from_rotation(matrix: np.ndarray) -> np.ndarray:
    trace = np.trace(matrix)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array(
            [0.25 / s, (matrix[2, 1] - matrix[1, 2]) * s, (matrix[0, 2] - matrix[2, 0]) * s, (matrix[1, 0] - matrix[0, 1]) * s]
        )
    i = int(np.argmax(np.diag(matrix)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(max(1.0 + matrix[i, i] - matrix[j, j] - matrix[k, k], 1e-12))
    quat = np.empty(4)
    quat[0] = (matrix[k, j] - matrix[j, k]) / s
    quat[1 + i] = 0.25 * s
    quat[1 + j] = (matrix[j, i] + matrix[i, j]) / s
    quat[1 + k] = (matrix[k, i] + matrix[i, k]) / s
    return quat / max(np.linalg.norm(quat), 1e-12)


def euler_from_quat(q: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = q
    roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    pitch = np.arcsin(np.clip(2 * (q0 * q2 - q3 * q1), -1.0, 1.0))
    yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
    return np.array([roll, pitch, yaw])


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(values, dtype=float)
    for dim in range(values.shape[1]):
        out[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return out


def interp_columns(t_out: np.ndarray, t_in: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.empty((len(t_out), values.shape[1]))
    for dim in range(values.shape[1]):
        out[:, dim] = np.interp(t_out, t_in, values[:, dim])
    return out


def read_bag(mcap_path: Path) -> dict[str, np.ndarray]:
    odo_t, pos, quat = [], [], []
    joy_t, axes = [], []
    ser_t, ser = [], []
    for msg in read_ros2_messages(str(mcap_path)):
        topic = msg.channel.topic
        stamp = msg.log_time_ns / 1e9
        if topic == "/cub1/odom":
            p = msg.ros_msg.pose.pose
            odo_t.append(stamp)
            pos.append([p.position.x, p.position.y, p.position.z])
            quat.append([p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z])
        elif topic == "/cub1/joy":
            joy_t.append(stamp)
            axes.append(list(msg.ros_msg.axes))
        elif topic == "/cub1/joy_serial_status":
            ser_t.append(stamp)
            ser.append(list(msg.ros_msg.data))
    return {
        "start_epoch_s": float(min(odo_t[0], joy_t[0])) if odo_t and joy_t else 0.0,
        "odo_t": np.asarray(odo_t),
        "pos_enu": np.asarray(pos),
        "quat_flu": np.asarray(quat),
        "joy_t": np.asarray(joy_t),
        "axes": np.asarray(axes),
        "ser_t": np.asarray(ser_t),
        "ser": np.asarray(ser, dtype=float),
    }


def convert_run(bag: dict[str, np.ndarray], name: str) -> dict[str, np.ndarray | str]:
    dt = 1.0 / SAMPLE_RATE_HZ
    t0 = max(bag["odo_t"][0], bag["joy_t"][0])
    t1 = min(bag["odo_t"][-1], bag["joy_t"][-1])
    time_s = np.arange(0.0, t1 - t0, dt)
    grid = time_s + t0

    # Motion capture drops out; resampling interpolates straight lines across
    # the gaps, which is fabricated data. Mark each grid sample as tracked
    # only when a real odometry sample is nearby, and dilate the invalid
    # region so smoothing/differentiation contamination at gap edges is also
    # excluded. Consumers must not train or score on untracked samples.
    nearest = np.clip(np.searchsorted(bag["odo_t"], grid), 1, len(bag["odo_t"]) - 1)
    distance = np.minimum(np.abs(grid - bag["odo_t"][nearest - 1]), np.abs(bag["odo_t"][nearest] - grid))
    untracked = distance > 0.06
    dilate = int(round(0.31 * SAMPLE_RATE_HZ)) | 1
    untracked = np.convolve(untracked.astype(float), np.ones(dilate), mode="same") > 0.0
    tracked = (~untracked).astype(np.int8)

    pos_enu = interp_columns(grid, bag["odo_t"], bag["pos_enu"])
    quat_flu = interp_columns(grid, bag["odo_t"], bag["quat_flu"])
    quat_flu /= np.maximum(np.linalg.norm(quat_flu, axis=1, keepdims=True), 1e-12)

    pos_ned = pos_enu @ ENU_TO_NED.T
    quat_ned = np.empty_like(quat_flu)
    for k in range(len(grid)):
        rotation_frd_to_ned = ENU_TO_NED @ quat_to_rotation(quat_flu[k]) @ FLU_TO_FRD
        quat_ned[k] = quat_from_rotation(rotation_frd_to_ned)
        if k > 0 and np.dot(quat_ned[k], quat_ned[k - 1]) < 0.0:
            quat_ned[k] = -quat_ned[k]

    pos_smooth = moving_average(pos_ned, (int(round(0.11 * SAMPLE_RATE_HZ)) | 1))
    quat_smooth = moving_average(quat_ned, (int(round(0.07 * SAMPLE_RATE_HZ)) | 1))
    quat_smooth /= np.maximum(np.linalg.norm(quat_smooth, axis=1, keepdims=True), 1e-12)
    pos_dot = np.gradient(pos_smooth, dt, axis=0, edge_order=2)
    quat_dot = np.gradient(quat_smooth, dt, axis=0, edge_order=2)

    x = np.full((len(grid), len(STATE_NAMES)), np.nan)
    x[:, 0:3] = pos_smooth
    x[:, 6:10] = quat_smooth
    euler = np.empty((len(grid), 3))
    for k in range(len(grid)):
        rotation = quat_to_rotation(quat_smooth[k])
        x[k, 3:6] = rotation.T @ pos_dot[k]
        q0, q1, q2, q3 = quat_smooth[k]
        qmat = 0.5 * np.array([[-q1, -q2, -q3], [q0, -q3, q2], [q3, q0, -q1], [-q2, q1, q0]])
        x[k, 10:13] = np.linalg.lstsq(qmat, quat_dot[k], rcond=None)[0]
        euler[k] = euler_from_quat(quat_smooth[k])

    axes = interp_columns(grid, bag["joy_t"], bag["axes"])
    pwm_joy = np.empty((len(grid), 5))
    for serial_channel, (axis, gain) in JOY_TO_PWM.items():
        pwm_joy[:, serial_channel] = 1500.0 + gain * axes[:, axis]
    # The serial echo is authoritative (it is what reached the aircraft) but
    # only ticks at ~8 Hz; zero-order-hold it to the grid.
    ser_idx = np.clip(np.searchsorted(bag["ser_t"], grid, side="right") - 1, 0, len(bag["ser_t"]) - 1)
    pwm_serial = bag["ser"][ser_idx]
    flight_mode = (pwm_serial[:, 4] > MODE_STABILIZED_PWM).astype(np.int8)

    throttle = np.clip((pwm_joy[:, 0] - 1000.0) / 1000.0, 0.0, 1.0)
    surfaces = (pwm_joy[:, 1:4] - 1500.0) / 500.0  # elevator, aileron, rudder in [-1, 1]
    u_cmd = np.column_stack([throttle, surfaces])
    control_meas = u_cmd[:, [0, 2, 1, 3]]

    mocap = np.column_stack([pos_ned, quat_ned])
    # Canonical pose convention (matching sportcub_mocap_4_17_26): ENU
    # position with body-FRD-to-ENU attitude. The raw Qualisys quaternion is
    # body-FLU-to-ENU, so append the FLU/FRD axis flip; storing the raw
    # quaternion renders the aircraft rolled 180 degrees in the web viewer.
    quat_enu_frd = np.empty_like(quat_flu)
    for k in range(len(quat_flu)):
        quat_enu_frd[k] = quat_from_rotation(quat_to_rotation(quat_flu[k]) @ FLU_TO_FRD)
        if k > 0 and np.dot(quat_enu_frd[k], quat_enu_frd[k - 1]) < 0.0:
            quat_enu_frd[k] = -quat_enu_frd[k]
    pose = np.column_stack([pos_enu, quat_enu_frd])
    return {
        "start_epoch_s": float(bag["start_epoch_s"]),
        "name": name,
        "time_s": time_s,
        "x": x,
        "u_cmd": u_cmd,
        "control_meas": control_meas,
        "control_pwm": pwm_serial,
        "flight_mode": flight_mode,
        "mocap_tracked": tracked,
        "mocap": mocap,
        "pose": pose,
        "euler": euler,
    }


def stack_runs(runs: list[dict[str, np.ndarray | str]]) -> dict[str, np.ndarray]:
    n_runs = len(runs)
    max_len = max(len(run["time_s"]) for run in runs)

    def ragged(key: str, width: int | None) -> np.ndarray:
        shape = (n_runs, max_len) if width is None else (n_runs, max_len, width)
        out = np.full(shape, np.nan)
        for index, run in enumerate(runs):
            n = len(run["time_s"])
            out[index, :n] = run[key]
        return out

    valid_mask = np.zeros((n_runs, max_len), dtype=bool)
    flight_mode = np.full((n_runs, max_len), -1, dtype=np.int8)
    mocap_tracked = np.zeros((n_runs, max_len), dtype=np.int8)
    for index, run in enumerate(runs):
        n = len(run["time_s"])
        valid_mask[index, :n] = True
        flight_mode[index, :n] = run["flight_mode"]
        mocap_tracked[index, :n] = run["mocap_tracked"]

    # Session time supports battery-discharge modeling: the 1S pack visibly
    # loses prop power over the session, so consumers need to know where each
    # run sits in the session timeline, not just time within the run.
    session_start = min(float(run["start_epoch_s"]) for run in runs)
    session_offset_s = np.asarray([float(run["start_epoch_s"]) - session_start for run in runs])

    x = ragged("x", len(STATE_NAMES))
    return {
        "time_s": ragged("time_s", None),
        "session_offset_s": session_offset_s,
        "valid_mask": valid_mask,
        "x_meas": x,
        "y_meas": x,
        "direct_state_meas": x,
        "mocap_meas": ragged("mocap", len(MOCAP_NAMES)),
        "pose_meas": ragged("pose", len(POSE_NAMES)),
        "u_cmd": ragged("u_cmd", len(INPUT_NAMES)),
        "control_meas": ragged("control_meas", len(CONTROL_NAMES)),
        "control_pwm": ragged("control_pwm", len(CONTROL_PWM_NAMES)),
        "flight_mode": flight_mode,
        "mocap_tracked": mocap_tracked,
        "euler_meas": ragged("euler", len(EULER_NAMES)),
        "segment_names": np.asarray([str(run["name"]) for run in runs]),
        "state_names": np.asarray(STATE_NAMES),
        "direct_state_names": np.asarray(STATE_NAMES),
        "input_names": np.asarray(INPUT_NAMES),
        "control_names": np.asarray(CONTROL_NAMES),
        "control_pwm_names": np.asarray(CONTROL_PWM_NAMES),
        "pose_names": np.asarray(POSE_NAMES),
        "mocap_names": np.asarray(MOCAP_NAMES),
        "euler_names": np.asarray(EULER_NAMES),
        "truth_available": np.asarray(False),
        "system_dof": np.asarray(6),
        "dataset_id": np.asarray(DATASET_ID),
        "split_name": np.asarray("flights"),
        "sample_period_s": np.asarray(1.0 / SAMPLE_RATE_HZ),
        "format_version": np.asarray("sysid.timeseries.ragged.v1"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = []
    for run_dir in sorted(args.raw_root.iterdir()):
        mcaps = sorted(run_dir.glob("*.mcap"))
        if not mcaps:
            continue
        bag = read_bag(mcaps[0])
        run = convert_run(bag, run_dir.name)
        manual_s = float(np.sum(np.asarray(run["flight_mode"]) == 0)) / SAMPLE_RATE_HZ
        print(
            f"{run_dir.name}: {len(run['time_s'])} samples ({len(run['time_s']) / SAMPLE_RATE_HZ:.1f} s), "
            f"manual {manual_s:.1f} s",
            flush=True,
        )
        runs.append(run)
    if not runs:
        raise SystemExit(f"no rosbag runs found under {args.raw_root}")
    payload = stack_runs(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    size_mb = args.output.stat().st_size / 1e6
    print(f"wrote {args.output} ({size_mb:.1f} MB, {len(runs)} flights)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
