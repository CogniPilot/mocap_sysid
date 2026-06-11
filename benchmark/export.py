"""Export benchmark CSV results into website-ready JSON bundles."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .schema import METHOD_RESULT_FIELDS, MODEL_FAMILY_3DOF, MODEL_FAMILY_6DOF, SCHEMA_VERSION
from .registry import all_method_metadata, metadata_to_dict
from .scenarios import SCENARIOS_6DOF, SIX_DOF_SCENARIO_TITLES
from dataset_tools.registry import discover_manifests


NUMERIC_FIELDS = {
    "validation_score",
    "train_elapsed_s",
    "train_cpu_s",
    "train_gpu_s",
    "gpu_memory_mb",
    "rollout_elapsed_s",
    "total_elapsed_s",
    "train_loss_final",
    "decision_variables",
    "train_samples",
    "rmse_V",
    "rmse_alpha",
    "rmse_gamma",
    "rmse_Q",
    "rmse_position_m",
    "rmse_velocity_mps",
    "rmse_quaternion",
    "rmse_rates_rad_s",
    "rmse_mocap_position_m",
    "rmse_mocap_quaternion",
    "mocap_rmse_x_pos",
    "mocap_rmse_z_pos",
    "mocap_rmse_theta",
    "coeff_residual_rmse_C_L",
    "coeff_residual_rmse_C_D",
    "coeff_residual_rmse_C_M",
    "max_abs_alpha_deg",
    "max_abs_theta_deg",
    "min_speed_mps",
    "max_speed_mps",
    "vertical_extent_m",
}

NED_TO_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def _rotation_body_to_inertial(quat_wxyz: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = quat_wxyz / max(float(np.linalg.norm(quat_wxyz)), 1e-12)
    return np.array(
        [
            [1.0 - 2.0 * (q2**2 + q3**2), 2.0 * (q1 * q2 - q0 * q3), 2.0 * (q1 * q3 + q0 * q2)],
            [2.0 * (q1 * q2 + q0 * q3), 1.0 - 2.0 * (q1**2 + q3**2), 2.0 * (q2 * q3 - q0 * q1)],
            [2.0 * (q1 * q3 - q0 * q2), 2.0 * (q2 * q3 + q0 * q1), 1.0 - 2.0 * (q1**2 + q2**2)],
        ]
    )


def _pose_ned_to_enu(pose_ned: np.ndarray) -> np.ndarray:
    pose = np.empty_like(pose_ned)
    pose[:, 0] = pose_ned[:, 1]
    pose[:, 1] = pose_ned[:, 0]
    pose[:, 2] = -pose_ned[:, 2]
    if pose_ned.shape[1] < 7:
        return pose
    for index, quat in enumerate(pose_ned[:, 3:7]):
        pose[index, 3:7] = _quat_wxyz_from_rotation(NED_TO_ENU @ _rotation_body_to_inertial(quat))
    return pose


def _git_sha(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _coerce_value(key: str, value: str | None) -> Any:
    if value is None:
        return None
    text = value.strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    if key in NUMERIC_FIELDS:
        try:
            number = float(text)
        except ValueError:
            return text
        if not math.isfinite(number):
            return None
        if key in {"decision_variables", "train_samples"} and number.is_integer():
            return int(number)
        return number
    return text


def _method_rows(
    results_dir: Path,
    dataset_modes: tuple[str, ...],
    dataset_titles: dict[str, str],
    method_training_modes: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in dataset_modes:
        path = results_dir / f"{scenario}_shared_method_comparison.csv"
        for raw in _read_csv(path):
            method = raw.get("method", "")
            clean_method = method.removesuffix(" (mocap)")
            record = {field: _coerce_value(field, raw.get(field)) for field in METHOD_RESULT_FIELDS}
            record["scenario"] = scenario
            record["scenario_title"] = dataset_titles.get(scenario, scenario.replace("_", " ").title())
            record["model_family"] = MODEL_FAMILY_3DOF
            record["method"] = method
            record["training_scenario"] = method_training_modes.get(clean_method)
            rows.append(record)
    rows.sort(
        key=lambda row: (
            str(row.get("scenario") or ""),
            str(row.get("state_source") or ""),
            float(row.get("validation_score") or math.inf),
            str(row.get("method") or ""),
        )
    )
    return rows


def _six_dof_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in _read_csv(results_dir / "aircraft6dof_method_comparison.csv"):
        record = {field: _coerce_value(field, raw.get(field)) for field in METHOD_RESULT_FIELDS}
        for key, value in raw.items():
            if key not in record:
                record[key] = _coerce_value(key, value)
        scenario = str(record.get("validation_scenario") or record.get("scenario") or "aircraft_6dof_aggressive")
        record["scenario"] = scenario
        record["scenario_title"] = SIX_DOF_SCENARIO_TITLES.get(scenario, scenario.replace("_", " ").title())
        record["model_family"] = MODEL_FAMILY_6DOF
        record["training_scenario"] = raw.get("training_scenario") or record.get("training_scenario") or "aircraft_6dof_aggressive"
        record["validation_scenario"] = scenario
        rows.append(record)
    return rows


def _real_dataset_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*_method_comparison.csv")):
        if path.name == "aircraft6dof_method_comparison.csv" or path.name.endswith("_shared_method_comparison.csv"):
            continue
        for raw in _read_csv(path):
            record = {field: _coerce_value(field, raw.get(field)) for field in METHOD_RESULT_FIELDS}
            for key, value in raw.items():
                if key not in record:
                    record[key] = _coerce_value(key, value)
            scenario = str(record.get("scenario") or path.stem.removesuffix("_method_comparison"))
            record["scenario"] = scenario
            record["scenario_title"] = record.get("scenario_title") or scenario.replace("_", " ").title()
            rows.append(record)
    return rows


def _dedupe_method_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the last row for each plotted method key.

    Dataset-specific result files are appended after aggregate files, so this
    lets refreshed real-data rows override stale aggregate rows without
    duplicating website table entries.
    """
    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str, str]] = []
    for row in rows:
        key = (
            str(row.get("model_family") or ""),
            str(row.get("scenario") or ""),
            str(row.get("state_source") or ""),
            str(row.get("method") or ""),
        )
        if key not in by_key:
            order.append(key)
        by_key[key] = row
    return [by_key[key] for key in order]


def _maneuver_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in _read_csv(results_dir / "benchmark_maneuver_summary.csv"):
        rows.append({key: _coerce_value(key, value) for key, value in raw.items()})
    return rows


def _quaternion_to_euler_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = quat_wxyz / max(float(np.linalg.norm(quat_wxyz)), 1e-12)
    roll = math.atan2(2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * (q1**2 + q2**2))
    pitch = math.asin(float(np.clip(2.0 * (q0 * q2 - q3 * q1), -1.0, 1.0)))
    yaw = math.atan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2**2 + q3**2))
    return np.rad2deg([roll, pitch, yaw])


def _playback_maneuver_rows(playback_rows: list[dict[str, Any]], existing_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing_modes = {str(row.get("mode")) for row in existing_rows}
    rows: list[dict[str, Any]] = []
    for track in playback_rows:
        title = str(track.get("title") or track.get("id") or "")
        if not title or title in existing_modes:
            continue
        positions: list[np.ndarray] = []
        speeds: list[np.ndarray] = []
        pitch_abs: list[np.ndarray] = []
        segments = track.get("segments") if isinstance(track.get("segments"), list) else [track]
        for segment in segments:
            if not isinstance(segment, dict) or not segment.get("position_enu_m"):
                continue
            time = np.asarray(segment.get("time_s", []), dtype=float)
            position = np.asarray(segment["position_enu_m"], dtype=float)
            if position.ndim != 2 or position.shape[0] < 2:
                continue
            positions.append(position)
            if len(time) == len(position) and len(time) > 1:
                velocity = np.gradient(position, time, axis=0)
                speeds.append(np.linalg.norm(velocity, axis=1))
            quat = np.asarray(segment.get("quaternion_wxyz", []), dtype=float)
            if quat.ndim == 2 and quat.shape[1] == 4:
                euler = np.asarray([_quaternion_to_euler_deg(row) for row in quat])
                pitch_abs.append(np.abs(euler[:, 1]))
        if not positions:
            continue
        all_position = np.concatenate(positions, axis=0)
        all_speeds = np.concatenate(speeds) if speeds else np.asarray([], dtype=float)
        all_pitch = np.concatenate(pitch_abs) if pitch_abs else np.asarray([], dtype=float)
        rows.append(
            {
                "mode": title,
                "max_abs_alpha_deg": None,
                "max_abs_theta_deg": float(np.nanmax(all_pitch)) if all_pitch.size else None,
                "min_speed_mps": float(np.nanmin(all_speeds)) if all_speeds.size else None,
                "max_speed_mps": float(np.nanmax(all_speeds)) if all_speeds.size else None,
                "vertical_extent_m": float(np.nanmax(all_position[:, 2]) - np.nanmin(all_position[:, 2])),
            }
        )
    return rows


def _generated_dataset_registry(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS_6DOF:
        local_files: dict[str, str] = {}
        train_path = scenario.default_path / "train.npz"
        validation_path = scenario.default_path / "validation.npz"
        if train_path.exists():
            local_files["train"] = str(train_path.relative_to(root))
        if validation_path.exists():
            local_files["validation"] = str(validation_path.relative_to(root))
        rows.append(
            {
                "id": scenario.id,
                "title": scenario.title,
                "status": "generated",
                "model_family": scenario.model_family,
                "source_type": "synthetic_simulation",
                "observation_type": "direct_state_and_pose",
                "local_data_dir": str(scenario.default_path.relative_to(root)),
                "generator": scenario.generator or (
                    "models.aircraft6dof.generate_dataset"
                    if scenario.model_family == MODEL_FAMILY_6DOF
                    else "dataset_tools.synthetic_3dof.compact"
                ),
                "local_data_files": local_files,
                "tags": list(scenario.tags),
            }
        )
    return rows


def _quat_wxyz_from_rotation(matrix: np.ndarray) -> np.ndarray:
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


def _quat_from_frd_angles(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    quat = np.empty((len(pitch), 4), dtype=float)
    for index, (phi, theta, psi) in enumerate(zip(roll, pitch, yaw, strict=True)):
        forward = np.array([np.cos(theta) * np.cos(psi), np.cos(theta) * np.sin(psi), np.sin(theta)])
        forward /= max(float(np.linalg.norm(forward)), 1e-12)
        right_level = np.array([np.sin(psi), -np.cos(psi), 0.0])
        right_level /= max(float(np.linalg.norm(right_level)), 1e-12)
        down_level = np.cross(forward, right_level)
        down_level /= max(float(np.linalg.norm(down_level)), 1e-12)
        right = right_level * np.cos(phi) + down_level * np.sin(phi)
        right /= max(float(np.linalg.norm(right)), 1e-12)
        down = np.cross(forward, right)
        down /= max(float(np.linalg.norm(down)), 1e-12)
        quat[index] = _quat_wxyz_from_rotation(np.column_stack([forward, right, down]))
    return quat


def _procedural_playback(scenario_id: str, model_family: str, title: str) -> dict[str, Any]:
    count = 180
    time_s = np.linspace(0.0, 18.0, count)
    phase = np.linspace(0.0, 2.0 * np.pi, count)
    if "sine_sweep" in scenario_id:
        amp = np.linspace(0.2, 1.0, count)
        y = 2.0 * np.sin(phase) * amp
        z = 0.7 * np.sin(2.5 * phase) * amp
        pitch = np.deg2rad(8.0) * np.sin(2.5 * phase) * amp
        roll = np.deg2rad(18.0) * np.sin(phase) * amp
    elif "trim_grid" in scenario_id:
        y = 0.5 * np.sin(phase)
        z = 0.3 * np.sin(2.0 * phase)
        pitch = np.deg2rad(3.0) * np.sin(2.0 * phase)
        roll = np.deg2rad(5.0) * np.sin(phase)
    elif "aggressive" in scenario_id or "safe_loop" in scenario_id:
        y = 2.7 * np.sin(phase)
        z = 1.5 * np.sin(2.0 * phase)
        pitch = np.deg2rad(22.0) * np.sin(2.0 * phase)
        roll = np.deg2rad(34.0) * np.sin(phase)
    else:
        y = 1.0 * np.sin(phase)
        z = 0.45 * np.sin(1.5 * phase)
        pitch = np.deg2rad(6.0) * np.sin(1.5 * phase)
        roll = np.deg2rad(8.0) * np.sin(phase)
    x = np.linspace(0.0, 12.0 if model_family == MODEL_FAMILY_6DOF else 8.0, count)
    yaw = np.arctan2(np.gradient(y), np.gradient(x))
    quat = _quat_from_frd_angles(roll, pitch, yaw)
    thrust = 0.55 + 0.15 * np.sin(phase + 0.3)
    aileron = np.clip(roll / np.deg2rad(35.0), -1.0, 1.0)
    elevator = np.clip(-pitch / np.deg2rad(25.0), -1.0, 1.0)
    rudder = np.clip(0.35 * np.sin(phase + 0.8), -1.0, 1.0)
    segment = {
        "name": "generated_preview",
        "time_s": np.round(time_s, 4).tolist(),
        "position_enu_m": np.round(np.column_stack([x, y, z]), 5).tolist(),
        "quaternion_wxyz": np.round(quat, 7).tolist(),
        "control_meas": np.round(np.column_stack([thrust, aileron, elevator, rudder]), 5).tolist(),
    }
    return {
        "id": scenario_id,
        "title": title,
        "model_family": model_family,
        "source": "generated_preview",
        "control_names": ["thrust", "aileron", "elevator", "rudder"],
        "pose_names": ["x_e", "y_n", "z_u", "q_w", "q_x", "q_y", "q_z"],
        "segments": [segment],
        **{key: value for key, value in segment.items() if key != "name"},
    }


def _string_list(data: np.lib.npyio.NpzFile, key: str, fallback: list[str]) -> list[str]:
    if key not in data.files:
        return fallback
    return [str(value) for value in np.asarray(data[key]).tolist()]


def _segment_payload(
    *,
    name: str,
    time_s: np.ndarray,
    pose: np.ndarray,
    control: np.ndarray | None,
    direct_state: np.ndarray | None,
) -> dict[str, Any] | None:
    if pose.shape[1] == 7:
        position = pose[:, 0:3]
        quat = pose[:, 3:7]
    elif pose.shape[1] == 3:
        position = np.column_stack([pose[:, 0], np.zeros(len(pose)), pose[:, 1]])
        quat = _quat_from_frd_angles(np.zeros(len(pose)), pose[:, 2], np.zeros(len(pose)))
    else:
        return None
    position = position - position[0]
    payload: dict[str, Any] = {
        "name": name,
        "time_s": np.round(time_s - time_s[0], 4).tolist(),
        "position_enu_m": np.round(position, 5).tolist(),
        "quaternion_wxyz": np.round(quat, 7).tolist(),
    }
    if control is not None:
        payload["control_meas"] = np.round(control, 5).tolist()
    if direct_state is not None:
        payload["direct_state_meas"] = np.round(direct_state, 6).tolist()
    return payload


def _npz_playback(root: Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    files = manifest.get("local_data_files") or {}
    validation = files.get("validation")
    if not validation:
        return None
    path = root / validation
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    segments: list[dict[str, Any]] = []
    if "pose_meas" in data.files and "time_s" in data.files and "valid_mask" in data.files:
        valid_mask = np.asarray(data["valid_mask"], dtype=bool)
        counts = np.sum(valid_mask, axis=1)
        if not len(counts) or int(np.max(counts)) < 2:
            return None
        segment_names = _string_list(data, "segment_names", [f"segment_{index + 1}" for index in range(len(counts))])
        for segment, count in enumerate(counts):
            n = int(count)
            if n < 2:
                continue
            stride = max(1, int(np.ceil(n / 120)))
            time_s = np.asarray(data["time_s"][segment, :n], dtype=float)[::stride]
            pose = np.asarray(data["pose_meas"][segment, :n, :], dtype=float)[::stride]
            control = np.asarray(data["control_meas"][segment, :n, :], dtype=float)[::stride] if "control_meas" in data.files else None
            direct_state = np.asarray(data["direct_state_meas"][segment, :n, :], dtype=float)[::stride] if "direct_state_meas" in data.files else None
            payload = _segment_payload(
                name=segment_names[segment] if segment < len(segment_names) else f"segment_{segment + 1}",
                time_s=time_s,
                pose=pose,
                control=control,
                direct_state=direct_state,
            )
            if payload is not None:
                segments.append(payload)
    elif "mocap_meas" in data.files and "t" in data.files:
        time = np.asarray(data["t"], dtype=float)
        mocap = np.asarray(data["mocap_meas"], dtype=float)
        controls_2d = np.asarray(data["u_act" if "u_act" in data.files else "u_cmd"], dtype=float) if {"u_act", "u_cmd"} & set(data.files) else None
        direct_state = np.asarray(data["y_meas" if "y_meas" in data.files else "x_true"], dtype=float) if {"y_meas", "x_true"} & set(data.files) else None
        for segment in range(mocap.shape[0]):
            stride = max(1, int(np.ceil(len(time) / 120)))
            control = None
            if controls_2d is not None:
                u = controls_2d[segment, :, :][::stride]
                control = np.column_stack([u[:, 0], np.zeros(len(u)), u[:, 1], np.zeros(len(u))])
            raw_pose = mocap[segment, :, :][::stride]
            pose = _pose_ned_to_enu(raw_pose) if manifest.get("model_family") == MODEL_FAMILY_6DOF else raw_pose
            payload = _segment_payload(
                name=f"validation_trial_{segment + 1}",
                time_s=time[::stride],
                pose=pose,
                control=control,
                direct_state=direct_state[segment, :, :][::stride] if direct_state is not None else None,
            )
            if payload is not None:
                segments.append(payload)
    else:
        return None
    if not segments:
        return None
    return {
        "id": str(manifest.get("id")),
        "title": str(manifest.get("title") or manifest.get("id")),
        "model_family": str(manifest.get("model_family")),
        "source": "validation_npz",
        "control_names": _string_list(data, "control_names", ["thrust", "aileron", "elevator", "rudder"]),
        "pose_names": _string_list(data, "pose_names", ["x_e", "y_n", "z_u", "q_w", "q_x", "q_y", "q_z"]),
        "direct_state_names": _string_list(data, "direct_state_names", []),
        "segments": segments,
        **{key: value for key, value in segments[0].items() if key != "name"},
    }


def _playback_registry(root: Path, dataset_manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actual_tracks: dict[str, dict[str, Any]] = {}
    playback: list[dict[str, Any]] = []
    for manifest in dataset_manifests:
        track = _npz_playback(root, manifest)
        if track is not None:
            actual_tracks[track["id"]] = track
    for scenario in SCENARIOS_6DOF:
        playback.append(actual_tracks.pop(scenario.id, None) or _procedural_playback(scenario.id, scenario.model_family, scenario.title))
    playback.extend(actual_tracks.values())
    return playback


def _method_trace_registry(results_dir: Path) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*method_traces*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, list):
            traces.extend(item for item in payload if isinstance(item, dict))
        elif isinstance(payload, dict):
            rows = payload.get("traces")
            if isinstance(rows, list):
                traces.extend(item for item in rows if isinstance(item, dict))
    return traces


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def export_web_data(
    *,
    root: Path,
    output_dir: Path,
    results_dir: Path,
    dataset_modes: tuple[str, ...],
    dataset_titles: dict[str, str],
    method_training_modes: dict[str, str],
) -> dict[str, Any]:
    """Write JSON files consumed by the static benchmark website."""

    output_dir.mkdir(parents=True, exist_ok=True)
    method_rows = _method_rows(results_dir, dataset_modes, dataset_titles, method_training_modes)
    method_rows.extend(_six_dof_rows(results_dir))
    method_rows.extend(_real_dataset_rows(results_dir))
    method_rows = _dedupe_method_rows(method_rows)
    maneuver_rows = _maneuver_rows(results_dir)
    dataset_manifests = [*_generated_dataset_registry(root), *discover_manifests(root / "dataset_tools")]
    playback_rows = _playback_registry(root, dataset_manifests)
    maneuver_rows.extend(_playback_maneuver_rows(playback_rows, maneuver_rows))
    trace_rows = _method_trace_registry(results_dir)
    generated_at = datetime.now(UTC).isoformat()
    git_sha = _git_sha(root)
    scenarios = [
        {
            "id": scenario,
            "title": dataset_titles.get(scenario, scenario.replace("_", " ").title()),
            "model_family": MODEL_FAMILY_3DOF,
            "method_result_count": sum(1 for row in method_rows if row.get("scenario") == scenario),
        }
        for scenario in dataset_modes
    ]
    for scenario, title in SIX_DOF_SCENARIO_TITLES.items():
        scenarios.append(
            {
                "id": scenario,
                "title": title,
                "model_family": MODEL_FAMILY_6DOF,
                "method_result_count": sum(1 for row in method_rows if row.get("scenario") == scenario),
            }
        )
    existing_scenarios = {scenario["id"] for scenario in scenarios}
    for manifest_row in dataset_manifests:
        dataset_id = str(manifest_row.get("id"))
        if not dataset_id or dataset_id in existing_scenarios:
            continue
        scenarios.append(
            {
                "id": dataset_id,
                "title": manifest_row.get("title", dataset_id.replace("_", " ").title()),
                "model_family": manifest_row.get("model_family"),
                "dataset_status": manifest_row.get("status"),
                "source_type": manifest_row.get("source_type"),
                "method_result_count": sum(1 for row in method_rows if row.get("scenario") == dataset_id),
            }
        )
    method_registry = [metadata_to_dict(method) for method in all_method_metadata(root / "methods" / "plugins")]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "git_sha": git_sha,
        "model_families": sorted(
            {str(row.get("model_family")) for row in method_rows if row.get("model_family")}
            | {str(row.get("model_family")) for row in dataset_manifests if row.get("model_family")}
            | {MODEL_FAMILY_6DOF}
        ),
        "method_registry": method_registry,
        "dataset_registry": dataset_manifests,
        "files": {
            "method_results": "method_results.json",
            "maneuver_summary": "maneuver_summary.json",
            "playback": "playback.json",
            "method_traces": "method_traces.json",
        },
        "scenarios": scenarios,
        "metric_definitions": {
            "validation_score": "Mean state NRMSE over open-loop validation rollouts; lower is better.",
            "train_elapsed_s": "Wall-clock fit time for the method on its assigned training dataset.",
            "rollout_elapsed_s": "Wall-clock validation prediction time after training is complete.",
        },
    }
    _write_json(output_dir / "method_results.json", method_rows)
    _write_json(output_dir / "maneuver_summary.json", maneuver_rows)
    _write_json(output_dir / "playback.json", playback_rows)
    _write_json(output_dir / "method_traces.json", trace_rows)
    _write_json(output_dir / "manifest.json", manifest)
    return manifest
