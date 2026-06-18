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
from .greybox import (
    SPORTCUB_PARAMETER_NAMES,
    Aircraft6DOFConfig,
    build_casadi_dynamics,
    euler_from_quaternion,
    nominal_rk4_step,
    normalize_quaternion,
    quaternion_from_euler,
    rotation_body_to_inertial,
)
from .greybox_oem_fit import (
    OEM_CONTROL_ORDER,
    euler_states_to_quat,
    fit_greybox,
    greybox_modelica_sources,
    quat_states_to_euler,
)
from .ground_model import fit_ground_effect, fit_ground_model, ground_modelica_sources, ground_rollout, planar_track
from .modelica.export_linear import (
    safe_closed_loop_modelica_source,
    linear_ss_modelica_source,
    poly_surrogate_modelica_source,
)
from .safe_controller import fit_safe_controller, safe_controller, safe_modelica_sources
from .segmentation import (
    GROUND_ALTITUDE_M,
    STABILIZED_MIN_WINDOW_S,
    GROUND_EFFECT_ALTITUDE_M,
    LABELS,
    STABILIZED_VALIDATION_PERIOD,
    euler_from_quat_array,
    is_autonomous,
    sample_labels,
    stabilized_windows,
)

FLIGHTS_DEFAULT = Path("data/sportcub_mocap_5_22_26_flights.npz")
TRAIN_DEFAULT = Path("data/sportcub_mocap_5_22_26_train.npz")
VALIDATION_DEFAULT = Path("data/sportcub_mocap_5_22_26_validation.npz")
OUTPUT_DEFAULT = ROOT / "site" / "public" / "data" / "flight_explorer.json"

DISPLAY_RATE_HZ = 10.0  # browser display rate; stride derived from the data rate
RIDGE = 1e-5
# 6DOF-Nominal is omitted: it is the synthetic benchmark's truth-minus-residual
# baseline and has no meaning on real flights (it remains the internal base of
# RidgeResidual and GP-RBF).
METHODS = (
    "6DOF-LinearSS",
    "6DOF-RidgeResidual",
    "6DOF-GreyBoxOEM",
    "6DOF-EquationError-LS",
    "6DOF-SINDy",
    "6DOF-Koopman-EDMD",
    "6DOF-Symbolic-Stepwise",
    "6DOF-Subspace-Hankel",
    "6DOF-GP-RBF",
)
HANKEL_LAG = 3


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


def train_safe_closed_loop(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, dict[str, list[dict[str, object]]], dict[str, float]]:
    """Identify the stabilized closed-loop dynamics directly.

    Free-running a bare-airframe model through SAFE segments requires the
    hidden-controller decomposition, whose inverse-dynamics identification is
    weakly conditioned, so the closed loop is fitted directly on the tracked
    stabilized data. The regression is restricted to the heading- and
    position-invariant state (body velocities, roll, pitch, body rates) plus
    the heading *increment*: flight dynamics do not depend on where the
    aircraft is or which way it points, and a global affine map fitted on raw
    position/quaternion states cannot represent the heading-dependent position
    update, so its free runs wander within seconds. Heading integrates the
    fitted increment and position integrates the rotated body velocity
    exactly, which lets free runs fly whole laps.

    Like the manual maneuver windows, the stabilized spans are cut into
    windows assigned round-robin per flight (every third held out), the model
    fits on the train windows only, and the held-out windows score it by
    free-run position error.

    Returns the 13x9 weight matrix ([u,v,w,phi,theta,p,q,r, stick(4), 1] ->
    [next u,v,w,phi,theta,p,q,r, dpsi]), the per-flight window membership for
    the website's Data Splits view, and the held-out validation scores.
    """
    dt = float(data["sample_period_s"])
    feats, targs = [], []
    membership: dict[str, list[dict[str, object]]] = {}
    holdouts: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    shoot_flights: list[tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]] = []
    for flight_index, name in enumerate(str(s) for s in data["segment_names"]):
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
        if is_autonomous(name):
            # The autopilot's lateral commands bypass the recorded sticks, so
            # these stabilized samples teach the model that a neutral stick
            # keeps turning.
            print(f"closed-loop SAFE fit: excluding {name} (autonomous)")
            continue
        windows = stabilized_windows(labels, tracked, dt)
        train_mask = np.zeros(len(x), dtype=bool)
        rows: list[dict[str, object]] = []
        for index, (start, stop) in enumerate(windows):
            held_out = len(windows) >= 2 and index % STABILIZED_VALIDATION_PERIOD == STABILIZED_VALIDATION_PERIOD - 1
            rows.append({
                "start_s": round(start * dt, 2),
                "stop_s": round(stop * dt, 2),
                "split": "validation" if held_out else "train",
            })
            if held_out:
                holdouts.append((x, sticks, start, stop))
            else:
                train_mask[start:stop] = True
        membership[name] = rows
        keep_next = train_mask[:-1] & train_mask[1:]
        euler = euler_from_quat_array(x[:, 6:10])
        invariant = np.column_stack([x[:, 3:6], euler[:, 0:2], x[:, 10:13]])
        dpsi = np.arctan2(np.sin(np.diff(euler[:, 2])), np.cos(np.diff(euler[:, 2])))
        feats.append(np.column_stack([invariant[:-1], sticks[:-1], np.ones(len(dpsi))])[keep_next])
        targs.append(np.column_stack([invariant[1:], dpsi])[keep_next])
        shoot_flights.append((sticks, euler, invariant, [(start, stop) for (start, stop), row in zip(windows, rows) if row["split"] == "train"]))
    features = np.concatenate(feats)
    targets = np.concatenate(targs)
    weights = np.linalg.solve(
        features.T @ features + RIDGE * np.eye(features.shape[1]), features.T @ targets
    )
    weights = refine_simulation_error(weights, shoot_flights, dt)

    # Score the held-out windows by free-run position error, the quantity the
    # viewer's Predict-here actually shows.
    step = make_safe_step(weights, dt)
    horizon = int(round(STABILIZED_MIN_WINDOW_S / dt))
    errors = []
    for x, sticks, start, stop in holdouts:
        if stop - start < horizon + 1:
            continue
        state = suite.local_state_estimate(x, dt, start, window=12)
        for k in range(horizon):
            state = step(state, sticks[start + k])
        if np.all(np.isfinite(state)):
            errors.append(float(np.linalg.norm(state[0:3] - x[start + horizon, 0:3])))
    scores = {
        "train_samples": int(len(features)),
        "validation_windows": len(errors),
        "validation_pos_err_5s_m": round(float(np.mean(errors)), 3) if errors else None,
    }
    print(
        f"closed-loop SAFE fit: {len(features)} train samples, "
        f"{len(errors)} held-out windows, mean 5 s position error "
        f"{scores['validation_pos_err_5s_m']} m"
    )
    return weights, membership, scores


def refine_simulation_error(weights: np.ndarray, shoot_flights, dt: float, horizon_s: float = 1.0, stride_s: float = 0.5, max_nfev: int = 100) -> np.ndarray:
    """Refine the one-step ridge fit against one-second free-run residuals.

    One-step regression on noisy states carries an attenuation bias that
    compounds over a 1,200-step rollout; re-optimizing the same 13x9 linear
    map against multi-step shooting residuals on the training windows
    (holdouts untouched) is the standard simulation-error cure. Measured:
    held-out 5 s position error 3.41 -> 2.98 m, with no structure change.
    """
    from scipy.optimize import least_squares

    horizon = int(round(horizon_s / dt))
    stride = int(round(stride_s / dt))
    inv0, psi0, controls, inv_t, psi_t = [], [], [], [], []
    for sticks, euler, invariant, train_windows in shoot_flights:
        for start, stop in train_windows:
            for k in range(start, stop - horizon - 1, stride):
                inv0.append(invariant[k])
                psi0.append(euler[k, 2])
                controls.append(sticks[k : k + horizon])
                inv_t.append(invariant[k + horizon])
                psi_t.append(euler[k + horizon, 2])
    if not inv0:
        return weights
    inv0 = np.asarray(inv0)
    psi0 = np.asarray(psi0)
    controls = np.asarray(controls)
    inv_t = np.asarray(inv_t)
    psi_t = np.asarray(psi_t)
    scale = np.append(np.std(inv_t, axis=0), max(np.std(psi_t), 1e-6))

    def residual(wflat: np.ndarray) -> np.ndarray:
        w = wflat.reshape(weights.shape)
        inv = inv0.copy()
        psi = psi0.copy()
        for k in range(horizon):
            z = np.concatenate([inv, controls[:, k, :], np.ones((len(inv), 1))], axis=1)
            out = z @ w
            psi = psi + out[:, 8]
            inv = out[:, :8]
        dpsi = np.arctan2(np.sin(psi - psi_t), np.cos(psi - psi_t))
        return np.concatenate([((inv - inv_t) / scale[:8]).ravel(), dpsi / scale[8]])

    fit = least_squares(residual, weights.ravel(), max_nfev=max_nfev)
    print(
        f"closed-loop SAFE fit: simulation-error refinement over {len(inv0)} "
        f"{horizon_s:.0f} s shooting windows, cost {0.5 * np.dot(residual(weights.ravel()), residual(weights.ravel())):.0f} -> {fit.cost:.0f}"
    )
    return fit.x.reshape(weights.shape)


def make_safe_step(safe_weights: np.ndarray, dt: float):
    """Full-state wrapper around the invariant closed-loop regression."""

    def safe_step(x: np.ndarray, stick: np.ndarray) -> np.ndarray:
        quat = normalize_quaternion(x[6:10])
        euler = euler_from_quaternion(quat)
        rot = rotation_body_to_inertial(quat)
        pos = x[0:3] + rot @ x[3:6] * dt
        z = np.concatenate([x[3:6], euler[0:2], x[10:13], stick, [1.0]])
        out = z @ safe_weights
        psi = np.arctan2(np.sin(euler[2] + out[8]), np.cos(euler[2] + out[8]))
        quat_next = quaternion_from_euler(np.array([out[3], out[4], psi]))
        return np.concatenate([pos, out[0:3], quat_next, out[5:8]])

    return safe_step


def train_methods(train_path: Path) -> dict[str, object]:
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
    print(f"grey-box OEM fit on {x.shape[0]} chunks", flush=True)
    greybox = fit_greybox(x, u, split.dt)

    # Generic surrogates, fitted exactly as the benchmark suite fits them
    # (position-free features, smoothed derivatives, increment targets), so
    # the browser free-runs reproduce the leaderboard rows.
    xk = x[:, :-1, :].reshape(-1, x.shape[-1])
    uk = u[:, :-1, :].reshape(-1, u.shape[-1])
    xkp1 = x[:, 1:, :].reshape(-1, x.shape[-1])
    dxdt = suite.savgol_derivative(x, split.dt)[:, :-1, :].reshape(-1, x.shape[-1])
    x_smooth = suite.savgol_states(x, split.dt)
    inc = (x_smooth[:, 1:, :] - x_smooth[:, :-1, :]).reshape(-1, x.shape[-1])
    protected = suite.PROTECTED_FEATURES
    phi_lin = suite.linear_features(xk, uk)
    phi_poly = suite.poly_features(xk, uk, degree=2)
    eq_w, eq_m, eq_s = suite.fit_standardized_ridge(phi_lin, dxdt[:, suite.DYNAMIC_ROWS], RIDGE)
    sindy_w, sindy_m, sindy_s = suite.stlsq_fit(phi_poly, dxdt[:, suite.DYNAMIC_ROWS], 10.0 * RIDGE, fraction=0.06, protected=protected)
    edmd_w, edmd_m, edmd_s = suite.fit_standardized_ridge(phi_poly, inc[:, suite.DYNAMIC_ROWS], 100.0 * RIDGE)
    sym_w, sym_m, sym_s = suite.stlsq_fit(phi_poly, inc[:, suite.DYNAMIC_ROWS], RIDGE, fraction=0.12, protected=protected)
    history, targets = [], []
    for trial in range(x.shape[0]):
        for index in range(HANKEL_LAG - 1, x.shape[1] - 1):
            history.append(np.concatenate((suite.invariant_state(x[trial, index - HANKEL_LAG + 1 : index + 1]).reshape(-1), u[trial, index], [1.0])))
            targets.append(x_smooth[trial, index + 1, suite.DYNAMIC_ROWS] - x_smooth[trial, index, suite.DYNAMIC_ROWS])
    hankel_w = suite.ridge_fit(np.asarray(history)[:, None, :], np.asarray(targets)[:, None, :], RIDGE)
    rng = np.random.default_rng(7)
    z_all = np.concatenate((suite.invariant_state(xk), uk), axis=1)
    centers = z_all[rng.choice(len(z_all), size=min(96, len(z_all)), replace=False)]
    length_scale = np.std(z_all, axis=0)
    length_scale = np.where(length_scale > 1e-6, length_scale, 1.0)
    phi_rbf = np.concatenate((suite.rbf_features(z_all, centers, length_scale), np.ones((len(z_all), 1))), axis=1)
    residual_flat = residual_target.reshape(-1, x.shape[-1])
    gp_w = np.linalg.solve(phi_rbf.T @ phi_rbf + 5.0 * RIDGE * np.eye(phi_rbf.shape[1]), phi_rbf.T @ residual_flat[:, suite.DYNAMIC_ROWS])
    surrogates = {
        "6DOF-GP-RBF": {"kind": "rbf_residual", "weights": gp_w, "centers": centers, "length_scale": length_scale},
        "6DOF-EquationError-LS": {"kind": "derivative", "degree": 1, "weights": eq_w, "mean": eq_m, "scale": eq_s},
        "6DOF-SINDy": {"kind": "derivative", "degree": 2, "weights": sindy_w, "mean": sindy_m, "scale": sindy_s},
        "6DOF-Koopman-EDMD": {"kind": "increment", "degree": 2, "weights": edmd_w, "mean": edmd_m, "scale": edmd_s},
        "6DOF-Symbolic-Stepwise": {"kind": "increment", "degree": 2, "weights": sym_w, "mean": sym_m, "scale": sym_s},
        "6DOF-Subspace-Hankel": {"kind": "hankel", "lag": HANKEL_LAG, "weights": hankel_w},
    }
    return {
        "6DOF-NominalGreyBox": None,
        "6DOF-LinearSS": linear,
        "6DOF-RidgeResidual": residual,
        "6DOF-GreyBoxOEM": greybox,
        "surrogates": surrogates,
    }


def make_stepper(method: str, weights, dt: float, config: Aircraft6DOFConfig):
    if method == "6DOF-GreyBoxOEM":
        _dynamics, rk4 = build_casadi_dynamics(weights["spec"], dt)
        theta_full = weights["theta_full"]

        def greybox_step(x: np.ndarray, u: np.ndarray) -> np.ndarray:
            x_euler = quat_states_to_euler(x[None, None, :])[0, 0]
            x_euler = np.asarray(rk4(x_euler, u[OEM_CONTROL_ORDER], theta_full)).ravel()
            return euler_states_to_quat(x_euler[None, None, :])[0, 0]

        return greybox_step

    if isinstance(weights, dict) and "kind" in weights:
        spec = weights
        if spec["kind"] == "hankel":
            lag = spec["lag"]
            memory = {"hist": [], "last": None}

            def hankel_step(x: np.ndarray, u: np.ndarray) -> np.ndarray:
                # Re-seed the lag window whenever the caller jumps to a new
                # state (a fresh segment or a SAFE-model handoff).
                if memory["last"] is None or not np.array_equal(memory["last"], x):
                    memory["hist"] = [np.asarray(x, dtype=float).copy()] * lag
                phi = np.concatenate((suite.invariant_state(np.asarray(memory["hist"][-lag:])).reshape(-1), u, [1.0]))
                nxt = suite.kinematic_step(x, x[suite.DYNAMIC_ROWS] + phi @ spec["weights"], dt)
                memory["hist"] = (memory["hist"] + [nxt.copy()])[-lag:]
                memory["last"] = nxt
                return nxt

            return hankel_step

        if spec["kind"] == "rbf_residual":

            def rbf_step(x: np.ndarray, u: np.ndarray) -> np.ndarray:
                base = nominal_rk4_step(x, u, dt, config)
                z = np.concatenate((suite.invariant_state(x[None, :])[0], u))[None, :]
                phi = np.concatenate((suite.rbf_features(z, spec["centers"], spec["length_scale"]), np.ones((1, 1))), axis=1)[0]
                return suite.kinematic_step(x, base[suite.DYNAMIC_ROWS] + phi @ spec["weights"], dt)

            return rbf_step

        def surrogate_step(x: np.ndarray, u: np.ndarray) -> np.ndarray:
            phi = (
                suite.linear_features(x[None, :], u[None, :])
                if spec["degree"] == 1
                else suite.poly_features(x[None, :], u[None, :], degree=2)
            )[0]
            delta = suite.apply_standardized(phi, spec["weights"], spec["mean"], spec["scale"])
            gain = dt if spec["kind"] == "derivative" else 1.0
            return suite.kinematic_step(x, x[suite.DYNAMIC_ROWS] + gain * delta, dt)

        return surrogate_step

    def step(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        if method == "6DOF-NominalGreyBox":
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
    modes: np.ndarray,
    bias: np.ndarray,
    safe_step,
) -> np.ndarray:
    """Integrate the bare airframe while the pilot flies manual and the
    directly identified closed-loop model while SAFE is engaged.

    The handoff keys off the recorded mode channel, not the altitude-based
    segmentation class: a low SAFE pass is still closed-loop flight, and the
    state carries over continuously at every switch.
    """
    pred = np.empty((len(sticks), len(x0)))
    pred[0] = suite.normalize_state(x0)
    for k in range(len(sticks) - 1):
        if modes[k] == 1:
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


def load_window_records(paths: dict[str, Path]) -> list[dict[str, object]]:
    """Maneuver-window datasets (4/17) as flight-like records.

    Each window becomes a track: all-manual mode (the collection has no
    SAFE channel), tracked everywhere, and the train/validation membership
    taken from which split file the window lives in.
    """
    records = []
    for split_name, path in paths.items():
        data = np.load(path, allow_pickle=False)
        for index, name in enumerate(str(s) for s in data["segment_names"]):
            mask = np.asarray(data["valid_mask"][index], dtype=bool)
            records.append({
                "name": name,
                "x": np.asarray(data["x_meas"][index][mask], dtype=float),
                "pose": np.asarray(data["pose_meas"][index][mask], dtype=float),
                "sticks": np.asarray(data["u_cmd"][index][mask], dtype=float),
                "split": split_name,
                "dt": float(data["sample_period_s"]),
            })
    return records


SAFE_CONTROLLER_CACHE = ROOT / "results" / "sportcub_safe_controller.json"


def load_or_fit_safe_controller(args, greybox_fit: dict, safe_weights: np.ndarray) -> dict:
    """The joint simulation-error fit takes ~10 minutes; reuse the committed
    result unless asked to refit (the standalone CLI also refreshes it)."""
    if SAFE_CONTROLLER_CACHE.exists() and not args.refit_controller:
        print(f"SAFE controller: using cached {SAFE_CONTROLLER_CACHE}")
        return json.loads(SAFE_CONTROLLER_CACHE.read_text())
    greybox_payload = {
        "parameter_names": greybox_fit["parameter_names"],
        "parameters": greybox_fit["theta"],
        "cr_std": greybox_fit["cr_std"],
        "fixed_parameters": greybox_fit["spec"].fixed_parameters,
        "max_deflection_deg": greybox_fit["spec"].max_deflection_deg,
    }
    controller = fit_safe_controller(args.flights, greybox_payload, safe_weights)
    SAFE_CONTROLLER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SAFE_CONTROLLER_CACHE.write_text(json.dumps(controller, indent=2, sort_keys=True) + "\n")
    return controller


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["5_22", "4_17"], default="5_22")
    parser.add_argument("--flights", type=Path, default=FLIGHTS_DEFAULT)
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument("--validation", type=Path, default=VALIDATION_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--refit-controller", action="store_true", help="rerun the SAFE controller joint fit instead of reading results/sportcub_safe_controller.json")
    args = parser.parse_args()
    windows_mode = args.dataset == "4_17"
    if windows_mode:
        args.train = Path("data/sportcub_mocap_4_17_26_train.npz")
        args.validation = Path("data/sportcub_mocap_4_17_26_validation.npz")
        args.output = ROOT / "site" / "public" / "data" / "flight_explorer_4_17.json"

    if windows_mode:
        window_records = load_window_records({"train": args.train, "validation": args.validation})
        dt = window_records[0]["dt"]
        safe_weights, safe_membership, safe_scores = None, {}, None
        safe_step = None
        gains = None
        ground, ground_effect = None, None
    else:
        data = np.load(args.flights, allow_pickle=False)
        dt = float(data["sample_period_s"])
    downsample = max(1, int(round(1.0 / (DISPLAY_RATE_HZ * dt))))
    config = Aircraft6DOFConfig()
    weights = train_methods(args.train)
    if not windows_mode:
        safe_weights, safe_membership, safe_scores = train_safe_closed_loop(data)
        safe_step = make_safe_step(safe_weights, dt)
        controller = load_or_fit_safe_controller(args, weights["6DOF-GreyBoxOEM"], safe_weights)
        gains = controller["gains"]
        greybox_payload = {
            "parameter_names": weights["6DOF-GreyBoxOEM"]["parameter_names"],
            "parameters": weights["6DOF-GreyBoxOEM"]["theta"],
            "fixed_parameters": weights["6DOF-GreyBoxOEM"]["spec"].fixed_parameters,
            "max_deflection_deg": weights["6DOF-GreyBoxOEM"]["spec"].max_deflection_deg,
        }
        fixed = greybox_payload["fixed_parameters"]
        ground = fit_ground_model(data, fixed["m"], fixed["g"])
        ground_effect = fit_ground_effect(data, greybox_payload)
        print(f"ground model: {ground['scores']}", flush=True)
    membership = manual_window_splits({"train": args.train, "validation": args.validation})

    if windows_mode:
        record_iter = [(i, r["name"]) for i, r in enumerate(window_records)]
    else:
        record_iter = list(enumerate(str(s) for s in data["segment_names"]))

    flights_payload = []
    for flight_index, name in record_iter:
        if windows_mode:
            record = window_records[flight_index]
            x = record["x"]
            pose = record["pose"]
            sticks = record["sticks"]
            mode = np.zeros(len(x), dtype=np.int8)
            tracked = np.ones(len(x), dtype=bool)
        else:
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
        if windows_mode:
            # Maneuver windows are airborne manual excerpts by construction;
            # the altitude heuristic misreads their mocap-local origin as ground.
            labels = np.full(len(x), 3, dtype=np.int8)
        else:
            labels = sample_labels(x, mode)
        mode_int = np.asarray(mode, dtype=int)
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

        steppers = {
            method: make_stepper(method, weights.get(method, weights["surrogates"].get(method)), dt, config)
            for method in METHODS
        }
        segment_rows = []
        freeruns: dict[str, dict[str, object]] = {}
        for seg_index, (start, stop, label) in enumerate(segments):
            row: dict[str, object] = {
                "kind": LABELS[label],
                "start_s": round(start * dt, 2),
                "stop_s": round(stop * dt, 2),
                "split": (window_records[flight_index]["split"] if windows_mode and label == 3 else membership.get((name, start))),
            }
            keep = tracked[start:stop]
            if label in (1, 2, 3) and stop - start >= 30 and keep.sum() >= 10:
                x0 = suite.local_state_estimate(x, dt, start, window=12)
                scores = {}
                for method in METHODS:
                    pred = rollout(steppers[method], x0, sticks[start:stop], mode_int[start:stop], bias, safe_step)
                    pred_aligned = suite.align_quaternion_signs(pred, x[start:stop])
                    # Score only on genuinely tracked samples: interpolated
                    # dropout spans are fabricated data.
                    scores[method] = round(float(suite.nrmse_score(pred_aligned[keep], x[start:stop][keep])), 4)
                row["scores"] = scores
            elif label == 0 and ground is not None and stop - start >= 30 and tracked[start:stop].all():
                pn, pe, psi, v_h = planar_track(x[start:stop])
                params = [ground["parameters"][n] for n in ("kT", "mu", "cv", "ks", "k0")]
                pred = ground_rollout(params, ground["fixed"]["mass"], ground["fixed"]["g"], (pn[0], pe[0], psi[0], v_h[0]), sticks[start:stop], stop - start - 1, dt)
                err = np.hypot(pred[:, 0] - pn[1:], pred[:, 1] - pe[1:])
                row["scores"] = {"GroundRoll": round(float(np.mean(err)), 4)}
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
                # Full-rate sticks, labels, and mode drive the on-the-fly
                # rollouts; the manual/SAFE model handoff keys off the mode.
                "stick_full": np.round(sticks, 3).tolist(),
                "labels_full": labels.tolist(),
                "labels": labels[ds].tolist(),
                "mode": mode_int[ds].tolist(),
                "mode_full": mode_int.tolist(),
                "tracked": tracked[ds].astype(int).tolist(),
                "tracked_full": tracked.astype(int).tolist(),
                "bias": np.round(bias, 4).tolist(),
                "autonomous": is_autonomous(name),
                "segments": segment_rows,
                # Stabilized train/validation windows of the closed-loop SAFE
                # model, mirrored in the Data Splits view.
                "stabilized_splits": safe_membership.get(name, []),
                # Ground train/validation windows of the planar rolling model.
                "ground_splits": (ground["membership"].get(name, []) if ground else []),
                "ground_z": (ground["ground_z"].get(name) if ground else None),
            }
        )
        print(f"{name}: {len(segment_rows)} segments", flush=True)

    payload = {
        "dataset": "sportcub_mocap_4_17_26" if windows_mode else "sportcub_mocap_5_22_26",
        "labels": list(LABELS),
        "methods": list(METHODS),
        "models": {
            "linear_weights": np.round(weights["6DOF-LinearSS"], 6).tolist(),
            # Global affine state-space -> standalone generated Modelica (the weight
            # matrix is the whole structure; der(x) = (W*[x,u,1]-x)/dt).
            "linear_modelica": {
                "generated_name": "LinearSSIdentified",
                "generated_source": linear_ss_modelica_source(weights["6DOF-LinearSS"], dt),
            },
            "residual_weights": np.round(weights["6DOF-RidgeResidual"], 6).tolist(),
            **({
                "safe_gains": {axis: np.round(np.asarray(coef, dtype=float), 5).tolist() for axis, coef in gains.items()},
                "safe_controller": {
                    **{key: controller[key] for key in ("surface_lag_s", "scores", "airframe_corrections", "implied_law") if key in controller},
                    "modelica": safe_modelica_sources(
                        controller["gains"], controller["surface_lag_s"],
                        provenance="SAFE inner-loop controller fit (pd_command + surface lag) on the stabilized windows",
                    ),
                },
                "ground": {
                    **{key: ground[key] for key in ("parameters", "fixed", "scores")},
                    "modelica": ground_modelica_sources(
                        ground["parameters"], ground["fixed"],
                        provenance="planar ground-roll fit on the tracked ground windows",
                    ),
                },
                "ground_effect": ground_effect,
                "safe_invariant_weights": np.round(safe_weights, 8).tolist(),
                "safe_scores": safe_scores,
                # The closed-loop fit is a linear map; emit it as a generated
                # standalone Modelica model (no hand-written baseline -- the
                # weight matrix IS the structure).
                "safe_closed_loop_modelica": {
                    "generated_name": "SafeClosedLoopIdentified",
                    "generated_source": safe_closed_loop_modelica_source(safe_weights, dt),
                },
            } if not windows_mode else {}),
            "surrogates": {
                method: {
                    **{
                        key: (np.round(value, 7).tolist() if isinstance(value, np.ndarray) else value)
                        for key, value in spec.items()
                    },
                    # Invariant-feature polynomial surrogates (derivative/increment)
                    # fold to a standalone generated .mo; kernel surrogates (GP-RBF,
                    # lagged Subspace-Hankel) have no polynomial form -- skip them.
                    **(
                        {
                            "modelica": {
                                "generated_name": method.replace("6DOF-", "").replace("-", "") + "Identified",
                                "generated_source": poly_surrogate_modelica_source(
                                    spec, dt,
                                    model_name=method.replace("6DOF-", "").replace("-", "") + "Identified",
                                ),
                            }
                        }
                        if spec.get("kind") in ("derivative", "increment")
                        else {}
                    ),
                }
                for method, spec in weights["surrogates"].items()
            },
            "greybox": {
                "parameter_names": weights["6DOF-GreyBoxOEM"]["parameter_names"],
                "parameters": np.round(weights["6DOF-GreyBoxOEM"]["theta"], 6).tolist(),
                "cr_std": np.round(weights["6DOF-GreyBoxOEM"]["cr_std"], 6).tolist(),
                "couplings": weights["6DOF-GreyBoxOEM"]["couplings"],
                "uncertainty_note": "Cramer-Rao lower bounds from the output-error Jacobian; residual coloring uncorrected (optimistic).",
                "fixed_parameters": weights["6DOF-GreyBoxOEM"]["spec"].fixed_parameters,
                "max_deflection_deg": weights["6DOF-GreyBoxOEM"]["spec"].max_deflection_deg,
                # Actual Modelica source (single source of truth) + the identified
                # model with these fitted parameters baked in, for the inspector.
                "modelica": greybox_modelica_sources(
                    weights["6DOF-GreyBoxOEM"]["theta_full"],
                    provenance="GreyBoxOEM output-error fit on the manual training chunks",
                ),
            },
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
