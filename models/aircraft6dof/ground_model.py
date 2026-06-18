#!/usr/bin/env python3
"""Planar ground-roll model and ground-effect increments from the 5/22 flights.

Ground windows (tracked, label ``ground``) are cut and split exactly like the
stabilized windows. The rolling model is planar with states
(p_n, p_e, psi, V):

    dV/dt   = kT * (T_max/m) * thr^1.45 - mu * g - c_v * V^2
    dpsi/dt = (k_s * rudder + k_0) * V

with position integrating along heading. Thrust uses the nominal Sport Cub
map as a stated prior with a fitted scale ``kT`` (the throttle contrast on
the ground is too small to separate thrust scale from rolling resistance;
see ground_roll_analysis.py for the sensitivity study). The steering rate is
proportional to ground speed, as a castering tail-dragger's heading rate is,
and ``k_0`` absorbs wheel-alignment/thrust asymmetry. Parameters are fitted
by simulation error over the train windows and validated by held-out 5 s
free-run position error -- the same metric as the airborne models, with
hold-position and constant-velocity baselines for context.

Ground-effect (label ``ground_effect``) spans only a few seconds of
rotation/flare transition, so no separate rollout model is identifiable.
Instead the in-band lift/drag increments are identified by equation error
against the fitted grey-box: the band's force-coefficient residuals relative
to the airborne-manual reference give effective dCL/dCD with standard
errors. The recovered sign (lift deficit, drag excess) reflects the
high-alpha rotation/flare maneuvers that dominate the band rather than
classical ground-effect lift augmentation, and is reported as a diagnostic.

Outputs ``results/sportcub_ground_model.json`` via the CLI; the website
exporter calls ``fit_ground_model``/``fit_ground_effect`` directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from benchmark.paths import RESULTS
from .safe_controller import GreyboxAirframe, smooth_columns
from .segmentation import (
    euler_from_quat_array,
    sample_labels,
    split_stabilized_windows,
    stabilized_windows,
)

FLIGHTS_DEFAULT = Path("data/sportcub_mocap_5_22_26_flights.npz")
EXPLORER_JSON_DEFAULT = Path("site/public/data/flight_explorer.json")
MAX_THRUST_N = 0.32
THRUST_EXPONENT = 1.45
VALIDATION_HORIZON_S = 5.0
MIN_GE_AIRSPEED_MPS = 2.5
PARAM_NAMES = ("kT", "mu", "cv", "ks", "k0")
PARAM_BOUNDS = ([0.2, 0.0, 0.0, -5.0, -1.0], [3.0, 1.0, 1.0, 5.0, 1.0])
PARAM_INIT = np.array([1.0, 0.15, 0.05, 0.5, 0.0])


def ground_windows(labels: np.ndarray, tracked: np.ndarray, dt: float) -> list[tuple[int, int]]:
    """Ground spans cut with the stabilized-window machinery (label 0)."""
    relabeled = np.where(labels == 0, 2, -1).astype(np.int8)
    return stabilized_windows(relabeled, tracked, dt)


def planar_track(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(p_n, p_e, psi, V_horizontal) from the 13-state record."""
    eul = euler_from_quat_array(x[:, 6:10])
    v_h = np.linalg.norm(x[:, 3:5], axis=1)
    return x[:, 0], x[:, 1], eul[:, 2], v_h


def ground_rollout(params, mass: float, g: float, x0, sticks, n: int, dt: float) -> np.ndarray:
    """Planar rollout; sticks in u_cmd order (thr, elev, ail, rud)."""
    kT, mu, cv, ks, k0 = params
    pn, pe, psi, V = x0
    out = np.empty((n, 4))
    for k in range(n):
        thr = max(float(sticks[k, 0]), 0.0)
        rud = float(sticks[k, 3])
        acc = kT * (MAX_THRUST_N / mass) * thr**THRUST_EXPONENT - mu * g - cv * V * V
        V = max(V + dt * acc, 0.0)
        psi = psi + dt * (ks * rud + k0) * V
        pn += dt * V * np.cos(psi)
        pe += dt * V * np.sin(psi)
        out[k] = (pn, pe, psi, V)
    return out


def fit_ground_model(data, mass: float, g: float) -> dict:
    dt = float(data["sample_period_s"])
    windows = []
    membership: dict[str, list[dict[str, object]]] = {}
    ground_z: dict[str, float] = {}
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
        labels = sample_labels(x, mode)
        if ((labels == 0) & tracked).any():
            ground_z[name] = float(np.median(x[(labels == 0) & tracked, 2]))
        rows = []
        for a, b, split in split_stabilized_windows(ground_windows(labels, tracked, dt)):
            rows.append({"start_s": round(a * dt, 2), "stop_s": round(b * dt, 2), "split": split})
            windows.append((x, sticks, a, b, split))
        if rows:
            membership[name] = rows
    if not windows:
        raise SystemExit("no tracked ground windows found")

    horizon = int(round(VALIDATION_HORIZON_S / dt))
    every = int(round(0.1 / dt))

    def residuals(params):
        res = []
        for x, sticks, a, b, split in windows:
            if split != "train":
                continue
            pn, pe, psi, V = planar_track(x[a:b])
            pred = ground_rollout(params, mass, g, (pn[0], pe[0], psi[0], V[0]), sticks[a:b], b - a - 1, dt)
            err = (pred - np.column_stack([pn[1:], pe[1:], psi[1:], V[1:]]))[::every]
            err[:, 2] = np.arctan2(np.sin(err[:, 2]), np.cos(err[:, 2])) * 2.0
            res.append(err.ravel())
        return np.concatenate(res)

    def validate(params):
        model, hold, drift = [], [], []
        for x, sticks, a, b, split in windows:
            if split != "validation" or b - a < horizon + 1:
                continue
            pn, pe, psi, V = planar_track(x[a:b])
            pred = ground_rollout(params, mass, g, (pn[0], pe[0], psi[0], V[0]), sticks[a:b], horizon, dt)
            model.append(float(np.hypot(pred[-1, 0] - pn[horizon], pred[-1, 1] - pe[horizon])))
            hold.append(float(np.hypot(pn[horizon] - pn[0], pe[horizon] - pe[0])))
            t_h = horizon * dt
            drift.append(float(np.hypot(pn[0] + V[0] * np.cos(psi[0]) * t_h - pn[horizon], pe[0] + V[0] * np.sin(psi[0]) * t_h - pe[horizon])))
        return model, hold, drift

    fit = least_squares(residuals, PARAM_INIT, bounds=PARAM_BOUNDS, method="trf")
    model, hold, drift = validate(fit.x)
    return {
        "parameters": {n: round(float(v), 5) for n, v in zip(PARAM_NAMES, fit.x)},
        "fixed": {"max_thrust_n": MAX_THRUST_N, "thrust_exponent": THRUST_EXPONENT, "mass": mass, "g": g},
        "scores": {
            "ground_pos_err_5s_m": round(float(np.mean(model)), 3),
            "hold_position_baseline_m": round(float(np.mean(hold)), 3),
            "constant_velocity_baseline_m": round(float(np.mean(drift)), 3),
            "validation_windows": len(model),
            "train_windows": sum(1 for w in windows if w[4] == "train"),
        },
        "membership": membership,
        "ground_z": {k: round(v, 3) for k, v in ground_z.items()},
    }


def fit_ground_effect(data, greybox: dict) -> dict:
    """In-band force-coefficient increments by equation error vs the grey-box."""
    airframe = GreyboxAirframe(greybox)
    dt = float(data["sample_period_s"])
    mass = airframe.fixed["m"]
    res = {1: ([], []), 3: ([], [])}
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
        labels = sample_labels(x, mode)
        eul = euler_from_quat_array(x[:, 6:10])
        s12 = np.column_stack([x[:, 0:3], x[:, 3:6], eul, x[:, 10:13]])
        vdot = np.gradient(smooth_columns(x[:, 3:6], 9), dt, axis=0)
        pred = airframe.rhs(s12, sticks)
        speed = np.linalg.norm(x[:, 3:6], axis=1)
        qbar_s = np.maximum(0.5 * airframe.fixed["rho"] * speed**2 * airframe.fixed["S"], 1e-6)
        for lab in (1, 3):
            keep = (labels == lab) & tracked & (speed > MIN_GE_AIRSPEED_MPS)
            res[lab][0].append(((vdot[:, 2] - pred[:, 5]) * mass / qbar_s)[keep])
            res[lab][1].append(((vdot[:, 0] - pred[:, 3]) * mass / qbar_s)[keep])
    gw, gu = (np.concatenate(r) for r in res[1])
    bw, bu = (np.concatenate(r) for r in res[3])
    dcl = -(float(np.mean(gw)) - float(np.mean(bw)))
    dcd = -(float(np.mean(gu)) - float(np.mean(bu)))
    sem = lambda v: float(np.std(v) / np.sqrt(len(v)))  # noqa: E731
    return {
        "dCL": round(dcl, 4),
        "dCL_sem": round(float(np.hypot(sem(gw), sem(bw))), 4),
        "dCD": round(dcd, 4),
        "dCD_sem": round(float(np.hypot(sem(gu), sem(bu))), 4),
        "band_samples": int(len(gw)),
        "band_seconds": round(len(gw) * dt, 1),
        "note": (
            "Equation-error contrast of in-band force-coefficient residuals against the "
            "airborne-manual reference. The band is dominated by rotation/flare maneuvers, "
            "so the recovered lift deficit / drag excess reflects those high-alpha "
            "transitions rather than classical ground-effect augmentation; reported as a "
            "diagnostic, not applied to the prediction models."
        ),
    }


def ground_modelica_sources(parameters: dict, fixed: dict, *, provenance=None) -> dict:
    """Baseline + identified GroundRoll Modelica source (fitted params baked in).

    Maps the planar ground-roll fit (kT, mu, cv, ks, k0) plus the fixed mass/g
    onto ``GroundRoll.mo``'s parameters, so the identified ground model
    round-trips as a recompilable ``.mo`` -- the same treatment as the airframe
    and the SAFE controller. (Ground *effect* is a dCL/dCD diagnostic, not a
    dynamical model, so it has no ``.mo``.)
    """
    from .modelica.export_identified import HERE as _MO_DIR, identified_model_source

    base_mo = _MO_DIR / "GroundRoll.mo"
    values = {
        "kT": float(parameters["kT"]), "mu": float(parameters["mu"]),
        "cv": float(parameters["cv"]), "ks": float(parameters["ks"]),
        "k0": float(parameters["k0"]),
        "m": float(fixed["mass"]), "g": float(fixed["g"]),
    }
    return {
        "baseline_name": "GroundRoll",
        "baseline_source": base_mo.read_text(),
        "identified_name": "GroundRollIdentified",
        "identified_source": identified_model_source(
            base_mo, values, "GroundRollIdentified", provenance=provenance
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flights", type=Path, default=FLIGHTS_DEFAULT)
    parser.add_argument("--explorer-json", type=Path, default=EXPLORER_JSON_DEFAULT)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()

    data = np.load(args.flights, allow_pickle=False)
    greybox = json.loads(args.explorer_json.read_text())["models"]["greybox"]
    fixed = greybox["fixed_parameters"]
    ground = fit_ground_model(data, fixed["m"], fixed["g"])
    effect = fit_ground_effect(data, greybox)
    print("ground model:", ground["parameters"])
    print("scores:", ground["scores"])
    print(f"ground effect: dCL {effect['dCL']} +/- {effect['dCL_sem']}, dCD {effect['dCD']} +/- {effect['dCD_sem']} over {effect['band_seconds']} s")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    output = args.results_dir / "sportcub_ground_model.json"
    output.write_text(json.dumps({"ground": ground, "ground_effect": effect}, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")

    provenance = f"ground-roll fit: 5s pos err {ground['scores']['ground_pos_err_5s_m']} m"
    mo_out = args.results_dir / "GroundRollIdentified.mo"
    mo_out.write_text(ground_modelica_sources(ground["parameters"], ground["fixed"], provenance=provenance)["identified_source"])
    print(f"wrote {mo_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
