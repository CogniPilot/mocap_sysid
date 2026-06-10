#!/usr/bin/env python3
"""Identify ground-roll dynamics (rolling resistance and thrust) from takeoff rolls.

The 2026-05-22 Sport Cub dataset includes automatically segmented ground
windows. Takeoff rolls — throttle onset to the instant of liftoff — are the
cleanest friction/thrust identification windows: the input is a known ramp,
the motion is monotonic acceleration on the wheels, and the window ends at
rotation. The longitudinal ground-roll model fitted here is

    dV/dt = c0 + c1 * delta_T + c2 * V,

where the rolling-resistance and drag terms oppose the thrust term. In this
session the pilot held essentially one throttle setting on the ground (about
0.5, with takeoff near the same level), so the thrust scale and rolling
resistance are not separately identifiable from the recorded ground data: any
unconstrained regression returns sign-indefinite coefficients. The analysis
therefore adopts the nominal thrust map T = T_max * delta_T^1.45 as a stated
prior (k_T = 1) and identifies the effective rolling resistance and quadratic
drag conditioned on it,

    dV/dt - T(delta_T)/m = -mu_eff * g - c_v * V^2 - k_batt * tau * T(delta_T)/m,

reporting the sensitivity d(mu_eff)/d(k_T) so the prior is transparent: if the
true static thrust is, say, 20% below nominal, mu_eff drops by 0.2 times the
reported sensitivity. The k_batt term models session-wide battery sag: the
airframe runs on a 1S LiPo whose prop power visibly drops over the session,
so the available thrust decays with session time tau. Because the ground
windows span the whole session at a roughly common throttle level, tau
provides the thrust contrast that throttle itself does not.

A second stage separates the battery effect per flight, and is the physically
meaningful one: pack endurance is only 5-10 minutes, so batteries are swapped
or recharged between flights and the session-wide term is a crude cross-pack
trend, not one battery discharging. Each flight starts at its own pack
voltage and the voltage droops within the flight, lowering the achievable
motor rpm and thrust. Voltage is
not telemetered, so the per-flight effective thrust scale is the observable
proxy: with mu and c_v fixed from the pooled stage, the thrust-side residual
a + mu*g + c_v*V^2 is regressed per flight on [T/m, t_flight * T/m], giving
the initial thrust scale k_f (initial-voltage proxy) and the within-flight
decay rate lambda_f for every flight with enough ground samples. Turning taxi samples are gated out by yaw rate. Ground
data is also highly susceptible to floor disturbances (wheel debris, surface
seams), so the fit drops the segmenter's robust outlier flags and then applies
Huber iteratively-reweighted least squares so residual events do not steer the
coefficients.

Outputs: ``results/sportcub_ground_roll.csv`` and
``latex/fig/ground_roll_takeoff.svg/.png``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark.paths import LATEX_FIG, RESULTS

DATASET_DEFAULT = Path("data/sportcub_mocap_5_22_26_ground.npz")
GRAVITY = 9.81
MIN_ROLL_SPEED_MPS = 0.2
MAX_YAW_RATE_RAD_S = 0.5  # straight-rolling gate: turning taxi adds steering scrub
SMOOTH_WINDOW_S = 0.25
HUBER_DELTA = 1.0
IRLS_ITERATIONS = 10
# Nominal Sport Cub thrust map from models/aircraft6dof/model.py
MAX_THRUST_N = 0.32
MASS_KG = 0.075
THRUST_EXPONENT = 1.45


def smooth(values: np.ndarray, dt: float, window_s: float = SMOOTH_WINDOW_S) -> np.ndarray:
    window = max(3, int(round(window_s / dt)) | 1)
    kernel = np.ones(window) / window
    pad = window // 2
    return np.convolve(np.pad(values, pad, mode="edge"), kernel, mode="valid")


def huber_fit(phi: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    weights = np.ones(len(target))
    coef = np.linalg.lstsq(phi, target, rcond=None)[0]
    for _ in range(IRLS_ITERATIONS):
        residual = target - phi @ coef
        scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 1e-6)
        z = residual / (scale * HUBER_DELTA)
        weights = np.where(np.abs(z) <= 1.0, 1.0, 1.0 / np.maximum(np.abs(z), 1e-9))
        w_phi = phi * weights[:, None]
        coef = np.linalg.lstsq(w_phi.T @ phi, w_phi.T @ target, rcond=None)[0]
    rmse = float(np.sqrt(np.mean((target - phi @ coef) ** 2)))
    return coef, rmse


def collect_samples(
    data: np.lib.npyio.NpzFile,
    indices: list[int],
    dt: float,
    *,
    throttle_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gated (speed, accel, throttle, session time, flight time, flight id) samples."""
    offsets = data["session_offset_s"] if "session_offset_s" in data.files else np.zeros(len(data["segment_names"]))
    names = [str(name) for name in data["segment_names"]]
    speeds, accels, throttles, taus, flight_times, flight_ids = [], [], [], [], [], []
    for index in indices:
        mask = np.asarray(data["valid_mask"][index], dtype=bool)
        x = data["x_meas"][index][mask]
        u = data["u_cmd"][index][mask]
        outlier = np.asarray(data["outlier_mask"][index][mask], dtype=bool)
        speed = smooth(np.linalg.norm(x[:, 3:5], axis=1), dt)
        accel = np.gradient(speed, dt)
        yaw_rate = np.abs(x[:, 12])
        keep = (
            (~outlier)
            & (speed > MIN_ROLL_SPEED_MPS)
            & (yaw_rate < MAX_YAW_RATE_RAD_S)
            & (u[:, 0] >= throttle_range[0])
            & (u[:, 0] <= throttle_range[1])
        )
        speeds.append(speed[keep])
        accels.append(accel[keep])
        throttles.append(u[keep, 0])
        taus.append(float(offsets[index]) + np.flatnonzero(keep) * dt)
        # Time within the flight: the segment's start offset within its own
        # recording is encoded by the window start index in the segment name.
        window_start = int(names[index].rsplit("_", 1)[-1]) * dt if names[index].rsplit("_", 1)[-1].isdigit() else 0.0
        flight_times.append(window_start + np.flatnonzero(keep) * dt)
        flight_ids.extend([names[index].split("__")[0]] * int(keep.sum()))
    if not speeds:
        return np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), []
    return (
        np.concatenate(speeds),
        np.concatenate(accels),
        np.concatenate(throttles),
        np.concatenate(taus),
        np.concatenate(flight_times),
        flight_ids,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--fig-dir", type=Path, default=LATEX_FIG)
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=False)
    dt = float(data["sample_period_s"])
    kinds = [str(kind) for kind in data["window_kinds"]]
    names = [str(name) for name in data["segment_names"]]
    rolls = [index for index, kind in enumerate(kinds) if kind == "takeoff_roll"]
    if not rolls:
        raise SystemExit("no takeoff_roll windows in the ground dataset")

    rollouts = [index for index, kind in enumerate(kinds) if kind == "landing_rollout"]
    windows = rolls + rollouts

    speed, accel, throttle, tau, flight_time, flight_ids = collect_samples(data, windows, dt, throttle_range=(0.0, 1.0))
    if len(accel) < 100:
        raise SystemExit(f"only {len(accel)} gated rolling samples; cannot identify ground-roll model")
    thrust_accel = MAX_THRUST_N * np.clip(throttle, 0.0, 1.0) ** THRUST_EXPONENT / MASS_KG
    target = accel - thrust_accel  # = -mu*g - c_v*V^2 - k_batt*tau*T/m under the k_T = 1 prior
    phi = np.column_stack([np.ones(len(speed)), speed**2, tau * thrust_accel])
    coef_fit, rmse = huber_fit(phi, target)
    mu = -coef_fit[0] / GRAVITY
    c_v2 = -coef_fit[1]
    k_batt = -coef_fit[2]
    sensitivity = float(thrust_accel.mean() / GRAVITY)

    # The session has a single ground throttle level (quantiles collapse to
    # 0.5), so mu and the thrust scale k_T are confounded: only the line
    # mu_true = mu_eff + (k_T - 1) * dmu_dkT is identified. At 240 Hz the
    # reduced smoothing attenuation exposes that the k_T = 1 prior is
    # optimistic (mu_eff goes slightly negative); for a plausible rolling
    # resistance band this implies k_T ~ 1.4-1.7, consistent with the
    # per-flight thrust scales below.
    k_t_for_mu = {mu_ref: 1.0 + (mu_ref - mu) / sensitivity for mu_ref in (0.05, 0.08, 0.12)}

    rows = [
        {
            "scope": "pooled",
            "takeoff_windows": len(rolls),
            "rollout_windows": len(rollouts),
            "samples": len(accel),
            "mu_eff": mu,
            "cv2_per_m": c_v2,
            "k_batt_per_s": k_batt,
            "thrust_loss_30min_pct": 100.0 * k_batt * 1800.0,
            "dmu_dkT": sensitivity,
            "thrust_prior_kT": 1.0,
            "kT_if_mu_0.05": round(k_t_for_mu[0.05], 3),
            "kT_if_mu_0.08": round(k_t_for_mu[0.08], 3),
            "kT_if_mu_0.12": round(k_t_for_mu[0.12], 3),
            "rmse_mps2": rmse,
        }
    ]
    # Per-flight battery stage: initial thrust scale (initial-voltage proxy)
    # and within-flight decay, with mu and c_v anchored by the pooled fit.
    thrust_residual = accel + mu * GRAVITY + c_v2 * speed**2
    for flight in sorted(set(flight_ids)):
        members = np.asarray([identifier == flight for identifier in flight_ids])
        if members.sum() < 100:
            continue
        phi_f = np.column_stack([thrust_accel[members], flight_time[members] * thrust_accel[members]])
        coef_f, rmse_f = huber_fit(phi_f, thrust_residual[members])
        k_flight = float(coef_f[0])
        decay = float(-coef_f[1] / k_flight) if abs(k_flight) > 1e-6 else float("nan")
        rows.append(
            {
                "scope": flight,
                "takeoff_windows": "",
                "rollout_windows": "",
                "samples": int(members.sum()),
                "mu_eff": "",
                "cv2_per_m": "",
                "k_batt_per_s": decay,
                "thrust_loss_30min_pct": "",
                "dmu_dkT": "",
                "thrust_prior_kT": k_flight,
                "rmse_mps2": rmse_f,
            }
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.results_dir / "sportcub_ground_roll.csv"
    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output_csv}")
    print(
        "  identifiability: single ground throttle level; mu_true = mu_eff + (k_T - 1)*"
        f"{sensitivity:.3f}; k_T for mu in {{0.05, 0.08, 0.12}} = "
        + ", ".join(f"{k_t_for_mu[m]:.2f}" for m in (0.05, 0.08, 0.12))
    )
    print(
        f"  mu_eff={mu:+.4f} (nominal-thrust prior, dmu/dkT={sensitivity:.3f}), cV2={c_v2:+.4f} /m, "
        f"k_batt={k_batt:+.2e} /s (thrust loss {100.0*k_batt*1800.0:+.1f}% per 30 min), "
        f"rmse={rmse:.3f} m/s^2 ({len(accel)} samples from {len(rolls)} rolls + {len(rollouts)} rollouts)"
    )
    for row in rows[1:]:
        print(
            f"  {row['scope']:42s} k_flight={row['thrust_prior_kT']:+.2f} (initial-voltage proxy), "
            f"within-flight decay={row['k_batt_per_s']:+.2e} /s, rmse={row['rmse_mps2']:.3f} ({row['samples']} samples)"
        )
    coef = np.array([1.0, -mu * GRAVITY, -c_v2])

    # Figure: the longest takeoff roll with outliers marked, plus the pooled fit.
    longest = max(rolls, key=lambda index: int(np.asarray(data["valid_mask"][index]).sum()))
    mask = np.asarray(data["valid_mask"][longest], dtype=bool)
    x = data["x_meas"][longest][mask]
    u = data["u_cmd"][longest][mask]
    outlier = np.asarray(data["outlier_mask"][longest][mask], dtype=bool)
    time = np.arange(mask.sum()) * dt
    speed = smooth(np.linalg.norm(x[:, 3:5], axis=1), dt)
    accel = np.gradient(speed, dt)
    thrust_accel = MAX_THRUST_N * np.clip(u[:, 0], 0.0, 1.0) ** THRUST_EXPONENT / MASS_KG
    predicted = coef[0] * thrust_accel + coef[1] + coef[2] * speed**2

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    axes[0].plot(time, speed, color="#4c78a8", lw=1.2, label="ground speed")
    axes[0].plot(time, u[:, 0] * speed.max(), color="0.6", lw=0.9, label="throttle (scaled)")
    axes[0].scatter(time[outlier], speed[outlier], s=12, color="#d62728", zorder=3, label="outlier flagged")
    axes[0].set_ylabel("speed (m/s)")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].set_title(f"Takeoff roll: {names[longest]}")
    axes[1].plot(time, accel, color="#4c78a8", lw=0.9, label="measured dV/dt")
    axes[1].plot(time, predicted, color="#f58518", lw=1.4, label="pooled Huber fit")
    axes[1].scatter(time[outlier], accel[outlier], s=12, color="#d62728", zorder=3)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("acceleration (m/s$^2$)")
    axes[1].legend(fontsize=8, loc="upper left")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".svg", ".png"):
        fig.savefig(args.fig_dir / f"ground_roll_takeoff{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.fig_dir / 'ground_roll_takeoff.svg'} (+.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
