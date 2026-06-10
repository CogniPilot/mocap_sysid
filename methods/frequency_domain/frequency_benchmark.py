#!/usr/bin/env python3
"""Frequency-domain diagnostic figures for the aircraft benchmark.

This is not presented as a fair numerical competitor to the time-domain methods
on the closed-loop benchmark. It creates the frequency-response/coherence figure
the review asked for and documents why a frequency-tailored excitation is needed.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import coherence, csd, welch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.benchmark import make_frequency_case
from common.paths import FIG_DIR, RESULTS_DIR
from common.plotting import save_figure


def transfer_estimate(input_signal: np.ndarray, output_signal: np.ndarray, fs: float, nperseg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # H1 estimate: scipy csd(x, y) averages conj(X) * Y, so the input must be
    # the first argument or the returned phase is conjugated.
    freq, pxy = csd(input_signal, output_signal, fs=fs, nperseg=nperseg, detrend="constant")
    _, pxx = welch(input_signal, fs=fs, nperseg=nperseg, detrend="constant")
    _, coh = coherence(input_signal, output_signal, fs=fs, nperseg=nperseg, detrend="constant")
    return freq, pxy / pxx, coh


def plot_frequency(case, fs: float, nperseg: int) -> list[dict[str, float]]:
    elevator = case.u_id[:, 1] - np.mean(case.u_id[:, 1])
    q_rate = case.y_meas[:, 3] - np.mean(case.y_meas[:, 3])
    gamma = case.y_meas[:, 2] - np.mean(case.y_meas[:, 2])
    freq_q, h_q, coh_q = transfer_estimate(elevator, q_rate, fs, nperseg)
    freq_g, h_g, coh_g = transfer_estimate(elevator, gamma, fs, nperseg)

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.2), sharex=True)
    axes[0].semilogx(freq_q[1:], 20.0 * np.log10(np.abs(h_q[1:])), label=r"$\delta_e \rightarrow Q$")
    axes[0].semilogx(freq_g[1:], 20.0 * np.log10(np.abs(h_g[1:])), label=r"$\delta_e \rightarrow \gamma$")
    axes[0].set_ylabel("magnitude [dB]")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].semilogx(freq_q[1:], np.rad2deg(np.unwrap(np.angle(h_q[1:]))), label=r"$Q$")
    axes[1].semilogx(freq_g[1:], np.rad2deg(np.unwrap(np.angle(h_g[1:]))), label=r"$\gamma$")
    axes[1].set_ylabel("phase [deg]")
    axes[1].grid(True, which="both", alpha=0.25)

    axes[2].semilogx(freq_q[1:], coh_q[1:], label=r"$Q$")
    axes[2].semilogx(freq_g[1:], coh_g[1:], label=r"$\gamma$")
    axes[2].set_ylabel("coherence")
    axes[2].set_xlabel("frequency [Hz]")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].grid(True, which="both", alpha=0.25)
    axes[2].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "frequency_response_diagnostic")

    rows = []
    for label, freq, h, coh in [("Q", freq_q, h_q, coh_q), ("gamma", freq_g, h_g, coh_g)]:
        valid = freq > 0.0
        if np.any(valid):
            idx = np.argmax(np.abs(h[valid]))
            valid_freq = freq[valid]
            valid_h = h[valid]
            valid_coh = coh[valid]
            rows.append(
                {
                    "output": label,
                    "peak_frequency_hz": float(valid_freq[idx]),
                    "peak_magnitude_db": float(20.0 * np.log10(np.abs(valid_h[idx]))),
                    "mean_coherence": float(np.mean(valid_coh)),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=80.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--nperseg", type=int, default=512)
    args = parser.parse_args()
    start = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    case = make_frequency_case(args.duration, args.dt, args.seed)
    rows = plot_frequency(case, fs=1.0 / args.dt, nperseg=min(args.nperseg, len(case.t) // 2))
    for row in rows:
        row["method"] = "frequency_welch_etfe"
        row["case"] = case.name
        row["elapsed_s"] = time.perf_counter() - start
    with (RESULTS_DIR / "frequency_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {FIG_DIR / 'frequency_response_diagnostic.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
