#!/usr/bin/env python3
"""Universal differential equation / neural residual benchmark.

Identification from measurements only: the parametric nominal model (starting
from initial_theta()) and a neural state-derivative residual are trained
jointly with the ODE solver in the loop, using multiple-shooting segments of
the measured states. Derivative matching is not used because the benchmark's
pitch dynamics put dQ/dt content at the sampling Nyquist frequency, making
finite-difference derivative targets unusable. The true parameters are never
used.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.benchmark import PARAMETER_NAMES, STATE_LABELS, STATE_NAMES, Aircraft, Bounds, eom, initial_theta, make_cases, true_theta
from common.metrics import aggregate_trajectory_score, finite_difference_derivative, percent_error, rmse
from common.paths import FIG_DIR, RESULTS_DIR
from common.plotting import save_figure


class ResidualNet(torch.nn.Module):
    def __init__(self, width: int, depth: int):
        super().__init__()
        layers: list[torch.nn.Module] = [torch.nn.Linear(6, width), torch.nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([torch.nn.Linear(width, width), torch.nn.Tanh()])
        layers.append(torch.nn.Linear(width, 4))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, xu: torch.Tensor) -> torch.Tensor:
        return self.net(xu)


def smooth_data(y: np.ndarray, dt: float, window: int, polyorder: int) -> tuple[np.ndarray, np.ndarray]:
    window = min(window, len(y) - (1 - len(y) % 2))
    if window % 2 == 0:
        window -= 1
    min_window = polyorder + 2
    if min_window % 2 == 0:
        min_window += 1
    window = max(window, min_window)
    x_smooth = savgol_filter(y, window_length=window, polyorder=polyorder, axis=0, mode="interp")
    dxdt = finite_difference_derivative(x_smooth, dt)
    return x_smooth, dxdt


def torch_eom(x: torch.Tensor, u: torch.Tensor, theta: torch.Tensor, aircraft: Aircraft) -> torch.Tensor:
    v = torch.clamp(x[:, 0], min=3.0)
    alpha = x[:, 1]
    gamma = x[:, 2]
    q_rate = x[:, 3]
    thrust = u[:, 0]
    elevator = u[:, 1]
    cl0, cla, cd0, k_drag, cm0, cma, cmq, cme = theta
    qbar = 0.5 * aircraft.rho * v**2
    q_hat = aircraft.chord * q_rate / (2.0 * v)
    cl = cl0 + cla * alpha
    cd = cd0 + k_drag * cl**2
    cm = cm0 + cma * alpha + cmq * q_hat + cme * elevator
    lift = cl * qbar * aircraft.wing_area
    drag = cd * qbar * aircraft.wing_area
    moment = cm * qbar * aircraft.wing_area * aircraft.chord
    v_dot = (-drag + thrust * torch.cos(alpha) - aircraft.mass * aircraft.gravity * torch.sin(gamma)) / aircraft.mass
    gamma_dot = (lift + thrust * torch.sin(alpha) - aircraft.mass * aircraft.gravity * torch.cos(gamma)) / (aircraft.mass * v)
    q_dot = moment / aircraft.jy
    alpha_dot = q_rate - gamma_dot
    return torch.stack((v_dot, alpha_dot, gamma_dot, q_dot), dim=1)


def train_case(case, args, device: torch.device) -> dict[str, object]:
    aircraft = Aircraft()
    bounds = Bounds()
    x_smooth, _ = smooth_data(case.y_meas, args.dt, args.smooth_window, args.polyorder)
    n_time = len(case.t)
    seg_len = max(2, int(args.segment_steps))
    stride = max(1, int(args.segment_stride))
    starts = np.arange(0, n_time - seg_len, stride)

    y = torch.tensor(case.y_meas, dtype=torch.float32, device=device)
    u = torch.tensor(case.u_id, dtype=torch.float32, device=device)
    noise_std = torch.tensor(case.noise_std, dtype=torch.float32, device=device)
    x0_seg = torch.tensor(x_smooth[starts], dtype=torch.float32, device=device)
    x_mean = y.mean(dim=0)
    x_scale = y.std(dim=0)
    x_scale = torch.where(x_scale > 1e-6, x_scale, torch.ones_like(x_scale))
    u_mean = u.mean(dim=0)
    u_scale = u.std(dim=0)
    u_scale = torch.where(u_scale > 1e-6, u_scale, torch.ones_like(u_scale))
    # residual output scale: one state-spread per second
    res_scale = x_scale

    net = ResidualNet(args.width, args.depth).to(device)
    # start from the pure parametric model: zero-init the output layer
    torch.nn.init.zeros_(net.net[-1].weight)
    torch.nn.init.zeros_(net.net[-1].bias)
    lower = torch.tensor(bounds.theta_lower, dtype=torch.float32, device=device)
    upper = torch.tensor(bounds.theta_upper, dtype=torch.float32, device=device)
    # optimize a normalized parameter vector so Adam steps are comparable
    # across parameters whose magnitudes span two orders of magnitude
    theta_base = torch.tensor(initial_theta(), dtype=torch.float32, device=device)
    theta_span = 0.5 * (upper - lower)
    theta_norm = torch.nn.Parameter(torch.zeros(8, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([*net.parameters(), theta_norm], lr=args.lr, weight_decay=args.weight_decay)
    dt = args.dt

    def rhs(x: torch.Tensor, u_now: torch.Tensor, theta_now: torch.Tensor) -> torch.Tensor:
        xu = torch.cat(((x - x_mean) / x_scale, (u_now - u_mean) / u_scale), dim=1)
        return torch_eom(x, u_now, theta_now, aircraft) + net(xu) * res_scale

    history = []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        theta = theta_base + theta_span * theta_norm
        theta_c = torch.minimum(torch.maximum(theta, lower), upper)
        x = x0_seg
        loss = torch.zeros((), device=device)
        for k in range(seg_len):
            u0 = u[starts + k]
            u1 = u[starts + k + 1]
            umid = 0.5 * (u0 + u1)
            k1 = rhs(x, u0, theta_c)
            k2 = rhs(x + 0.5 * dt * k1, umid, theta_c)
            k3 = rhs(x + 0.5 * dt * k2, umid, theta_c)
            k4 = rhs(x + dt * k3, u1, theta_c)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            loss = loss + torch.mean(((x - y[starts + k + 1]) / noise_std) ** 2)
        loss = loss / seg_len + 1e3 * torch.mean(((theta - theta_c) / theta_span) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*net.parameters(), theta_norm], 10.0)
        optimizer.step()
        if epoch % max(1, args.log_every) == 0 or epoch == args.epochs - 1:
            history.append((epoch, float(loss.detach())))
    elapsed = time.perf_counter() - start

    theta = theta_base + theta_span * theta_norm
    theta_hat = torch.minimum(torch.maximum(theta, lower), upper).detach().cpu().numpy()
    x_mean_np = x_mean.cpu().numpy()
    x_scale_np = x_scale.cpu().numpy()
    u_mean_np = u_mean.cpu().numpy()
    u_scale_np = u_scale.cpu().numpy()
    res_scale_np = res_scale.cpu().numpy()

    def residual_fn(x: np.ndarray, u_now: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xu_np = np.concatenate(((x - x_mean_np) / x_scale_np, (u_now - u_mean_np) / u_scale_np))[None, :]
            pred = net(torch.tensor(xu_np, dtype=torch.float32, device=device))
            return pred.cpu().numpy().ravel() * res_scale_np

    trajectory = simulate_ude(case.y_meas[0], case.u_id, args.dt, residual_fn, theta_hat)
    return {
        "case": case.name,
        "method": "UDE",
        "trajectory": trajectory,
        "history": history,
        "theta": theta_hat,
        "elapsed_s": elapsed,
        "decision_variables": sum(p.numel() for p in net.parameters()) + theta_hat.size,
        "train_score": aggregate_trajectory_score(trajectory, case.x_true),
    }


def ude_derivative(x: np.ndarray, u: np.ndarray, residual_fn, theta: np.ndarray) -> np.ndarray:
    return eom(x, u, theta, Aircraft()) + residual_fn(x, u)


def simulate_ude(x0: np.ndarray, u: np.ndarray, dt: float, residual_fn, theta: np.ndarray | None = None) -> np.ndarray:
    theta = initial_theta() if theta is None else theta
    x = np.empty((len(u), 4))
    x[0] = x0
    for k in range(len(u) - 1):
        u0, u1 = u[k], u[k + 1]
        umid = 0.5 * (u0 + u1)
        k1 = ude_derivative(x[k], u0, residual_fn, theta)
        k2 = ude_derivative(x[k] + 0.5 * dt * k1, umid, residual_fn, theta)
        k3 = ude_derivative(x[k] + 0.5 * dt * k2, umid, residual_fn, theta)
        k4 = ude_derivative(x[k] + dt * k3, u1, residual_fn, theta)
        x[k + 1] = x[k] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.all(np.isfinite(x[k + 1])) or np.linalg.norm(x[k + 1]) > 1e4:
            x[k + 1 :] = x[k]
            break
    return x


def plot_ude_trajectories(cases, results) -> None:
    fig, axes = plt.subplots(4, len(cases), figsize=(8.2, 5.6), sharex=True)
    if len(cases) == 1:
        axes = axes[:, None]
    for col, case in enumerate(cases):
        pred = results[case.name]["trajectory"]
        for row, label in enumerate(STATE_LABELS):
            ax = axes[row, col]
            truth = case.x_true[:, row].copy()
            meas = case.y_meas[:, row].copy()
            y_hat = pred[:, row].copy()
            if row in (1, 2, 3):
                truth = np.rad2deg(truth)
                meas = np.rad2deg(meas)
                y_hat = np.rad2deg(y_hat)
            ax.plot(case.t, truth, color="black", linewidth=1.4, label="Truth")
            ax.plot(case.t, meas, color="0.75", linewidth=0.5, alpha=0.75, label="Measured")
            ax.plot(case.t, y_hat, color="#9467bd", linewidth=1.1, label="UDE")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title(case.name.replace("_", " "))
            if row == 3:
                ax.set_xlabel("time [s]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, FIG_DIR / "ude_trajectories")


def plot_loss(results) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(8.2, 3.2), sharey=True)
    if len(results) == 1:
        axes = [axes]
    for ax, (case_name, result) in zip(axes, results.items()):
        hist = np.array(result["history"], dtype=float)
        ax.semilogy(hist[:, 0], hist[:, 1])
        ax.set_title(case_name.replace("_", " "))
        ax.set_xlabel("epoch")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("residual loss")
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "ude_training_loss")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--segment-steps", type=int, default=10, help="multiple-shooting segment length in samples")
    parser.add_argument("--segment-stride", type=int, default=5, help="stride between shooting segment starts")
    parser.add_argument("--smooth-window", type=int, default=17)
    parser.add_argument("--polyorder", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cases = make_cases(args.duration, args.dt, args.seed)
    results = {}
    rows = []
    for case in cases:
        result = train_case(case, args, device)
        results[case.name] = result
        state_rmse = rmse(result["trajectory"], case.x_true)
        row = {
            "case": case.name,
            "method": "UDE",
            "elapsed_s": result["elapsed_s"],
            "decision_variables": result["decision_variables"],
            "train_score": result["train_score"],
        }
        row.update({f"rmse_{name}": value for name, value in zip(STATE_NAMES, state_rmse)})
        row.update({f"theta_{name}": value for name, value in zip(PARAMETER_NAMES, result["theta"])})
        row.update({f"errpct_{name}": value for name, value in zip(PARAMETER_NAMES, percent_error(result["theta"], true_theta()))})
        rows.append(row)
        print(f"{case.name}: score={row['train_score']:.4g}, time={row['elapsed_s']:.2f}s")
    with (RESULTS_DIR / "ude_fit_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_ude_trajectories(cases, results)
    plot_loss(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
