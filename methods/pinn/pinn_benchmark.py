#!/usr/bin/env python3
"""Inverse PINN benchmark for the shared longitudinal aircraft dataset."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.benchmark import PARAMETER_NAMES, STATE_LABELS, STATE_NAMES, Aircraft, Bounds, initial_theta, make_cases, true_theta
from common.metrics import aggregate_trajectory_score, percent_error, rmse
from common.paths import FIG_DIR, RESULTS_DIR
from common.plotting import save_figure


class MLP(torch.nn.Module):
    def __init__(self, width: int, depth: int, output_dim: int):
        super().__init__()
        layers: list[torch.nn.Module] = [torch.nn.Linear(1, width), torch.nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([torch.nn.Linear(width, width), torch.nn.Tanh()])
        layers.append(torch.nn.Linear(width, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, t_norm: torch.Tensor) -> torch.Tensor:
        return self.net(t_norm)


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
    t = torch.tensor(case.t[:, None], dtype=torch.float32, device=device)
    t_scale = float(case.t[-1] - case.t[0])
    t_norm = ((t - t[0]) / t_scale).detach().clone().requires_grad_(True)
    y = torch.tensor(case.y_meas, dtype=torch.float32, device=device)
    u = torch.tensor(case.u_id, dtype=torch.float32, device=device)
    noise_std = torch.tensor(case.noise_std, dtype=torch.float32, device=device)
    x_mean = y.mean(dim=0)
    x_scale = y.std(dim=0)
    x_scale = torch.where(x_scale > 1e-6, x_scale, torch.ones_like(x_scale))

    net = MLP(args.width, args.depth, 4).to(device)
    bounds = Bounds()
    lower = torch.tensor(bounds.theta_lower, dtype=torch.float32, device=device)
    upper = torch.tensor(bounds.theta_upper, dtype=torch.float32, device=device)
    # optimize a normalized parameter vector so Adam steps are comparable
    # across parameters whose magnitudes span two orders of magnitude
    theta_base = torch.tensor(initial_theta(), dtype=torch.float32, device=device)
    theta_span = 0.5 * (upper - lower)
    theta_norm = torch.nn.Parameter(torch.zeros(8, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([*net.parameters(), theta_norm], lr=args.lr)
    history = []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        x_hat = net(t_norm) * x_scale + x_mean
        dx_cols = []
        for idx in range(4):
            grad = torch.autograd.grad(x_hat[:, idx].sum(), t_norm, create_graph=True)[0][:, 0] / t_scale
            dx_cols.append(grad)
        dxdt = torch.stack(dx_cols, dim=1)
        theta = theta_base + theta_span * theta_norm
        theta_clamped = torch.minimum(torch.maximum(theta, lower), upper)
        physics = torch_eom(x_hat, u, theta_clamped, aircraft)
        data_loss = torch.mean(((x_hat - y) / noise_std) ** 2)
        physics_loss = torch.mean(((dxdt - physics) / noise_std) ** 2)
        ic_loss = torch.mean(((x_hat[0] - y[0]) / noise_std) ** 2)
        bound_loss = torch.mean(((theta - theta_clamped) / theta_span) ** 2)
        loss = args.data_weight * data_loss + args.physics_weight * physics_loss + args.ic_weight * ic_loss + 1e3 * bound_loss
        loss.backward()
        optimizer.step()
        if epoch % max(1, args.log_every) == 0 or epoch == args.epochs - 1:
            history.append((epoch, float(loss.detach()), float(data_loss.detach()), float(physics_loss.detach()), float(ic_loss.detach())))
    elapsed = time.perf_counter() - start
    with torch.no_grad():
        x_hat = (net(t_norm) * x_scale + x_mean).detach().cpu().numpy()
        theta = theta_base + theta_span * theta_norm
        theta_hat = torch.minimum(torch.maximum(theta, lower), upper).detach().cpu().numpy()
    return {
        "case": case.name,
        "method": "PINN",
        "trajectory": x_hat,
        "theta": theta_hat,
        "history": history,
        "elapsed_s": elapsed,
        "decision_variables": sum(p.numel() for p in net.parameters()) + theta.numel(),
        "train_score": aggregate_trajectory_score(x_hat, case.x_true),
    }


def plot_pinn_trajectories(cases, results) -> None:
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
            ax.plot(case.t, y_hat, color="#2ca02c", linewidth=1.1, label="PINN")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title(case.name.replace("_", " "))
            if row == 3:
                ax.set_xlabel("time [s]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, FIG_DIR / "pinn_trajectories")


def plot_loss(results) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(8.2, 3.2), sharey=True)
    if len(results) == 1:
        axes = [axes]
    for ax, (case_name, result) in zip(axes, results.items()):
        hist = np.array(result["history"], dtype=float)
        ax.semilogy(hist[:, 0], hist[:, 1], label="total")
        ax.semilogy(hist[:, 0], hist[:, 2], label="data")
        ax.semilogy(hist[:, 0], hist[:, 3], label="physics")
        ax.set_title(case_name.replace("_", " "))
        ax.set_xlabel("epoch")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("loss")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "pinn_training_loss")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--data-weight", type=float, default=1.0)
    parser.add_argument("--physics-weight", type=float, default=0.05)
    parser.add_argument("--ic-weight", type=float, default=1.0)
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
        theta_error = percent_error(result["theta"], true_theta())
        state_rmse = rmse(result["trajectory"], case.x_true)
        row = {
            "case": case.name,
            "method": "PINN",
            "elapsed_s": result["elapsed_s"],
            "decision_variables": result["decision_variables"],
            "train_score": result["train_score"],
        }
        row.update({f"rmse_{name}": value for name, value in zip(STATE_NAMES, state_rmse)})
        row.update({f"theta_{name}": value for name, value in zip(PARAMETER_NAMES, result["theta"])})
        row.update({f"errpct_{name}": value for name, value in zip(PARAMETER_NAMES, theta_error)})
        rows.append(row)
        print(f"{case.name}: score={row['train_score']:.4g}, time={row['elapsed_s']:.2f}s")
    with (RESULTS_DIR / "pinn_fit_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_pinn_trajectories(cases, results)
    plot_loss(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
