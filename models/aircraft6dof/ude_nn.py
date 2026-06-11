"""Torch ODE-in-the-loop UDE for the 6DOF Sport Cub grey-box.

The universal-differential-equation premise implemented honestly at 6DOF:
the fitted grey-box airframe provides the known physics as a differentiable
torch port of its continuous dynamics, a small MLP adds a residual to the
six dynamic-state derivatives on the heading/position-invariant features,
and both integrate together through RK4 over multiple-shooting segments of
the measured training chunks (simulation-error objective). Derivative
targets are never formed; validation is a frozen-network open-loop rollout.

Requires torch (the project venv); the suite row is skipped gracefully when
torch is unavailable.
"""

from __future__ import annotations

import numpy as np

from .greybox import (
    FIXED_PARAMETER_NAMES,
    SPORTCUB_PARAMETER_NAMES,
    sportcub_greybox_spec,
)


def _torch():
    import torch

    return torch


def greybox_rhs_torch(x, u, p, torch):
    """Batched grey-box Euler-state RHS, mirroring build_casadi_dynamics.

    x: (..., 12) Euler states; u: (..., 4) controls in CONTROL_NAMES_OEM
    order (aileron, elevator, throttle, rudder); p: dict of parameter
    tensors/floats. Returns (..., 12) state derivatives.
    """
    u_b, v_b, w_b = x[..., 3], x[..., 4], x[..., 5]
    phi, theta = x[..., 6], x[..., 7]
    psi = x[..., 8]
    p_r, q_r, r_r = x[..., 9], x[..., 10], x[..., 11]
    ail_cmd, elev_cmd, thr_cmd, rud_cmd = u[..., 0], u[..., 1], u[..., 2], u[..., 3]

    deg = np.pi / 180.0
    thr = torch.clamp(thr_cmd, min=0.0)
    elev_rad = p["max_e"] * deg * elev_cmd
    ail_rad = p["max_a"] * deg * ail_cmd
    rud_rad = p["max_r"] * deg * rud_cmd

    speed = torch.sqrt(u_b**2 + v_b**2 + w_b**2 + 1e-9).clamp(min=1e-3)
    alpha = torch.atan2(w_b, u_b)
    beta = torch.asin(torch.clamp(v_b / speed, -0.99, 0.99))
    qbar = 0.5 * p["rho"] * speed**2
    c_a, s_a = torch.cos(alpha), torch.sin(alpha)
    c_b, s_b = torch.cos(beta), torch.sin(beta)

    CL = p["CL0"] + p["CLa"] * alpha
    CD = p["CD0"] + p["CDCLS"] * CL**2
    lift = qbar * p["S"] * CL
    drag = qbar * p["S"] * CD
    side = qbar * p["S"] * (p["CYb"] * beta)
    thrust = p["KT"] * p["m"] * thr

    fx = -drag * c_a * c_b - side * c_a * s_b + lift * s_a + thrust
    fy = -drag * s_b + side * c_b
    fz = -drag * s_a * c_b - side * s_a * s_b - lift * c_a

    c_phi, s_phi = torch.cos(phi), torch.sin(phi)
    c_th, s_th = torch.cos(theta), torch.sin(theta)
    c_psi, s_psi = torch.cos(psi), torch.sin(psi)

    u_dot = fx / p["m"] - p["g"] * s_th + r_r * v_b - q_r * w_b
    v_dot = fy / p["m"] + p["g"] * s_phi * c_th + p_r * w_b - r_r * u_b
    w_dot = fz / p["m"] + p["g"] * c_phi * c_th + q_r * u_b - p_r * v_b

    bV = p["b"] / (2.0 * speed)
    cV = p["cbar"] / (2.0 * speed)
    roll_acc = qbar * (p["KL0"] + p["KLb"] * beta + p["KLp"] * bV * p_r + p["KLr"] * bV * r_r + p["KLda"] * ail_rad + p["KLdr"] * rud_rad)
    pitch_acc = qbar * (p["KM0"] + p["KMa"] * alpha + p["KMq"] * cV * q_r + p["KMe"] * elev_rad)
    yaw_acc = qbar * (p["KN0"] + p["KNb"] * beta + p["KNp"] * bV * p_r + p["KNr"] * bV * r_r + p["KNda"] * ail_rad + p["KNdr"] * rud_rad)

    p_dot = roll_acc + ((p["Iyy"] - p["Izz"]) / p["Ixx"]) * q_r * r_r + (p["Ixz"] / p["Ixx"]) * p_r * q_r
    q_dot = pitch_acc + ((p["Izz"] - p["Ixx"]) / p["Iyy"]) * p_r * r_r + (p["Ixz"] / p["Iyy"]) * (r_r**2 - p_r**2)
    r_dot = yaw_acc + ((p["Ixx"] - p["Iyy"]) / p["Izz"]) * p_r * q_r + (p["Ixz"] / p["Izz"]) * q_r * r_r

    c_th_safe = torch.sign(c_th) * torch.clamp(torch.abs(c_th), min=1e-3)
    common = q_r * s_phi + r_r * c_phi
    phi_dot = p_r + (s_th / c_th_safe) * common
    theta_dot = q_r * c_phi - r_r * s_phi
    psi_dot = common / c_th_safe

    pn_dot = c_th * c_psi * u_b + (s_phi * s_th * c_psi - c_phi * s_psi) * v_b + (c_phi * s_th * c_psi + s_phi * s_psi) * w_b
    pe_dot = c_th * s_psi * u_b + (s_phi * s_th * s_psi + c_phi * c_psi) * v_b + (c_phi * s_th * s_psi - s_phi * c_psi) * w_b
    pd_dot = -s_th * u_b + s_phi * c_th * v_b + c_phi * c_th * w_b

    return torch.stack([pn_dot, pe_dot, pd_dot, u_dot, v_dot, w_dot, phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot], dim=-1)


def invariant_features_torch(x, u, torch):
    """[u, v, w, gx, gy, gz, p, q, r, controls]: gravity direction from Euler."""
    phi, theta = x[..., 6], x[..., 7]
    gx = -torch.sin(theta)
    gy = torch.sin(phi) * torch.cos(theta)
    gz = torch.cos(phi) * torch.cos(theta)
    return torch.cat([x[..., 3:6], torch.stack([gx, gy, gz], dim=-1), x[..., 9:12], u], dim=-1)


def train_greybox_ude(
    x_euler: np.ndarray,
    u_oem: np.ndarray,
    theta_full: np.ndarray,
    dt: float,
    *,
    stride: int = 4,
    horizon_steps: int = 30,
    hidden: int = 32,
    epochs: int = 300,
    seed: int = 0,
) -> dict:
    """Train the residual MLP RK4-in-the-loop over shooting segments.

    x_euler: (chunks, samples, 12) measured Euler states; u_oem matching
    controls in OEM order; theta_full the 35-long fitted grey-box parameter
    vector. Shooting segments of ``horizon_steps`` model steps (at
    ``dt*stride``) start from measured states; the loss is the weighted
    multi-step state error (positions, attitude, velocities, rates).
    """
    torch = _torch()
    torch.manual_seed(seed)
    device = torch.device("cpu")
    names = tuple(FIXED_PARAMETER_NAMES) + tuple(SPORTCUB_PARAMETER_NAMES)
    spec = sportcub_greybox_spec()
    p = {name: float(v) for name, v in zip(names, theta_full)}
    p["max_e"] = spec.max_deflection_deg["elevator"]
    p["max_a"] = spec.max_deflection_deg["aileron"]
    p["max_r"] = spec.max_deflection_deg["rudder"]

    h = dt * stride
    xs = x_euler[:, ::stride, :]
    us = u_oem[:, ::stride, :]
    n_chunks, n_steps = xs.shape[0], xs.shape[1]
    starts = []
    for c in range(n_chunks):
        for k in range(0, n_steps - horizon_steps - 1, horizon_steps // 2):
            starts.append((c, k))
    x0 = torch.tensor(np.array([xs[c, k] for c, k in starts]), dtype=torch.float32, device=device)
    useq = torch.tensor(np.array([us[c, k : k + horizon_steps] for c, k in starts]), dtype=torch.float32, device=device)
    target = torch.tensor(np.array([xs[c, k + 1 : k + horizon_steps + 1] for c, k in starts]), dtype=torch.float32, device=device)

    net = torch.nn.Sequential(
        torch.nn.Linear(13, hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, 6),
    ).to(device)
    # Start as a near-zero perturbation of the grey-box.
    with torch.no_grad():
        net[-1].weight *= 1e-3
        net[-1].bias.zero_()

    dyn_rows = [3, 4, 5, 9, 10, 11]
    weight = torch.tensor([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 2.0, 2.0, 1.0, 0.1, 0.1, 0.1], device=device)

    def rhs(x, u):
        base = greybox_rhs_torch(x, u, p, torch)
        res = net(invariant_features_torch(x, u, torch))
        out = base.clone()
        out[..., dyn_rows] = base[..., dyn_rows] + res
        return out

    def rk4(x, u):
        k1 = rhs(x, u)
        k2 = rhs(x + 0.5 * h * k1, u)
        k3 = rhs(x + 0.5 * h * k2, u)
        k4 = rhs(x + h * k3, u)
        return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        x = x0
        loss = x.new_zeros(())
        for k in range(horizon_steps):
            x = rk4(x, useq[:, k, :])
            err = (x - target[:, k, :]) * weight
            # wrap psi
            err_psi = torch.atan2(torch.sin(x[..., 8] - target[:, k, 8]), torch.cos(x[..., 8] - target[:, k, 8]))
            loss = loss + (err[..., :8] ** 2).mean() + (err[..., 9:] ** 2).mean() + (err_psi**2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss))
    return {"net": net, "p": p, "stride": stride, "h": h, "losses": losses, "torch": torch, "dyn_rows": dyn_rows}


def rollout_ude_quat(model: dict, x0_quat: np.ndarray, u_cmd_oem: np.ndarray, dt: float) -> np.ndarray:
    """Frozen-network open-loop rollout returning 13-state predictions."""
    from .greybox_oem_fit import euler_states_to_quat, quat_states_to_euler

    torch = model["torch"]
    net = model["net"]
    p = model["p"]
    dyn_rows = model["dyn_rows"]
    stride = model["stride"]
    h = dt * stride

    def rhs(x, u):
        base = greybox_rhs_torch(x, u, p, torch)
        res = net(invariant_features_torch(x, u, torch))
        out = base.clone()
        out[..., dyn_rows] = base[..., dyn_rows] + res
        return out

    n, length = u_cmd_oem.shape[:2]
    x = torch.tensor(quat_states_to_euler(x0_quat[:, None, :])[:, 0, :], dtype=torch.float32)
    pred = np.empty((n, length, 12))
    pred[:, 0, :] = x.numpy()
    with torch.no_grad():
        for k in range(0, length - stride, stride):
            u = torch.tensor(u_cmd_oem[:, k, :], dtype=torch.float32)
            k1 = rhs(x, u)
            k2 = rhs(x + 0.5 * h * k1, u)
            k3 = rhs(x + 0.5 * h * k2, u)
            k4 = rhs(x + h * k3, u)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            # Physical clamps, matching the other surrogate rollouts.
            x[..., 3:6] = torch.clamp(x[..., 3:6], -15.0, 15.0)
            x[..., 6:8] = torch.clamp(x[..., 6:8], -1.5, 1.5)
            x[..., 9:12] = torch.clamp(x[..., 9:12], -20.0, 20.0)
            xn = x.numpy()
            for j in range(1, stride + 1):
                if k + j < length:
                    pred[:, k + j, :] = xn
    return euler_states_to_quat(pred)
