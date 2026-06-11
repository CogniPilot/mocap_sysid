"""Framework-owned 6DOF grey-box aircraft model specifications.

This module holds the reusable physical model pieces for 6DOF grey-box OEM
methods. Benchmark code should import this module directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from .model import (
    COEFFICIENT_NAMES,
    INPUT_NAMES,
    MAX_SPEED,
    MIN_SPEED,
    STATE_NAMES,
    Aircraft6DOFConfig,
    aerodynamic_coefficients,
    airdata,
    control_schedule,
    euler_from_quaternion,
    forces_and_moments,
    nominal_rk4_step,
    normalize_quaternion,
    quaternion_from_euler,
    rhs,
    rk4_step,
    rotation_body_to_inertial,
    simulate_smoke,
)


STATE_NAMES_EULER = (
    "p_n",
    "p_e",
    "p_d",
    "u",
    "v",
    "w",
    "phi",
    "theta",
    "psi",
    "p",
    "q",
    "r",
)
CONTROL_NAMES_OEM = ("aileron", "elevator", "throttle", "rudder")
OUTPUT_NAMES_MOCAP = ("p_n", "p_e", "p_d", "phi", "theta", "psi")
LATENT_INITIAL_NAMES = ("u0", "v0", "w0", "p0", "q0", "r0", "phi0", "theta0", "psi0")

FIXED_PARAMETER_NAMES = ("m", "S", "b", "cbar", "rho", "g", "Ixx", "Iyy", "Izz", "Ixz")
SPORTCUB_PARAMETER_NAMES = (
    "CL0",
    "CLa",
    "CD0",
    "CDCLS",
    "CYb",
    "KT",
    "KL0",
    "KLb",
    "KLp",
    "KLr",
    "KLda",
    "KLdr",
    "KM0",
    "KMa",
    "KMq",
    "KMe",
    "KN0",
    "KNb",
    "KNp",
    "KNr",
    "KNda",
    "KNdr",
)


@dataclass(frozen=True)
class Bounds1D:
    lower: float
    initial: float
    upper: float

    def clipped_initial(self) -> float:
        return float(np.clip(self.initial, self.lower, self.upper))


@dataclass(frozen=True)
class SportCubGreyboxConfig:
    """6DOF lumped-parameter grey-box model for the Sport Cub S 2 dataset."""

    fixed_parameters: dict[str, float] = field(
        default_factory=lambda: {
            "m": 0.063,
            "S": 0.05553,
            "b": 0.617,
            "cbar": 0.09,
            "rho": 1.225,
            "g": 9.81,
            "Ixx": 6.9e-4,
            "Iyy": 6.0e-4,
            "Izz": 1.25e-3,
            "Ixz": 3.5e-5,
        }
    )
    max_deflection_deg: dict[str, float] = field(
        default_factory=lambda: {"elevator": 23.0, "aileron": 25.0, "rudder": 30.0}
    )
    default_parameter_bounds: dict[str, Bounds1D] = field(
        default_factory=lambda: {
            "CL0": Bounds1D(0.05, 0.50, 1.60),
            "CLa": Bounds1D(1.50, 4.00, 7.00),
            "CD0": Bounds1D(0.02, 0.08, 0.25),
            "CDCLS": Bounds1D(0.00, 0.05, 0.30),
            "CYb": Bounds1D(-1.20, -0.30, 0.20),
            "KT": Bounds1D(0.50, 2.50, 9.00),
            "KL0": Bounds1D(-2.00, 0.00, 2.00),
            "KLb": Bounds1D(-50.00, -2.00, 50.00),
            "KLp": Bounds1D(-200.00, -50.00, 0.00),
            "KLr": Bounds1D(-50.00, 5.00, 50.00),
            "KLda": Bounds1D(-200.00, 50.00, 200.00),
            "KLdr": Bounds1D(-50.00, 2.00, 50.00),
            "KM0": Bounds1D(-2.00, 0.00, 2.00),
            "KMa": Bounds1D(-20.00, -2.00, 0.00),
            "KMq": Bounds1D(-400.00, -25.00, 0.00),
            "KMe": Bounds1D(-60.00, -0.40, 60.00),
            "KN0": Bounds1D(-2.00, 0.00, 2.00),
            "KNb": Bounds1D(-50.00, 3.00, 50.00),
            "KNp": Bounds1D(-50.00, -1.00, 50.00),
            "KNr": Bounds1D(-100.00, -8.00, 20.00),
            "KNda": Bounds1D(-30.00, -1.00, 30.00),
            "KNdr": Bounds1D(-50.00, -10.00, 50.00),
        }
    )
    literature_parameter_bounds: dict[str, Bounds1D] = field(
        default_factory=lambda: {
            "CL0": Bounds1D(0.05, 0.50, 1.60),
            "CLa": Bounds1D(1.50, 4.00, 7.00),
            "CYb": Bounds1D(-1.20, -0.30, 0.20),
        }
    )
    output_sigma: tuple[float, float, float, float, float, float] = (
        0.10,
        0.10,
        0.10,
        float(np.deg2rad(2.0)),
        float(np.deg2rad(2.0)),
        float(np.deg2rad(2.0)),
    )
    normalize_segment_costs: bool = True

    @property
    def inertia_ratios(self) -> dict[str, float]:
        p = self.fixed_parameters
        return {
            "c_qr_p": (p["Iyy"] - p["Izz"]) / p["Ixx"],
            "c_pq_p": p["Ixz"] / p["Ixx"],
            "c_pr_q": (p["Izz"] - p["Ixx"]) / p["Iyy"],
            "c_p2r2_q": p["Ixz"] / p["Iyy"],
            "c_pq_r": (p["Ixx"] - p["Iyy"]) / p["Izz"],
            "c_qr_r": p["Ixz"] / p["Izz"],
        }

    @property
    def output_weight_diagonal(self) -> np.ndarray:
        sigma = np.asarray(self.output_sigma, dtype=float)
        return 1.0 / np.square(sigma)

    def fixed_parameter_vector(self) -> np.ndarray:
        return np.array([self.fixed_parameters[name] for name in FIXED_PARAMETER_NAMES], dtype=float)

    def full_parameter_vector(self, estimated_parameters: np.ndarray) -> np.ndarray:
        theta = np.asarray(estimated_parameters, dtype=float)
        if theta.shape != (len(SPORTCUB_PARAMETER_NAMES),):
            raise ValueError(f"expected {len(SPORTCUB_PARAMETER_NAMES)} estimated parameters, got {theta.shape}")
        return np.concatenate((self.fixed_parameter_vector(), theta))

    def default_parameter_setup(self) -> list[tuple[str, float, float, float]]:
        return [
            (name, bounds.lower, bounds.initial, bounds.upper)
            for name, bounds in ((name, self.default_parameter_bounds[name]) for name in SPORTCUB_PARAMETER_NAMES)
        ]


def sportcub_greybox_spec() -> SportCubGreyboxConfig:
    """Return the framework-owned Sport Cub 6DOF grey-box model spec."""

    return SportCubGreyboxConfig()


def wrap_angle_np(angle: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(angle), np.cos(angle))


def euler_output_residual_np(predicted: np.ndarray, measured: np.ndarray) -> np.ndarray:
    """Residual for [p_n, p_e, p_d, phi, theta, psi] outputs."""

    residual = np.asarray(predicted, dtype=float) - np.asarray(measured, dtype=float)
    residual[..., 5] = wrap_angle_np(residual[..., 5])
    return residual


def euler_output_residual_ca(predicted, measured):
    """CasADi residual for [p_n, p_e, p_d, phi, theta, psi] outputs."""

    d_psi = predicted[5] - measured[5]
    return ca.vertcat(
        predicted[0] - measured[0],
        predicted[1] - measured[1],
        predicted[2] - measured[2],
        predicted[3] - measured[3],
        predicted[4] - measured[4],
        ca.atan2(ca.sin(d_psi), ca.cos(d_psi)),
    )


def build_casadi_dynamics(config: SportCubGreyboxConfig, dt: float):
    """Build CasADi continuous dynamics and fixed-step RK4 functions.

    State order is ``STATE_NAMES_EULER`` and control order is
    ``CONTROL_NAMES_OEM``.  The parameter vector is fixed parameters followed by
    ``SPORTCUB_PARAMETER_NAMES``.
    """

    n_states = len(STATE_NAMES_EULER)
    n_params = len(FIXED_PARAMETER_NAMES) + len(SPORTCUB_PARAMETER_NAMES)
    x_sym = ca.SX.sym("x", n_states)
    u_sym = ca.SX.sym("u", len(CONTROL_NAMES_OEM))
    p_sym = ca.SX.sym("p", n_params)

    u_b, v_b, w_b = x_sym[3], x_sym[4], x_sym[5]
    phi, theta, psi = x_sym[6], x_sym[7], x_sym[8]
    p_r, q_r, r_r = x_sym[9], x_sym[10], x_sym[11]
    ail_cmd, elev_cmd, thr_cmd, rud_cmd = u_sym[0], u_sym[1], u_sym[2], u_sym[3]

    (
        m,
        S,
        b_span,
        cbar,
        rho,
        g,
        Ixx,
        Iyy,
        Izz,
        Ixz,
        CL0,
        CLa,
        CD0,
        CDCLS,
        CYb,
        KT,
        KL0,
        KLb,
        KLp,
        KLr,
        KLda,
        KLdr,
        KM0,
        KMa,
        KMq,
        KMe,
        KN0,
        KNb,
        KNp,
        KNr,
        KNda,
        KNdr,
    ) = (p_sym[i] for i in range(n_params))

    max_defl = config.max_deflection_deg
    thr = ca.fmax(thr_cmd, 0.0)
    elev_rad = max_defl["elevator"] * (ca.pi / 180.0) * elev_cmd
    ail_rad = max_defl["aileron"] * (ca.pi / 180.0) * ail_cmd
    rud_rad = max_defl["rudder"] * (ca.pi / 180.0) * rud_cmd

    speed = ca.sqrt(u_b**2 + v_b**2 + w_b**2 + 1e-9)
    speed_safe = ca.fmax(speed, 1e-3)
    alpha = ca.atan2(w_b, u_b)
    beta = ca.asin(ca.fmin(ca.fmax(v_b / speed_safe, -0.99), 0.99))
    qbar = 0.5 * rho * speed_safe**2

    c_a = ca.cos(alpha)
    s_a = ca.sin(alpha)
    c_b = ca.cos(beta)
    s_b = ca.sin(beta)

    CL = CL0 + CLa * alpha
    CD = CD0 + CDCLS * CL**2
    CY = CYb * beta
    lift = qbar * S * CL
    drag = qbar * S * CD
    side = qbar * S * CY
    thrust = KT * m * thr

    # Aerodynamic forces rotate from wind axes; thrust is body-fixed along x
    # (a propeller does not rotate with the airflow).
    force_x_b = -drag * c_a * c_b - side * c_a * s_b + lift * s_a + thrust
    force_y_b = -drag * s_b + side * c_b
    force_z_b = -drag * s_a * c_b - side * s_a * s_b - lift * c_a

    c_phi = ca.cos(phi)
    s_phi = ca.sin(phi)
    c_th = ca.cos(theta)
    s_th = ca.sin(theta)
    c_psi = ca.cos(psi)
    s_psi = ca.sin(psi)

    u_dot = force_x_b / m - g * s_th + r_r * v_b - q_r * w_b
    v_dot = force_y_b / m + g * s_phi * c_th + p_r * w_b - r_r * u_b
    w_dot = force_z_b / m + g * c_phi * c_th + q_r * u_b - p_r * v_b

    roll_accel = qbar * (
        KL0
        + KLb * beta
        + KLp * (b_span / (2.0 * speed_safe)) * p_r
        + KLr * (b_span / (2.0 * speed_safe)) * r_r
        + KLda * ail_rad
        + KLdr * rud_rad
    )
    pitch_accel = qbar * (KM0 + KMa * alpha + KMq * (cbar / (2.0 * speed_safe)) * q_r + KMe * elev_rad)
    yaw_accel = qbar * (
        KN0
        + KNb * beta
        + KNp * (b_span / (2.0 * speed_safe)) * p_r
        + KNr * (b_span / (2.0 * speed_safe)) * r_r
        + KNda * ail_rad
        + KNdr * rud_rad
    )

    p_dot = roll_accel + ((Iyy - Izz) / Ixx) * q_r * r_r + (Ixz / Ixx) * p_r * q_r
    q_dot = pitch_accel + ((Izz - Ixx) / Iyy) * p_r * r_r + (Ixz / Iyy) * (r_r**2 - p_r**2)
    r_dot = yaw_accel + ((Ixx - Iyy) / Izz) * p_r * q_r + (Ixz / Izz) * q_r * r_r

    c_th_safe = ca.sign(c_th) * ca.fmax(ca.fabs(c_th), 1e-3)
    common = q_r * s_phi + r_r * c_phi
    phi_dot = p_r + (s_th / c_th_safe) * common
    theta_dot = q_r * c_phi - r_r * s_phi
    psi_dot = common / c_th_safe

    r00 = c_th * c_psi
    r01 = s_phi * s_th * c_psi - c_phi * s_psi
    r02 = c_phi * s_th * c_psi + s_phi * s_psi
    r10 = c_th * s_psi
    r11 = s_phi * s_th * s_psi + c_phi * c_psi
    r12 = c_phi * s_th * s_psi - s_phi * c_psi
    r20 = -s_th
    r21 = s_phi * c_th
    r22 = c_phi * c_th

    xdot = ca.vertcat(
        r00 * u_b + r01 * v_b + r02 * w_b,
        r10 * u_b + r11 * v_b + r12 * w_b,
        r20 * u_b + r21 * v_b + r22 * w_b,
        u_dot,
        v_dot,
        w_dot,
        phi_dot,
        theta_dot,
        psi_dot,
        p_dot,
        q_dot,
        r_dot,
    )
    dynamics = ca.Function("sportcub_6dof_greybox_rhs", [x_sym, u_sym, p_sym], [xdot], ["x", "u", "p"], ["xdot"])

    k1 = dynamics(x_sym, u_sym, p_sym)
    k2 = dynamics(x_sym + 0.5 * dt * k1, u_sym, p_sym)
    k3 = dynamics(x_sym + 0.5 * dt * k2, u_sym, p_sym)
    k4 = dynamics(x_sym + dt * k3, u_sym, p_sym)
    x_next = x_sym + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next = ca.vertcat(x_next[:8], ca.atan2(ca.sin(x_next[8]), ca.cos(x_next[8])), x_next[9:])
    rk4_step = ca.Function("sportcub_6dof_greybox_rk4", [x_sym, u_sym, p_sym], [x_next], ["x", "u", "p"], ["x_next"])
    return dynamics, rk4_step


def main() -> None:
    """Print the registered Sport Cub grey-box model summary."""

    spec = sportcub_greybox_spec()
    print("Sport Cub 6DOF grey-box model")
    print(f"  states: {len(STATE_NAMES_EULER)}")
    print(f"  controls: {len(CONTROL_NAMES_OEM)}")
    print(f"  estimated parameters: {len(SPORTCUB_PARAMETER_NAMES)}")
    print(f"  fixed parameters: {len(FIXED_PARAMETER_NAMES)}")
    print(f"  output weights: {spec.output_weight_diagonal.tolist()}")


if __name__ == "__main__":
    main()
