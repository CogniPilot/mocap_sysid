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
    normalize_quaternion,
    quaternion_from_euler,
    rhs,
    rotation_body_to_inertial,
    simulate_smoke,
)

def rk4_step(x, u_cmd, dt, config):
    """Full nonlinear truth 6DOF RK4 step (the synthetic-data generator's model).

    Generated from ``modelica/Aircraft6DOF.mo`` (nl=1) via Rumoca -- the single
    source of truth. ``model.py``'s numpy version is retained only as the
    independent parity oracle in ``modelica/check_parity.py``.
    """
    from .modelica.dynamics import rk4_step as _modelica_rk4

    return _modelica_rk4(x, u_cmd, dt, config)


def nominal_rk4_step(x, u_cmd, dt, config):
    """Attached-flow nominal 6DOF RK4 step (the residual/hybrid methods' baseline).

    Generated from ``modelica/Aircraft6DOF.mo`` (nl=0) via Rumoca -- the single
    source of truth. ``model.py``'s numpy version is retained only as the
    independent parity oracle in ``modelica/check_parity.py``.
    """
    from .modelica.dynamics import nominal_rk4_step as _modelica_nominal

    return _modelica_nominal(x, u_cmd, dt, config)


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
MEASURED_FIXED_PARAMETER_NAMES = (
    "m",
    "S",
    "b",
    "cbar",
    "rho",
    "g",
    "wing_incidence",
    "thr_max",
    "max_defl_ail",
    "max_defl_elev",
    "max_defl_rud",
)
INERTIA_PARAMETER_NAMES = ("Ixx", "Iyy", "Izz", "Ixz")
GROUND_PARAMETER_NAMES = ("ground_k", "ground_c", "roll_fric", "side_fric", "ground_contact_eps")
GROUND_ESTIMATED_PARAMETER_NAMES = ("ground_contact_eps",)
SPORTCUB_PARAMETER_NAMES = (
    "wing_incidence",
    "thr_max",
    "ground_k",
    "ground_c",
    "roll_fric",
    "side_fric",
    "ground_contact_eps",
    "CL0",
    "CLa",
    "CD0",
    "k_ind",
    "CD0_fp",
    "Cm0",
    "Cma",
    "Cmq",
    "Cmde",
    "CYb",
    "CYda",
    "CYdr",
    "CYp",
    "CYr",
    "CY_fp_coef",
    "Clb",
    "Clp",
    "Clr",
    "Clda",
    "Cldr",
    "Cnb",
    "Cnp",
    "Cnr",
    "Cndr",
    "Cnda",
    "alpha_stall",
    "blend_width",
    "max_defl_ail",
    "max_defl_elev",
    "max_defl_rud",
)
AERO_PARAMETER_NAMES = tuple(
    name for name in SPORTCUB_PARAMETER_NAMES
    if name not in MEASURED_FIXED_PARAMETER_NAMES and name not in GROUND_PARAMETER_NAMES
)
GREYBOX_ESTIMATED_PARAMETER_NAMES = GROUND_ESTIMATED_PARAMETER_NAMES + INERTIA_PARAMETER_NAMES + AERO_PARAMETER_NAMES


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
            "Ixx": Bounds1D(1.0e-4, 6.9e-4, 2.5e-3),
            "Iyy": Bounds1D(1.0e-4, 6.0e-4, 2.5e-3),
            "Izz": Bounds1D(2.0e-4, 1.25e-3, 5.0e-3),
            "Ixz": Bounds1D(-8.0e-4, 3.5e-5, 8.0e-4),
            "wing_incidence": Bounds1D(-0.20, 0.10472, 0.30),
            "thr_max": Bounds1D(0.02, 0.32, 2.00),
            "ground_k": Bounds1D(10.0, 140.0, 3000.0),
            "ground_c": Bounds1D(0.5, 7.0, 150.0),
            "roll_fric": Bounds1D(0.0, 0.02, 2.0),
            "side_fric": Bounds1D(0.0, 1.2, 50.0),
            "ground_contact_eps": Bounds1D(1.0e-6, 1.0e-4, 2.0e-3),
            "CL0": Bounds1D(0.05, 0.50, 1.60),
            "CLa": Bounds1D(1.50, 4.70, 7.00),
            "CD0": Bounds1D(0.02, 0.06, 0.25),
            "k_ind": Bounds1D(0.00, 0.09, 0.40),
            "CD0_fp": Bounds1D(0.10, 0.30, 1.50),
            "Cm0": Bounds1D(-1.00, 0.00, 1.00),
            "Cma": Bounds1D(-5.00, -0.80, 1.00),
            "Cmq": Bounds1D(-80.00, -12.00, 0.00),
            "Cmde": Bounds1D(-4.00, 0.30, 4.00),
            "CYb": Bounds1D(-1.20, -0.30, 0.20),
            "CYda": Bounds1D(-0.20, 0.004, 0.20),
            "CYdr": Bounds1D(-0.30, -0.015, 0.30),
            "CYp": Bounds1D(-2.00, -0.15, 1.00),
            "CYr": Bounds1D(-1.00, 0.20, 2.00),
            "CY_fp_coef": Bounds1D(0.00, 0.50, 2.00),
            "Clb": Bounds1D(-5.00, -0.25, 2.00),
            "Clp": Bounds1D(-5.00, -0.50, 0.00),
            "Clr": Bounds1D(-2.00, 0.15, 2.00),
            "Clda": Bounds1D(-2.00, 0.05, 2.00),
            "Cldr": Bounds1D(-1.00, 0.006, 1.00),
            "Cnb": Bounds1D(-2.00, 0.06, 2.00),
            "Cnp": Bounds1D(-2.00, 0.010, 2.00),
            "Cnr": Bounds1D(-5.00, -0.15, 0.00),
            "Cndr": Bounds1D(-2.00, 0.015, 2.00),
            "Cnda": Bounds1D(-2.00, 0.006, 2.00),
            "alpha_stall": Bounds1D(0.10, 0.349, 0.70),
            "blend_width": Bounds1D(0.01, 0.0873, 0.30),
            "max_defl_ail": Bounds1D(0.10, 0.5236, 1.20),
            "max_defl_elev": Bounds1D(0.10, 0.4189, 1.20),
            "max_defl_rud": Bounds1D(0.10, 0.349, 1.20),
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

    def default_sportcub_parameters(self) -> dict[str, float]:
        return {
            name: self.default_parameter_bounds[name].initial
            for name in SPORTCUB_PARAMETER_NAMES
        }

    def default_full_parameter_mapping(self) -> dict[str, float]:
        values = dict(self.fixed_parameters)
        values.update(self.default_sportcub_parameters())
        return values

    def full_parameter_vector_from_mapping(self, parameters: dict[str, float]) -> np.ndarray:
        values = self.default_full_parameter_mapping()
        values.update({name: float(value) for name, value in parameters.items()})
        return np.array(
            [values[name] for name in FIXED_PARAMETER_NAMES + SPORTCUB_PARAMETER_NAMES],
            dtype=float,
        )

    def full_parameter_vector(self, estimated_parameters: np.ndarray) -> np.ndarray:
        theta = np.asarray(estimated_parameters, dtype=float)
        if theta.shape != (len(SPORTCUB_PARAMETER_NAMES),):
            raise ValueError(f"expected {len(SPORTCUB_PARAMETER_NAMES)} estimated parameters, got {theta.shape}")
        return np.concatenate((self.fixed_parameter_vector(), theta))

    def default_parameter_setup(self, names=SPORTCUB_PARAMETER_NAMES) -> list[tuple[str, float, float, float]]:
        return [
            (name, bounds.lower, bounds.initial, bounds.upper)
            for name, bounds in ((name, self.default_parameter_bounds[name]) for name in names)
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

    The physics is owned by ``modelica/SportCubGreybox.mo`` and generated through
    Rumoca; this returns that Modelica-backed kernel.
    """
    from .modelica.dynamics import build_casadi_dynamics as _modelica_build

    return _modelica_build(config, dt)


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
