"""Rumoca-backed Modelica dynamics for the 6DOF benchmark.

This is the integration layer between Rumoca-compiled continuous kernels and
the benchmark's existing call sites. The Modelica sources in this directory are
the single source of truth for the physics; generated Python kernels are local
cache files produced on demand by ``generate.py``. This module adds only the
fixed-step RK4 discretization and the post-step guards
(quaternion renormalization / speed & rate clamps for the truth model, psi-wrap
for the grey-box) -- which are integration details, not continuous dynamics, and
are intentionally kept out of the ``.mo`` models.

Two public surfaces, each a drop-in for an existing benchmark API:

* ``Aircraft6DOFKernel`` -- mirrors ``model.py``'s ``rhs`` / ``rk4_step`` /
  ``nominal_rk4_step`` (numpy in, numpy out) for the nonlinear truth model.
* ``build_casadi_dynamics(config, dt)`` -- mirrors
  ``greybox.build_casadi_dynamics``, returning ``(dynamics, rk4_step)`` CasADi
  Functions for the Euler grey-box.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

GENERATED = Path(__file__).resolve().parent / "generated"


def _load_generated(model_stem: str, suffix: str):
    """Import a local Rumoca-generated kernel module.

    The module exposes ``rhs(x, u, p) -> xdot`` plus ``STATE_NAMES`` /
    ``INPUT_NAMES`` / ``PARAM_NAMES`` and ``N_Y`` / ``N_U`` / ``N_P``. We wrap the
    latter in a ``meta`` namespace matching the metadata the call sites expect, so
    nothing downstream changes.
    """
    from .generate import ensure_generated

    ensure_generated(model_stem)
    path = GENERATED / f"{model_stem}_{suffix}.py"
    spec = importlib.util.spec_from_file_location(f"_rumoca_{model_stem}_{suffix}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    meta = SimpleNamespace(
        state_names=tuple(mod.STATE_NAMES),
        input_names=tuple(mod.INPUT_NAMES),
        param_names=tuple(mod.PARAM_NAMES),
        n_x=mod.N_Y, n_u=mod.N_U, n_p=mod.N_P,
    )
    return mod.rhs, meta


@lru_cache(maxsize=None)
def _casadi_kernel(model_stem: str):
    """CasADi ``rhs(x, u, p)`` Function + metadata (generated ``*_casadi_solve.py``)."""
    return _load_generated(model_stem, "casadi_solve")


@lru_cache(maxsize=None)
def load_jax_kernel(model_stem: str):
    """JAX ``rhs(x, u, p)`` closure + metadata (generated ``*_jax_solve.py``)."""
    return _load_generated(model_stem, "jax_solve")


# --------------------------------------------------------------------------- #
# Nonlinear truth model (13-state quaternion) -- mirrors model.py
# --------------------------------------------------------------------------- #
from ..model import _post_step  # exact post-step guards (quaternion renorm / clamps)


# Truth param order is meta.param_names; build it from an Aircraft6DOFConfig.
# Memoized because the suite calls this once per RK4 step in tight rollout loops;
# the vector depends only on the (frozen, hashable) config and the nl flag.
@lru_cache(maxsize=None)
def _truth_param_vector(config, nonlinear: bool) -> np.ndarray:
    values = {
        "mass": config.mass, "g": config.gravity,
        "ixx": config.inertia[0], "iyy": config.inertia[1], "izz": config.inertia[2],
        "ixz": config.inertia_xz, "rho": config.rho, "S": config.wing_area,
        "b": config.wing_span, "c": config.mean_chord, "prop_arm": config.prop_arm,
        "max_thrust": config.max_thrust, "prop_wash_gain": config.prop_wash_gain,
        "alpha_stall_deg": config.alpha_stall_deg, "stall_width_deg": config.stall_width_deg,
        "cl_max": config.cl_max, "cl_min": config.cl_min,
        "nl": 1.0 if nonlinear else 0.0,
    }
    _, meta = _casadi_kernel("Aircraft6DOF")
    vec = np.array([values[name] for name in meta.param_names], dtype=float)
    vec.setflags(write=False)  # cached value must not be mutated by callers
    return vec


@lru_cache(maxsize=None)
def _truth_rk4_fn(dt: float):
    """Fused fixed-step RK4 as a single CasADi Function (one eval per step).

    Excludes the post-step guards (quaternion renorm / clamps), which stay in
    numpy via ``_post_step`` to match ``model.py`` bit-for-bit.
    """
    import casadi as ca

    rhs, meta = _casadi_kernel("Aircraft6DOF")
    x = ca.SX.sym("x", meta.n_x)
    u = ca.SX.sym("u", meta.n_u)
    p = ca.SX.sym("p", meta.n_p)
    k1 = rhs(x, u, p)
    k2 = rhs(x + 0.5 * dt * k1, u, p)
    k3 = rhs(x + 0.5 * dt * k2, u, p)
    k4 = rhs(x + dt * k3, u, p)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return ca.Function("aircraft6dof_rk4_core", [x, u, p], [x_next])


class Aircraft6DOFKernel:
    """Numpy-facing nonlinear 6DOF truth dynamics, generated from Aircraft6DOF.mo.

    Behaviourally identical to ``model.py`` (verified to ~1e-13); reuses
    ``model.py``'s exact post-step so trajectories match bit-for-bit in intent.
    """

    def __init__(self):
        self._rhs, self.meta = _casadi_kernel("Aircraft6DOF")

    def rhs(self, x, u_cmd, config, *, nonlinear: bool = True) -> np.ndarray:
        p = _truth_param_vector(config, nonlinear)
        out = self._rhs(np.asarray(x, dtype=float), np.asarray(u_cmd, dtype=float), p)
        return np.asarray(out).flatten()

    def _rk4(self, x, u_cmd, dt, config, nonlinear: bool):
        p = _truth_param_vector(config, nonlinear)
        x_next = _truth_rk4_fn(float(dt))(
            np.asarray(x, dtype=float), np.asarray(u_cmd, dtype=float), p
        )
        return _post_step(np.asarray(x_next).flatten())

    def rk4_step(self, x, u_cmd, dt, config) -> np.ndarray:
        return self._rk4(x, u_cmd, dt, config, nonlinear=True)

    def nominal_rk4_step(self, x, u_cmd, dt, config) -> np.ndarray:
        return self._rk4(x, u_cmd, dt, config, nonlinear=False)


_TRUTH: Aircraft6DOFKernel | None = None


def truth_kernel() -> Aircraft6DOFKernel:
    global _TRUTH
    if _TRUTH is None:
        _TRUTH = Aircraft6DOFKernel()
    return _TRUTH


def rk4_step(x, u_cmd, dt, config) -> np.ndarray:
    """Module-level nonlinear truth RK4 step (drop-in for ``model.rk4_step``)."""
    return truth_kernel().rk4_step(x, u_cmd, dt, config)


def nominal_rk4_step(x, u_cmd, dt, config) -> np.ndarray:
    """Module-level attached-flow nominal RK4 step (drop-in for ``model.nominal_rk4_step``)."""
    return truth_kernel().nominal_rk4_step(x, u_cmd, dt, config)


# --------------------------------------------------------------------------- #
# Euler grey-box (12-state) -- mirrors greybox.build_casadi_dynamics
# --------------------------------------------------------------------------- #
def _euler_dynamics(model_stem: str, dt: float, rk4_name: str):
    """``(dynamics, rk4_step)`` for an Euler-state grey-box, with the psi (index 8)
    wrap the reference RK4 applies."""
    import casadi as ca

    dynamics_raw, meta = _casadi_kernel(model_stem)
    x = ca.SX.sym("x", dynamics_raw.size1_in(0))
    u = ca.SX.sym("u", dynamics_raw.size1_in(1))
    # Keep the public grey-box API on fixed(10) + SPORTCUB_PARAMETER_NAMES.
    # Rumoca may retain internal guard/cache constants in the generated solve
    # parameter vector; those are not fitted/user parameters, so the wrapper
    # pads them after the public vector instead of exposing them downstream.
    user_param_names = tuple(name for name in meta.param_names if not name.startswith("__pre__."))
    p_user = ca.SX.sym("p", len(user_param_names))
    pad = dynamics_raw.size1_in(2) - len(user_param_names)
    p = ca.vertcat(p_user, ca.SX.zeros(pad)) if pad > 0 else p_user
    rhs = dynamics_raw(x, u, p)
    dynamics = ca.Function(
        f"{model_stem}_rhs_public",
        [x, u, p_user],
        [rhs],
        ["x", "u", "p"],
        ["xdot"],
    )

    k1 = rhs
    k2 = dynamics_raw(x + 0.5 * dt * k1, u, p)
    k3 = dynamics_raw(x + 0.5 * dt * k2, u, p)
    k4 = dynamics_raw(x + dt * k3, u, p)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    # Wrap psi (state index 8) to (-pi, pi], matching the reference RK4.
    x_next = ca.vertcat(x_next[:8], ca.atan2(ca.sin(x_next[8]), ca.cos(x_next[8])), x_next[9:])
    rk4_step = ca.Function(rk4_name, [x, u, p_user], [x_next], ["x", "u", "p"], ["x_next"])
    return dynamics, rk4_step


def build_casadi_dynamics(config, dt: float):
    """Return ``(dynamics, rk4_step)`` CasADi Functions for the Sport Cub grey-box.

    Drop-in replacement for ``greybox.build_casadi_dynamics``: same state order
    (STATE_NAMES_EULER), control order (CONTROL_NAMES_OEM), and parameter vector
    (fixed(10) followed by SPORTCUB_PARAMETER_NAMES). ``dynamics(x, u, p)`` is the
    Modelica-generated continuous RHS; ``rk4_step`` is the fixed-step RK4 with the
    same psi (yaw) wrap as the reference.
    """
    return _euler_dynamics("SportCubGreybox", dt, "sportcub_6dof_greybox_rk4")
