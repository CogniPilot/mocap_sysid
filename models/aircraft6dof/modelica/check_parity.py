"""Lock the Modelica-generated dynamics to the reference implementations.

The Modelica sources are the single source of truth for the 6DOF physics. This
check fails if a generated kernel ever drifts from its hand-written reference,
so ``model.py`` / the legacy CasADi grey-box and the ``.mo`` models cannot
silently diverge. Run directly or from CI::

    python -m models.aircraft6dof.modelica.check_parity

Exits non-zero on any mismatch above tolerance.
"""

from __future__ import annotations

import sys

import numpy as np

from .. import greybox as gb
from .. import model as truth
from . import dynamics as md

TOL = 1e-9
GEN = __import__("pathlib").Path(__file__).resolve().parent / "generated"


def _truth_base(cfg) -> list[float]:
    return [cfg.mass, cfg.gravity, *cfg.inertia, cfg.inertia_xz, cfg.rho, cfg.wing_area,
            cfg.wing_span, cfg.mean_chord, cfg.prop_arm, cfg.max_thrust, cfg.prop_wash_gain,
            cfg.alpha_stall_deg, cfg.stall_width_deg, cfg.cl_max, cfg.cl_min]


def check_truth_rhs(n: int = 500) -> float:
    rhs, _ = md._casadi_kernel("Aircraft6DOF")
    cfg = truth.Aircraft6DOFConfig()
    base = _truth_base(cfg)
    rng = np.random.default_rng(0)
    worst = 0.0
    for nl, nonlinear in ((1.0, True), (0.0, False)):
        p0 = np.array(base + [nl])
        for _ in range(n):
            spd = rng.uniform(2.6, 11.5)
            vel = np.array([spd, rng.uniform(-2, 2), rng.uniform(-3, 3)])
            q = rng.normal(size=4); q /= np.linalg.norm(q)
            x = np.concatenate([rng.uniform(-5, 5, 3), vel, q, rng.uniform(-5, 5, 3)])
            u = np.array([rng.uniform(0, 1), rng.uniform(-.65, .65), rng.uniform(-.75, .75), rng.uniform(-.65, .65)])
            ref = truth.rhs(x, u, cfg, nonlinear=nonlinear)
            got = np.array(rhs(x, u, p0)).flatten()
            worst = max(worst, float(np.max(np.abs(ref - got))))
    return worst


def check_truth_rk4(steps: int = 300) -> float:
    """Roll out both the nonlinear (nl=1) and nominal/attached-flow (nl=0) RK4
    trajectories; the nominal path is the residual/hybrid methods' baseline."""
    cfg = truth.Aircraft6DOFConfig()
    k = md.truth_kernel()
    x0 = np.concatenate([[0, 0, 0], [cfg.wing_speed, 0, 0], [1, 0, 0, 0], [0, 0, 0]])
    worst = 0.0
    for ref_step, mod_step in (
        (truth.rk4_step, k.rk4_step),                  # nl=1 nonlinear truth
        (truth.nominal_rk4_step, k.nominal_rk4_step),  # nl=0 nominal baseline (family B)
    ):
        x_ref = x0.copy()
        x_mod = x0.copy()
        for i in range(steps):
            u = truth.control_schedule(i * cfg.dt)
            x_ref = ref_step(x_ref, u, cfg.dt, cfg)
            x_mod = mod_step(x_mod, u, cfg.dt, cfg)
            worst = max(worst, float(np.max(np.abs(x_ref - x_mod))))
    return worst


def _random_aero(rng, cfg):
    aero = np.array([cfg.default_parameter_bounds[m].initial for m in gb.SPORTCUB_PARAMETER_NAMES])
    return cfg.full_parameter_vector(aero * rng.uniform(0.5, 1.5, size=aero.shape))


def check_greybox(n: int = 500) -> float:
    cfg = gb.SportCubGreyboxConfig()
    dt = 0.02
    _, rk4_ref = gb._build_casadi_dynamics_legacy(cfg, dt)
    _, rk4_mod = md.build_casadi_dynamics(cfg, dt)
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(n):
        p = _random_aero(rng, cfg)
        spd = rng.uniform(5, 25)
        x = np.array([*rng.uniform(-20, 20, 3), spd, rng.uniform(-3, 3), rng.uniform(-4, 4),
                      rng.uniform(-.5, .5), rng.uniform(-.5, .5), rng.uniform(-np.pi, np.pi),
                      rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-2, 2)])
        u = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0, 1), rng.uniform(-1, 1)])
        ref = np.array(rk4_ref(x, u, p)).flatten()
        got = np.array(rk4_mod(x, u, p)).flatten()
        worst = max(worst, float(np.max(np.abs(ref - got))))
    return worst


def check_greybox_lag(n: int = 500) -> float:
    cfg = gb.SportCubGreyboxConfig()
    dt = 0.02
    _, rk4_ref = gb._build_casadi_dynamics_lag_legacy(cfg, dt)
    _, rk4_mod = md.build_casadi_dynamics_lag(cfg, dt)
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(n):
        p = _random_aero(rng, cfg)
        spd = rng.uniform(5, 25)
        x = np.array([*rng.uniform(-20, 20, 3), spd, rng.uniform(-3, 3), rng.uniform(-4, 4),
                      rng.uniform(-.5, .5), rng.uniform(-.5, .5), rng.uniform(-np.pi, np.pi),
                      rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-2, 2),
                      rng.uniform(-.4, .4), rng.uniform(-.4, .4), rng.uniform(-.5, .5), rng.uniform(0, 1)])
        u = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0, 1), rng.uniform(-1, 1), rng.uniform(0, 600)])
        ref = np.array(rk4_ref(x, u, p)).flatten()
        got = np.array(rk4_mod(x, u, p)).flatten()
        worst = max(worst, float(np.max(np.abs(ref - got))))
    return worst


def check_base_models() -> float:
    """Lock the BaseModel wiring: every type-A/B method's base resolves correctly.

    Asserts (a) the registry kinds are valid and every one resolves, (b)
    `nominal_base` resolves to a step matching model.py's nominal_rk4_step (the
    independent oracle), and (c) `greybox_base` resolves to a genuinely different
    model (so it can't silently collapse to the nominal). Returns the worst
    nominal-vs-oracle error.
    """
    from .. import comparison_suite as cs

    cfg = truth.Aircraft6DOFConfig()
    dt = cfg.dt
    # (a) registry well-formed + every kind resolvable
    valid = {"nominal", "greybox"}
    bad = {m: k for m, k in cs.METHOD_BASE_MODEL.items() if k not in valid}
    if bad:
        raise AssertionError(f"METHOD_BASE_MODEL has invalid kinds: {bad}")

    rng = np.random.default_rng(3)
    aero = np.array([gb.SportCubGreyboxConfig().default_parameter_bounds[m].initial
                     for m in gb.SPORTCUB_PARAMETER_NAMES])
    theta_full = gb.SportCubGreyboxConfig().full_parameter_vector(aero)
    nominal_step = cs.resolve_base_step(cs.nominal_base(cfg), dt)
    greybox_step = cs.resolve_base_step(cs.greybox_base(theta_full), dt)

    worst = 0.0
    diff_gb = 0.0
    for _ in range(200):
        spd = rng.uniform(2.6, 11.5)
        vel = np.array([spd, rng.uniform(-2, 2), rng.uniform(-3, 3)])
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        x = np.concatenate([rng.uniform(-5, 5, 3), vel, q, rng.uniform(-1, 1, 3)])
        u = np.array([rng.uniform(0, 1), rng.uniform(-.6, .6), rng.uniform(-.6, .6), rng.uniform(-.6, .6)])
        ref = truth.nominal_rk4_step(x, u, dt, cfg)            # numpy oracle
        worst = max(worst, float(np.max(np.abs(ref - nominal_step(x, u)))))
        diff_gb = max(diff_gb, float(np.max(np.abs(nominal_step(x, u) - greybox_step(x, u)))))
    if diff_gb < 1e-6:
        raise AssertionError("greybox_base resolves to the same step as nominal_base — wiring collapsed")
    return worst


def check_jax() -> float | None:
    """Literal JAX kernel vs model.py for the truth model. Returns None (skip) if
    JAX is unavailable -- the NumPy backend (verified above) is its exact mirror."""
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
    except Exception:
        return None
    rhs_j, _ = md.load_jax_kernel("Aircraft6DOF")
    cfg = truth.Aircraft6DOFConfig()
    base = _truth_base(cfg)
    rng = np.random.default_rng(9)
    worst = 0.0
    for _ in range(200):
        spd = rng.uniform(2.6, 11.5)
        vel = np.array([spd, rng.uniform(-2, 2), rng.uniform(-3, 3)])
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        x = np.concatenate([rng.uniform(-5, 5, 3), vel, q, rng.uniform(-5, 5, 3)])
        u = np.array([rng.uniform(0, 1), rng.uniform(-.65, .65), rng.uniform(-.75, .75), rng.uniform(-.65, .65)])
        ref = truth.rhs(x, u, cfg, nonlinear=True)
        got = np.array(rhs_j(x, u, np.array(base + [1.0])))
        worst = max(worst, float(np.max(np.abs(ref - got))))
    return worst


def main() -> int:
    checks = {
        "truth rhs    (Aircraft6DOF.mo vs model.py)": check_truth_rhs(),
        "truth rk4    (trajectory vs model.py)": check_truth_rk4(),
        "greybox rk4  (SportCubGreybox.mo vs legacy CasADi)": check_greybox(),
        "greybox-lag rk4 (SportCubGreyboxLag.mo vs legacy CasADi)": check_greybox_lag(),
        "base-model wiring (nominal_base vs model.py; greybox_base distinct)": check_base_models(),
    }
    ok = True
    for name, worst in checks.items():
        status = "PASS" if worst < TOL else "FAIL"
        if worst >= TOL:
            ok = False
        print(f"  [{status}] {name}: worst |err| = {worst:.3e}")
    jax_worst = check_jax()
    if jax_worst is None:
        print("  [SKIP] jax rhs    (Aircraft6DOF.mo vs model.py): jax unavailable (numpy mirror covers it)")
    else:
        status = "PASS" if jax_worst < TOL else "FAIL"
        if jax_worst >= TOL:
            ok = False
        print(f"  [{status}] jax rhs    (Aircraft6DOF.mo vs model.py): worst |err| = {jax_worst:.3e}")
    print("modelica parity:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
