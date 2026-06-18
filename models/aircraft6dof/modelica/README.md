# Modelica-generated 6DOF dynamics

The `.mo` models here are the **single source of truth** for the 6DOF aircraft
physics. Continuous dynamics are written once in Modelica and compiled with the
[Rumoca](https://github.com/cognipilot/rumoca) compiler into an explicit
`xdot = f(x, u, p)` kernel that the benchmark consumes through CasADi (and,
optionally, JAX).

## Models

| Source | Reference it reproduces | State |
|---|---|---|
| `Aircraft6DOF.mo` | `model.py` (nonlinear truth model) | 13-state quaternion |
| `SportCubGreybox.mo` | `greybox._build_casadi_dynamics_legacy` | 12-state Euler |

`Aircraft6DOF.mo` has an `nl` parameter (`1` = nonlinear truth, `0` = attached-flow
nominal) matching `model.py`'s `nonlinear=` flag.

Post-integration guards (quaternion renorm / speed & rate clamps for the truth
model, yaw wrap for the grey-box) are **integration details, not continuous
dynamics**, and live in the Python RK4 wrapper (`dynamics.py`), not the `.mo`.

## How it works

```
Aircraft6DOF.mo ──rumoca compile --target casadi-solve / jax-solve──┐
                                                                     ▼
                       generated/<Model>_casadi_solve.py   (ca.Function rhs(x,u,p))
                       generated/<Model>_jax_solve.py       (jit/grad/vmap rhs(x,u,p))
                                          │
                       dynamics.py  (RK4 + post-step wrappers)
                                          │
                  greybox.build_casadi_dynamics  ·  comparison_suite
```

The explicit ODE `xdot = rhs(x, u, p)` is rendered directly from Rumoca's
scalarized, causalized **solve IR** (the same source the C / FMI / Rust backends
use) — no DAE residual, no rootfinder, no runtime interpreter. The generated
kernels are self-contained and committed, so **the benchmark does not need Rumoca
installed at runtime** — only to regenerate after a `.mo` change. A
`generated/*.solve.json` IR dump is also written for inspection.

## Workflow

Regenerate the cached IR after editing a `.mo` (needs `rumoca` on PATH or `$RUMOCA`):

```bash
python -m models.aircraft6dof.modelica.generate
```

Verify the generated kernels still match the reference implementations
(also run automatically by `./results.py check-setup`):

```bash
python -m models.aircraft6dof.modelica.check_parity
```

## Backends

Each model is rendered from the solve IR into standalone kernel modules, both
exposing `rhs(x, u, p) -> xdot` (states in `STATE_NAMES`, inputs in
`INPUT_NAMES`, params in `PARAM_NAMES` / model-declaration order):

- `generated/<Model>_casadi_solve.py` → CasADi `Function('rhs', [x,u,p], [xdot])`
- `generated/<Model>_jax_solve.py` → pure-`jnp`, `jit`/`grad`/`vmap`-compatible `rhs`

`dynamics.py` imports these via `_casadi_kernel(stem)` / `load_jax_kernel(stem)`.

## Notes

- These Modelica kernels are the only runtime path. `greybox.build_casadi_dynamics`,
  `build_casadi_dynamics_lag`, and `nominal_rk4_step` dispatch here with no
  fallback. The hand-written numpy (`model.py`) and CasADi
  (`greybox._build_casadi_dynamics*_legacy`) implementations are kept solely as the
  independent parity oracles in `check_parity.py`.
- `SportCubGreybox.mo` / `SportCubGreyboxLag.mo` bake the default
  `max_deflection_deg` as constants; a non-default config would not be honoured by
  the Modelica path (edit the `.mo` and regenerate instead).
- The synthetic-data *truth* generator (`model.rk4_step`, nl=1) still runs on numpy
  `model.py` — it is the data oracle, not a method input. `Aircraft6DOF.mo` (nl=1)
  is verified bit-identical to it.
