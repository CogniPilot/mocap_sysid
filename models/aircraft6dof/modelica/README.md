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
| `SportCubGreybox.mo` | Rumoca interactive fixed-wing plant, ported to this benchmark's state/input convention | 12-state Euler |

`Aircraft6DOF.mo` has an `nl` parameter (`1` = nonlinear truth, `0` = attached-flow
nominal) matching `model.py`'s `nonlinear=` flag.

Post-integration guards (quaternion renorm / speed & rate clamps for the truth
model, yaw wrap for the grey-box) are **integration details, not continuous
dynamics**, and live in the Python RK4 wrapper (`dynamics.py`), not the `.mo`.

## How it works

```
Aircraft6DOF.mo ──rumoca Python API / CLI target render──┐
                                                                     ▼
                       ignored local cache: generated/<Model>_casadi_solve.py
                                            generated/<Model>_jax_solve.py
                                          │
                       dynamics.py  (RK4 + post-step wrappers)
                                          │
                  greybox.build_casadi_dynamics  ·  comparison_suite
```

The explicit ODE `xdot = rhs(x, u, p)` is rendered directly from Rumoca's
scalarized, causalized **solve IR** (the same source the C / FMI / Rust backends
use) — no DAE residual, no rootfinder, no runtime interpreter. The generated
kernels are local cache artifacts and are not committed; the project pins
`rumoca==0.9.8` so a normal Python install can regenerate them on demand. A
`generated/*.solve.json` IR dump is also written locally for inspection.

## Workflow

Regenerate the local cache after editing a `.mo` (uses the `rumoca` Python
package, with `$RUMOCA` / `rumoca` CLI fallback):

```bash
python -m models.aircraft6dof.modelica.generate
```

Verify the generated kernels still match the reference implementations
(also run automatically by `./results.py check-setup`):

```bash
python -m models.aircraft6dof.modelica.check_parity
```

## Backends

Each model is rendered by Rumoca into standalone local cache modules, both
exposing `rhs(x, u, p) -> xdot` (states in `STATE_NAMES`, inputs in
`INPUT_NAMES`, params in `PARAM_NAMES` / model-declaration order):

- `generated/<Model>_casadi_solve.py` → CasADi `Function('rhs', [x,u,p], [xdot])`
- `generated/<Model>_jax_solve.py` → pure-`jnp`, `jit`/`grad`/`vmap`-compatible `rhs`

`dynamics.py` ensures the cache is present, then imports these via
`_casadi_kernel(stem)` / `load_jax_kernel(stem)`.

## Notes

- These Modelica kernels are the only runtime path. `greybox.build_casadi_dynamics`,
  and `nominal_rk4_step` dispatch here with no fallback. The hand-written numpy
  truth model (`model.py`) is kept solely as the independent parity oracle for
  `Aircraft6DOF.mo`.
- `SportCubGreybox.mo` bakes the default
  `max_deflection_deg` as constants; a non-default config would not be honoured by
  the Modelica path (edit the `.mo` and regenerate instead).
- The synthetic-data *truth* generator (`model.rk4_step`, nl=1) still runs on numpy
  `model.py` — it is the data oracle, not a method input. `Aircraft6DOF.mo` (nl=1)
  is verified bit-identical to it.
