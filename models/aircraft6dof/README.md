# Aircraft 6DOF Model

This package contains the coupled 6DOF aircraft benchmark family.

The current implementation is a deterministic nonlinear small-aircraft model with:

- inertial position,
- body velocity,
- unit quaternion attitude,
- body angular rates,
- throttle, elevator, aileron, and rudder commands.

The truth model uses body-axis aerodynamic force and moment coefficients,
smooth lift rollover, post-stall drag rise, control-effectiveness loss, nonlinear
lateral-directional coupling, first-order actuator lag, noisy direct-state
measurements, and mocap-style position/quaternion measurements. The comparison
suite also exposes an attached-flow nominal model so residual methods can be
tested against hidden nonlinear stall effects.

Run the smoke simulation from the repository root:

```bash
PYTHONPATH=. python3 -m models.aircraft6dof.smoke
```

Generate the default aggressive 6DOF dataset:

```bash
./results.py simulate-6dof
```

Generate the full synthetic 6DOF dataset family:

```bash
./results.py simulate-6dof --dataset-modes open_loop sine_sweep aggressive trim_grid
```

Run the implemented 6DOF baseline methods on an existing dataset:

```bash
./results.py suite-6dof
```

By default this runs the baseline methods on the standard 6DOF dataset family
and aggregates the figures/tables across `open_loop`, `sine_sweep`,
`aggressive`, and `trim_grid`. To run only one dataset, pass it as the sole
mode, for example:

```bash
./results.py suite-6dof --dataset-modes aggressive
```

Run the full local 6DOF workflow:

```bash
./results.py all-6dof
```

The full workflow generates the standard dataset family, runs the baseline
comparison with explicit method-specific training splits and validation splits,
exports GitHub Pages JSON, and refreshes LaTeX-ready tables and figures.
Generated datasets are written to
`work/data/aircraft_6dof_*` and are intentionally ignored by git. The
available 6DOF dataset modes are:

- `open_loop`: small multi-sine pilot inputs around near-trim conditions.
- `sine_sweep`: chirp-like elevator, throttle, aileron, and rudder excitation.
- `aggressive`: large pulses and multisine inputs that drive stall onset, drag rise, pull-up, and recovery.
- `trim_grid`: local small-deviation inputs around a grid of speed, angle-of-attack, and sideslip operating points.

Training/validation assignment:

- no-fit rows such as `6DOF-Nominal` use no training split;
- local linear, model-stitching, subspace, and frequency-stitching rows train on `aircraft_6dof_trim_grid`;
- nonlinear residual, surrogate, symbolic, SINDy, and output-error-style rows train on `aircraft_6dof_aggressive`;
- every trained model is validated open-loop against each requested validation dataset using only the validation initial condition and pilot commands.

Current baseline methods:

- `6DOF-Nominal`: attached-flow RK4 rollout with the supplied pilot-command history.
- `6DOF-LinearSS`: global affine discrete state-space ridge fit.
- `6DOF-Model-Stitching`: airdata-scheduled local affine state-space models.
- `6DOF-Subspace-Hankel`: lagged ARX/Hankel predictor.
- `6DOF-Frequency-Welch`: regularized identified-realization frequency baseline.
- `6DOF-Frequency-Stitching`: airdata-scheduled local realization residuals.
- `6DOF-Koopman-EDMD`: quadratic lifted one-step predictor.
- `6DOF-EquationError-LS`: affine derivative regression.
- `6DOF-EKF-ParamID`: fitted residual parameter vector with open-loop validation.
- `6DOF-Fisher-UQ`: uncertainty-wrapper row around the fitted residual parameter model.
- `6DOF-OEM-SS`: lightweight output-error state-space analogue.
- `6DOF-RidgeResidual`: attached-flow RK4 rollout plus a ridge one-step residual.
- `6DOF-Variational-Mocap`: smoothed weak-form derivative baseline.
- `6DOF-SINDy`: sparse quadratic-library derivative model.
- `6DOF-Symbolic-Stepwise`: sparse quadratic one-step predictor.
- `6DOF-GP-RBF`: sparse RBF residual surrogate.
- `6DOF-UDE-Residual`: nominal dynamics plus quadratic residual closure.
- `6DOF-PINN-Closure`: physics-structured sparse residual closure.
- `6DOF-NN-Surrogate`: random-feature neural residual surrogate.

## Grey-Box OEM Models

The Sport Cub real-flight OEM work should use the framework-owned model
specification in `models.aircraft6dof.greybox`. That module defines the 6DOF
Euler-state model, control order, mocap output order, fixed aircraft constants,
22 estimated lumped aerodynamic parameters, bounds, output weights, and CasADi
RK4 dynamics.

The method family is still OEM/grey-box output error.  The Sport Cub addition is
the 6DOF grey-box model specification and EEM-initialized parameterization, not
a separate ad hoc method family.
