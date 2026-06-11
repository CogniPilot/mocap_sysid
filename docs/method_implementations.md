# Method Implementations: Decisions and Review Guide

This document details every implemented identification method, the reasoning
behind each implementation decision, and the concrete steps each method takes,
so a reviewer can audit the code against the stated algorithm. File references
are to functions (stable across edits); the authoritative code is
`models/aircraft6dof/comparison_suite.py` (6DOF suite),
`models/aircraft6dof/greybox_oem_fit.py` (grey-box OEM),
`models/aircraft6dof/flight_explorer_export.py` (browser export),
`site/src/explorer.js` (browser ports), and root `comparison_suite.py` (3DOF
suite).

## 0. Shared rules that bound every method

These rules were each added in response to a verified failure mode; they apply
to all methods unless stated.

**R1 — Validation data is initial-conditions-and-inputs only.** Every
validation rollout receives one initial state (`validation_x0`) and the pilot
command history (`u_cmd`). No method filters, smooths, or conditions on
validation measurements mid-rollout; measured states appear again only in
scoring. Audited 2026-06-11; the one violation found (3DOF suite defaulting to
`u_act`, the post-controller actuator signal) was fixed — `u_cmd` is the
default and `u_act` is documented as an oracle diagnostic.
*Why:* a predictor with access to realtime measurements (e.g. a Kalman filter
fed validation positions) beats every honest model; the benchmark question is
prediction from inputs alone.

**R2 — Known kinematics are never fitted.** The position derivative
(`ṗ = R(q)v`) and quaternion derivative (`q̇ = ½ q⊗ω`) are exact rigid-body
identities. Generic surrogates fit only the six dynamic states
(u̇, v̇, ẇ, ṗ, q̇, ṙ); rollouts integrate position from the rotated body
velocity and attitude with an exact axis-angle quaternion step
(`kinematic_step` / `kinematicStepJS`).
*Why (measured):* when position increments were regression outputs, a bad
quadratic map translated at 66 m/s against a 12 m/s velocity clamp
(physically impossible 134 m errors in 3 s); when quaternion rows were fitted,
linear-feature methods could not represent the bilinear q⊗ω at all. This is
also the textbook structure: classical equation error (Klein/Morelli) fits
force/moment equations only.

**R3 — Features are heading- and position-invariant.** Surrogate features are
`[u, v, w, g_b, p, q, r, sticks]` where `g_b = Rᵀ(q)·e_z` is the body-frame
gravity direction (`invariant_state`). Inertial positions and raw quaternion
components are excluded.
*Why:* rigid-body flight dynamics are invariant to translation and heading;
attitude enters only through gravity (and airflow already captured by u,v,w).
Raw quaternions let small datasets overfit heading — the 4/17 dataset (1.4k
samples) was the measured victim — and position features made rollouts
structurally divergent (worst published score 9,073 before the fix).

**R4 — Derivative/increment targets are smoothed.** Derivative targets use a
Savitzky–Golay quadratic filter (50 ms window, `savgol_derivative`);
increment targets difference SG-smoothed states (`savgol_states`).
*Why (measured):* raw one-step differences of 240 Hz mocap-derived states
have signal-to-noise of 0.24–0.77 per channel on the 5/22 data — a regression
on raw increments fits 2–4× more noise than dynamics. The 50 ms window is
~12 samples, far below the shortest dynamics of interest (~0.5 s roll mode).

**R5 — Physical envelopes catch divergence honestly.** 6DOF rollouts clamp
speed and rates (`normalize_state`); 3DOF rollouts freeze when the state
leaves a generous physical box (V∈[0.5,100] m/s, |γ|≤2, |α|≤1.5, |Q|≤50
rad/s), keeping one clipped sample so the `diverged` flag triggers.
*Why:* before this, exponential blowups posted "scores" like α-RMSE = 1,160
radians with `diverged=False`.

**R6 — Real-data preprocessing is identical for all methods.**
`trim_to_input_onset` splits each manual maneuver into a quasi-steady lead-in
and an actuation window; per-flight stick trim bias is the pooled
data-compatibility estimate from lead-ins (`estimate_input_bias`, solving
J_u·b = f(x̄,ū) − ẋ_obs over input-free spans); actuation windows are cut
into 0.6–1.2 s gap-free chunks with the initial state estimated by a local
fit at each chunk start. Train/validation assignment is round-robin per
flight (every third manual window held out). Mocap dropout samples are never
trained on or scored.

**Honesty labels.** Every result row carries `implementation_status`
(`implemented` / `analogue` / `placeholder`) and `diverged`. `analogue`
means the row is a defensible stand-in that does not run the named
algorithm's full machinery; the notes state exactly what it is.

---

## 1. 6DOF-GreyBoxOEM (`greybox_oem_fit.py`) — implemented

**What it is.** Segment-wise output-error estimation of the 22-coefficient
physical Sport Cub model (`greybox.py`: CL0, CLa, CD0, CDCLS, CYb, KT, and
roll/pitch/yaw moment-coefficient groups with b/2V and c̄/2V rate terms).

**Steps.**
1. Load the manual training chunks (R6); convert quaternion states to the
   grey-box's 12 Euler states (`quat_states_to_euler`).
2. Reorder controls from the benchmark's `INPUT_NAMES` order to the model's
   `CONTROL_NAMES_OEM` via `OEM_CONTROL_ORDER` — derived from the name lists,
   never hardcoded (a hardcoded map once silently swapped aileron/elevator;
   scores barely moved but parameter signs were unphysical).
3. Build a CasADi RK4 step at a 60 Hz model rate (`build_casadi_dynamics`).
   60 Hz is 4× the fastest fitted dynamics and 4× cheaper than 240 Hz.
4. Roll each chunk open-loop from its measured initial state
   (`chunk_rollouts`); no shooting-node continuity variables are needed
   because chunk ICs are measured (this is segment-wise output error, not
   multiple shooting — the paper says so explicitly).
5. Residuals on mocap-observable outputs only — position and Euler attitude,
   ψ wrapped — weighted by the spec's output sigmas (`output_residuals`).
6. Solve with scipy `least_squares` (TRF) inside physical parameter bounds.
   Bounds were widened after the fit pinned KT/CL0/CLa (the original KT cap
   of 0.85 m/s² per throttle could not balance cruise drag ≈ 2.3); the
   fitter prints and records `parameters_at_bounds` (within 1% of the box)
   so bound starvation cannot pass silently. CDCLS pinned at its *physical*
   zero bound is accepted (induced drag is weakly identifiable over the
   narrow CL range).
7. Thrust acts along body-x (a propeller does not rotate with the airflow);
   this was a verified bug fix worth 19% of validation NRMSE.

**Known structural limits (stated in the paper):** no actuator/servo lag, no
thrust airspeed/battery dependence, no prop-wash tail augmentation, no CLq or
α̇ terms.

## 2. 6DOF-LinearSS — implemented

One-step affine map x[k+1] = Wᵀ[x, u, 1] by ridge regression
(`design_matrix` + `ridge_fit`) on the training chunks; open-loop validation
rollout with `normalize_state` clamps. Positions remain in this feature set
*by contract* (the browser's `affinePredict` and the historical rows share
the 18-column layout); the near-identity discrete map keeps the position
columns benign, unlike the derivative/increment surrogates. The Model
Inspector shows the map as continuous-time equations ((W−I)/dt).

## 3. 6DOF-RidgeResidual — implemented

The attached-flow nominal model's RK4 step plus a ridge one-step residual on
the same affine features. The residual absorbs actuator lag and (on synthetic
data) hidden stall aerodynamics. On real data its base model has wrong-signed
lateral control derivatives — documented; the honest fix is the UDE row.

## 4. 6DOF-EquationError-LS — implemented

Classical equation-error least squares (Klein/Morelli), correctly scoped:
1. Derivative targets = SG-smoothed derivatives (R4) of the six dynamic
   states only (R2).
2. Features = linear invariant features (R3) — note the gravity-direction
   components are what allow a *linear* feature set to represent the gravity
   term at all.
3. Standardized ridge solve (`fit_standardized_ridge`); validation by
   explicit-Euler integration of the fitted derivative model inside the
   kinematic stepper.
Remaining honest weakness: a single global linear model in derivative form
accumulates bias over long rollouts where the dynamics are
amplitude-dependent; the score reflects that, not an implementation error.

## 5. 6DOF-SINDy — implemented

Sequentially thresholded least squares (Brunton et al.), actually iterated:
1. Quadratic library on invariant features (`poly_features`, 105 terms).
2. Targets: smoothed derivatives of the six dynamic rows.
3. `stlsq_fit`: ridge fit → magnitude threshold (per-output quantile,
   constant+linear block protected) → ridge **refit on the surviving
   features** → repeat to fixed point (≤5 iterations). A single post-hoc
   prune (the previous implementation) leaves dense-fit coefficient values
   in place and is not STLSQ; the backend string only says `numpy-stlsq`
   since this fix.
4. Library choice: generic quadratic rather than aerodynamic-coefficient
   structure — deliberate, to test structure *discovery*; the structured
   alternative is the grey-box row.

## 6. 6DOF-Koopman-EDMD — analogue

Quadratic one-step **increment** predictor (smoothed targets, R4) on
invariant features with heavy ridge (100×). Labeled `analogue` because lifted
observables are not propagated (predictions return to the state space each
step) — i.e., this tests the EDMD feature class, not the Koopman operator
machinery. Remaining high error on the 4/17 dataset is small-data quadratic
overfit (1.4k samples vs 105 features) and is reported, not hidden.

## 7. 6DOF-Symbolic-Stepwise — analogue

Same library/targets as SINDy but with `stlsq_fit` at a milder sparsity
(12%); labeled `analogue` because it does not run statistical forward
selection (no PSE/PRESS criteria à la Klein's stepwise regression). Exists to
separate "sparsity level" from "library" effects against SINDy.

## 8. 6DOF-Subspace-Hankel — analogue

Lag-3 ARX on the invariant state history predicting smoothed dynamic-state
increments, ridge-fit, increment-integrated under R2 kinematics. Labeled
`analogue`: no Hankel SVD or N4SID/MOESP realization. The browser port
re-seeds its lag window whenever the incoming state is not its own last
output (fresh anchor or SAFE handoff). After R2/R3 it ties the grey-box on
real data (0.106 vs 0.102) — a strong argument that most of the predictable
signal is short-memory linear.

## 9. 6DOF-GP-RBF — implemented (deterministic core)

Sparse-GP posterior mean with fixed hyperparameters:
96 RBF centers subsampled from training inputs, per-dimension length scales
set to feature standard deviations, ridge weights; the kernel residual
corrects the six dynamic rows of the nominal model's step. Honest limitations
in the notes: no marginal-likelihood hyperparameter optimization, no
propagated predictive variance. (The paper's kernel-methods subsection states
the same.)

## 10. 6DOF-UDE-Residual — analogue (deterministic)

The UDE premise is *best known physics + learned residual*
(Rackauckas et al.). Implementation decisions:
1. **Base model:** on real Sport Cub scenarios the base is the *fitted
   grey-box* (one shared fit, hoisted before this block); on synthetic
   scenarios it is the attached-flow nominal. Using the synthetic nominal on
   real data forced the residual to overpower a base with wrong-signed
   lateral derivatives — measured at 12.1 NRMSE before, 0.109 after.
2. **Residual targets:** smoothed next dynamic states minus the base model's
   one-step prediction (`greybox_one_step`).
3. **Shrinkage scaled by sample count:** the standardized gram grows with n,
   so a fixed λ of 1e-4 was ~1e-5 of the gram diagonal — effectively
   unregularized. λ = 1.0·n shrinks a noise-dominated residual toward zero,
   correctly returning ≈ the base when there is nothing to learn, plus a
   small genuine correction (0.095 vs base 0.102 on 5/22).
4. **Rollout:** base step, residual added to the six dynamic rows, kinematics
   integrated from the corrected state (otherwise attitude drifts with
   uncorrected rates before the residual lands — measured failure).
Labeled `analogue` because the residual is ridge, not a trained network; the
3DOF tier has the genuine ODE-in-the-loop torch UDE.

## 11. 6DOF placeholders (labeled, excluded from claims)

`Frequency-Welch`/`Frequency-Stitching` (regularized realizations, no
spectral content), `EKF-ParamID`/`Fisher-UQ`/`OEM-SS` (one ridge fit reported
under three names — notes say so), `Variational-Mocap` (smoothed weak-form
ridge), `PINN-Closure` (sparsified copy of the UDE weights),
`NN-Surrogate` (random-feature closed form). Each is `placeholder` in the
CSV/website; none is presented as the named method.

## 12. SAFE closed-loop model (`flight_explorer_export.py`) — stabilized handoff

One shared linear model for SAFE-mode flight, used by every method (free runs
hand off on the recorded mode channel, state continuous across the switch):
1. **Why direct closed-loop ID:** stabilized sticks are attitude commands;
   the bare airframe cannot replay them. Composing the separately identified
   controller with each airframe was measured worse (13.6 m vs 2.9 m at
   5 s) because the inverse-dynamics controller estimate is weakly
   conditioned. This is the textbook *direct method* of closed-loop
   identification.
2. **Invariant formulation:** regress only [u,v,w,φ,θ,p,q,r] plus the
   heading *increment* Δψ from [state, stick, 1]; heading integrates Δψ and
   position integrates R(q)v exactly. A raw-state affine fit cannot
   represent heading-dependent position kinematics and wandered within
   seconds (8.3 m → 2.9 m at 5 s after this change).
3. **Validation discipline:** stabilized spans are cut into 10 s windows,
   round-robin per flight, every third held out; the model trains on train
   windows only and reports held-out 5 s free-run position error (3.4 m).
4. **Exclusions:** the autonomous flight (its lateral commands bypass the
   recorded sticks — rudder channel literally constant while the aircraft
   banks) and mocap dropouts.

## 13. SAFE controller grey-box (`safe_controller.py`) — interpretable decomposition

Guessed structure δ = Kp·(sat(scale·stick, ±envelope) − attitude) − Kd·rate
+ bias per axis (rudder: stick gain, yaw-rate gain, bias), fitted by a staged
grid over (scale, envelope) with Huber regression for the gains. Retained for
interpretability and as the fallback when no closed-loop fit exists; its
gains were identified against the nominal airframe, which is why composition
currently loses to the direct fit. Re-identifying against the fitted
grey-box is the planned upgrade toward per-method SAFE handoff.

## 14. 3DOF tier (root `comparison_suite.py`, `methods/`)

The 3DOF methods are documented in the paper's formulation sections; the
implementation decisions worth review:
- **OEM single vs multiple shooting:** the suite's OEM is CasADi multiple
  shooting (soft continuity residuals, IPOPT budget 200, convergence status
  recorded; rows are not published from non-converged solves).
- **Filter-error EKF:** augmented-state EKF (Joseph form, CasADi Jacobians)
  runs on *training* data only; validation is a frozen-θ open-loop rollout.
- **UDE/PINN (torch):** trained RK4-in-the-loop over multiple-shooting
  segments of measured states — derivative targets are never formed (the
  old derivative-matching design was unusable at the original dt and was
  redesigned; the benchmark dt is 0.02 s for Nyquist/RK4-stability reasons).
- **SINDy:** trig features replaced by Taylor residuals (sin α − α, …) to
  break library collinearity; thresholding and the reported active mask are
  consistent; defaults tuned at dt = 0.02.
- **Trim:** solves the full equilibrium including the lift equation
  (residual ~1e-16), not the legacy thrust/moment-only point.
- **Frequency-domain fit:** ETFE with correct `csd(input, output)` ordering
  (phase was conjugated before), input-power weighting (not claimed to be
  coherence), and unstable identified poles reflected into the left
  half-plane before open-loop simulation.
- **Envelope/diverged:** R5.

## 15. Browser ports (`site/src/explorer.js`)

Every exported model has a JavaScript stepper implementing the same equations
as its Python counterpart; the grey-box port is verified against the CasADi
RK4 step to machine precision (0 diff), and all ten steppers are smoke-tested
in node for bounded, finite 3 s free runs from real anchors. Steppers share
`kinematicStepJS` (R2), `invariantStateJS` (R3), and `postStep` clamps. The
free-run loop (`rolloutFrom`) switches between the manual-segment airframe
and the SAFE closed-loop model on the recorded mode channel and stops at
ground contact (no gear model).
