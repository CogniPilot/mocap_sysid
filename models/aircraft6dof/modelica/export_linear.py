"""Emit a learned **linear** model as a Modelica file.

The structured models (grey-box, SAFE controller, ground) bake fitted parameters
into a hand-written ``.mo``. A linear/affine model has no hand-written structure
-- its structure *is* the weight matrix -- so the whole ``.mo`` is generated from
the weights. This module does that for the **SAFE closed-loop** invariant fit
(``safe_invariant_weights``): a discrete affine map

    next = W^T @ z,   z = [u,v,w, phi,theta, p,q,r, thr,elev,ail,rud, 1]   (13 features)
    out  = [next u,v,w, next phi,theta, next p,q,r, dpsi]                   (9 outputs)

rendered as the continuous model the inspector already displays,
``der(state) = (W^T z - state)/dt``, with heading integrating the increment and
position integrating the rotated body velocity (the exact ``make_safe_step``
semantics). The result is a standalone, Rumoca-recompilable ``.mo``.
"""

from __future__ import annotations

import numpy as np

# Feature order of z (the first 12; the 13th is the constant bias term).
_FEATURES = ("u", "v", "w", "phi", "theta", "p", "q", "r", "throttle", "elevator", "aileron", "rudder")
# Output columns -> the state whose next value (or increment, for psi) they predict.
_OUT_STATE = ("u", "v", "w", "phi", "theta", "p", "q", "r")  # cols 0..7; col 8 = dpsi


def _affine_expr(col: np.ndarray) -> str:
    """Render one weight column as ``c0*u + c1*v + ... + bias`` (parens guard signs)."""
    terms = [f"({col[k]!r})*{name}" for k, name in enumerate(_FEATURES)]
    terms.append(f"({col[-1]!r})")  # bias (the 13th feature is the constant 1)
    return " + ".join(terms)


def safe_closed_loop_modelica_source(weights, dt: float, *, model_name: str = "SafeClosedLoopIdentified") -> str:
    """Modelica source for the SAFE closed-loop linear fit (13x9 ``weights``)."""
    W = np.asarray(weights, dtype=float)
    if W.shape != (13, 9):
        raise ValueError(f"expected a 13x9 SAFE weight matrix, got {W.shape}")
    dt = float(dt)

    der_lines = []
    for col, state in enumerate(_OUT_STATE):
        der_lines.append(f"  der({state}) = (({_affine_expr(W[:, col])}) - {state})/dt;")
    # heading: integrate the predicted increment (no -psi term)
    der_lines.append(f"  der(psi) = ({_affine_expr(W[:, 8])})/dt;")
    der_block = "\n".join(der_lines)

    return f'''// Generated from the SAFE closed-loop invariant fit (safe_invariant_weights).
// A direct, lumped closed-loop linear model of the SAFE-stabilized dynamics:
// der(state) = (W*[u,v,w,phi,theta,p,q,r, sticks, 1] - state)/dt, with heading
// integrating the fitted increment and position integrating the rotated body
// velocity. No controller/airframe split -- the whole stabilized response is the
// weight matrix. The runtime uses a discrete one-step form (this is its
// continuous idealization, identical at the data rate).
model {model_name}
  parameter Real dt = {dt!r} "Fit sample period [s]";

  input Real throttle;
  input Real elevator;
  input Real aileron;
  input Real rudder;

  Real p_n(start = 0);
  Real p_e(start = 0);
  Real p_d(start = 0);
  Real u(start = 16);
  Real v(start = 0);
  Real w(start = 0);
  Real phi(start = 0);
  Real theta(start = 0);
  Real psi(start = 0);
  Real p(start = 0);
  Real q(start = 0);
  Real r(start = 0);

protected
  Real cphi, sphi, cth, sth, cpsi, spsi;

equation
  // learned linear closed-loop dynamics (8 invariant states + heading increment)
{der_block}

  // position integrates the rotated body velocity (heading-dependent)
  cphi = cos(phi); sphi = sin(phi);
  cth = cos(theta); sth = sin(theta);
  cpsi = cos(psi); spsi = sin(psi);
  der(p_n) = (cth*cpsi)*u + (sphi*sth*cpsi - cphi*spsi)*v + (cphi*sth*cpsi + sphi*spsi)*w;
  der(p_e) = (cth*spsi)*u + (sphi*sth*spsi + cphi*cpsi)*v + (cphi*sth*spsi - sphi*cpsi)*w;
  der(p_d) = (-sth)*u + (sphi*cth)*v + (cphi*cth)*w;
end {model_name};
'''


# State order of the benchmark 13-state (matches model.py STATE_NAMES).
_STATE13 = ("p_n", "p_e", "p_d", "u", "v", "w", "q_w", "q_x", "q_y", "q_z", "p", "q", "r")
_CTRL = ("throttle", "elevator", "aileron", "rudder")


def linear_ss_modelica_source(weights, dt: float, *, model_name: str = "LinearSSIdentified") -> str:
    """Modelica source for the global affine state-space (LinearSS), 18x13 weights.

    Discrete map next = W^T @ [state(13), controls(4), 1]; rendered continuous as
    der(state) = (W^T phi - state)/dt for every state (position and quaternion
    are predicted linearly too -- no kinematic split)."""
    W = np.asarray(weights, dtype=float)
    if W.shape != (18, 13):
        raise ValueError(f"expected an 18x13 LinearSS weight matrix, got {W.shape}")
    feats = [*_STATE13, *_CTRL]  # 17; the 18th is the bias
    der_lines = []
    for col, state in enumerate(_STATE13):
        terms = [f"({W[k, col]!r})*{name}" for k, name in enumerate(feats)]
        terms.append(f"({W[17, col]!r})")
        der_lines.append(f"  der({state}) = (({' + '.join(terms)}) - {state})/dt;")
    decls = "\n".join(f"  Real {s}(start = {16.0 if s == 'u' else (1.0 if s == 'q_w' else 0.0)!r});" for s in _STATE13)
    return f'''// Generated from the LinearSS fit: a single global affine discrete state-space
// x[k+1] = A x[k] + B u + c (here as the continuous der(x) = (W*[x,u,1]-x)/dt).
// No structure beyond the weight matrix; recompiles through Rumoca.
model {model_name}
  parameter Real dt = {float(dt)!r} "Fit sample period [s]";
{chr(10).join(f"  input Real {c};" for c in _CTRL)}
{decls}
equation
{chr(10).join(der_lines)}
end {model_name};
'''


# Invariant base features z = [u,v,w, gx,gy,gz, p,q,r, controls]; gx/gy/gz are the
# body-frame gravity direction from the quaternion (see comparison_suite).
_BASE = ("u", "v", "w", "gx", "gy", "gz", "p", "q", "r", *_CTRL)  # 13


def _poly_term_exprs(degree: int) -> list[str]:
    """Feature expressions in poly_features / linear_features order."""
    if degree == 1:
        return [*_BASE, "1.0"]  # linear_features: invariant(9)+controls(4)+bias = 14
    terms = ["1.0", *_BASE]  # poly_features: 1, z(13), then products i<=j
    for i in range(len(_BASE)):
        for j in range(i, len(_BASE)):
            terms.append(f"({_BASE[i]})*({_BASE[j]})")
    return terms  # 105 for 13 base features


def poly_surrogate_modelica_source(spec: dict, dt: float, *, model_name: str) -> str:
    """Modelica source for an invariant-feature polynomial surrogate.

    Covers SINDy / EquationError (kind='derivative') and Koopman / Symbolic-Stepwise
    (kind='increment'), degree 1 or 2. The standardization ((phi-mean)/scale @ W) is
    folded into raw-feature coefficients. Predicts the 6 dynamic-state derivatives;
    position and quaternion integrate kinematically (as in the rollout)."""
    W = np.asarray(spec["weights"], dtype=float)
    mean = np.asarray(spec["mean"], dtype=float)
    scale = np.asarray(spec["scale"], dtype=float)
    degree = int(spec["degree"])
    increment = spec["kind"] == "increment"
    terms = _poly_term_exprs(degree)
    if W.shape != (len(terms), 6):
        raise ValueError(f"weights {W.shape} do not match {len(terms)} features x 6 outputs")
    c = W / scale[:, None]                       # raw-feature coefficients
    off = -(W * (mean / scale)[:, None]).sum(axis=0)
    dt = float(dt)
    inv_dt = (1.0 / dt) if increment else 1.0    # increment -> divide by dt for a rate

    # der(vel) = outputs 0..2, der(omega) = outputs 3..5
    targets = ("u", "v", "w", "p", "q", "r")
    der_lines = []
    for col, tgt in enumerate(targets):
        parts = [f"({off[col] * inv_dt!r})"] if abs(off[col]) > 1e-12 else []
        for k, term in enumerate(terms):
            if abs(c[k, col]) > 1e-12:
                parts.append(f"({c[k, col] * inv_dt!r})*{term}")
        der_lines.append(f"  der({tgt}) = {' + '.join(parts) if parts else '0.0'};")
    der_block = "\n".join(der_lines)

    return f'''// Generated from the {model_name} fit. Invariant-feature {"quadratic" if degree == 2 else "linear"}
// {"derivative" if not increment else "one-step"} model: der([u,v,w,p,q,r]) from heading/position-invariant
// features [u,v,w, body-gravity gx,gy,gz, p,q,r, controls] (+ their products).
// Position and quaternion integrate kinematically. Standardization folded into the
// coefficients; recompiles through Rumoca.
model {model_name}
  input Real throttle;
  input Real elevator;
  input Real aileron;
  input Real rudder;

  Real p_n(start = 0);
  Real p_e(start = 0);
  Real p_d(start = 0);
  Real u(start = 16);
  Real v(start = 0);
  Real w(start = 0);
  Real q_w(start = 1);
  Real q_x(start = 0);
  Real q_y(start = 0);
  Real q_z(start = 0);
  Real p(start = 0);
  Real q(start = 0);
  Real r(start = 0);

protected
  Real qn, q0, q1, q2, q3, gx, gy, gz;
  Real R11, R12, R13, R21, R22, R23, R31, R32, R33;

equation
  // normalized quaternion + body-frame gravity direction (the invariant attitude)
  qn = max(sqrt(q_w^2 + q_x^2 + q_y^2 + q_z^2), 1e-12);
  q0 = q_w/qn; q1 = q_x/qn; q2 = q_y/qn; q3 = q_z/qn;
  gx = 2.0*(q1*q3 - q0*q2);
  gy = 2.0*(q2*q3 + q0*q1);
  gz = 1.0 - 2.0*(q1^2 + q2^2);

  // learned dynamic-state derivatives
{der_block}

  // body->inertial rotation and kinematic position/quaternion integration
  R11 = 1.0 - 2.0*(q2^2 + q3^2); R12 = 2.0*(q1*q2 - q0*q3); R13 = 2.0*(q1*q3 + q0*q2);
  R21 = 2.0*(q1*q2 + q0*q3); R22 = 1.0 - 2.0*(q1^2 + q3^2); R23 = 2.0*(q2*q3 - q0*q1);
  R31 = 2.0*(q1*q3 - q0*q2); R32 = 2.0*(q2*q3 + q0*q1); R33 = 1.0 - 2.0*(q1^2 + q2^2);
  der(p_n) = R11*u + R12*v + R13*w;
  der(p_e) = R21*u + R22*v + R23*w;
  der(p_d) = R31*u + R32*v + R33*w;
  der(q_w) = 0.5*(-q1*p - q2*q - q3*r);
  der(q_x) = 0.5*( q0*p + q2*r - q3*q);
  der(q_y) = 0.5*( q0*q - q1*r + q3*p);
  der(q_z) = 0.5*( q0*r + q1*q - q2*p);
end {model_name};
'''
