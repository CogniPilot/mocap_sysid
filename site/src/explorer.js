// Flight explorer: full-flight segmentation viewer with on-the-fly model rollouts.
//
// Loads flight_explorer.json (truth, per-sample segmentation labels, full-rate
// sticks, fitted model parameters) and integrates method predictions in the
// browser from any clicked segment or time: manual segments use the recorded
// sticks re-referenced by the per-flight trim bias; stabilized segments close
// the loop with the identified SAFE inner-loop controller model, because the
// bare airframe alone cannot represent stabilized flight. Computing rollouts
// client-side avoids exporting a trace per (segment, method) pair.

const DATA_URL = "./public/data/flight_explorer.json";
const LABEL_COLORS = { ground: "#8d6e63", ground_effect: "#26a69a", stabilized: "#5c7cfa", manual: "#f08c00" };
const METHOD_COLORS = { "6DOF-Nominal": "#d62728", "6DOF-LinearSS": "#2ca02c", "6DOF-RidgeResidual": "#9467bd", "6DOF-GreyBoxOEM": "#e8a838", "6DOF-EquationError-LS": "#17becf", "6DOF-SINDy": "#e377c2", "6DOF-Koopman-EDMD": "#bcbd22", "6DOF-Symbolic-Stepwise": "#8c564b", "6DOF-Subspace-Hankel": "#1f77b4", "6DOF-GP-RBF": "#f7b6d2" };
const MIN_SPEED = 2.5;
const MAX_SPEED = 12.0;

const ex = {
  data: null,
  flightIndex: 0,
  anchorTimeS: null,
  selectedMethods: new Set(),
  predictions: {},
  // Whether the playback is showing this module's dataset. While another
  // dataset (3DOF, synthetic 6DOF) is displayed the explorer must stay
  // silent: publishing an overlay would hijack the view back to the Sport
  // Cub flights.
  active: true,
};

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function normQuat(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

function eulerFromQuat(q) {
  const [q0, q1, q2, q3] = normQuat(q);
  return [
    Math.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2)),
    Math.asin(clamp(2 * (q0 * q2 - q3 * q1), -1, 1)),
    Math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3)),
  ];
}

function rotationBodyToInertial(q) {
  const [q0, q1, q2, q3] = q;
  return [
    [1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
    [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 - q0 * q1)],
    [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2)],
  ];
}

function matVec(m, v) {
  return [m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2], m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2], m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]];
}

function matTVec(m, v) {
  return [m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2], m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2], m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2]];
}

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

// Attached-flow nominal 6DOF dynamics ported from models/aircraft6dof/model.py
// (the nonlinear=False path: no stall gate, no hidden residual terms).
function nominalRhs(x, u, cfg) {
  const vel = [x[3], x[4], x[5]];
  const quat = normQuat([x[6], x[7], x[8], x[9]]);
  const rates = [x[10], x[11], x[12]];
  const speed = Math.max(Math.hypot(...vel), 1e-6);
  const alpha = Math.atan2(vel[2], Math.max(vel[0], 1e-6));
  const beta = Math.asin(clamp(vel[1] / speed, -0.98, 0.98));
  const throttle = clamp(u[0], 0, 1);
  const [, elevator, aileron, rudder] = u;
  const rateScale = Math.max(2 * speed, 1e-6);
  const pHat = (cfg.wing_span * rates[0]) / rateScale;
  const qHat = (cfg.mean_chord * rates[1]) / rateScale;
  const rHat = (cfg.wing_span * rates[2]) / rateScale;

  const cl = 0.27 + 5.2 * alpha + 0.36 * elevator + 3.1 * qHat;
  const cd = 0.045 + 0.075 * cl * cl + 0.42 * beta * beta + 0.018 * (aileron * aileron + rudder * rudder) + 0.01 * elevator * elevator;
  const cm = 0.03 - 1.05 * alpha - 1.15 * elevator - 9.0 * qHat;
  const cy = -0.82 * beta + 0.3 * rudder + 0.12 * aileron - 0.35 * rHat;
  const clRoll = -0.12 * beta + 0.42 * aileron - 0.5 * pHat + 0.1 * rHat;
  const cn = 0.18 * beta - 0.26 * rudder - 0.08 * aileron - 0.42 * rHat;
  const cx = -cd * Math.cos(alpha) + cl * Math.sin(alpha);
  const cz = -cd * Math.sin(alpha) - cl * Math.cos(alpha);

  const qbar = 0.5 * cfg.rho * speed * speed;
  const propWash = 1 + cfg.prop_wash_gain * throttle;
  const thrust = cfg.max_thrust * Math.pow(throttle, 1.45);
  const force = [
    propWash * qbar * cfg.wing_area * cx + thrust,
    propWash * qbar * cfg.wing_area * cy,
    propWash * qbar * cfg.wing_area * cz,
  ];
  const moment = [
    propWash * qbar * cfg.wing_area * cfg.wing_span * clRoll,
    propWash * qbar * cfg.wing_area * cfg.mean_chord * cm + cfg.prop_arm * thrust,
    propWash * qbar * cfg.wing_area * cfg.wing_span * cn,
  ];

  const rot = rotationBodyToInertial(quat);
  const posDot = matVec(rot, vel);
  const gravBody = matTVec(rot, [0, 0, cfg.gravity]);
  const velDot = [
    force[0] / cfg.mass + gravBody[0] - (rates[1] * vel[2] - rates[2] * vel[1]),
    force[1] / cfg.mass + gravBody[1] - (rates[2] * vel[0] - rates[0] * vel[2]),
    force[2] / cfg.mass + gravBody[2] - (rates[0] * vel[1] - rates[1] * vel[0]),
  ];
  const [ix, iy, iz] = cfg.inertia;
  const ixz = cfg.inertia_xz;
  const h = [ix * rates[0] - ixz * rates[2], iy * rates[1], iz * rates[2] - ixz * rates[0]];
  const torque = [moment[0] - (rates[1] * h[2] - rates[2] * h[1]), moment[1] - (rates[2] * h[0] - rates[0] * h[2]), moment[2] - (rates[0] * h[1] - rates[1] * h[0])];
  // Solve [ix,0,-ixz;0,iy,0;-ixz,0,iz] wdot = torque (2x2 block + scalar).
  const det = ix * iz - ixz * ixz;
  const ratesDot = [(iz * torque[0] + ixz * torque[2]) / det, torque[1] / iy, (ixz * torque[0] + ix * torque[2]) / det];
  const [q0, q1, q2, q3] = quat;
  const [p, qr, r] = rates;
  const quatDot = [
    0.5 * (-q1 * p - q2 * qr - q3 * r),
    0.5 * (q0 * p + q2 * r - q3 * qr),
    0.5 * (q0 * qr - q1 * r + q3 * p),
    0.5 * (q0 * r + q1 * qr - q2 * p),
  ];
  return [...posDot, ...velDot, ...quatDot, ...ratesDot];
}

function quatFromEuler(roll, pitch, yaw) {
  const cr = Math.cos(0.5 * roll), sr = Math.sin(0.5 * roll);
  const cp = Math.cos(0.5 * pitch), sp = Math.sin(0.5 * pitch);
  const cy = Math.cos(0.5 * yaw), sy = Math.sin(0.5 * yaw);
  return normQuat([
    cr * cp * cy + sr * sp * sy,
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
  ]);
}

// Sport Cub grey-box OEM dynamics ported from models/aircraft6dof/greybox.py
// (build_casadi_dynamics). Twelve Euler states (pos NED, body uvw, roll/pitch/
// yaw, body rates); u is the stick vector in (throttle, elevator, aileron,
// rudder) order; gb carries the fitted+fixed parameters by name.
function greyboxRhs(s, u, gb) {
  const p = gb.p;
  const [uB, vB, wB] = [s[3], s[4], s[5]];
  const [phi, theta, psi] = [s[6], s[7], s[8]];
  const [pR, qR, rR] = [s[9], s[10], s[11]];
  const thr = Math.max(u[0], 0);
  const elevRad = gb.maxDefl.elevator * (Math.PI / 180) * u[1];
  const ailRad = gb.maxDefl.aileron * (Math.PI / 180) * u[2];
  const rudRad = gb.maxDefl.rudder * (Math.PI / 180) * u[3];

  const speed = Math.max(Math.sqrt(uB * uB + vB * vB + wB * wB + 1e-9), 1e-3);
  const alpha = Math.atan2(wB, uB);
  const beta = Math.asin(clamp(vB / speed, -0.99, 0.99));
  const qbar = 0.5 * p.rho * speed * speed;
  const cA = Math.cos(alpha), sA = Math.sin(alpha);
  const cB = Math.cos(beta), sB = Math.sin(beta);

  const CL = p.CL0 + p.CLa * alpha;
  const CD = p.CD0 + p.CDCLS * CL * CL;
  const lift = qbar * p.S * CL;
  const drag = qbar * p.S * CD;
  const side = qbar * p.S * (p.CYb * beta);
  const thrust = p.KT * p.m * thr;

  // Aerodynamic forces rotate from wind axes; thrust is body-fixed along x.
  const fx = -drag * cA * cB - side * cA * sB + lift * sA + thrust;
  const fy = -drag * sB + side * cB;
  const fz = -drag * sA * cB - side * sA * sB - lift * cA;

  const cPhi = Math.cos(phi), sPhi = Math.sin(phi);
  const cTh = Math.cos(theta), sTh = Math.sin(theta);
  const cPsi = Math.cos(psi), sPsi = Math.sin(psi);

  const uDot = fx / p.m - p.g * sTh + rR * vB - qR * wB;
  const vDot = fy / p.m + p.g * sPhi * cTh + pR * wB - rR * uB;
  const wDot = fz / p.m + p.g * cPhi * cTh + qR * uB - pR * vB;

  const bV = p.b / (2 * speed);
  const cV = p.cbar / (2 * speed);
  const rollAccel = qbar * (p.KL0 + p.KLb * beta + p.KLp * bV * pR + p.KLr * bV * rR + p.KLda * ailRad + p.KLdr * rudRad);
  const pitchAccel = qbar * (p.KM0 + p.KMa * alpha + p.KMq * cV * qR + p.KMe * elevRad);
  const yawAccel = qbar * (p.KN0 + p.KNb * beta + p.KNp * bV * pR + p.KNr * bV * rR + p.KNda * ailRad + p.KNdr * rudRad);

  const pDot = rollAccel + ((p.Iyy - p.Izz) / p.Ixx) * qR * rR + (p.Ixz / p.Ixx) * pR * qR;
  const qDot = pitchAccel + ((p.Izz - p.Ixx) / p.Iyy) * pR * rR + (p.Ixz / p.Iyy) * (rR * rR - pR * pR);
  const rDot = yawAccel + ((p.Ixx - p.Iyy) / p.Izz) * pR * qR + (p.Ixz / p.Izz) * qR * rR;

  const cThSafe = Math.sign(cTh || 1) * Math.max(Math.abs(cTh), 1e-3);
  const common = qR * sPhi + rR * cPhi;
  const phiDot = pR + (sTh / cThSafe) * common;
  const thetaDot = qR * cPhi - rR * sPhi;
  const psiDot = common / cThSafe;

  return [
    cTh * cPsi * uB + (sPhi * sTh * cPsi - cPhi * sPsi) * vB + (cPhi * sTh * cPsi + sPhi * sPsi) * wB,
    cTh * sPsi * uB + (sPhi * sTh * sPsi + cPhi * cPsi) * vB + (cPhi * sTh * sPsi - sPhi * cPsi) * wB,
    -sTh * uB + sPhi * cTh * vB + cPhi * cTh * wB,
    uDot, vDot, wDot,
    phiDot, thetaDot, psiDot,
    pDot, qDot, rDot,
  ];
}

export function makeGreyboxStepper(greybox, dt) {
  const p = { ...greybox.fixed_parameters };
  greybox.parameter_names.forEach((name, i) => { p[name] = greybox.parameters[i]; });
  const gb = { p, maxDefl: greybox.max_deflection_deg };
  return (x, u) => {
    const e = eulerFromQuat([x[6], x[7], x[8], x[9]]);
    let s = [x[0], x[1], x[2], x[3], x[4], x[5], e[0], e[1], e[2], x[10], x[11], x[12]];
    const k1 = greyboxRhs(s, u, gb);
    const k2 = greyboxRhs(s.map((v, i) => v + 0.5 * dt * k1[i]), u, gb);
    const k3 = greyboxRhs(s.map((v, i) => v + 0.5 * dt * k2[i]), u, gb);
    const k4 = greyboxRhs(s.map((v, i) => v + dt * k3[i]), u, gb);
    s = s.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
    s[8] = Math.atan2(Math.sin(s[8]), Math.cos(s[8]));
    const q = quatFromEuler(s[6], s[7], s[8]);
    return [s[0], s[1], s[2], s[3], s[4], s[5], q[0], q[1], q[2], q[3], s[9], s[10], s[11]];
  };
}

// Heading/position-invariant closed-loop SAFE model: the regression predicts
// the next body velocities, roll/pitch, rates, and the heading increment from
// [u,v,w,phi,theta,p,q,r, stick, 1]; heading integrates the increment and
// position integrates the rotated body velocity exactly, so free runs can fly
// whole laps instead of wandering off the affine fit's operating point.
export function makeSafeStepper(W, dt) {
  return (x, stick) => {
    const quat = normQuat([x[6], x[7], x[8], x[9]]);
    const e = eulerFromQuat(quat);
    const rot = rotationBodyToInertial(quat);
    const step = matVec(rot, [x[3], x[4], x[5]]);
    const z = [x[3], x[4], x[5], e[0], e[1], x[10], x[11], x[12], stick[0], stick[1], stick[2], stick[3], 1];
    const out = new Array(9).fill(0);
    for (let i = 0; i < z.length; i++) {
      const row = W[i];
      for (let j = 0; j < 9; j++) out[j] += z[i] * row[j];
    }
    const psi = Math.atan2(Math.sin(e[2] + out[8]), Math.cos(e[2] + out[8]));
    const q = quatFromEuler(out[3], out[4], psi);
    return [
      x[0] + step[0] * dt, x[1] + step[1] * dt, x[2] + step[2] * dt,
      out[0], out[1], out[2],
      q[0], q[1], q[2], q[3],
      out[5], out[6], out[7],
    ];
  };
}

function postStep(x) {
  const out = x.slice();
  const q = normQuat([out[6], out[7], out[8], out[9]]);
  out[6] = q[0]; out[7] = q[1]; out[8] = q[2]; out[9] = q[3];
  const speed = Math.hypot(out[3], out[4], out[5]);
  if (speed > MAX_SPEED) {
    for (let i = 3; i < 6; i++) out[i] *= MAX_SPEED / speed;
  } else if (speed > 1e-9 && speed < MIN_SPEED) {
    for (let i = 3; i < 6; i++) out[i] *= MIN_SPEED / speed;
  }
  for (let i = 10; i < 13; i++) out[i] = clamp(out[i], -8, 8);
  return out;
}

function nominalStep(x, u, dt, cfg) {
  const k1 = nominalRhs(x, u, cfg);
  const x2 = x.map((v, i) => v + 0.5 * dt * k1[i]);
  const k2 = nominalRhs(x2, u, cfg);
  const x3 = x.map((v, i) => v + 0.5 * dt * k2[i]);
  const k3 = nominalRhs(x3, u, cfg);
  const x4 = x.map((v, i) => v + dt * k3[i]);
  const k4 = nominalRhs(x4, u, cfg);
  return postStep(x.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])));
}

function affinePredict(x, u, weights) {
  const phi = [...x, ...u, 1];
  const out = new Array(13).fill(0);
  for (let i = 0; i < phi.length; i++) {
    const row = weights[i];
    const value = phi[i];
    for (let j = 0; j < 13; j++) out[j] += value * row[j];
  }
  return out;
}

// Generic neural-network evaluator for browser-side prediction. NumPy-trained
// nets (random-feature readouts, MLPs, RBF expansions) deploy as JSON weight
// specs and run through this forward pass; torch-trained models deploy as
// ONNX and run through onnxruntime-web instead, with the same integration
// loop calling the session per step.
function evalNet(spec, input) {
  if (spec.type === "rbf") {
    const z = input.map((v, i) => (v - spec.x_mean[i]) / spec.x_scale[i]);
    const phi = [1];
    for (const center of spec.centers) {
      let d2 = 0;
      for (let i = 0; i < z.length; i++) d2 += (z[i] - center[i]) ** 2;
      phi.push(Math.exp(-0.5 * d2 / (spec.length_scale * spec.length_scale)));
    }
    return spec.weights[0].map((_, j) => {
      let acc = 0;
      for (let i = 0; i < phi.length; i++) acc += phi[i] * spec.weights[i][j];
      return acc * spec.y_scale[j] + spec.y_mean[j];
    });
  }
  let h = input;
  for (const layer of spec.layers) {
    const out = layer.b.slice();
    for (let i = 0; i < h.length; i++) {
      const row = layer.w[i];
      for (let j = 0; j < out.length; j++) out[j] += h[i] * row[j];
    }
    if (layer.act === "tanh") h = out.map(Math.tanh);
    else if (layer.act === "relu") h = out.map((v) => Math.max(0, v));
    else h = out;
  }
  return h;
}

// Position-free feature maps mirroring the benchmark suite's linear_features
// and poly_features (rigid-body dynamics are translation-invariant).
function linearFeaturesJS(x, u) {
  return [...x.slice(3), ...u, 1];
}

function polyFeaturesJS(x, u) {
  const z = [...x.slice(3), ...u];
  const out = [1, ...z];
  for (let i = 0; i < z.length; i++) {
    for (let j = i; j < z.length; j++) out.push(z[i] * z[j]);
  }
  return out;
}

function applyStandardizedJS(phi, spec) {
  const out = new Array(spec.weights[0].length).fill(0);
  for (let i = 0; i < phi.length; i++) {
    const v = (phi[i] - spec.mean[i]) / spec.scale[i];
    if (v === 0) continue;
    const row = spec.weights[i];
    for (let j = 0; j < out.length; j++) out[j] += v * row[j];
  }
  return out;
}

function makeSurrogateStepper(spec, dt, cfg) {
  if (spec.kind === "hankel") {
    // Lagged ARX on position-free state history, increment-integrated. The
    // window re-seeds whenever the incoming state is not our own last output
    // (fresh anchor or SAFE-model handoff).
    let hist = null;
    let last = null;
    return (x, u) => {
      if (last !== x) hist = new Array(spec.lag).fill(x);
      const phi = [];
      for (const h of hist) phi.push(...h.slice(3));
      phi.push(...u, 1);
      const delta = new Array(x.length).fill(0);
      for (let i = 0; i < phi.length; i++) {
        if (phi[i] === 0) continue;
        const row = spec.weights[i];
        for (let j = 0; j < delta.length; j++) delta[j] += phi[i] * row[j];
      }
      const next = postStep(x.map((v, j) => v + delta[j]));
      hist = [...hist.slice(1), next];
      last = next;
      return next;
    };
  }
  if (spec.kind === "rbf_residual") {
    return (x, u) => {
      const base = nominalStep(x, u, dt, cfg);
      const z = [...x.slice(3), ...u];
      const phi = spec.centers.map((center) => {
        let d2 = 0;
        for (let i = 0; i < z.length; i++) {
          const d = (z[i] - center[i]) / spec.length_scale[i];
          d2 += d * d;
        }
        return Math.exp(-0.5 * d2);
      });
      phi.push(1);
      const out = new Array(x.length).fill(0);
      for (let i = 0; i < phi.length; i++) {
        const row = spec.weights[i];
        for (let j = 0; j < out.length; j++) out[j] += phi[i] * row[j];
      }
      return postStep(base.map((v, j) => v + out[j]));
    };
  }
  return (x, u) => {
    const phi = spec.degree === 1 ? linearFeaturesJS(x, u) : polyFeaturesJS(x, u);
    const delta = applyStandardizedJS(phi, spec);
    if (spec.kind === "derivative") return postStep(x.map((v, j) => v + dt * delta[j]));
    return postStep(x.map((v, j) => v + delta[j]));
  };
}

export function makeStepper(method, models, dt) {
  const cfg = models.config;
  if (models.surrogates && models.surrogates[method]) {
    return makeSurrogateStepper(models.surrogates[method], dt, cfg);
  }
  if (models.nets && models.nets[method]) {
    const spec = models.nets[method];
    return (x, u) => {
      const base = spec.residual ? nominalStep(x, u, dt, cfg) : new Array(13).fill(0);
      const out = evalNet(spec, [...x, ...u]);
      return postStep(base.map((v, i) => v + out[i]));
    };
  }
  if (method === "6DOF-Nominal") return (x, u) => nominalStep(x, u, dt, cfg);
  if (method === "6DOF-GreyBoxOEM" && models.greybox) return makeGreyboxStepper(models.greybox, dt);
  if (method === "6DOF-LinearSS") return (x, u) => postStep(affinePredict(x, u, models.linear_weights));
  return (x, u) => {
    const base = nominalStep(x, u, dt, cfg);
    const res = affinePredict(x, u, models.residual_weights);
    return postStep(base.map((v, i) => v + res[i]));
  };
}

function safeController(gains) {
  // SAFE self-level: the stick commands attitude through the envelope clip
  // ([Kp, cmd_scale, envelope_limit, Kd, offset] per attitude axis), the loop
  // closes on attitude error with rate damping, and surfaces saturate.
  return (stick, x) => {
    const euler = eulerFromQuat([x[6], x[7], x[8], x[9]]);
    const ge = gains.elevator;
    const ga = gains.aileron;
    const gr = gains.rudder;
    const thetaCmd = clamp(ge[1] * stick[1], -ge[2], ge[2]);
    const phiCmd = clamp(ga[1] * stick[2], -ga[2], ga[2]);
    return [
      stick[0],
      clamp(ge[0] * (thetaCmd - euler[1]) - ge[3] * x[11] + ge[4], -0.65, 0.65),
      clamp(ga[0] * (phiCmd - euler[0]) - ga[3] * x[10] + ga[4], -0.75, 0.75),
      clamp(gr[0] * stick[3] + gr[1] * x[12] + gr[2], -0.65, 0.65),
    ];
  };
}

function estimateInitialState(flight, timeS) {
  // Interpolate the measured state at timeS between its bracketing samples.
  // (A wide local-fit window biases the position toward the inside of turns,
  // so the free run would start visibly off the measured track; the converter
  // already smooths velocities, leaving nothing for a fit to clean up.)
  const t = flight.time;
  let hi = t.findIndex((v) => v >= timeS);
  if (hi < 0) hi = t.length - 1;
  const lo = Math.max(0, hi - 1);
  const span = t[hi] - t[lo];
  const w = span > 1e-9 ? clamp((timeS - t[lo]) / span, 0, 1) : 0;
  const a = flight.state[lo];
  const b = flight.state[hi];
  const x0 = a.map((v, j) => v + (b[j] - v) * w);
  if (a[6] * b[6] + a[7] * b[7] + a[8] * b[8] + a[9] * b[9] < 0) {
    // Antipodal quaternions: lerp through the short way before normalizing.
    for (let j = 6; j < 10; j++) x0[j] = a[j] + (-b[j] - a[j]) * w;
  }
  const q = normQuat([x0[6], x0[7], x0[8], x0[9]]);
  x0[6] = q[0]; x0[7] = q[1]; x0[8] = q[2]; x0[9] = q[3];
  return x0;
}

const Q_NED_TO_ENU = [0, Math.SQRT1_2, Math.SQRT1_2, 0]; // 180 deg about (1,1,0)/sqrt(2)

function quatMul(a, b) {
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ];
}

function rolloutFrom(flight, models, method, timeS) {
  const dt = flight.dt_full;
  const startIdx = Math.max(0, Math.round(timeS / dt));
  const sticks = flight.stick_full;
  const labels = flight.labels_full;
  const modes = flight.mode_full || labels.map((label) => (label === 2 ? 1 : 0));
  // Stop at the next ground contact: the airframe models have no gear model.
  let endIdx = sticks.length;
  for (let k = startIdx + Math.round(1 / dt); k < labels.length; k++) {
    if (labels[k] === 0) { endIdx = k; break; }
  }
  const stepper = makeStepper(method, models, dt);
  const safeStep = models.safe_invariant_weights ? makeSafeStepper(models.safe_invariant_weights, dt) : null;
  const ctrl = safeController(models.safe_gains);
  const bias = flight.bias;
  let x = estimateInitialState(flight, timeS);
  const stride = Math.max(1, Math.round(0.1 / dt));
  const times = [];
  const altitude = [];
  const pitch = [];
  const posEnu = [];
  const quatEnu = [];
  for (let k = startIdx; k < endIdx - 1; k++) {
    if ((k - startIdx) % stride === 0) {
      times.push(k * dt);
      altitude.push(-x[2]);
      pitch.push(eulerFromQuat([x[6], x[7], x[8], x[9]])[1]);
      posEnu.push([x[1], x[0], -x[2]]);
      quatEnu.push(quatMul(Q_NED_TO_ENU, normQuat([x[6], x[7], x[8], x[9]])));
    }
    const stick = sticks[k];
    if (modes[k] === 1 && safeStep) {
      // SAFE engaged: the directly identified closed-loop model replaces the
      // bare airframe + provisional controller decomposition. The handoff
      // keys off the recorded mode channel (a low SAFE pass is still
      // closed-loop flight) and the state carries over continuously.
      x = safeStep(x, stick);
    } else {
      const u = modes[k] === 1 ? ctrl(stick, x) : stick.map((v, i) => v - bias[i]);
      x = stepper(x, u);
    }
    if (!x.every(Number.isFinite)) break;
  }
  return { times, altitude, pitch, posEnu, quatEnu };
}

function flight() {
  return ex.data.flights[ex.flightIndex];
}

function predictionMethods() {
  // Free-run the leaderboard-selected methods the browser has parameters
  // for; with no usable selection, fall back to LinearSS so Predict here
  // always shows something.
  const usable = Array.from(ex.selectedMethods).filter((m) => ex.data.methods.includes(m));
  return usable.length ? usable : ["6DOF-LinearSS"];
}

function recomputePredictions() {
  ex.predictions = {};
  if (ex.anchorTimeS == null) return;
  for (const method of predictionMethods()) {
    ex.predictions[method] = rolloutFrom(flight(), ex.data.models, method, ex.anchorTimeS);
  }
}

function safeScoreNote() {
  const scores = ex.data?.models?.safe_scores;
  if (!scores || scores.validation_pos_err_5s_m == null) return "";
  return ` (currently ${scores.validation_pos_err_5s_m.toFixed(1)} m mean over ${scores.validation_windows} held-out windows)`;
}

function renderSplitsView() {
  // "Data Splits" tab: every flight as a timeline colored by segmentation
  // class, with the manual maneuver windows marked by their train/validation
  // membership, so the provenance of every fitted model is visible.
  const chart = document.querySelector("#splits-chart");
  const notes = document.querySelector("#splits-notes");
  if (!chart || !ex.data) return;
  chart.innerHTML = "";
  const maxDuration = Math.max(...ex.data.flights.map((f) => f.time[f.time.length - 1] || 1));
  const colorAt = (f, k) => (f.tracked && !f.tracked[k] ? "#4a5159" : LABEL_COLORS[ex.data.labels[f.labels[k]]] || "#666");
  for (const f of ex.data.flights) {
    const row = document.createElement("div");
    row.className = "splits-row";
    const name = document.createElement("div");
    name.className = "splits-name";
    name.textContent = f.name + (f.autonomous ? " (autonomous)" : "");
    row.append(name);
    const lane = document.createElement("div");
    lane.className = "splits-lane";
    const duration = f.time[f.time.length - 1] || 1;
    lane.style.width = `${(100 * duration) / maxDuration}%`;
    // Segmentation gradient, like the playback time bar.
    const stops = [];
    let runStart = 0;
    for (let k = 1; k <= f.labels.length; k += 1) {
      if (k === f.labels.length || colorAt(f, k) !== colorAt(f, runStart)) {
        const a = ((f.time[runStart] / duration) * 100).toFixed(2);
        const b = ((f.time[Math.min(k, f.time.length - 1)] / duration) * 100).toFixed(2);
        stops.push(`${colorAt(f, runStart)} ${a}% ${b}%`);
        runStart = k;
      }
    }
    const bar = document.createElement("div");
    bar.className = "splits-bar";
    bar.style.background = `linear-gradient(to right, ${stops.join(", ")})`;
    lane.append(bar);
    // Train/validation membership strips: manual maneuver windows (bare
    // airframe methods) above the bar, stabilized windows (closed-loop SAFE
    // model) below it.
    const addStrip = (start, stop, split, title, below) => {
      const strip = document.createElement("div");
      strip.className = `splits-window splits-${split}${below ? " splits-below" : ""}`;
      strip.style.left = `${(100 * start) / duration}%`;
      strip.style.width = `${(100 * (stop - start)) / duration}%`;
      strip.title = title;
      lane.append(strip);
    };
    for (const segment of f.segments) {
      if (!segment.split) continue;
      addStrip(segment.start_s, segment.stop_s, segment.split,
        `${segment.kind} ${segment.start_s}-${segment.stop_s} s -> ${segment.split} (airframe methods)`, false);
    }
    for (const window of f.stabilized_splits || []) {
      addStrip(window.start_s, window.stop_s, window.split,
        `stabilized ${window.start_s}-${window.stop_s} s -> ${window.split} (closed-loop SAFE model)`, true);
    }
    row.append(lane);
    chart.append(row);
  }
  const legend = document.createElement("div");
  legend.className = "splits-legend";
  legend.innerHTML = [
    ...Object.entries(LABEL_COLORS).map(([label, color]) => `<span><i style="background:${color}"></i>${label.replace("_", " ")}</span>`),
    '<span><i style="background:#4a5159"></i>mocap dropout</span>',
    '<span><i class="splits-train-key"></i>train window (above: airframe methods, below: SAFE model)</span>',
    '<span><i class="splits-validation-key"></i>validation window</span>',
  ].join("");
  chart.append(legend);
  if (notes) {
    notes.innerHTML = `
      <p>Manual maneuver windows (orange) are detected from the transmitter mode channel. Each window is split
      into a quasi-steady <em>lead-in</em> — pooled per flight to estimate the stick trim bias — and the
      <em>control actuation</em> portion, which is cut into 0.6&ndash;1.2&nbsp;s gap-free chunks whose initial states are
      estimated at each chunk start. Windows are assigned round-robin within each flight (every third manual
      window becomes validation), so both splits span multiple flights and battery states; models are fitted on
      the train chunks only and scored on held-out validation chunks.</p>
      <p>Stabilized segments (blue) never train the bare-airframe methods: the SAFE inner loop adds hidden surface
      corrections. They train the separate <em>closed-loop SAFE model</em> instead, with the same discipline as the
      manual windows: the tracked stabilized spans are cut into ~10&nbsp;s windows (strips below the bar), every third
      window per flight is held out, the model fits on the train windows only, and the held-out windows score it by
      5&nbsp;s free-run position error${safeScoreNote()}. The autonomous flight is excluded: its lateral commands
      bypass the recorded sticks. Ground (brown) windows feed the rolling-friction and thrust analysis, ground
      effect (teal) is kept out of all airframe fits, and mocap dropouts (gray) are never trained on or scored.</p>`;
  }
}

function svgEl(tag, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function renderAll() {
  const status = document.querySelector("#explorer-status");
  if (ex.anchorTimeS == null) {
    status.textContent = "Scrub the colored timeline and press Predict here to set the prediction initial condition.";
  } else {
    const used = Object.keys(ex.predictions).map((m) => m.replace("6DOF-", "")).join(", ");
    const fallback = ex.selectedMethods.size ? "" : " (no browser-runnable method selected in the leaderboard; using LinearSS)";
    const note = ex.anchorNote ? ` [${ex.anchorNote}]` : "";
    const dead = flight().autonomous
      ? " Warning: this autonomous flight was flown by an offboard autopilot whose lateral commands are not in the recorded sticks, so free runs cannot anticipate its turns."
      : "";
    status.textContent = `Free run from t = ${ex.anchorTimeS.toFixed(2)} s with ${used}${fallback}${note}; SAFE controller closes the loop through stabilized segments.${dead}`;
  }
}

const METHOD_COLORS_HEX = { "6DOF-Nominal": 0xd62728, "6DOF-LinearSS": 0x2ca02c, "6DOF-RidgeResidual": 0x9467bd, "6DOF-GreyBoxOEM": 0xe8a838, "6DOF-EquationError-LS": 0x17becf, "6DOF-SINDy": 0xe377c2, "6DOF-Koopman-EDMD": 0xbcbd22, "6DOF-Symbolic-Stepwise": 0x8c564b, "6DOF-Subspace-Hankel": 0x1f77b4, "6DOF-GP-RBF": 0xf7b6d2 };

function publishOverlay(timeS) {
  // Publish the segmentation-colored full-flight track and the free-run
  // predictions so the 3D playback can draw them. Selecting a flight (before
  // any click) publishes the colored track alone.
  const f = flight();
  const overlay = {
    stamp: `${f.name}@${timeS == null ? "none" : timeS.toFixed(2)}#${Array.from(ex.selectedMethods).join("|")}`,
    track: f.pos,
    origin: f.pos[0],
    dtFull: f.dt_full,
    anchored: timeS != null && ex.anchorTimeS != null,
    labels: f.labels,
    tracked: f.tracked || f.labels.map(() => 1),
    predictions: Object.entries(ex.predictions).map(([method, pred]) => ({
      color: METHOD_COLORS_HEX[method] ?? 0x444444,
      points: pred.posEnu,
      times: pred.times,
      quats: pred.quatEnu,
    })),
  };
  if (!ex.playbackTrack) ex.playbackTrack = buildPlaybackTrack();
  window.dispatchEvent(
    new CustomEvent("explorer-set-ic", {
      detail: {
        flight: f.name,
        flightIndex: ex.flightIndex,
        timeS: timeS == null ? 0 : timeS,
        overlay,
        // Browser-runnable methods, so the playback can offer a quick picker.
        methods: ex.data.methods,
        // Carry the full-flight track with the event so registration can
        // never be lost to module load order.
        track: ex.playbackTrack,
      },
    }),
  );
}

function buildPlaybackTrack() {
  // Full flights as first-class playback tracks: the 3D animation flies the
  // whole record (positions re-zeroed per flight; the overlay carries the
  // matching origin), instead of only the benchmark chunk windows.
  return {
    id: "sportcub_flights_5_22",
    model_family: "aircraft6dof",
    source: "mocap full record",
    title: "Sport Cub full flights (2026-05-22)",
    segments: ex.data.flights.map((f) => ({
      name: f.name,
      time_s: f.time,
      position_enu_m: f.pos.map((p) => [p[0] - f.pos[0][0], p[1] - f.pos[0][1], p[2] - f.pos[0][2]]),
      quaternion_wxyz: f.quat,
      labels: f.labels,
      tracked: f.tracked,
      mode: f.mode,
      control_meas: f.stick_full
        .filter((_, index) => index % Math.max(1, Math.round(0.1 / f.dt_full)) === 0)
        .map((u) => [u[0], u[2], u[1], u[3]]),
    })),
  };
}

function firstFlyableTime(f, timeS) {
  // The airframe models have no gear physics and mocap dropouts have no
  // state: anchor free-runs at the first airborne, tracked sample at or
  // after the requested time.
  const dtFull = f.dt_full;
  let k = Math.max(0, Math.round(timeS / dtFull));
  const tracked = f.tracked_full || null;
  const okAt = (index) => f.labels_full[index] !== 0 && (!tracked || tracked[index]);
  // Require a short run of clean samples so the anchor never sits on a
  // tracking-reacquisition edge where smoothed attitude is contaminated.
  const margin = Math.round(0.3 / dtFull);
  while (k < f.labels_full.length) {
    let run = 0;
    while (k + run < f.labels_full.length && okAt(k + run) && run < margin) run += 1;
    if (run >= margin) return (k + margin) * dtFull;
    k += run + 1;
  }
  return null;
}

function setAnchor(timeS) {
  const snapped = firstFlyableTime(flight(), timeS);
  if (snapped == null) {
    ex.anchorTimeS = null;
    ex.predictions = {};
    ex.anchorNote = "no airborne data after the requested time";
    renderAll();
    publishOverlay(null);
    return;
  }
  ex.anchorNote = snapped - timeS > 0.1 ? "anchor moved past ground/dropout to the first airborne sample" : "";
  ex.anchorTimeS = snapped;
  recomputePredictions();
  renderAll();
  publishOverlay(snapped);
}

function bind() {
  const wrap = document.querySelector("#explorer-flight-wrap");
  if (wrap) wrap.hidden = false;
  const select = document.querySelector("#explorer-flight");
  select.innerHTML = ex.data.flights.map((f, i) => `<option value="${i}">${f.name}</option>`).join("");
  select.value = String(ex.flightIndex);
  select.addEventListener("change", (event) => {
    ex.flightIndex = parseInt(event.target.value, 10);
    ex.anchorTimeS = null;
    ex.predictions = {};
    renderAll();
    publishOverlay(null);
  });
}

export async function initExplorer() {
  const host = document.querySelector("#explorer-flight");
  if (!host) return;
  try {
    const response = await fetch(DATA_URL);
    if (!response.ok) throw new Error(`${response.status}`);
    ex.data = await response.json();
  } catch (error) {
    const status = document.querySelector("#explorer-status");
    if (status) status.textContent = `Flight data unavailable (${error.message}).`;
    return;
  }
  // Default to the flight with the cleanest full trajectory.
  const preferred = ex.data.flights.findIndex((f) => f.name.startsWith("elev3211_2026_05"));
  if (preferred >= 0) ex.flightIndex = preferred;
  bind();
  renderAll();
  renderSplitsView();
  // Handshake with the playback module: announce the full-flight view and
  // retry until acknowledged, so no module load order or transient error can
  // leave the 3D viewer on the benchmark-window view.
  let playbackLinked = false;
  window.addEventListener("playback-ack", () => {
    playbackLinked = true;
    console.debug("explorer: 3D playback linked");
  });
  const announce = () => {
    window.dispatchEvent(new CustomEvent("explorer-flights-ready", { detail: { track: buildPlaybackTrack() } }));
    publishOverlay(ex.anchorTimeS);
  };
  const tryAnnounce = (remaining) => {
    if (playbackLinked || remaining <= 0) {
      if (!playbackLinked) {
        document.querySelector("#explorer-status").textContent =
          "3D playback link failed; check the browser console for errors.";
      }
      return;
    }
    announce();
    setTimeout(() => tryAnnounce(remaining - 1), 500);
  };
  tryAnnounce(40);
  window.addEventListener("playback-ready", () => tryAnnounce(40));
  // The leaderboard owns method selection; free-run the selected methods the
  // browser has model parameters for.
  window.addEventListener("playback-context-changed", (event) => {
    const wasActive = ex.active;
    ex.active = Boolean(event.detail.explorerActive);
    const wrap = document.querySelector("#explorer-flight-wrap");
    if (wrap) wrap.hidden = !ex.active;
    // Returning to the Sport Cub dataset restores the segmentation overlay
    // (and any anchored free run) that the other dataset's view cleared.
    if (ex.active && !wasActive) publishOverlay(ex.anchorTimeS);
  });
  window.addEventListener("explorer-anchor-request", (event) => {
    if (!ex.active) return;
    // Clicking Predict here again at (nearly) the same time clears the run.
    if (ex.anchorTimeS != null && Math.abs(event.detail.timeS - ex.anchorTimeS) < 0.05) {
      ex.anchorTimeS = null;
      ex.predictions = {};
      renderAll();
      publishOverlay(null);
      return;
    }
    setAnchor(event.detail.timeS);
  });
  window.addEventListener("methods-changed", (event) => {
    const available = new Set(ex.data.methods);
    ex.selectedMethods = new Set((event.detail.methods || []).filter((m) => available.has(m)));
    if (!ex.active) return;
    recomputePredictions();
    renderAll();
    if (ex.anchorTimeS != null) publishOverlay(ex.anchorTimeS);
  });
  window.addEventListener("resize", renderAll);
}

initExplorer();
