model SafeController
  "SAFE self-level inner-loop controller; port of safe_controller.py (pd_command + surface lag).

   The hidden Horizon Hobby SAFE controller reshapes the pilot sticks into surface
   deflections. Per attitude axis (elevator->pitch, aileron->roll):

     cmd      = sat(c * stick, +/- limit)            (stick maps to an attitude setpoint)
     delta_raw= Kp * (cmd - attitude) - Kd * rate + b
   the rudder is stick passthrough plus yaw-rate damping:
     delta_raw= gs * stick_rud + Kr * r + b
   each delta_raw is clipped to the physical command limit and then passed through a
   first-order surface lag (the states of this model):

     der(delta) = (clip(delta_raw) - delta) / tau

   The lagged delta_e/a/r are the effective surface commands fed to the airframe
   (SportCubGreybox.mo) to close the loop. Inputs are the pilot sticks plus the
   sensed attitude/rates from the airframe.

   NOTE: the Python runtime applies the surface lag as a discrete forward-Euler
   filter (delta += (dt/max(tau,dt))*(cmd-delta)) sampled at the data rate; this is
   its continuous idealization (der(delta) = (cmd-delta)/tau). The control LAW
   (sat/PD/clip) is identical; for tau >> dt the lag matches too."

  // ---- elevator gains [Kp, c, limit, Kd, b] ----
  parameter Real Kp_e = 1.0;
  parameter Real c_e = 0.7;
  parameter Real limit_e = 0.6;
  parameter Real Kd_e = 0.1;
  parameter Real b_e = 0.0;
  // ---- aileron gains [Kp, c, limit, Kd, b] ----
  parameter Real Kp_a = 1.0;
  parameter Real c_a = 0.7;
  parameter Real limit_a = 0.6;
  parameter Real Kd_a = 0.1;
  parameter Real b_a = 0.0;
  // ---- rudder gains [gs, Kr, b] ----
  parameter Real gs_r = 1.0;
  parameter Real Kr_r = 0.0;
  parameter Real b_r = 0.0;
  // ---- first-order surface lags [s] ----
  parameter Real tau_e = 0.1;
  parameter Real tau_a = 0.1;
  parameter Real tau_r = 0.1;

  // ---- physical surface command limits (SURFACE_LIMITS), baked ----
  constant Real SURF_LIMIT_E = 0.65;
  constant Real SURF_LIMIT_A = 0.75;
  constant Real SURF_LIMIT_R = 0.65;

  // ---- inputs: pilot sticks + sensed attitude/rates from the airframe ----
  input Real stick_elevator;
  input Real stick_aileron;
  input Real stick_rudder;
  input Real phi;     // roll  [rad]
  input Real theta;   // pitch [rad]
  input Real p;       // roll rate  [rad/s]
  input Real q;       // pitch rate [rad/s]
  input Real r;       // yaw rate   [rad/s]

  // ---- states: lagged surface commands (the controller's effective output) ----
  Real delta_e(start = 0);
  Real delta_a(start = 0);
  Real delta_r(start = 0);

protected
  Real th_cmd, ph_cmd;
  Real de_raw, da_raw, dr_raw;
  Real de_cmd, da_cmd, dr_cmd;

equation
  // attitude setpoints from sticks (saturated to +/- limit)
  th_cmd = min(abs(limit_e), max(-abs(limit_e), c_e*stick_elevator));
  ph_cmd = min(abs(limit_a), max(-abs(limit_a), c_a*stick_aileron));

  // per-axis PD law (elevator->pitch, aileron->roll, rudder passthrough+damp)
  de_raw = Kp_e*(th_cmd - theta) - Kd_e*q + b_e;
  da_raw = Kp_a*(ph_cmd - phi) - Kd_a*p + b_a;
  dr_raw = gs_r*stick_rudder + Kr_r*r + b_r;

  // clip to physical command limits
  de_cmd = min(SURF_LIMIT_E, max(-SURF_LIMIT_E, de_raw));
  da_cmd = min(SURF_LIMIT_A, max(-SURF_LIMIT_A, da_raw));
  dr_cmd = min(SURF_LIMIT_R, max(-SURF_LIMIT_R, dr_raw));

  // first-order surface lags -> effective surface commands
  der(delta_e) = (de_cmd - delta_e)/tau_e;
  der(delta_a) = (da_cmd - delta_a)/tau_a;
  der(delta_r) = (dr_cmd - delta_r)/tau_r;
end SafeController;
