model GroundRoll
  "Planar ground-roll model; port of ground_model.py (ground_rollout).

   A castering tail-dragger rolling on the runway, states (p_n, p_e, psi, V):

     der(V)   = kT * (max_thrust/m) * thr^1.45 - mu * g - cv * V^2
     der(psi) = (ks * rudder + k0) * V
     der(p_n) = V * cos(psi)
     der(p_e) = V * sin(psi)

   Thrust uses the nominal Sport Cub map (max_thrust * thr^1.45) with a fitted
   scale kT; mu is rolling resistance, cv quadratic drag; steering rate is
   proportional to ground speed (ks rudder gain, k0 wheel-alignment/thrust
   asymmetry). Fitted by simulation error over the tracked ground windows.

   NOTE: the Python runtime integrates this with forward Euler and clamps V>=0
   each step; this is its continuous form. The right-hand side (the equations
   above) is identical -- only the integrator/guard live in the wrapper."

  // ---- fitted parameters ----
  parameter Real kT = 1.0 "Thrust scale";
  parameter Real mu = 0.15 "Rolling resistance";
  parameter Real cv = 0.05 "Quadratic drag";
  parameter Real ks = 0.5 "Steering rudder gain";
  parameter Real k0 = 0.0 "Steering bias (wheel/thrust asymmetry)";
  // ---- fixed parameters ----
  parameter Real m = 0.063 "Mass [kg]";
  parameter Real g = 9.81 "Gravity [m/s^2]";

  // ---- baked constants (the stated thrust prior) ----
  constant Real max_thrust = 0.32 "Max thrust [N] (MAX_THRUST_N)";
  constant Real thrust_exponent = 1.45 "Throttle exponent (THRUST_EXPONENT)";

  // ---- inputs (u_cmd order subset: throttle, rudder) ----
  input Real throttle;
  input Real rudder;

  // ---- states ----
  Real p_n(start = 0) "North position [m]";
  Real p_e(start = 0) "East position [m]";
  Real psi(start = 0) "Heading [rad]";
  Real V(start = 0) "Ground speed [m/s]";

protected
  Real thr, acc;

equation
  thr = max(throttle, 0.0);
  acc = kT*(max_thrust/m)*thr^thrust_exponent - mu*g - cv*V^2;
  der(p_n) = V*cos(psi);
  der(p_e) = V*sin(psi);
  der(psi) = (ks*rudder + k0)*V;
  der(V) = acc;
end GroundRoll;
