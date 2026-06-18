model SportCubGreybox
  "6DOF lumped-parameter grey-box; faithful port of greybox.build_casadi_dynamics.

   State order matches STATE_NAMES_EULER:
     pos[3]   = p_n, p_e, p_d      (NED position)
     vel[3]   = u, v, w            (body-frame velocity)
     att[3]   = phi, theta, psi    (Euler angles)
     rates[3] = p, q, r            (body angular rates)
   Input order matches CONTROL_NAMES_OEM: aileron, elevator, throttle, rudder.

   Parameter order matches config.full_parameter_vector():
     10 fixed parameters  (m, S, b, cbar, rho, g, Ixx, Iyy, Izz, Ixz)
     followed by 25 estimated aero parameters (SPORTCUB_PARAMETER_NAMES).
   TAUS/TAUM/KB are declared for vector-layout parity but, like the CasADi
   reference, are not used by the rigid-body dynamics (servo/motor lag lives in
   the actuator model, not here).

   Control-surface max deflections are `constant` (baked, as in the reference),
   not estimated parameters, so the parameter vector stays [fixed(10), aero(25)].
   The psi-wrap post-step in the reference RK4 stays in the Python wrapper."

  // ---- fixed parameters ----
  parameter Real m = 0.063;
  parameter Real S = 0.05553;
  parameter Real b = 0.617;
  parameter Real cbar = 0.09;
  parameter Real rho = 1.225;
  parameter Real g = 9.81;
  parameter Real Ixx = 6.9e-4;
  parameter Real Iyy = 6.0e-4;
  parameter Real Izz = 1.25e-3;
  parameter Real Ixz = 3.5e-5;

  // ---- estimated aero parameters (defaults = bounds.initial) ----
  parameter Real CL0 = 0.50;
  parameter Real CLa = 4.00;
  parameter Real CD0 = 0.08;
  parameter Real CDCLS = 0.05;
  parameter Real CYb = -0.30;
  parameter Real KT = 2.50;
  parameter Real KL0 = 0.00;
  parameter Real KLb = -2.00;
  parameter Real KLp = -50.00;
  parameter Real KLr = 5.00;
  parameter Real KLda = 50.00;
  parameter Real KLdr = 2.00;
  parameter Real KM0 = 0.00;
  parameter Real KMa = -2.00;
  parameter Real KMq = -25.00;
  parameter Real KMe = -0.40;
  parameter Real KN0 = 0.00;
  parameter Real KNb = 3.00;
  parameter Real KNp = -1.00;
  parameter Real KNr = -8.00;
  parameter Real KNda = -1.00;
  parameter Real KNdr = -10.00;
  parameter Real TAUS = 0.08;
  parameter Real TAUM = 0.15;
  parameter Real KB = -0.0005;

  // ---- baked constants ----
  constant Real DEG2RAD = 0.017453292519943295;
  constant Real MAXDEF_ELEV = 23.0;
  constant Real MAXDEF_AIL = 25.0;
  constant Real MAXDEF_RUD = 30.0;

  // ---- inputs ----
  input Real aileron;
  input Real elevator;
  input Real throttle;
  input Real rudder;

  // ---- states ----
  Real pos[3](start = {0, 0, 0});
  Real vel[3](start = {16, 0, 0});
  Real att[3](start = {0, 0, 0});
  Real rates[3](start = {0, 0, 0});

protected
  Real thr, elev_rad, ail_rad, rud_rad;
  Real speed, speed_safe, alpha, beta, qbar;
  Real c_a, s_a, c_b, s_b;
  Real CL, CD, CY, lift, drag, side, thrust;
  Real fx, fy, fz;
  Real c_phi, s_phi, c_th, s_th, c_psi, s_psi, c_th_safe, common;
  Real roll_accel, pitch_accel, yaw_accel;
  Real r00, r01, r02, r10, r11, r12, r20, r21, r22;

equation
  // command conditioning
  thr = max(throttle, 0.0);
  elev_rad = MAXDEF_ELEV*DEG2RAD*elevator;
  ail_rad = MAXDEF_AIL*DEG2RAD*aileron;
  rud_rad = MAXDEF_RUD*DEG2RAD*rudder;

  // airdata
  speed = sqrt(vel[1]^2 + vel[2]^2 + vel[3]^2 + 1e-9);
  speed_safe = max(speed, 1e-3);
  alpha = atan2(vel[3], vel[1]);
  beta = asin(min(0.99, max(-0.99, vel[2]/speed_safe)));
  qbar = 0.5*rho*speed_safe^2;

  c_a = cos(alpha);
  s_a = sin(alpha);
  c_b = cos(beta);
  s_b = sin(beta);

  // aerodynamic forces (wind axes) + body-fixed thrust
  CL = CL0 + CLa*alpha;
  CD = CD0 + CDCLS*CL^2;
  CY = CYb*beta;
  lift = qbar*S*CL;
  drag = qbar*S*CD;
  side = qbar*S*CY;
  thrust = KT*m*thr;
  fx = -drag*c_a*c_b - side*c_a*s_b + lift*s_a + thrust;
  fy = -drag*s_b + side*c_b;
  fz = -drag*s_a*c_b - side*s_a*s_b - lift*c_a;

  // attitude trig
  c_phi = cos(att[1]);
  s_phi = sin(att[1]);
  c_th = cos(att[2]);
  s_th = sin(att[2]);
  c_psi = cos(att[3]);
  s_psi = sin(att[3]);

  // body translational dynamics
  der(vel[1]) = fx/m - g*s_th + rates[3]*vel[2] - rates[2]*vel[3];
  der(vel[2]) = fy/m + g*s_phi*c_th + rates[1]*vel[3] - rates[3]*vel[1];
  der(vel[3]) = fz/m + g*c_phi*c_th + rates[2]*vel[1] - rates[1]*vel[2];

  // angular dynamics
  roll_accel = qbar*(KL0 + KLb*beta + KLp*(b/(2.0*speed_safe))*rates[1]
               + KLr*(b/(2.0*speed_safe))*rates[3] + KLda*ail_rad + KLdr*rud_rad);
  pitch_accel = qbar*(KM0 + KMa*alpha + KMq*(cbar/(2.0*speed_safe))*rates[2] + KMe*elev_rad);
  yaw_accel = qbar*(KN0 + KNb*beta + KNp*(b/(2.0*speed_safe))*rates[1]
              + KNr*(b/(2.0*speed_safe))*rates[3] + KNda*ail_rad + KNdr*rud_rad);
  der(rates[1]) = roll_accel + ((Iyy - Izz)/Ixx)*rates[2]*rates[3] + (Ixz/Ixx)*rates[1]*rates[2];
  der(rates[2]) = pitch_accel + ((Izz - Ixx)/Iyy)*rates[1]*rates[3] + (Ixz/Iyy)*(rates[3]^2 - rates[1]^2);
  der(rates[3]) = yaw_accel + ((Ixx - Iyy)/Izz)*rates[1]*rates[2] + (Ixz/Izz)*rates[2]*rates[3];

  // Euler kinematics
  c_th_safe = sign(c_th)*max(abs(c_th), 1e-3);
  common = rates[2]*s_phi + rates[3]*c_phi;
  der(att[1]) = rates[1] + (s_th/c_th_safe)*common;
  der(att[2]) = rates[2]*c_phi - rates[3]*s_phi;
  der(att[3]) = common/c_th_safe;

  // position kinematics: der(pos) = R(euler) * vel
  r00 = c_th*c_psi;
  r01 = s_phi*s_th*c_psi - c_phi*s_psi;
  r02 = c_phi*s_th*c_psi + s_phi*s_psi;
  r10 = c_th*s_psi;
  r11 = s_phi*s_th*s_psi + c_phi*c_psi;
  r12 = c_phi*s_th*s_psi - s_phi*c_psi;
  r20 = -s_th;
  r21 = s_phi*c_th;
  r22 = c_phi*c_th;
  der(pos[1]) = r00*vel[1] + r01*vel[2] + r02*vel[3];
  der(pos[2]) = r10*vel[1] + r11*vel[2] + r12*vel[3];
  der(pos[3]) = r20*vel[1] + r21*vel[2] + r22*vel[3];
end SportCubGreybox;
