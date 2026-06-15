model SportCubGreyboxLag
  "6DOF grey-box with first-order actuator lags and battery decay.
   Faithful port of greybox.build_casadi_dynamics_lag.

   State order matches STATE_NAMES_LAG (16 states):
     pos[3]   = p_n, p_e, p_d
     vel[3]   = u, v, w
     att[3]   = phi, theta, psi
     rates[3] = p, q, r
     delta_e, delta_a, delta_r = lagged surface deflections [rad]
     thr_f    = filtered throttle
   Input order matches CONTROL_NAMES_LAG: aileron, elevator, throttle, rudder,
   t_flight (flight time [s], enters only the battery thrust multiplier).

   Parameter order matches config.full_parameter_vector(): 10 fixed parameters
   followed by 25 estimated aero parameters (SPORTCUB_PARAMETER_NAMES). Unlike
   the lag-free model, this one uses TAUS, TAUM, KB. Max deflections are baked
   constants; the psi-wrap post-step lives in the Python RK4 wrapper."

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
  input Real t_flight;

  // ---- states ----
  Real pos[3](start = {0, 0, 0});
  Real vel[3](start = {16, 0, 0});
  Real att[3](start = {0, 0, 0});
  Real rates[3](start = {0, 0, 0});
  Real delta_e(start = 0);
  Real delta_a(start = 0);
  Real delta_r(start = 0);
  Real thr_f(start = 0);

protected
  Real speed, speed_safe, alpha, beta, qbar;
  Real c_a, s_a, c_b, s_b;
  Real CL, CD, CY, lift, drag, side, battery, thrust;
  Real fx, fy, fz;
  Real c_phi, s_phi, c_th, s_th, c_psi, s_psi, c_th_safe, common;
  Real bV, cV, roll_accel, pitch_accel, yaw_accel;
  Real r00, r01, r02, r10, r11, r12, r20, r21, r22;

equation
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

  // aerodynamic forces (use filtered throttle + battery decay)
  CL = CL0 + CLa*alpha;
  CD = CD0 + CDCLS*CL^2;
  CY = CYb*beta;
  lift = qbar*S*CL;
  drag = qbar*S*CD;
  side = qbar*S*CY;
  battery = max(1.0 + KB*t_flight, 0.3);
  thrust = KT*m*max(thr_f, 0.0)*battery;
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

  // angular dynamics (moments use lagged surface deflections)
  bV = b/(2.0*speed_safe);
  cV = cbar/(2.0*speed_safe);
  roll_accel = qbar*(KL0 + KLb*beta + KLp*bV*rates[1] + KLr*bV*rates[3] + KLda*delta_a + KLdr*delta_r);
  pitch_accel = qbar*(KM0 + KMa*alpha + KMq*cV*rates[2] + KMe*delta_e);
  yaw_accel = qbar*(KN0 + KNb*beta + KNp*bV*rates[1] + KNr*bV*rates[3] + KNda*delta_a + KNdr*delta_r);
  der(rates[1]) = roll_accel + ((Iyy - Izz)/Ixx)*rates[2]*rates[3] + (Ixz/Ixx)*rates[1]*rates[2];
  der(rates[2]) = pitch_accel + ((Izz - Ixx)/Iyy)*rates[1]*rates[3] + (Ixz/Iyy)*(rates[3]^2 - rates[1]^2);
  der(rates[3]) = yaw_accel + ((Ixx - Iyy)/Izz)*rates[1]*rates[2] + (Ixz/Izz)*rates[2]*rates[3];

  // Euler kinematics
  c_th_safe = sign(c_th)*max(abs(c_th), 1e-3);
  common = rates[2]*s_phi + rates[3]*c_phi;
  der(att[1]) = rates[1] + (s_th/c_th_safe)*common;
  der(att[2]) = rates[2]*c_phi - rates[3]*s_phi;
  der(att[3]) = common/c_th_safe;

  // position kinematics
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

  // first-order actuator lags + filtered throttle
  der(delta_e) = (MAXDEF_ELEV*DEG2RAD*elevator - delta_e)/TAUS;
  der(delta_a) = (MAXDEF_AIL*DEG2RAD*aileron - delta_a)/TAUS;
  der(delta_r) = (MAXDEF_RUD*DEG2RAD*rudder - delta_r)/TAUS;
  der(thr_f) = (min(1.0, max(0.0, throttle)) - thr_f)/TAUM;
end SportCubGreyboxLag;
