model Aircraft6DOF
  "Nonlinear 6DOF small-aircraft benchmark; faithful port of models/aircraft6dof/model.py.

   State order matches model.py STATE_NAMES:
     pos[3]  = x_n, y_e, z_d        (NED position, inertial)
     vel[3]  = u, v, w              (body-frame velocity)
     quat[4] = q_w, q_x, q_y, q_z   (body->inertial quaternion)
     omega[3]= p, q, r              (body angular rates)
   Input order matches model.py INPUT_NAMES: throttle, elevator, aileron, rudder.

   Set nl = 0 to recover the attached-flow 'nominal' model (model.py nonlinear=False);
   nl = 1 is the full nonlinear truth model used for synthetic data generation.

   Post-integration guards in model.py (_post_step: quaternion renormalization,
   speed clamp, rate clip) are NOT part of the continuous dynamics and are applied
   by the Python RK4 wrapper, not here."

  // ---- Parameters (Aircraft6DOFConfig defaults) ----
  parameter Real mass = 0.075 "Mass [kg]";
  parameter Real g = 9.81 "Gravity [m/s^2]";
  parameter Real ixx = 0.0016 "Inertia xx [kg m^2]";
  parameter Real iyy = 0.0022 "Inertia yy [kg m^2]";
  parameter Real izz = 0.0036 "Inertia zz [kg m^2]";
  parameter Real ixz = 0.00005 "Inertia xz product [kg m^2]";
  parameter Real rho = 1.18 "Air density [kg/m^3]";
  parameter Real S = 0.066 "Wing area [m^2]";
  parameter Real b = 0.62 "Wing span [m]";
  parameter Real c = 0.11 "Mean chord [m]";
  parameter Real prop_arm = 0.035 "Propeller moment arm [m]";
  parameter Real max_thrust = 0.32 "Maximum thrust [N]";
  parameter Real prop_wash_gain = 0.22 "Prop-wash aero scaling gain";
  parameter Real alpha_stall_deg = 14.0 "Stall angle [deg]";
  parameter Real stall_width_deg = 2.2 "Stall transition width [deg]";
  parameter Real cl_max = 1.28 "Max lift coefficient";
  parameter Real cl_min = -0.95 "Min lift coefficient";
  parameter Real nl = 1.0 "Nonlinear gate: 1=truth, 0=attached-flow nominal";

  // deg2rad factor (avoids pulling in Modelica.Constants)
  constant Real DEG2RAD = 0.017453292519943295;

  // ---- Inputs ----
  input Real throttle;
  input Real elevator;
  input Real aileron;
  input Real rudder;

  // ---- States ----
  Real pos[3](start = {0, 0, 0});
  Real vel[3](start = {4.5, 0, 0});
  Real quat[4](start = {1, 0, 0, 0});
  Real omega[3](start = {0, 0, 0});

protected
  Real speed, alpha, beta;
  Real rate_scale, p_hat, q_hat, r_hat;
  Real cl_attached, cd_attached, cm_attached;
  Real alpha_stall, width, gate_arg, stall_gate, gate;
  Real alpha_sign, alpha_excess, cl_limit, cl_post, control_eff;
  Real cl, cd, cm, cy, cl_roll, cn, cx, cz;
  Real throttle_c, qbar, prop_wash, thrust;
  Real Faero[3], Maero[3], Fbody[3], Mbody[3];
  Real q0, q1, q2, q3, qnorm;
  Real R11, R12, R13, R21, R22, R23, R31, R32, R33;
  Real Jw[3], corr[3], Meff[3], denom;

equation
  // ---- airdata ----
  speed = max(sqrt(vel[1]^2 + vel[2]^2 + vel[3]^2), 1e-6);
  alpha = atan2(vel[3], max(vel[1], 1e-6));
  beta = asin(min(0.98, max(-0.98, vel[2]/speed)));

  rate_scale = max(2.0*speed, 1e-6);
  p_hat = b*omega[1]/rate_scale;
  q_hat = c*omega[2]/rate_scale;
  r_hat = b*omega[3]/rate_scale;

  // ---- stall gate ----
  alpha_stall = alpha_stall_deg*DEG2RAD;
  width = stall_width_deg*DEG2RAD;
  gate_arg = min(60.0, max(-60.0, (abs(alpha) - alpha_stall)/max(width, 1e-6)));
  stall_gate = 1.0/(1.0 + exp(-gate_arg));
  gate = nl*stall_gate;
  alpha_sign = noEvent(if alpha >= 0.0 then 1.0 else -1.0);
  alpha_excess = max(abs(alpha) - alpha_stall, 0.0);
  cl_limit = noEvent(if alpha >= 0.0 then cl_max else abs(cl_min));
  cl_post = alpha_sign*max(0.28, cl_limit - 1.65*alpha_excess);
  control_eff = 1.0 - 0.58*gate;

  // ---- attached-flow coefficients (+ nonlinear stall corrections) ----
  cl_attached = 0.27 + 5.20*alpha + 0.36*elevator + 3.10*q_hat;
  cd_attached = 0.045 + 0.075*cl_attached^2 + 0.42*beta^2 + 0.018*(aileron^2 + rudder^2)
                + 0.010*elevator^2
                + nl*(gate*(0.16 + 1.45*alpha_excess + 5.5*alpha_excess^2));
  cm_attached = 0.030 - 1.05*alpha - 1.15*elevator - 9.0*q_hat
                + nl*(-gate*alpha_sign*(0.18 + 0.90*alpha_excess) - 0.035*gate*tanh(4.0*elevator));

  cl = (1.0 - gate)*cl_attached + gate*cl_post
       + nl*((1.0 - 0.45*gate)*(0.045*sin(2.0*alpha)*cos(beta) + 0.020*elevator*q_hat));
  cd = cd_attached;
  cm = cm_attached;
  cy = -0.82*beta + 0.30*rudder + 0.12*aileron - 0.35*r_hat
       + nl*(gate*(-0.22*beta*abs(beta) + 0.08*sin(3.0*alpha)*rudder));
  cl_roll = -0.12*beta + 0.42*control_eff*aileron - 0.50*p_hat + 0.10*r_hat
       + nl*(gate*(-0.18*beta + 0.06*rudder));
  cn = 0.18*beta - 0.26*control_eff*rudder - 0.08*aileron - 0.42*r_hat
       + nl*(gate*(0.10*beta - 0.05*aileron));

  cx = -cd*cos(alpha) + cl*sin(alpha);
  cz = -cd*sin(alpha) - cl*cos(alpha);

  // ---- forces and moments ----
  throttle_c = min(1.0, max(0.0, throttle));
  qbar = 0.5*rho*speed^2;
  Faero = qbar*S*{cx, cy, cz};
  Maero = qbar*S*{b*cl_roll, c*cm, b*cn};
  prop_wash = 1.0 + prop_wash_gain*throttle_c;
  thrust = max_thrust*throttle_c^1.45;
  Fbody = prop_wash*Faero + {thrust, 0.0, 0.0};
  Mbody = prop_wash*Maero + {0.0, prop_arm*thrust, 0.0};

  // ---- normalized quaternion and body->inertial rotation ----
  qnorm = max(sqrt(quat[1]^2 + quat[2]^2 + quat[3]^2 + quat[4]^2), 1e-12);
  q0 = quat[1]/qnorm;
  q1 = quat[2]/qnorm;
  q2 = quat[3]/qnorm;
  q3 = quat[4]/qnorm;
  R11 = 1.0 - 2.0*(q2^2 + q3^2);
  R12 = 2.0*(q1*q2 - q0*q3);
  R13 = 2.0*(q1*q3 + q0*q2);
  R21 = 2.0*(q1*q2 + q0*q3);
  R22 = 1.0 - 2.0*(q1^2 + q3^2);
  R23 = 2.0*(q2*q3 - q0*q1);
  R31 = 2.0*(q1*q3 - q0*q2);
  R32 = 2.0*(q2*q3 + q0*q1);
  R33 = 1.0 - 2.0*(q1^2 + q2^2);

  // ---- position kinematics: der(pos) = R * vel ----
  der(pos[1]) = R11*vel[1] + R12*vel[2] + R13*vel[3];
  der(pos[2]) = R21*vel[1] + R22*vel[2] + R23*vel[3];
  der(pos[3]) = R31*vel[1] + R32*vel[2] + R33*vel[3];

  // ---- velocity dynamics: F/m + gravity_body - omega x vel ----
  // gravity_body = R^T * [0,0,g] = g * [R31, R32, R33]
  der(vel[1]) = Fbody[1]/mass + g*R31 - (omega[2]*vel[3] - omega[3]*vel[2]);
  der(vel[2]) = Fbody[2]/mass + g*R32 - (omega[3]*vel[1] - omega[1]*vel[3]);
  der(vel[3]) = Fbody[3]/mass + g*R33 - (omega[1]*vel[2] - omega[2]*vel[1]);

  // ---- quaternion kinematics: 0.5 * Omega(omega) * q ----
  der(quat[1]) = 0.5*(-q1*omega[1] - q2*omega[2] - q3*omega[3]);
  der(quat[2]) = 0.5*( q0*omega[1] + q2*omega[3] - q3*omega[2]);
  der(quat[3]) = 0.5*( q0*omega[2] - q1*omega[3] + q3*omega[1]);
  der(quat[4]) = 0.5*( q0*omega[3] + q1*omega[2] - q2*omega[1]);

  // ---- angular dynamics: I*domega = M - omega x (I*omega), with Ixz coupling ----
  Jw = {ixx*omega[1] - ixz*omega[3], iyy*omega[2], izz*omega[3] - ixz*omega[1]};
  corr = {omega[2]*Jw[3] - omega[3]*Jw[2],
          omega[3]*Jw[1] - omega[1]*Jw[3],
          omega[1]*Jw[2] - omega[2]*Jw[1]};
  Meff = Mbody - corr;
  denom = ixx*izz - ixz^2;
  der(omega[1]) = (izz*Meff[1] + ixz*Meff[3])/denom;
  der(omega[2]) = Meff[2]/iyy;
  der(omega[3]) = (ixz*Meff[1] + ixx*Meff[3])/denom;
end Aircraft6DOF;
