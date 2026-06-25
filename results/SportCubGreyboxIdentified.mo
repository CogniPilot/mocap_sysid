// Identified model generated from SportCubGreybox.mo.
// fit: train_nrmse=0.2941, validation_nrmse=0.3067, cost=7981.38, nfev=100, status=ground=skipped; air=Maximum_Iterations_Exceeded
model SportCubGreyboxIdentified
  "Sport Cub grey-box using the same plant structure as Rumoca's interactive FixedWingPlant.

   This is a standalone NED/FRD port of examples/interactive/fixedwing/FixedWingSIL.mo:
   wind-axis aerodynamic force rotation, wing incidence, smooth stall/flat-plate
   blending, full lateral/directional terms, physical moments, and tricycle gear
   spring-damper contact. It keeps the benchmark API:

     pos[3]   = p_n, p_e, p_d      (NED position, down positive)
     vel[3]   = u, v, w            (body FRD velocity)
     att[3]   = phi, theta, psi    (Euler angles)
     rates[3] = p, q, r            (body FRD rates)

   Input order matches CONTROL_NAMES_OEM: aileron, elevator, throttle, rudder."

  // ---- fixed parameters ----
  parameter Real m = 0.063;
  parameter Real S = 0.05553;
  parameter Real b = 0.617;
  parameter Real cbar = 0.09;
  parameter Real rho = 1.225;
  parameter Real g = 9.81;
  parameter Real Ixx = 0.0018855591563213705;
  parameter Real Iyy = 0.0015461099727885994;
  parameter Real Izz = 0.0036191016996440827;
  parameter Real Ixz = 0.0002050714628894117;

  // ---- estimated parameters: Rumoca FixedWingPlant coefficient structure ----
  parameter Real wing_incidence = 0.10472 "Wing incidence angle [rad]";
  parameter Real thr_max = 0.32 "Maximum thrust [N]";

  parameter Real ground_k = 140.0 "Gear stiffness per wheel [N/m]";
  parameter Real ground_c = 7.0 "Gear normal damping per wheel [N*s/m]";
  parameter Real roll_fric = 0.02 "Rolling resistance [N/(m/s)]";
  parameter Real side_fric = 1.2 "Lateral tire grip [N/(m/s)]";

  parameter Real CL0 = 1.330834383129222 "Lift at zero AoA";
  parameter Real CLa = 5.796183914504367 "Lift slope [1/rad]";
  parameter Real CD0 = 0.04176753035472612 "Parasitic drag";
  parameter Real k_ind = 0.027107876434979222 "Induced drag factor";
  parameter Real CD0_fp = 1.1173970536911972 "Flat-plate drag";
  parameter Real Cm0 = -0.05926715018485296 "Pitch moment at alpha=0";
  parameter Real Cma = -0.08417984700267667 "Pitch stiffness [1/rad]";
  parameter Real Cmq = -55.049041514586506 "Pitch damping";
  parameter Real Cmde = -0.8170460411941961 "Elevator pitch effectiveness";

  parameter Real CYb = -1.1287553815264646 "Sideslip side force [1/rad]";
  parameter Real CYda = 0.09541383478659189 "Aileron side force";
  parameter Real CYdr = 0.025421820460296434 "Rudder side force";
  parameter Real CYp = -0.6091421882724996 "Roll-rate side force";
  parameter Real CYr = 0.7768811819544827 "Yaw-rate side force";
  parameter Real CY_fp_coef = 0.05195019014239133 "Flat-plate side force";
  parameter Real Clb = -0.0044208606816988905 "Dihedral effect";
  parameter Real Clp = -0.8941868966843818 "Roll damping";
  parameter Real Clr = -0.25785349311877626 "Yaw-roll coupling";
  parameter Real Clda = 0.005542570026855261 "Aileron roll effectiveness";
  parameter Real Cldr = 0.3017720072422754 "Rudder roll";
  parameter Real Cnb = 0.023792621000067123 "Weathercock stability";
  parameter Real Cnp = -0.15202222203851776 "Roll-yaw coupling";
  parameter Real Cnr = -0.8516129969142281 "Yaw damping";
  parameter Real Cndr = 1.5156784860258716 "Rudder yaw effectiveness";
  parameter Real Cnda = -0.05859350087449443 "Aileron adverse yaw";

  parameter Real alpha_stall = 0.11323610147506752 "Stall angle [rad]";
  parameter Real blend_width = 0.05214780102982927 "Stall blend width [rad]";
  parameter Real max_defl_ail = 0.5236 "Aileron travel [rad]";
  parameter Real max_defl_elev = 0.4189 "Elevator travel [rad]";
  parameter Real max_defl_rud = 0.349 "Rudder travel [rad]";

  // Wheel contact points in body FRD. The z offsets are positive down; when
  // level at pos[3] = -wheel_z, the wheels sit on ground_d.
  constant Real ground_d = 0.0 "Runway NED down coordinate [m]";
  constant Real wheel_x[3] = {0.10, -0.08, -0.08} "Wheel forward offsets nose,R,L [m]";
  constant Real wheel_y[3] = {0.0, 0.10, -0.10} "Wheel right offsets [m]";
  constant Real wheel_z[3] = {0.055, 0.055, 0.055} "Wheel down offsets [m]";
  constant Real eps = 1e-6;

  // ---- inputs ----
  input Real aileron;
  input Real elevator;
  input Real throttle;
  input Real rudder;

  // ---- states ----
  Real pos[3](start = {0, 0, -0.055});
  Real vel[3](start = {0, 0, 0});
  Real att[3](start = {0, 0, 0});
  Real rates[3](start = {0, 0, 0});

protected
  Real U, V_frd, W_frd, Vt, Vxz, alpha_body, alpha, beta, qbar, sigma;
  Real P_frd, Q_frd, R_frd;
  Real wx1, wx2, wx3, wy1, wy2, wy3, wz1, wz2, wz3;
  Real refx, refz, rdot, wzt1, wzt2, wzt3, nz;
  Real ail_rad, elev_rad, rud_rad, thr_out;
  Real CL_lin, CL_fp, CL, CD_lin, CD_fp, CD, CY_lin, CY_fp, CY;
  Real Cl_aero, Cm_aero, Cn_aero;
  Real FA_frd[3], MA_frd[3], F_thrust[3];
  Real fx, fy, fz, mx, my, mz;
  Real c_phi, s_phi, c_th, s_th, c_psi, s_psi, c_th_safe, common;
  Real r00, r01, r02, r10, r11, r12, r20, r21, r22;
  Real wh_h[3], wh_vbx[3], wh_vby[3], wh_vbz[3], wh_vwd[3], wh_Fn[3];
  Real wh_F[3, 3], wh_M[3, 3], F_ground[3], M_ground[3];
  Real hx, hy, hz, tx, ty, tz, detI;

equation
  // --- attitude trig and body FRD -> NED rotation ---
  c_phi = cos(att[1]);
  s_phi = sin(att[1]);
  c_th = cos(att[2]);
  s_th = sin(att[2]);
  c_psi = cos(att[3]);
  s_psi = sin(att[3]);
  r00 = c_th*c_psi;
  r01 = s_phi*s_th*c_psi - c_phi*s_psi;
  r02 = c_phi*s_th*c_psi + s_phi*s_psi;
  r10 = c_th*s_psi;
  r11 = s_phi*s_th*s_psi + c_phi*c_psi;
  r12 = c_phi*s_th*s_psi - s_phi*c_psi;
  r20 = -s_th;
  r21 = s_phi*c_th;
  r22 = c_phi*c_th;

  // --- FRD body velocity/rates in the same aero convention as FixedWingPlant ---
  U = vel[1];
  V_frd = vel[2];
  W_frd = vel[3];
  Vt = sqrt(U*U + V_frd*V_frd + W_frd*W_frd) + eps;
  Vxz = sqrt(U*U + W_frd*W_frd) + eps;
  alpha_body = atan2(W_frd, U);
  alpha = alpha_body + wing_incidence;
  beta = atan2(V_frd, Vxz);
  qbar = 0.5*rho*Vt*Vt;
  sigma = (1 + tanh((alpha - alpha_stall)/blend_width))/2;
  P_frd = rates[1];
  Q_frd = rates[2];
  R_frd = rates[3];

  // --- wind-frame axes from velocity (branch-free Gram-Schmidt, as in Rumoca) ---
  wx1 = U/Vt;
  wx2 = V_frd/Vt;
  wx3 = W_frd/Vt;
  refx = if abs(wx3) < abs(wx1) then 0 else 1;
  refz = if abs(wx3) < abs(wx1) then 1 else 0;
  rdot = refx*wx1 + refz*wx3;
  wzt1 = refx - rdot*wx1;
  wzt2 = -rdot*wx2;
  wzt3 = refz - rdot*wx3;
  nz = sqrt(wzt1*wzt1 + wzt2*wzt2 + wzt3*wzt3) + eps;
  wz1 = wzt1/nz;
  wz2 = wzt2/nz;
  wz3 = wzt3/nz;
  wy1 = wz2*wx3 - wz3*wx2;
  wy2 = wz3*wx1 - wz1*wx3;
  wy3 = wz1*wx2 - wz2*wx1;

  // --- surface deflections / throttle ---
  ail_rad = max_defl_ail*min(1, max(-1, aileron));
  elev_rad = max_defl_elev*min(1, max(-1, elevator));
  rud_rad = -max_defl_rud*min(1, max(-1, rudder));
  thr_out = min(1, max(0, throttle));

  // --- aerodynamic coefficients (same structure as Rumoca FixedWingPlant) ---
  CL_lin = CL0 + CLa*alpha;
  CL_fp = 2*sin(alpha)*cos(alpha);
  CL = (1 - sigma)*CL_lin + sigma*CL_fp;
  CD_lin = CD0 + k_ind*CL_lin*CL_lin;
  CD_fp = CD0_fp + 2*sin(alpha)*sin(alpha);
  CD = (1 - sigma)*CD_lin + sigma*CD_fp;
  CY_lin = CYb*beta + CYda*ail_rad + CYdr*rud_rad + CYp*(b/(2*Vt))*P_frd + CYr*(b/(2*Vt))*R_frd;
  CY_fp = CY_fp_coef*sin(beta)*cos(alpha);
  CY = (1 - sigma)*CY_lin + sigma*CY_fp;
  Cl_aero = Clda*ail_rad + Cldr*rud_rad + Clb*beta + Clp*(b/(2*Vt))*P_frd + Clr*(b/(2*Vt))*R_frd;
  Cm_aero = Cm0 + Cma*alpha + Cmde*elev_rad + Cmq*(cbar/(2*Vt))*Q_frd;
  Cn_aero = Cnb*beta + Cndr*rud_rad + Cnda*ail_rad + Cnp*(b/(2*Vt))*P_frd + Cnr*(b/(2*Vt))*R_frd;

  // --- wind axes -> body FRD, FA_wind = qbar*S*{-CD, CY, -CL} ---
  FA_frd[1] = qbar*S*(wx1*(-CD) + wy1*CY + wz1*(-CL));
  FA_frd[2] = qbar*S*(wx2*(-CD) + wy2*CY + wz2*(-CL));
  FA_frd[3] = qbar*S*(wx3*(-CD) + wy3*CY + wz3*(-CL));
  MA_frd[1] = qbar*S*b*Cl_aero;
  MA_frd[2] = qbar*S*cbar*Cm_aero;
  MA_frd[3] = qbar*S*b*Cn_aero;
  F_thrust = {thr_max*thr_out, 0, 0};

  // --- tricycle landing-gear contact, adapted from Z-up/FLU to NED/FRD ---
  for i in 1:3 loop
    wh_h[i] = pos[3] + r20*wheel_x[i] + r21*wheel_y[i] + r22*wheel_z[i] - ground_d;
    wh_vbx[i] = vel[1] + rates[2]*wheel_z[i] - rates[3]*wheel_y[i];
    wh_vby[i] = vel[2] + rates[3]*wheel_x[i] - rates[1]*wheel_z[i];
    wh_vbz[i] = vel[3] + rates[1]*wheel_y[i] - rates[2]*wheel_x[i];
    wh_vwd[i] = r20*wh_vbx[i] + r21*wh_vby[i] + r22*wh_vbz[i];
    wh_Fn[i] = if wh_h[i] > 0 then max(0, ground_k*wh_h[i] + ground_c*wh_vwd[i]) else 0;
    wh_F[1, i] = -wh_Fn[i]*r20 + (if wh_h[i] > 0 then -roll_fric*wh_vbx[i] else 0);
    wh_F[2, i] = -wh_Fn[i]*r21 + (if wh_h[i] > 0 then -side_fric*wh_vby[i] else 0);
    wh_F[3, i] = -wh_Fn[i]*r22;
    wh_M[1, i] = wheel_y[i]*wh_F[3, i] - wheel_z[i]*wh_F[2, i];
    wh_M[2, i] = wheel_z[i]*wh_F[1, i] - wheel_x[i]*wh_F[3, i];
    wh_M[3, i] = wheel_x[i]*wh_F[2, i] - wheel_y[i]*wh_F[1, i];
  end for;
  F_ground = {wh_F[1, 1] + wh_F[1, 2] + wh_F[1, 3],
              wh_F[2, 1] + wh_F[2, 2] + wh_F[2, 3],
              wh_F[3, 1] + wh_F[3, 2] + wh_F[3, 3]};
  M_ground = {wh_M[1, 1] + wh_M[1, 2] + wh_M[1, 3],
              wh_M[2, 1] + wh_M[2, 2] + wh_M[2, 3],
              wh_M[3, 1] + wh_M[3, 2] + wh_M[3, 3]};

  fx = FA_frd[1] + F_thrust[1] + F_ground[1];
  fy = FA_frd[2] + F_thrust[2] + F_ground[2];
  fz = FA_frd[3] + F_thrust[3] + F_ground[3];
  mx = MA_frd[1] + M_ground[1];
  my = MA_frd[2] + M_ground[2];
  mz = MA_frd[3] + M_ground[3];

  // --- translational dynamics in body FRD ---
  der(vel[1]) = fx/m - g*s_th + rates[3]*vel[2] - rates[2]*vel[3];
  der(vel[2]) = fy/m + g*s_phi*c_th + rates[1]*vel[3] - rates[3]*vel[1];
  der(vel[3]) = fz/m + g*c_phi*c_th + rates[2]*vel[1] - rates[1]*vel[2];

  // --- rigid-body angular dynamics with Ixz coupling ---
  hx = Ixx*rates[1] - Ixz*rates[3];
  hy = Iyy*rates[2];
  hz = Izz*rates[3] - Ixz*rates[1];
  tx = mx - (rates[2]*hz - rates[3]*hy);
  ty = my - (rates[3]*hx - rates[1]*hz);
  tz = mz - (rates[1]*hy - rates[2]*hx);
  detI = Ixx*Izz - Ixz*Ixz;
  der(rates[1]) = (Izz*tx + Ixz*tz)/detI;
  der(rates[2]) = ty/Iyy;
  der(rates[3]) = (Ixz*tx + Ixx*tz)/detI;

  // --- Euler and position kinematics ---
  c_th_safe = sign(c_th)*max(abs(c_th), 1e-3);
  common = rates[2]*s_phi + rates[3]*c_phi;
  der(att[1]) = rates[1] + (s_th/c_th_safe)*common;
  der(att[2]) = rates[2]*c_phi - rates[3]*s_phi;
  der(att[3]) = common/c_th_safe;
  der(pos[1]) = r00*vel[1] + r01*vel[2] + r02*vel[3];
  der(pos[2]) = r10*vel[1] + r11*vel[2] + r12*vel[3];
  der(pos[3]) = r20*vel[1] + r21*vel[2] + r22*vel[3];
end SportCubGreyboxIdentified;
