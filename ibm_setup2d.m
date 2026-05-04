function ibm = ibm_setup2d(coordX, coordY, dT, para)
<<<<<<< HEAD
% Setup an immersed boundary description on the SEM grid.
% Current version initializes one fixed circular rigid body.

ibm.enabled = true;
ibm.method = 'direct_forcing';
ibm.penalty = 1 / max(dT, eps);

% Geometry: centered cylinder (modifiable for other test cases)
ibm.center = [0.5 * (para.minx + para.maxx), 0.5 * (para.miny + para.maxy)];
ibm.radius = 0.10 * min(para.maxx - para.minx, para.maxy - para.miny);
ibm.smoothing_width = 2 * max((para.maxx - para.minx) / (para.nx_all - 1), ...
                              (para.maxy - para.miny) / (para.ny_all - 1));

radius_field = sqrt((coordX - ibm.center(1)).^2 + (coordY - ibm.center(2)).^2);
ibm.mask = 0.5 * (1 - tanh((radius_field - ibm.radius) / max(ibm.smoothing_width, eps)));

% Fixed rigid body velocity (placeholder for full FSI coupling)
ibm.target_u = zeros(size(coordX));
ibm.target_v = zeros(size(coordY));
ibm.body_velocity = [0, 0];
ibm.body_omega = 0;
=======
%IBM_SETUP2D Build a marker-based IBM model.
%   Set ibm.method to:
%       'lagrangian_structure'  Peskin-style elastic Lagrangian structure
%       'marker_direct_forcing' rigid direct-forcing body
%
%   The Lagrangian structure mode supports springs, torsional beams, target
%   points, and a Hill length-tension/force-velocity muscle model.

ibm.enabled = true;
ibm.method = 'lagrangian_structure';
ibm.sharp_enforce = false;

ibm.rho_f = 1.0;
ibm.rho_p = 1.5;
ibm.gravity = [0, -1];
ibm.motion = 'free';       % 'fixed', 'prescribed', or 'free'
ibm.prescribed_velocity = [0, 0];
ibm.prescribed_omega = 0;

domain_width = para.maxx - para.minx;
domain_height = para.maxy - para.miny;
ibm.center = [0.5 * (para.minx + para.maxx), ...
              0.8 * (para.miny + para.maxy)];
ibm.radius = 0.10 * min(domain_width, domain_height);

x = coordX(:, 1);
y = coordY(1, :);
dx_min = min(diff(x));
dy_min = min(diff(y));
h_min = min(dx_min, dy_min);
h_max = max(dx_min, dy_min);

ibm.kernel_width = 2.0 * h_max;
ibm.support_radius = 3.0 * ibm.kernel_width;
ibm.delta_hx = max(dx_min, eps);
ibm.delta_hy = max(dy_min, eps);
ibm.marker_count = max(64, ceil(2 * pi * ibm.radius / max(0.75 * h_min, eps)));
ibm.marker_thickness = max(h_min, ibm.kernel_width);

ibm.body_velocity = [0, 0];
ibm.body_omega = 0;
ibm.body_mass = ibm.rho_p * pi * ibm.radius^2;
ibm.body_inertia = 0.5 * ibm.body_mass * ibm.radius^2;
ibm.time = 0;

ibm.x = x(:);
ibm.y = y(:);
ibm.grid_weight = ibm_control_volume_weights2d(ibm.x, ibm.y);

ibm = ibm_refresh_markers2d(ibm);
ibm.reference_markers = ibm.markers;
ibm.previous_markers = ibm.markers;
ibm.marker_velocity = zeros(ibm.marker_count, 2);
ibm.model = ibm_default_lagrangian_models2d(ibm);
ibm = ibm_refresh_mask2d(ibm, coordX, coordY);
ibm.target_u = ibm.body_velocity(1) - ibm.body_omega * (coordY - ibm.center(2));
ibm.target_v = ibm.body_velocity(2) + ibm.body_omega * (coordX - ibm.center(1));

ibm.last_marker_force = zeros(ibm.marker_count, 2);
ibm.hydro_force = [0, 0];
ibm.hydro_torque = 0;
ibm.dT = dT;
>>>>>>> main

end
