function ibm = ibm_setup2d(coordX, coordY, dT, para)
% Setup an immersed boundary description on the SEM grid.
% Current version initializes one fixed circular rigid body.

ibm.enabled = true;
ibm.method = 'direct_forcing';
ibm.penalty = 100; % Fixed penalty for stability

% Geometry: centered cylinder (modifiable for other test cases)
ibm.center = [0.5 * (para.minx + para.maxx), 0.8 * (para.miny + para.maxy)]; % Position drop higher
ibm.radius = 0.10 * min(para.maxx - para.minx, para.maxy - para.miny);
ibm.smoothing_width = 2 * max((para.maxx - para.minx) / (para.nx_all - 1), ...
                              (para.maxy - para.miny) / (para.ny_all - 1));

radius_field = sqrt((coordX - ibm.center(1)).^2 + (coordY - ibm.center(2)).^2);
ibm.mask = 0.5 * (1 - tanh((radius_field - ibm.radius) / max(ibm.smoothing_width, eps)));

% Initial rigid body velocity
ibm.target_u = zeros(size(coordX));
ibm.target_v = zeros(size(coordY));
ibm.body_velocity = [0, 0]; % Initial velocity
ibm.body_omega = 0;

end
