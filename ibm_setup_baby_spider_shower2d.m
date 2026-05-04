function ibm = ibm_setup_baby_spider_shower2d(ibm, coordX, coordY, para)
%IBM_SETUP_BABY_SPIDER_SHOWER2D Configure the IB2d Baby Spider Shower case.

domain_width = para.maxx - para.minx;
domain_height = para.maxy - para.miny;
ib2d_nx = para.nx_all - 1;
ib2d_dx = domain_width / ib2d_nx;
ds = 0.5 * ib2d_dx;
web_length = 0.175 * domain_height;

y_top = para.miny + 0.85 * domain_height;
web_count = floor(web_length / ds) + 1;
y_web = y_top - (0:web_count-1).' * ds;
x_web = (para.minx + (0.25 / 1.5) * domain_width) * ones(size(y_web));
markers = [x_web, y_web];

ibm.case_name = 'baby_spider_shower';
ibm.method = 'lagrangian_structure';
ibm.motion = 'elastic';
ibm.rho_f = 1.225;
ibm.gravity = [0, -1];
ibm.markers = markers;
ibm.reference_markers = markers;
ibm.previous_markers = markers;
ibm.marker_count = size(markers, 1);
ibm.marker_ds = ds;
ibm.marker_thickness = max(ds, ibm.kernel_width);
ibm.marker_volume = ibm.marker_ds * ibm.marker_thickness ...
    * ones(ibm.marker_count, 1);
ibm.marker_velocity = zeros(ibm.marker_count, 2);
ibm.center = mean(markers, 1);
ibm.radius = max(sqrt(sum((markers - ibm.center).^2, 2)));

spring_count = ibm.marker_count - 1;
beam_count = ibm.marker_count - 2;
spring_ids = (1:spring_count).';
beam_ids = (1:beam_count).';

k_spring = 0.0375 / ds^2;
k_beam = 0.38 / ds^2;

ibm.model.springs.enabled = true;
ibm.model.springs.break_distance = 5 * ds;
ibm.model.springs.data = [spring_ids, spring_ids + 1, ...
    k_spring * ones(spring_count, 1), ds * ones(spring_count, 1), ...
    ones(spring_count, 1)];

ibm.model.beams.enabled = true;
ibm.model.beams.data = [beam_ids, beam_ids + 1, beam_ids + 2, ...
    k_beam * ones(beam_count, 1), zeros(beam_count, 1)];

ibm.model.targets.enabled = false;
ibm.model.targets.data = zeros(0, 4);

ibm.model.muscles.enabled = false;
ibm.model.muscles.data = zeros(0, 8);

ibm.model.masses.enabled = true;
ibm.model.masses.data = [ibm.marker_count, 1.0e4, 0.0225];

ibm.max_marker_speed = 0.5;
ibm.max_lagrangian_force_density = 2.0e4;
ibm.max_eulerian_acceleration = 50.0;

ibm.external_force.enabled = false;
ibm.external_force.k_stiff = 2.0e2;
ibm.external_force.u_max = 0.2;
ibm.external_force.x_min = para.minx + (0.05 / 1.5) * domain_width;
ibm.external_force.x_max = para.minx + (0.09 / 1.5) * domain_width;
ibm.external_force.y_min = para.miny;
ibm.external_force.y_max = para.maxy;
ibm.external_force.x_smooth_width = max(4 * ibm.delta_hx, ...
    0.25 * (ibm.external_force.x_max - ibm.external_force.x_min));
ibm.external_force.y_wall_taper = max(8 * ibm.delta_hy, 0.05 * domain_height);

ibm = ibm_refresh_mask2d(ibm, coordX, coordY);
ibm.target_u = zeros(size(coordX));
ibm.target_v = zeros(size(coordY));
ibm.last_marker_force = zeros(ibm.marker_count, 2);
ibm.hydro_force = [0, 0];
ibm.hydro_torque = 0;

end
