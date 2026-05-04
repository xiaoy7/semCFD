function [lag_force, ibm] = ibm_lagrangian_forces2d(ibm, dT)
%IBM_LAGRANGIAN_FORCES2D Structural force on Lagrangian markers.

nb = ibm.marker_count;
x = ibm.markers(:, 1);
y = ibm.markers(:, 2);
xp = ibm.previous_markers(:, 1);
yp = ibm.previous_markers(:, 2);

fx = zeros(nb, 1);
fy = zeros(nb, 1);
model = ibm.model;

if isfield(model, 'springs') && model.springs.enabled
    springs = model.springs.data;
    for k = 1:size(springs, 1)
        i = springs(k, 1);
        j = springs(k, 2);
        stiffness = springs(k, 3);
        rest_length = springs(k, 4);
        alpha = springs(k, 5);
        dx = x(j) - x(i);
        dy = y(j) - y(i);
        length_ij = max(sqrt(dx^2 + dy^2), eps);
        force_mag = 0.5 * (alpha + 1) * stiffness ...
            * (length_ij - rest_length)^alpha;
        sx = force_mag * dx / length_ij;
        sy = force_mag * dy / length_ij;
        fx(i) = fx(i) + sx;
        fy(i) = fy(i) + sy;
        fx(j) = fx(j) - sx;
        fy(j) = fy(j) - sy;
    end
end

if isfield(model, 'beams') && model.beams.enabled
    beams = model.beams.data;
    for k = 1:size(beams, 1)
        p = beams(k, 1);
        q = beams(k, 2);
        r = beams(k, 3);
        stiffness = beams(k, 4);
        curvature = beams(k, 5);
        cross_prod = (x(r) - x(q)) * (y(q) - y(p)) ...
            - (y(r) - y(q)) * (x(q) - x(p));
        bend = stiffness * (cross_prod - curvature);

        bfx_l = -bend * (y(r) - y(q));
        bfy_l =  bend * (x(r) - x(q));
        bfx_m =  bend * ((y(q) - y(p)) + (y(r) - y(q)));
        bfy_m = -bend * ((x(r) - x(q)) + (x(q) - x(p)));
        bfx_r = -bend * (y(q) - y(p));
        bfy_r =  bend * (x(q) - x(p));

        fx(p) = fx(p) - bfx_l;
        fy(p) = fy(p) - bfy_l;
        fx(q) = fx(q) + bfx_m;
        fy(q) = fy(q) + bfy_m;
        fx(r) = fx(r) - bfx_r;
        fy(r) = fy(r) - bfy_r;
    end
end

if isfield(model, 'targets') && model.targets.enabled
    targets = model.targets.data;
    for k = 1:size(targets, 1)
        i = targets(k, 1);
        stiffness = targets(k, 4);
        fx(i) = fx(i) + stiffness * (targets(k, 2) - x(i));
        fy(i) = fy(i) + stiffness * (targets(k, 3) - y(i));
    end
end

if isfield(model, 'masses') && model.masses.enabled
    masses = model.masses.data;
    for k = 1:size(masses, 1)
        i = masses(k, 1);
        mass_value = masses(k, 3);
        fx(i) = fx(i) + mass_value * ibm.gravity(1) / max(ibm.marker_ds, eps);
        fy(i) = fy(i) + mass_value * ibm.gravity(2) / max(ibm.marker_ds, eps);
    end
end

if isfield(model, 'muscles') && model.muscles.enabled
    muscles = model.muscles.data;
    for k = 1:size(muscles, 1)
        i = muscles(k, 1);
        j = muscles(k, 2);
        l_opt = muscles(k, 3);
        shape_coeff = muscles(k, 4);
        hill_a = muscles(k, 5);
        hill_b = muscles(k, 6);
        f_max = muscles(k, 7);
        activation = muscles(k, 8);

        dx = x(j) - x(i);
        dy = y(j) - y(i);
        length_ij = max(sqrt(dx^2 + dy^2), eps);
        prev_length = max(sqrt((xp(j) - xp(i))^2 + (yp(j) - yp(i))^2), eps);
        shortening_speed = (prev_length - length_ij) / max(dT, eps);

        length_tension = exp(-shape_coeff * ((length_ij / max(l_opt, eps)) - 1)^2);
        force_velocity = max(0, (hill_b - shortening_speed) ...
            / (hill_b + max(shortening_speed, 0) + hill_a));
        muscle_force = activation * f_max * length_tension * force_velocity;

        mx = muscle_force * dx / length_ij;
        my = muscle_force * dy / length_ij;
        fx(i) = fx(i) + mx;
        fy(i) = fy(i) + my;
        fx(j) = fx(j) - mx;
        fy(j) = fy(j) - my;
    end
end

lag_force = [fx, fy];
if isfield(ibm, 'max_lagrangian_force_density')
    force_mag = sqrt(sum(lag_force.^2, 2));
    capped = force_mag > ibm.max_lagrangian_force_density;
    if any(capped)
        scale = ibm.max_lagrangian_force_density ./ max(force_mag(capped), eps);
        lag_force(capped, :) = lag_force(capped, :) .* scale;
    end
end
if any(~isfinite(lag_force), 'all')
    error('IBM:NonFiniteLagrangianForce', ...
        'Non-finite Lagrangian force detected before spreading.');
end
ibm.last_marker_force = lag_force .* ibm.marker_ds;
ibm.hydro_force = -sum(ibm.last_marker_force, 1);
r = ibm.markers - mean(ibm.markers, 1);
ibm.hydro_torque = -sum(r(:, 1) .* ibm.last_marker_force(:, 2) ...
                      - r(:, 2) .* ibm.last_marker_force(:, 1));

end
