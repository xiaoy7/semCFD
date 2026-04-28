function ibm = ibm_update_rigid_body2d(ibm, Iter, dT, coordX, coordY)
% Update rigid-body motion in FSI coupling.
% For droplet simulation, apply gravity and update position/mask.

if ~ibm.enabled
    return
end

% Apply gravity acceleration (assuming g=1, downward)
ibm.body_velocity(2) = ibm.body_velocity(2) - 1 * dT;

% Update position
ibm.center = ibm.center + dT * ibm.body_velocity;

% Boundary collision check (domain [0,1] x [0,1])
minx = 0; maxx = 1; miny = 0; maxy = 1;
if ibm.center(1) - ibm.radius < minx
    ibm.center(1) = minx + ibm.radius;
    ibm.body_velocity(1) = 0;
elseif ibm.center(1) + ibm.radius > maxx
    ibm.center(1) = maxx - ibm.radius;
    ibm.body_velocity(1) = 0;
end
if ibm.center(2) - ibm.radius < miny
    ibm.center(2) = miny + ibm.radius;
    ibm.body_velocity(2) = 0;
elseif ibm.center(2) + ibm.radius > maxy
    ibm.center(2) = maxy - ibm.radius;
    ibm.body_velocity(2) = 0;
end

% Recalculate mask for new position
radius_field = sqrt((coordX - ibm.center(1)).^2 + (coordY - ibm.center(2)).^2);
ibm.mask = 0.5 * (1 - tanh((radius_field - ibm.radius) / max(ibm.smoothing_width, eps)));

% Update target velocities for rigid body motion
ibm.target_u = ibm.body_velocity(1) * ones(size(coordX));
ibm.target_v = ibm.body_velocity(2) * ones(size(coordY));

ibm.time = Iter * dT;

end
