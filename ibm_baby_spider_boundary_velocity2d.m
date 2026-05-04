function u_boundary = ibm_baby_spider_boundary_velocity2d(coordX, coordY, ibm, t)
%IBM_BABY_SPIDER_BOUNDARY_VELOCITY2D Smooth through-flow boundary velocity.

u_boundary = zeros(size(coordX));
if ~isfield(ibm, 'case_name') || ~strcmpi(ibm.case_name, 'baby_spider_shower')
    return
end

if isfield(ibm, 'external_force')
    u_max = ibm.external_force.u_max;
    y_taper = ibm.external_force.y_wall_taper;
else
    u_max = 0.2;
    y_taper = 0.05 * (max(coordY(:)) - min(coordY(:)));
end

y_min = min(coordY(:));
y_max = max(coordY(:));
y = coordY(1, :);

y_bottom = smoothstep((y - y_min) / max(y_taper, eps));
y_top = 1 - smoothstep((y - (y_max - y_taper)) / max(y_taper, eps));
profile = y_bottom .* y_top;
if max(profile) > eps
    profile = profile / max(profile);
end

u_wall = u_max * tanh(5 * t) * profile;
u_boundary(1, :) = u_wall;
u_boundary(end, :) = u_wall;

end

function y = smoothstep(x)
%SMOOTHSTEP Compact C1 ramp from 0 to 1 on x in [0,1].

x = min(max(x, 0), 1);
y = x.^2 .* (3 - 2 * x);

end

