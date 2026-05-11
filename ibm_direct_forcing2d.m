
function [force_u, force_v, ibm] = ibm_direct_forcing2d(u, v, ibm, dT)
%IBM_DIRECT_FORCING2D Marker-based direct-forcing immersed boundary source.
%   force_u and force_v are Eulerian acceleration fields to add to the
%   momentum RHS.  The Lagrangian marker force follows the NekIBM/Uhlmann
%   direct-forcing form:
%
%       F_l = rho_f * V_l * (U_body_l - u_tilde_l) / dt.
%
%   The equal and opposite hydrodynamic load on the body is accumulated in
%   ibm.hydro_force and ibm.hydro_torque.
% >>>>>>> main

if ~ibm.enabled
    force_u = zeros(size(u), 'like', u);
    force_v = zeros(size(v), 'like', v);
    fprintf('=== return ===\n');
    return
end

% <<<<<<< HEAD
penalty = ibm.penalty;
if isempty(penalty)
    penalty = 1 / max(dT, eps);
end

force_u = penalty * ibm.mask .* (ibm.target_u - u);
force_v = penalty * ibm.mask .* (ibm.target_v - v);
% =======
if isfield(ibm, 'method') && strcmpi(ibm.method, 'lagrangian_structure')
    [force_u, force_v, ibm] = ibm_spread_lagrangian_force2d(u, v, ibm, dT);
    return
end

input_on_gpu = isa(u, 'gpuArray') || isa(v, 'gpuArray');
u_cpu = gather(u);
v_cpu = gather(v);

force_u_cpu = zeros(size(u_cpu));
force_v_cpu = zeros(size(v_cpu));
marker_force = zeros(ibm.marker_count, 2);

Fu = griddedInterpolant({ibm.x, ibm.y}, u_cpu, 'linear', 'nearest');
Fv = griddedInterpolant({ibm.x, ibm.y}, v_cpu, 'linear', 'nearest');

sigma2 = ibm.kernel_width^2;
support2 = ibm.support_radius^2;
rho_f = ibm.rho_f;

for m = 1:ibm.marker_count
    xm = ibm.markers(m, 1);
    ym = ibm.markers(m, 2);

    uf = Fu(xm, ym);
    vf = Fv(xm, ym);
    ub = ibm.marker_velocity(m, 1);
    vb = ibm.marker_velocity(m, 2);

    marker_force(m, 1) = rho_f * ibm.marker_volume(m) * (ub - uf) / dT;
    marker_force(m, 2) = rho_f * ibm.marker_volume(m) * (vb - vf) / dT;

    dx = ibm.x - xm;
    dy = ibm.y - ym;
    ix = find(abs(dx) <= ibm.support_radius);
    iy = find(abs(dy) <= ibm.support_radius);

    if isempty(ix) || isempty(iy)
        continue
    end

    [DX, DY] = ndgrid(dx(ix), dy(iy));
    r2 = DX.^2 + DY.^2;
    delta = exp(-0.5 * r2 / sigma2) / (2 * pi * sigma2);
    delta(r2 > support2) = 0;

    weights = ibm.grid_weight(ix, iy);
    normalizer = sum(delta .* weights, 'all');
    if normalizer <= eps
        continue
    end
    delta = delta / normalizer;

    force_u_cpu(ix, iy) = force_u_cpu(ix, iy) ...
        + marker_force(m, 1) * delta / rho_f;
    force_v_cpu(ix, iy) = force_v_cpu(ix, iy) ...
        + marker_force(m, 2) * delta / rho_f;
end

ibm.last_marker_force = marker_force;
ibm.hydro_force = -sum(marker_force, 1);
r = ibm.markers - ibm.center;
ibm.hydro_torque = -sum(r(:, 1) .* marker_force(:, 2) ...
    - r(:, 2) .* marker_force(:, 1));

if input_on_gpu
    force_u = gpuArray(force_u_cpu);
    force_v = gpuArray(force_v_cpu);
else
    force_u = force_u_cpu;
    force_v = force_v_cpu;
end
% >>>>>>> main

end
