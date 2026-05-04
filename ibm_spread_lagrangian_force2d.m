function [force_u, force_v, ibm] = ibm_spread_lagrangian_force2d(u, v, ibm, dT)
%IBM_SPREAD_LAGRANGIAN_FORCE2D Spread elastic Lagrangian force to SEM grid.

input_on_gpu = isa(u, 'gpuArray') || isa(v, 'gpuArray');
u_cpu = gather(u);

[lag_force_density, ibm] = ibm_lagrangian_forces2d(ibm, dT);
lag_force = lag_force_density .* ibm.marker_ds;

force_u_cpu = zeros(size(u_cpu));
force_v_cpu = zeros(size(u_cpu));
rho_f = ibm.rho_f;
hx = ibm.delta_hx;
hy = ibm.delta_hy;

for m = 1:ibm.marker_count
    xm = ibm.markers(m, 1);
    ym = ibm.markers(m, 2);
    dx = ibm.x - xm;
    dy = ibm.y - ym;
    ix = find(abs(dx) < 2 * hx);
    iy = find(abs(dy) < 2 * hy);

    if isempty(ix) || isempty(iy)
        continue
    end

    [DX, DY] = ndgrid(dx(ix), dy(iy));
    delta = ibm_delta_peskin4_1d(DX, hx) .* ibm_delta_peskin4_1d(DY, hy);
    weights = ibm.grid_weight(ix, iy);
    normalizer = sum(delta .* weights, 'all');
    if normalizer <= eps
        continue
    end
    delta = delta / normalizer;

    force_u_cpu(ix, iy) = force_u_cpu(ix, iy) + lag_force(m, 1) * delta / rho_f;
    force_v_cpu(ix, iy) = force_v_cpu(ix, iy) + lag_force(m, 2) * delta / rho_f;
end

if isfield(ibm, 'max_eulerian_acceleration')
    force_u_cpu = min(max(force_u_cpu, -ibm.max_eulerian_acceleration), ...
        ibm.max_eulerian_acceleration);
    force_v_cpu = min(max(force_v_cpu, -ibm.max_eulerian_acceleration), ...
        ibm.max_eulerian_acceleration);
end
if any(~isfinite(force_u_cpu), 'all') || any(~isfinite(force_v_cpu), 'all')
    error('IBM:NonFiniteEulerianForce', ...
        'Non-finite Eulerian IBM force detected after spreading.');
end

if input_on_gpu
    force_u = gpuArray(force_u_cpu);
    force_v = gpuArray(force_v_cpu);
else
    force_u = force_u_cpu;
    force_v = force_v_cpu;
end

end
