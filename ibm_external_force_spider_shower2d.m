function [force_u, force_v] = ibm_external_force_spider_shower2d(u, v, ibm, t)
%IBM_EXTERNAL_FORCE_SPIDER_SHOWER2D Left-side target-velocity inflow force.

input_on_gpu = isa(u, 'gpuArray') || isa(v, 'gpuArray');
u_cpu = gather(u);
v_cpu = gather(v);

force_u_cpu = zeros(size(u_cpu));
force_v_cpu = zeros(size(v_cpu));

if ~isfield(ibm, 'external_force') || ~ibm.external_force.enabled
    if input_on_gpu
        force_u = gpuArray(force_u_cpu);
        force_v = gpuArray(force_v_cpu);
    else
        force_u = force_u_cpu;
        force_v = force_v_cpu;
    end
    return
end

cfg = ibm.external_force;
x_smooth = max(cfg.x_smooth_width, eps);
y_taper = max(cfg.y_wall_taper, eps);

x_rise = smoothstep((ibm.x - (cfg.x_min - x_smooth)) / x_smooth);
x_fall = 1 - smoothstep((ibm.x - cfg.x_max) / x_smooth);
x_window = x_rise .* x_fall;

y_bottom = smoothstep((ibm.y - cfg.y_min) / y_taper);
y_top = 1 - smoothstep((ibm.y - (cfg.y_max - y_taper)) / y_taper);
y_window = y_bottom .* y_top;

window = x_window * y_window.';
if max(window, [], 'all') > eps
    window = window / max(window, [], 'all');
end

target_u = cfg.u_max * tanh(5 * t);
force_u_cpu = -cfg.k_stiff * window .* (u_cpu - target_u);
force_v_cpu = -cfg.k_stiff * window .* v_cpu;

if isfield(ibm, 'max_eulerian_acceleration')
    force_u_cpu = min(max(force_u_cpu, -ibm.max_eulerian_acceleration), ...
        ibm.max_eulerian_acceleration);
    force_v_cpu = min(max(force_v_cpu, -ibm.max_eulerian_acceleration), ...
        ibm.max_eulerian_acceleration);
end
if any(~isfinite(force_u_cpu), 'all') || any(~isfinite(force_v_cpu), 'all')
    error('IBM:NonFiniteExternalForce', ...
        'Non-finite Baby Spider Shower external forcing detected.');
end

if input_on_gpu
    force_u = gpuArray(force_u_cpu);
    force_v = gpuArray(force_v_cpu);
else
    force_u = force_u_cpu;
    force_v = force_v_cpu;
end

end

function y = smoothstep(x)
%SMOOTHSTEP Compact C1 ramp from 0 to 1 on x in [0,1].

x = min(max(x, 0), 1);
y = x.^2 .* (3 - 2 * x);

end
