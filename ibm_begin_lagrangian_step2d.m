function ibm = ibm_begin_lagrangian_step2d(u, v, ibm, dT, coordX, coordY)
%IBM_BEGIN_LAGRANGIAN_STEP2D Predict half-step Lagrangian geometry.
%   This mirrors the first Lagrangian move in IB2d's IBM_Driver:
%       X_h = X_n + 0.5 * dt * U_n(X_n).
%   Structural forces are then assembled on X_h.

if ~ibm.enabled
    return
end

u_cpu = gather(u);
v_cpu = gather(v);
Fu = griddedInterpolant({ibm.x, ibm.y}, u_cpu, 'linear', 'nearest');
Fv = griddedInterpolant({ibm.x, ibm.y}, v_cpu, 'linear', 'nearest');

ibm.step_markers = ibm.markers;
u_lag = Fu(ibm.step_markers(:, 1), ibm.step_markers(:, 2));
v_lag = Fv(ibm.step_markers(:, 1), ibm.step_markers(:, 2));

ibm.half_marker_velocity = [u_lag, v_lag];
if any(~isfinite(ibm.half_marker_velocity), 'all')
    error('IBM:NonFiniteHalfStepVelocity', ...
        'Non-finite velocity interpolated to Lagrangian markers.');
end
if isfield(ibm, 'max_marker_speed')
    speed = sqrt(sum(ibm.half_marker_velocity.^2, 2));
    capped = speed > ibm.max_marker_speed;
    if any(capped)
        scale = ibm.max_marker_speed ./ max(speed(capped), eps);
        ibm.half_marker_velocity(capped, :) = ...
            ibm.half_marker_velocity(capped, :) .* scale;
    end
end
ibm.markers = ibm.step_markers + 0.5 * dT * ibm.half_marker_velocity;
ibm.markers(:, 1) = min(max(ibm.markers(:, 1), ibm.x(1)), ibm.x(end));
ibm.markers(:, 2) = min(max(ibm.markers(:, 2), ibm.y(1)), ibm.y(end));
ibm.center = mean(ibm.markers, 1);

ibm = ibm_refresh_mask2d(ibm, coordX, coordY);
ibm.target_u = zeros(size(coordX));
ibm.target_v = zeros(size(coordY));

end
