function ibm = ibm_update_lagrangian_markers2d(u, v, ibm, dT, coordX, coordY)
%IBM_UPDATE_LAGRANGIAN_MARKERS2D Complete IBM_Driver-style marker update.
%   The force was assembled at the half-step geometry.  Finish the
%   Lagrangian step with
%       X_np1 = X_n + dt * U_np1(X_h),
%   using the SEM velocity supplied by the caller.

if ~ibm.enabled
    return
end

u_cpu = gather(u);
v_cpu = gather(v);
Fu = griddedInterpolant({ibm.x, ibm.y}, u_cpu, 'linear', 'nearest');
Fv = griddedInterpolant({ibm.x, ibm.y}, v_cpu, 'linear', 'nearest');

half_markers = ibm.markers;
if isfield(ibm, 'step_markers')
    old_markers = ibm.step_markers;
else
    old_markers = ibm.markers;
end

u_lag = Fu(half_markers(:, 1), half_markers(:, 2));
v_lag = Fv(half_markers(:, 1), half_markers(:, 2));
ibm.marker_velocity = [u_lag, v_lag];
if any(~isfinite(ibm.marker_velocity), 'all')
    error('IBM:NonFiniteMarkerVelocity', ...
        'Non-finite velocity interpolated during Lagrangian marker update.');
end
if isfield(ibm, 'max_marker_speed')
    speed = sqrt(sum(ibm.marker_velocity.^2, 2));
    capped = speed > ibm.max_marker_speed;
    if any(capped)
        scale = ibm.max_marker_speed ./ max(speed(capped), eps);
        ibm.marker_velocity(capped, :) = ibm.marker_velocity(capped, :) .* scale;
    end
end
ibm.markers = old_markers + dT * ibm.marker_velocity;

ibm.markers(:, 1) = min(max(ibm.markers(:, 1), ibm.x(1)), ibm.x(end));
ibm.markers(:, 2) = min(max(ibm.markers(:, 2), ibm.y(1)), ibm.y(end));

ibm.center = mean(ibm.markers, 1);
ibm = ibm_refresh_mask2d(ibm, coordX, coordY);
ibm.target_u = zeros(size(coordX));
ibm.target_v = zeros(size(coordY));
ibm.time = ibm.time + dT;

ibm.previous_markers = half_markers;
if isfield(ibm, 'step_markers')
    ibm = rmfield(ibm, 'step_markers');
end
if isfield(ibm, 'half_marker_velocity')
    ibm = rmfield(ibm, 'half_marker_velocity');
end

end
