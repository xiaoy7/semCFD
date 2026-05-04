function ibm = ibm_refresh_markers2d(ibm)
%IBM_REFRESH_MARKERS2D Recompute circular body markers and no-slip targets.

theta = linspace(0, 2 * pi, ibm.marker_count + 1).';
theta(end) = [];

normal = [cos(theta), sin(theta)];
ibm.markers = ibm.center + ibm.radius * normal;
ibm.marker_ds = 2 * pi * ibm.radius / ibm.marker_count;
ibm.marker_volume = ibm.marker_ds * ibm.marker_thickness ...
    * ones(ibm.marker_count, 1);

r = ibm.markers - ibm.center;
ibm.marker_velocity = [ ...
    ibm.body_velocity(1) - ibm.body_omega * r(:, 2), ...
    ibm.body_velocity(2) + ibm.body_omega * r(:, 1)];

end

