function model = ibm_default_lagrangian_models2d(ibm)
%IBM_DEFAULT_LAGRANGIAN_MODELS2D Default IB2d-style structural models.

nb = ibm.marker_count;
ids = (1:nb).';
next_ids = [2:nb, 1].';
prev_ids = [nb, 1:nb-1].';

edge = ibm.markers(next_ids, :) - ibm.markers(ids, :);
rest_length = sqrt(sum(edge.^2, 2));

model.springs.enabled = true;
model.springs.data = [ids, next_ids, ...
    5.0e2 * ones(nb, 1), rest_length, ones(nb, 1)];

beam_curvature = ibm_compute_beam_curvature2d(ibm.markers, prev_ids, ids, next_ids);
model.beams.enabled = true;
model.beams.data = [prev_ids, ids, next_ids, ...
    1.0e-2 * ones(nb, 1), beam_curvature];

model.targets.enabled = true;
model.targets.data = [ids, ibm.reference_markers, 5.0e1 * ones(nb, 1)];

model.muscles.enabled = false;
model.muscles.data = [ids, next_ids, rest_length, ...
    ones(nb, 1), 0.25 * ones(nb, 1), 0.25 * ones(nb, 1), ...
    2.0e1 * ones(nb, 1), zeros(nb, 1)];

end

