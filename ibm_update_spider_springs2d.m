function ibm = ibm_update_spider_springs2d(ibm)
%IBM_UPDATE_SPIDER_SPRINGS2D Break web springs that stretch too far.

if ~isfield(ibm, 'model') || ~isfield(ibm.model, 'springs') ...
        || ~ibm.model.springs.enabled ...
        || ~isfield(ibm.model.springs, 'break_distance')
    return
end

springs = ibm.model.springs.data;
if isempty(springs)
    return
end

i = springs(:, 1);
j = springs(:, 2);
d = ibm.markers(j, :) - ibm.markers(i, :);
lengths = sqrt(sum(d.^2, 2));
springs(lengths > ibm.model.springs.break_distance, 3) = 0;
ibm.model.springs.data = springs;

end

