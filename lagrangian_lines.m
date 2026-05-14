function lines = lagrangian_lines(ibm)
lines = zeros(0, 2);
if isfield(ibm, 'model') && isfield(ibm.model, 'springs') ...
        && isfield(ibm.model.springs, 'data') ...
        && ~isempty(ibm.model.springs.data)
    springs = ibm.model.springs.data;
    active = springs(:, 3) ~= 0;
    lines = [lines; springs(active, 1:2) - 1];
end
end

