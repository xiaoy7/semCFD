function value = gpu_value(value)
if isstruct(value)
    names = fieldnames(value);
    for k = 1:numel(names)
        value.(names{k}) = gpu_value(value.(names{k}));
    end
else
    value = gpuArray(value);
end
end

