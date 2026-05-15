function out = gather_ns3d_struct(in)
out = in;
fieldNames = fieldnames(in);
for k = 1:numel(fieldNames)
    name = fieldNames{k};
    if isstruct(in.(name))
        out.(name) = gather_ns3d_struct(in.(name));
    elseif isa(in.(name), 'gpuArray')
        out.(name) = gather(in.(name));
    else
        out.(name) = in.(name);
    end
end
end
