function out = move_ns3d_struct_to_gpu(in)
out = in;
fieldNames = fieldnames(in);
for k = 1:numel(fieldNames)
    name = fieldNames{k};
    if isstruct(in.(name))
        out.(name) = move_ns3d_struct_to_gpu(in.(name));
    else
        out.(name) = gpuArray(in.(name));
    end
end
end
