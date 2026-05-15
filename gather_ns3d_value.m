function out = gather_ns3d_value(in)
if isa(in, 'gpuArray')
    out = gather(in);
else
    out = in;
end
end
