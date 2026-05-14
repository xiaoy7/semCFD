function write_structured_scalar(filename, nx, ny, name, pointst, data, time)
fid = fopen(filename, 'w');
assert(fid > 0, 'Could not open VTK file: %s', filename);
cleanup = onCleanup(@() fclose(fid));
write_structured_intro(fid,  nx, ny, name, pointst, time);
% [nx, ny] = size(coordX);
fprintf(fid, 'SCALARS %s float 1\n', name);
fprintf(fid, 'LOOKUP_TABLE default\n');
% for j = 1:ny
%     for i = 1:nx
%         fprintf(fid, '%.9e\n', data(i, j));
%     end
% end
% export the data by using matrix format, which is much faster
data = data';
data = data(:);
fprintf(fid, '%.9e\n', data);
end

