function write_lagrangian_points(filename, ibm, time, with_connections)
markers = ibm.markers;
n = size(markers, 1);
fid = fopen(filename, 'w');
assert(fid > 0, 'Could not open VTK file: %s', filename);
cleanup = onCleanup(@() fclose(fid));
write_header(fid, 'Lagrangian points', time);
fprintf(fid, 'DATASET POLYDATA\n');
fprintf(fid, 'POINTS %d float\n', n);
for i = 1:n
    fprintf(fid, '%.9e %.9e 0\n', markers(i, 1), markers(i, 2));
end

if with_connections
    lines = lagrangian_lines(ibm);
    fprintf(fid, 'LINES %d %d\n', size(lines, 1), 3 * size(lines, 1));
    for k = 1:size(lines, 1)
        fprintf(fid, '2 %d %d\n', lines(k, 1), lines(k, 2));
    end
else
    fprintf(fid, 'VERTICES %d %d\n', n, 2 * n);
    for i = 0:n-1
        fprintf(fid, '1 %d\n', i);
    end
end

fprintf(fid, '\nPOINT_DATA %d\n', n);
if isfield(ibm, 'marker_velocity') && size(ibm.marker_velocity, 1) == n
    fprintf(fid, 'VECTORS marker_velocity float\n');
    for i = 1:n
        fprintf(fid, '%.9e %.9e 0\n', ibm.marker_velocity(i, 1), ...
            ibm.marker_velocity(i, 2));
    end
end
if isfield(ibm, 'last_marker_force') && size(ibm.last_marker_force, 1) == n
    fprintf(fid, '\nVECTORS marker_force float\n');
    for i = 1:n
        fprintf(fid, '%.9e %.9e 0\n', ibm.last_marker_force(i, 1), ...
            ibm.last_marker_force(i, 2));
    end
end
end

