function sem_write_ibm_vtk2d(vtk_dir, dump_id, time, coordX, coordY, ...
    U, V, P, Omega, force_u, force_v, ibm)
%SEM_WRITE_IBM_VTK2D Write IB2d-style VTK output for semCFD.

if ~exist(vtk_dir, 'dir')
    mkdir(vtk_dir);
end

tag = sprintf('%04d', dump_id);
u_mag = sqrt(U.^2 + V.^2);
f_mag = sqrt(force_u.^2 + force_v.^2);

write_structured_vector(fullfile(vtk_dir, ['u.' tag '.vtk']), ...
    'u', coordX, coordY, U, V, time);
write_structured_scalar(fullfile(vtk_dir, ['P.' tag '.vtk']), ...
    'P', coordX, coordY, P, time);
write_structured_scalar(fullfile(vtk_dir, ['Omega.' tag '.vtk']), ...
    'Omega', coordX, coordY, Omega, time);
write_structured_scalar(fullfile(vtk_dir, ['uMag.' tag '.vtk']), ...
    'uMag', coordX, coordY, u_mag, time);
write_structured_scalar(fullfile(vtk_dir, ['fX.' tag '.vtk']), ...
    'fX', coordX, coordY, force_u, time);
write_structured_scalar(fullfile(vtk_dir, ['fY.' tag '.vtk']), ...
    'fY', coordX, coordY, force_v, time);
write_structured_scalar(fullfile(vtk_dir, ['fMag.' tag '.vtk']), ...
    'fMag', coordX, coordY, f_mag, time);

if isfield(ibm, 'markers') && ~isempty(ibm.markers)
    write_lagrangian_points(fullfile(vtk_dir, ['lagsPts.' tag '.vtk']), ...
        ibm, time, false);
    write_lagrangian_points(fullfile(vtk_dir, ['lagPtsConnect.' tag '.vtk']), ...
        ibm, time, true);
end

end

function write_header(fid, title, time)
fprintf(fid, '# vtk DataFile Version 2.0\n');
fprintf(fid, '%s time=%g\n', title, time);
fprintf(fid, 'ASCII\n');
end

function write_structured_intro(fid, title, coordX, coordY, time)
write_header(fid, title, time);
[nx, ny] = size(coordX);
fprintf(fid, 'DATASET STRUCTURED_GRID\n');
fprintf(fid, 'DIMENSIONS %d %d 1\n', nx, ny);
fprintf(fid, 'POINTS %d float\n', nx * ny);
for j = 1:ny
    for i = 1:nx
        fprintf(fid, '%.9e %.9e 0\n', coordX(i, j), coordY(i, j));
    end
end
fprintf(fid, '\nPOINT_DATA %d\n', nx * ny);
end

function write_structured_scalar(filename, name, coordX, coordY, data, time)
fid = fopen(filename, 'w');
assert(fid > 0, 'Could not open VTK file: %s', filename);
cleanup = onCleanup(@() fclose(fid));
write_structured_intro(fid, name, coordX, coordY, time);
[nx, ny] = size(coordX);
fprintf(fid, 'SCALARS %s float 1\n', name);
fprintf(fid, 'LOOKUP_TABLE default\n');
for j = 1:ny
    for i = 1:nx
        fprintf(fid, '%.9e\n', data(i, j));
    end
end
end

function write_structured_vector(filename, name, coordX, coordY, U, V, time)
fid = fopen(filename, 'w');
assert(fid > 0, 'Could not open VTK file: %s', filename);
cleanup = onCleanup(@() fclose(fid));
write_structured_intro(fid, name, coordX, coordY, time);
[nx, ny] = size(coordX);
fprintf(fid, 'VECTORS %s float\n', name);
for j = 1:ny
    for i = 1:nx
        fprintf(fid, '%.9e %.9e 0\n', U(i, j), V(i, j));
    end
end
end

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

