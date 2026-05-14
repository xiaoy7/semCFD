function sem_write_ibm_vtk2d(vtk_dir, nx, ny, dump_id, time, pointst, ...
    U, V, P, Omega, force_u, force_v, ibm)
%SEM_WRITE_IBM_VTK2D Write IB2d-style VTK output for semCFD.

if ~exist(vtk_dir, 'dir')
    mkdir(vtk_dir);
end

tag = sprintf('%04d', dump_id);
u_mag = sqrt(U.^2 + V.^2);
f_mag = sqrt(force_u.^2 + force_v.^2);

write_structured_vector(fullfile(vtk_dir, ['u.' tag '.vtk']), ...
   nx, ny, 'u', pointst, U, V, time);
write_structured_scalar(fullfile(vtk_dir, ['P.' tag '.vtk']), ...
    nx, ny, 'P', pointst, P, time);
write_structured_scalar(fullfile(vtk_dir, ['Omega.' tag '.vtk']), ...
    nx, ny, 'Omega', pointst, Omega, time);
write_structured_scalar(fullfile(vtk_dir, ['uMag.' tag '.vtk']), ...
    nx, ny, 'uMag', pointst, u_mag, time);
write_structured_scalar(fullfile(vtk_dir, ['fX.' tag '.vtk']), ...
    nx, ny, 'fX', pointst, force_u, time);
write_structured_scalar(fullfile(vtk_dir, ['fY.' tag '.vtk']), ...
     nx, ny,'fY', pointst, force_v, time);
write_structured_scalar(fullfile(vtk_dir, ['fMag.' tag '.vtk']), ...
    nx, ny, 'fMag', pointst, f_mag, time);

if isfield(ibm, 'markers') && ~isempty(ibm.markers)
    write_lagrangian_points(fullfile(vtk_dir, ['lagsPts.' tag '.vtk']), ...
        ibm, time, false);
    write_lagrangian_points(fullfile(vtk_dir, ['lagPtsConnect.' tag '.vtk']), ...
        ibm, time, true);
end

end


