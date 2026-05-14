function write_header(fid, title, time)
fprintf(fid, '# vtk DataFile Version 2.0\n');
fprintf(fid, '%s time=%g\n', title, time);
fprintf(fid, 'ASCII\n');
end
