function write_structured_intro(fid, nx, ny, title, pointst, time)
write_header(fid, title, time);
fprintf(fid, 'DATASET STRUCTURED_GRID\n');
fprintf(fid, 'DIMENSIONS %d %d 1\n', nx, ny);
fprintf(fid, 'POINTS %d float\n', nx * ny);
% for j = 1:ny
%     for i = 1:nx
%         fprintf(fid, '%.9e %.9e 0\n', coordX(i, j), coordY(i, j));
%     end
% end
% export the points by using matrix format, which is much faster
% points = [reshape(coordX, [], 1), reshape(coordY, [], 1), zeros(nx * ny, 1)];
fprintf(fid, '%.9e %.9e %.9e\n', pointst);
fprintf(fid, '\nPOINT_DATA %d\n', nx * ny);
end

