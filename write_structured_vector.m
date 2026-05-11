function write_structured_vector(filename, nx, ny, name, pointst, U, V, time)
fid = fopen(filename, 'w');
assert(fid > 0, 'Could not open VTK file: %s', filename);
cleanup = onCleanup(@() fclose(fid));
write_structured_intro(fid, nx, ny, name, pointst, time);
fprintf(fid, 'VECTORS %s float\n', name);
% for j = 1:ny
%     for i = 1:nx
%         fprintf(fid, '%.9e %.9e 0\n', U(i, j), V(i, j));
%     end
% end
% export the data by using matrix format, which is much faster
U_vec = reshape(U', [], 1);
V_vec = reshape(V', [], 1); 
zero_vec = zeros(size(U_vec));
data = [U_vec, V_vec, zero_vec]';
fprintf(fid, '%.9e %.9e %.9e\n', data);
end

