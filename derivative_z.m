function uz = derivative_z(Dmatrixz,uvw,nx_all,ny_all,nz_all)
% Vectorized derivative along z to avoid nested loops
uvw_cols = reshape(permute(uvw, [3 1 2]), nz_all, []);   % each column is uvw(i,j,:)
uz_cols  = Dmatrixz * uvw_cols;
uz = permute(reshape(uz_cols, nz_all, nx_all, ny_all), [2 3 1]);
