%% xiao yao v2025.5.28
% Solve the 3D Helmholtz equation with mixed non-homogeneous Neumann /
% Dirichlet boundary conditions using a spectral element method (SEM).
%
% PDE: -Δu + α u = f   in Ω = [0,1]^3
% Neumann faces:   x = 0,1   (∂u/∂n = g_N)
% Dirichlet faces: y = 0,1 and z = 0,1 (u = g_D)

clc
clear
addpath(genpath("D:\semMatlab"))

% -------------------------------------------------------------------------
% bookkeeping
currentLocation = pwd;
cd ..
fileLocation = pwd;
currentTime = datetime('now', 'Format', 'yyyyMMdd_HHmm_ss');
outDir = fullfile(fileLocation, ['time_' char(currentTime)]);
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
cd(currentLocation)
copyfile("*.m", outDir)

% -------------------------------------------------------------------------
% problem parameters
alpha = 1;
coeffx = pi;
coeffy = pi;
coeffz = 2 * pi;

geom.Np = 4;
geom.basis = 'SEM';
geom.minx = 0;  geom.maxx = 1;
geom.miny = 0;  geom.maxy = 2;
geom.minz = 0;  geom.maxz = 3;
geom.Ncellx = 10;
geom.Ncelly = 20;
geom.Ncellz = 30;
geom.Npx = geom.Np;
geom.Npy = geom.Np;
geom.Npz = geom.Np;
geom.nx_all = geom.Ncellx * geom.Npx + 1;
geom.ny_all = geom.Ncelly * geom.Npy + 1;
geom.nz_all = geom.Ncellz * geom.Npz + 1;

if gpuDeviceCount("available") < 1
    geom.device = 'cpu';
else
    geom.device = 'gpu';
    geom.device_id = 1;
end

% axis-specific parameters
paramX = geom;
paramX.bc = 'neumann';
paramX.Ncellx = geom.Ncellx;
paramX.Ncelly = geom.Ncelly;
paramX.Ncellz = geom.Ncellz;
paramX.Npx = geom.Npx;
paramX.Npy = geom.Npy;
paramX.Npz = geom.Npz;
paramX.nx = geom.Ncellx * geom.Npx + 1;
paramX.nx_all = geom.nx_all;
paramX.freeNodesx = 1:geom.nx_all;

paramY = geom;
paramY.bc = 'dirichlet';
paramY.Ncellx = geom.Ncellx;
paramY.Ncelly = geom.Ncelly;
paramY.Ncellz = geom.Ncellz;
paramY.Npx = geom.Npx;
paramY.Npy = geom.Npy;
paramY.Npz = geom.Npz;
paramY.ny = geom.Ncelly * geom.Npy - 1;
paramY.ny_all = geom.ny_all;
paramY.freeNodesy = 2:geom.ny_all - 1;
paramY.bcNodesy = [1, geom.ny_all];

paramZ = geom;
paramZ.bc = 'dirichlet';
paramZ.Ncellx = geom.Ncellx;
paramZ.Ncelly = geom.Ncelly;
paramZ.Ncellz = geom.Ncellz;
paramZ.Npx = geom.Npx;
paramZ.Npy = geom.Npy;
paramZ.Npz = geom.Npz;
paramZ.nz = geom.Ncellz * geom.Npz - 1;
paramZ.nz_all = geom.nz_all;
paramZ.freeNodesz = 2:geom.nz_all - 1;
paramZ.bcNodesz = [1, geom.nz_all];

freeY = paramY.freeNodesy;
freeZ = paramZ.freeNodesz;
ny_free = numel(freeY);
nz_free = numel(freeZ);

fprintf('Laplacian is Q%d spectral element method\n', geom.Np);

% one-dimensional SEM operators
[dx, x, Tx, ~, lambda_x, Dmatrixx, ~] = cal_matrix2('x', paramX);
[dy, y, Ty, ~, lambda_y, Dmatrixy, ~] = cal_matrix2('y', paramY);
[dz, z, Tz, ~, lambda_z, Dmatrixz, ~] = cal_matrix2('z', paramZ);
Dmatrixx = full(Dmatrixx);
Dmatrixy = full(Dmatrixy);
Dmatrixz = full(Dmatrixz);

% quadrature weights (GLL, accumulated over elements)
wx = assemble_sem_weights(geom.Npx, geom.Ncellx, geom.minx, geom.maxx);
wy_full = assemble_sem_weights(geom.Npy, geom.Ncelly, geom.miny, geom.maxy);
wz_full = assemble_sem_weights(geom.Npz, geom.Ncellz, geom.minz, geom.maxz);
wy_free = wy_full(freeY);
wz_free = wz_full(freeZ);

% tensor grid
[X, Y, Z] = ndgrid(x, y, z);

% manufactured exact solution
exact_sol = sin(coeffx * X) .*  cos(coeffy * Y) .*  cos(coeffz * Z);
duxexact_sol = coeffx * cos(coeffx * X) .*  cos(coeffy * Y) .*  cos(coeffz * Z);
duyexact_sol = sin(coeffx * X) .* (-coeffy * sin(coeffy * Y)) .*  cos(coeffz * Z);
duzexact_sol = sin(coeffx * X) .*  cos(coeffy * Y) .* (-coeffz * sin(coeffz * Z));
d2ux = -coeffx^2 * sin(coeffx * X) .*  cos(coeffy * Y) .*  cos(coeffz * Z);
d2uy = -coeffy^2 * sin(coeffx * X) .* cos(coeffy * Y) .*  cos(coeffz * Z);
d2uz = -coeffz^2 * sin(coeffx * X) .*  cos(coeffy * Y) .* cos(coeffz * Z);
lap_exact = d2ux + d2uy + d2uz;
f_pde = -lap_exact + alpha * exact_sol;

% Improved lifting: use the exact y and z dependence so interior v should be ~0
amp_y =  cos(coeffy * Y);          % exact y factor
amp_z =  cos(coeffz * Z);          % exact z factor
u_lift = sin(coeffx * X) .* amp_y .* amp_z;  % equals exact_sol
dux_lift = coeffx * cos(coeffx * X) .* amp_y .* amp_z;
% Laplacian of u_lift (use product rule with exact amplitudes)
amp_y_yy = -coeffy^2 * cos(coeffy * Y); % second derivative of  cos)
amp_z_zz = -coeffz^2 * cos(coeffz * Z); % second derivative of  cos)
lap_lift = (-coeffx^2) * sin(coeffx * X) .* amp_y .* amp_z + ...
    sin(coeffx * X) .* amp_y_yy .* amp_z + ...
    sin(coeffx * X) .* amp_y .* amp_z_zz;

% reduced problem RHS for v = u - u_lift
f_v_full = f_pde - (-lap_lift + alpha * u_lift); % should be ~0
dv_dx_exact = duxexact_sol - dux_lift;           % should be ~0 on Neumann faces

% assemble Neumann boundary contribution (x-faces) for v = u - u_lift.
% Two candidate scalings: with wx factor (weak-form mass weighting) and without (strong-form collocation).
side_weights = reshape(wy_free, [1, ny_free, 1]) .* reshape(wz_free, [1, 1, nz_free]);
Fg_no = zeros(geom.nx_all, ny_free, nz_free);
Fg_no(1, :, :)  = - dv_dx_exact(1, freeY, freeZ)  .* side_weights;
Fg_no(end, :, :) = dv_dx_exact(end, freeY, freeZ) .* side_weights;

mass_diag = reshape(wx, [geom.nx_all, 1, 1]) .* ...
    reshape(wy_free, [1, ny_free, 1]) .* ...
    reshape(wz_free, [1, 1, nz_free]);
f_solver = f_v_full(:, freeY, freeZ) + Fg_no ./ mass_diag;


invTx = pinv(Tx);
invTy = pinv(Ty);
invTz = pinv(Tz);

useGPU = strcmp(geom.device, 'gpu');
if useGPU
    fprintf('GPU detected; transferring data but performing solve on CPU for stability\n')
    Device = gpuDevice(geom.device_id);
    f_solver_gpu = gpuArray(f_solver);
    u_lift_gpu = gpuArray(u_lift);
    exact_gpu = gpuArray(exact_sol);
    dux_gpu = gpuArray(duxexact_sol);
    duy_gpu = gpuArray(duyexact_sol);
    duz_gpu = gpuArray(duzexact_sol);
    wait(Device);
    f_solver_cpu = gather(f_solver_gpu);
    u_lift = gather(u_lift_gpu);
    exact_sol = gather(exact_gpu);
    duxexact_sol = gather(dux_gpu);
    duyexact_sol = gather(duy_gpu);
    duzexact_sol = gather(duz_gpu);
else
    f_solver_cpu = f_solver;
end

Lambda3D = reshape(lambda_x, [geom.nx_all, 1, 1]) + ...
    reshape(lambda_y, [1, ny_free, 1]) + ...
    reshape(lambda_z, [1, 1, nz_free]);

tic;
% Ensure inverse transform matrices correspond to interior (free) nodes for y,z
ny_free = size(f_solver_cpu,2); nz_free = size(f_solver_cpu,3);

% Forward transforms (physical -> modal) along y, z, then x
rhs_modal = f_solver_cpu; % size: nx_all x ny_free x nz_free
% y-transform
tmp = permute(rhs_modal,[1 3 2]); % x,z,y
tmp = reshape(tmp, geom.nx_all * nz_free, ny_free); % (x*z) x y
tmp = tmp * invTy'; % apply invTy' on y dim
tmp = reshape(tmp, geom.nx_all, nz_free, ny_free);
rhs_modal = permute(tmp,[1 3 2]); % back to x,y,z
% z-transform
tmp = reshape(rhs_modal, geom.nx_all * ny_free, nz_free); % (x*y) x z
tmp = tmp * invTz';
rhs_modal = reshape(tmp, geom.nx_all, ny_free, nz_free);
% x-transform
tmp = reshape(rhs_modal, geom.nx_all, ny_free * nz_free); % x x (y*z)
tmp = invTx * tmp;
rhs_modal = reshape(tmp, geom.nx_all, ny_free, nz_free);

% Divide by eigenvalues
rhs_modal = rhs_modal ./ (Lambda3D + alpha);

% Inverse transforms (modal -> physical) along x, z, then y
v_free = rhs_modal;
% inverse x
tmp = reshape(v_free, geom.nx_all, ny_free * nz_free);
tmp = Tx * tmp;
v_free = reshape(tmp, geom.nx_all, ny_free, nz_free);
% inverse z
tmp = reshape(v_free, geom.nx_all * ny_free, nz_free);
tmp = tmp * Tz';
v_free = reshape(tmp, geom.nx_all, ny_free, nz_free);
% inverse y
tmp = permute(v_free,[1 3 2]); % x,z,y
tmp = reshape(tmp, geom.nx_all * nz_free, ny_free);
tmp = tmp * Ty';
tmp = reshape(tmp, geom.nx_all, nz_free, ny_free);
v_free = permute(tmp,[1 3 2]); % x,y,z

time = toc;

v_all = zeros(geom.nx_all, geom.ny_all, geom.nz_all);
v_all(:, freeY, freeZ) = v_free;
u_all = v_all + u_lift;

err = u_all - exact_sol;
l2_err = norm(err(:)) * sqrt(dx * dy * dz);
l_inf_err = norm(err(:), inf);
fprintf('The ell-2 norm error is %.6e\n', l2_err);
fprintf('The ell-infty norm error is %.6e\n', l_inf_err);
vol_weight = sum(wx) * sum(wy_full) * sum(wz_full);
wl2_err = sqrt( sum( (err(:)).^2 ) * dx * dy * dz );
fprintf('Weighted L2 (physical measure) error: %.6e\n', wl2_err);

if useGPU
    fprintf('The CPU time of solver (with GPU staging) is %.3f s\n', time);
else
    fprintf('The CPU time of solver is %.3f s\n', time);
end

[ux_num, uy_num, uz_num] = sem_tensor_ops.grad(u_all, {Dmatrixx, Dmatrixy, Dmatrixz});
err_ux = ux_num - duxexact_sol;
err_uy = uy_num - duyexact_sol;
err_uz = uz_num - duzexact_sol;
fprintf('The ell-2 norm error of ux is %.6e\n', norm(err_ux(:)) * sqrt(dx * dy * dz));
fprintf('The ell-2 norm error of uy is %.6e\n', norm(err_uy(:)) * sqrt(dx * dy * dz));
fprintf('The ell-2 norm error of uz is %.6e\n', norm(err_uz(:)) * sqrt(dx * dy * dz));
fprintf('The ell-inf norm error of ux is %.6e\n', norm(err_ux(:), inf));
fprintf('The ell-inf norm error of uy is %.6e\n', norm(err_uy(:), inf));
fprintf('The ell-inf norm error of uz is %.6e\n', norm(err_uz(:), inf));

OUTPUT_Tecplot3d(1, outDir, [X(:), Y(:), Z(:)], ...
    geom.nx_all, geom.ny_all, geom.nz_all, ...
    'u,uexact,ux,uy,uz', u_all(:), exact_sol(:), ux_num(:), uy_num(:), uz_num(:));

fprintf('done\n');

