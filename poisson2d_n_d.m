%% xiao yao v2025.5.28
% Solve the Helmholtz equation with mixed non-homogeneous Neumann/Dirichlet
% boundary conditions using a spectral element method (SEM).
%
% PDE: -Δu + α u = f in Ω = [0,1] x [0,1]
% Left/right (x = 0, 1):  ∂u/∂n = g_N  (Neumann)
% Bottom/top (y = 0, 1):  u = g_D      (Dirichlet)

clc
clear
addpath(genpath("D:\semMatlab")) % Ensure this path is correct

% -------------------------------------------------------------------------
% set file location
currentLocation = pwd;
cd ..
fileLocation = pwd;
currentTime = datetime('now', 'Format', 'yyyyMMdd_HHmm_ss');
pathname = fullfile(fileLocation, ['time_' char(currentTime)]);
if ~exist(pathname, 'dir')
    mkdir(pathname);
end
cd(currentLocation)
copyfile("*.m", pathname)

stage = 1;
fprintf('=== %d Program Starts ===\n', stage);
stage = stage + 1;

alpha = 1;
coeffx = pi;
coeffy = pi;

geom.Np = 4;            % polynomial degree
geom.basis = 'SEM';
geom.minx = 0;
geom.maxx = 1;
geom.miny = 0;
geom.maxy = 1;
geom.Ncellx = 20;
geom.Ncelly = 20;
geom.Npx = geom.Np;
geom.Npy = geom.Np;
geom.nx_all = geom.Ncellx * geom.Np + 1;
geom.ny_all = geom.Ncelly * geom.Np + 1;

if gpuDeviceCount("available") < 1
    geom.device = 'cpu';
else
    geom.device = 'gpu';
    geom.device_id = 1; % Choose appropriate GPU ID
end

% build separate 1-D parameter sets for Neumann (x) and Dirichlet (y)
para_n = geom;
para_n.bc = 'neumann';
para_n = parameter_bc2d(para_n);

para_d = geom;
para_d.bc = 'dirichlet';
para_d = parameter_bc2d(para_d);
freeY = para_d.freeNodesy;

% quadrature weights (GLL) aggregated across elements
wx = assemble_sem_weights(geom.Npx, geom.Ncellx, geom.minx, geom.maxx);   % nx_all x 1
wy_full = assemble_sem_weights(geom.Npy, geom.Ncelly, geom.miny, geom.maxy); % ny_all x 1
wy_free = wy_full(freeY);  % interior weights (Dirichlet boundaries removed)

fprintf('Laplacian is Q%d spectral element method \n', geom.Np)
if geom.Np < 2
    fprintf('It is also classical second order discrete Laplacian \n')
else
    fprintf('It is also a %d-th order accurate finite difference scheme \n', geom.Np + 2)
end

[dx, x, Tx, ~, lambda_x, Dmatrixx, ~] = cal_matrix2('x', para_n);
[dy, y, Ty, ~, lambda_y, Dmatrixy, ~] = cal_matrix2('y', para_d);

[coordY, coordX] = meshgrid(y, x); % size nx_all x ny_all

% manufactured exact solution (satisfies mixed BCs)
exact_sol = sin(coeffx * coordX) .* (0.5 + cos(coeffy * coordY));
duxexact_sol = coeffx * cos(coeffx * coordX) .* (0.5 + cos(coeffy * coordY));
duyexact_sol = sin(coeffx * coordX) .* (-coeffy * sin(coeffy * coordY));
lap_exact = -coeffx^2 * sin(coeffx * coordX) .* (0.5 + cos(coeffy * coordY)) ...
            -coeffy^2 * sin(coeffx * coordX) .* cos(coeffy * coordY);
f_pde = -lap_exact + alpha * exact_sol;

% lifting function for inhomogeneous Dirichlet data on y = 0,1
eta = (coordY - geom.miny) / (geom.maxy - geom.miny);
factor_bottom = 0.5 + cos(coeffy * geom.miny);
factor_top = 0.5 + cos(coeffy * geom.maxy);
lift_profile = (1 - eta) * factor_bottom + eta * factor_top;
u_lift = sin(coeffx * coordX) .* lift_profile;
dux_lift = coeffx * cos(coeffx * coordX) .* lift_profile;
lap_lift = -coeffx^2 * sin(coeffx * coordX) .* lift_profile; % d^2/dy^2 = 0

% adjusted RHS and Neumann data for v = u - u_lift (homogeneous Dirichlet)
f_v_full = f_pde - (-lap_lift + alpha * u_lift);
dv_dx_exact = duxexact_sol - dux_lift;

Fg = zeros(geom.nx_all, numel(freeY));
Fg(1, :) = Fg(1, :) - dv_dx_exact(1, freeY) .* wy_free.';
Fg(end, :) = Fg(end, :) + dv_dx_exact(end, freeY) .* wy_free.';

mass_diag = wx * wy_free.';
f_solver = f_v_full(:, freeY) + Fg ./ mass_diag;

invTx = pinv(Tx);
invTy = pinv(Ty);

switch geom.device
    case 'gpu'
        fprintf('GPU computation: starting to load matrices/data \n')
        Device = gpuDevice(geom.device_id);
        Tx = gpuArray(Tx);
        Ty = gpuArray(Ty);
        invTx = gpuArray(invTx);
        invTy = gpuArray(invTy);
        lambda_x = gpuArray(lambda_x);
        lambda_y = gpuArray(lambda_y);
        f_solver = gpuArray(f_solver);
end



Lambda2D = bsxfun(@plus, lambda_x, lambda_y');
tic;
v_hat = invTx * f_solver * invTy';
v_hat = v_hat ./ (Lambda2D + alpha);
v_free = Tx * v_hat * Ty';
switch geom.device
    case 'gpu'
        wait(Device);
        v_free = gather(v_free);
end
time = toc;

v_all = zeros(geom.nx_all, geom.ny_all);
v_all(:, freeY) = v_free;
u_all = v_all + u_lift;

err = u_all - exact_sol;
l2_err = norm(err, 'fro') * sqrt(dx * dy);
fprintf('The ell-2 norm error is %.6e \n', l2_err)
l_infty_err = norm(err(:), inf);
fprintf('The ell-infty norm error is %.6e \n', l_infty_err)

switch geom.device
    case 'gpu'
        fprintf('The GPU time of solver is %.3f s\n', time)
    case 'cpu'
        fprintf('The CPU time of solver is %.3f s\n', time)
end

fprintf('\n')

% gradients and error analysis
uxx = Dmatrixx * u_all;
uyy = u_all * Dmatrixy';
err_uxx = uxx - duxexact_sol;
err_uyy = uyy - duyexact_sol;
l2_err_uxx = norm(err_uxx, 'fro') * sqrt(dx * dy);
fprintf('The ell-2 norm error of uxx is %.6e \n', l2_err_uxx)
l_infty_err_uxx = norm(err_uxx(:), inf);
fprintf('The ell-infty norm error of uxx is %.6e \n', l_infty_err_uxx)
l2_err_uyy = norm(err_uyy, 'fro') * sqrt(dx * dy);
fprintf('The ell-2 norm error of uyy is %.6e \n', l2_err_uyy)
l_infty_err_uyy = norm(err_uyy(:), inf);
fprintf('The ell-infty norm error of uyy is %.6e \n', l_infty_err_uyy)

% export data to Tecplot
varName = 'u,uexact,uxx,uexactxx,uyy,uexactyy\n';
OUTPUT_Tecplot2D4(2, pathname, geom.nx_all, geom.ny_all, varName, ...
    coordX(:), coordY(:), u_all(:), exact_sol(:), uxx(:), duxexact_sol(:), uyy(:), duyexact_sol(:));

fprintf('done \n');
