%% xiao yao v2025.5.28
% Solve the Poisson/Helmholtz equation with non-homogeneous Neumann
% boundary conditions using the spectral element method (SEM).
% PDE: -Delta u + alpha * u = f  in Omega
% du/dn = g on the left and right side (neumann boundary)
% u = 0 on the top and right side (dirichlet boundary)
% Domain: Omega = [0,1] x [0,1]

clc
clear
addpath(genpath("D:\semMatlab")) % Ensure this path is correct

% set file location -----------------------------------------
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

para.Np = 4; % polynomial degree in each direction
para.basis = 'SEM'; % 'SEM' or 'FFT'
para.bc = 'neumann';
para.minx = 0;
para.maxx = 0.5;
para.miny = 0;
para.maxy = 0.5;
para.Ncellx = 20;
para.Ncelly = 20;
para.Npx = para.Np;
para.Npy = para.Np;
para.nx_all = para.Ncellx * para.Np + 1;
para.ny_all = para.Ncelly * para.Np + 1;

if gpuDeviceCount("available") < 1
    para.device = 'cpu';
else
    para.device = 'gpu';
    para.device_id = 1; % Choose appropriate GPU ID
end

para = parameter_bc2d(para);
wx = assemble_sem_weights(para.Npx, para.Ncellx, para.minx, para.maxx);
wy = assemble_sem_weights(para.Npy, para.Ncelly, para.miny, para.maxy);

fprintf('Laplacian is Q%d spectral element method \n', para.Np)
if para.Np < 2
    fprintf('It is also classical second order discrete Laplacian \n')
else
    fprintf('It is also a %d-th order accurate finite difference scheme \n', para.Np + 2)
end

[dx, x, Tx, ~, lambda_x, Dmatrixx, ex] = cal_matrix2('x', para);
[dy, y, Ty, ~, lambda_y, Dmatrixy, ey] = cal_matrix2('y', para);

[coordY, coordX] = meshgrid(y, x);

% exact values
u1x = sin(coeffx * x);
u1y = sin(coeffy * y);
exact_sol = squeeze(u1x * u1y');

% derivative of exact values
du1x = coeffx * cos(coeffx * x);
du1y = coeffy * cos(coeffy * y);
duxexact_sol = squeeze(du1x * u1y');
duyexact_sol = squeeze(u1x * du1y');

f_pde = (alpha + coeffx^2 + coeffy^2) * exact_sol;

% Construct F_g for non-homogeneous Neumann boundary conditions
Fx = zeros(para.nx_all, para.ny_all);
Fy = zeros(para.nx_all, para.ny_all);
Fx(1, :) = -duxexact_sol(1, :) .* wy';
Fx(end, :) = duxexact_sol(end, :) .* wy';
Fy(:, 1) = -duyexact_sol(:, 1) .* wx;
Fy(:, end) = duyexact_sol(:, end) .* wx;
Fg = Fx + Fy;


mass_diag = wx * wy';
f_solver = f_pde + Fg ./ mass_diag;

invTx = pinv(Tx);
invTy = pinv(Ty);

switch para.device
    case 'gpu'
        fprintf('GPU computation: starting to load matrices/data \n')
        Device = gpuDevice(para.device_id);
        Tx = gpuArray(Tx);
        Ty = gpuArray(Ty);
        invTx = gpuArray(invTx);
        invTy = gpuArray(invTy);
        lambda_x = gpuArray(lambda_x);
        lambda_y = gpuArray(lambda_y);
        f_solver = gpuArray(f_solver);
        Lambda2D = bsxfun(@plus, lambda_x, lambda_y');
    otherwise
        Lambda2D = bsxfun(@plus, lambda_x, lambda_y');
end

tic;
u_hat = invTx * f_solver * invTy';
u_hat = u_hat ./ (Lambda2D + alpha);
u = Tx * u_hat * Ty';
switch para.device
    case 'gpu'
        wait(Device);
        u = gather(u);
    otherwise
        % already on CPU
end
time = toc;

u_all = u;
err = u_all - exact_sol;
l2_err = norm(err, 'fro') * sqrt(dx * dy);
fprintf('The ell-2 norm error is %.6e \n', l2_err)
l_infty_err = norm(err(:), inf);
fprintf('The ell-infty norm error is %.6e \n', l_infty_err)

switch para.device
    case 'gpu'
        fprintf('The GPU time of solver is %.3f s\n', time)
    case 'cpu'
        fprintf('The CPU time of solver is %.3f s\n', time)
end

fprintf('\n')

% calculate the derivative
uxx = Dmatrixx * u_all;
uyy = u_all * Dmatrixy';

% error analysis
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
OUTPUT_Tecplot2D4(2, pathname, para.nx_all, para.ny_all, varName, ...
    coordX(:), coordY(:), u_all(:), exact_sol(:), uxx(:), duxexact_sol(:), uyy(:), duyexact_sol(:));

fprintf('done \n');

