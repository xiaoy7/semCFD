%% xiao yao v2025.5.28
% solve the poisson equation with non-homogenous Neumann boundary


clc
clear
addpath(genpath("D:\semMatlab")) % Ensure this path is correct


% set file location -----------------------------------------
currentLocation = pwd;
cd ..
fileLocation = pwd;
currentTime = datetime('now', 'Format', 'yyyyMMdd_HHmm_ss');% 获取当前日期和时间
pathname = [fileLocation, '\time' char(currentTime)];
cd(currentLocation)
copyfile("*.m", pathname)



stage = 1;
fprintf('=== %d Program Starts ===\n', stage); %[output:8139b378]
stage = stage + 1;
%%
% set dirichlets

frePrint = 10;
freOut = frePrint*10;
freMat = freOut * 20;


delta = 1e-3;    % Time step
steps = 2000; % Max number of steps (increased for potential convergence)
tol = 1e-4;   % Convergence tolerance (tightened slightly)
% alpha_helmholtz = 1 / delta;  % Coefficient for Helmholtz equation (implicit time term)
varName = 'U,V,PRE,phi,psi\n'; % export data
LL1 = 1; % the characteristic length


Cx = 0.5;     % Center of circle in x direction
Cy = 0.5;       % Center of circle in y direction
radius = 0.2; % Radius of circle

Cn = 0.02;
eta = Cn * LL1;  % Smoothing dirichlet for tanh

coeffx = pi;
coeffy = pi; % 2 *
alpha = 1;
dirichlet.Np = 4; % polynomial degree


dirichlet.basis = 'SEM'; %'FFT' 'SEM'
if dirichlet.basis == 'FFT'
    dirichlet.length = 1; % Assuming square domain if FFT
    dirichlet.minx = -dirichlet.length;
    dirichlet.maxx = dirichlet.length;
    dirichlet.miny = -dirichlet.length;
    dirichlet.maxy = dirichlet.length;
    dirichlet.Ncellx = 1;
    dirichlet.Ncelly = 1; % FFT implies single domain usually
else
    dirichlet.minx = 0;
    dirichlet.maxx = 1;
    dirichlet.miny = 0;
    dirichlet.maxy = 1; % Square domain

    % dirichlet.Ncell = 10; % number of cells in finite element (default)
    % number of cells in finite element
    dirichlet.Ncellx = 20;
    dirichlet.Ncelly = 20;

end


if gpuDeviceCount('available') < 1
    dirichlet.device = 'cpu';
else
    dirichlet.device = 'gpu';
    dirichlet.device_id = 1; % Choose appropriate GPU ID
end


% polynomial degree
dirichlet.Npx = dirichlet.Np;
dirichlet.Npy = dirichlet.Np;
dirichlet.nx_all = dirichlet.Ncellx * dirichlet.Np + 1;
dirichlet.ny_all = dirichlet.Ncelly * dirichlet.Np + 1;


dirichlet.bc = 'dirichlet'; % Boundary condition type for velocity
neumann = dirichlet; % Use same domain/discretization dirichlets initially
neumann.bc = 'neumann'; % Boundary condition type for pressure correction

% Calculate free/boundary nodes based on BC type
dirichlet = parameter_bc2d(dirichlet);
neumann = parameter_bc2d(neumann); % Pressure uses Neumann


% the domain is [Lminx, Lmaxx] x [Lminy, Lmaxy]
%%
% Qk finite element with (k+1)-point Gauss-Lobatto quadrature
fprintf('Laplacian is Q%d spectral element method \n', dirichlet.Np) %[output:3f84157a]

if (dirichlet.Np < 2) %[output:group:13fba25a]
    fprintf('It is also classical second order discrete Laplacian \n')
else
    fprintf('It is also a %d-th order accurate finite difference scheme \n', dirichlet.Np + 2) %[output:345ecd30]
end %[output:group:13fba25a]

% Get matrices for Velocity (Dirichlet BCs applied internally by cal_matrix2 for INTERIOR solve)
[dx, x, Tx, ~, lambda_xd, Dmatrixx, ex] = cal_matrix2('x', dirichlet); % Hx not needed directly
[dy, y, Ty, ~, lambda_yd, Dmatrixy, ey] = cal_matrix2('y', dirichlet); % Hy not needed directly
DmatrixyT = Dmatrixy'; % Transpose for convenience

% Get matrices for Pressure (Neumann BCs - use cal_matrix2_1 or similar)
% Ensure cal_matrix2_1 correctly handles Neumann for pressure
[dxn, exn, Txn, lambda_xn] = cal_matrix2_1('x', neumann,x);
[dyn, eyn, Tyn, lambda_yn] = cal_matrix2_1('y', neumann,y);

[X, Y] = meshgrid(y, x);
% coordx = repmat(x, ny * nz, 1);
% coordy1 = repmat(y', nx, nz);
% coordy = coordy1(:);
% coordz1 = repmat(z, 1, nx * ny)';
% coordz = coordz1(:);
coord(:, 1) = X(:);
coord(:, 2) = Y(:);

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
% F_g represents the boundary integral term integral(g * phi_i dGamma)
% where g is du/dn.
% ex and ey are 1D quadrature weights from cal_matrix2.
Fg = zeros(dirichlet.nx_all, dirichlet.ny_all);

% Contributions from x-boundaries (du/dx = g_x specified)
% Left boundary: x = x(1) (index 1). Normal n=(-1,0). du/dn = -du/dx.
Fg(1, :) = Fg(1, :) - duxexact_sol(1, :) .* eyn'; % ey is ny_all x 1, so ey' is 1 x ny_all
% Right boundary: x = x(end) (index nx_all). Normal n=(1,0). du/dn = du/dx.
Fg(dirichlet.nx_all, :) = Fg(dirichlet.nx_all, :) + duxexact_sol(dirichlet.nx_all, :) .* eyn';

% Contributions from y-boundaries (du/dy = g_y specified)
% Bottom boundary: y = y(1) (index 1). Normal n=(0,-1). du/dn = -du/dy.
Fg(:, 1) = Fg(:, 1) - duyexact_sol(:, 1) .* exn; % ex is nx_all x 1
% Top boundary: y = y(end) (index ny_all). Normal n=(0,1). du/dn = du/dy.
Fg(:, dirichlet.ny_all) = Fg(:, dirichlet.ny_all) + duyexact_sol(:, dirichlet.ny_all) .* exn;

% Corner contributions are naturally included by summing contributions from intersecting edges.

% Modified source term for the solver: f_solver = f_pde + M_inv * Fg
% M_inv_diag = 1 ./ (ex * ey');
f_solver = f_pde + Fg ./ (exn * eyn'); % ex and ey are CPU arrays of weights here.

u_all = exact_sol*0;
% The following line for setting Dirichlet BCs has no effect for pure Neumann
% as bcNodesx/bcNodesy are empty. It can be kept or removed.
u_all(dirichlet.bcNodesx,dirichlet.bcNodesy) = exact_sol(dirichlet.bcNodesx,dirichlet.bcNodesy);
% generate f as a 2D tensor [nx, ny] --- This comment is now outdated by f_solver
% f = (alpha + 2 * pi*pi) * exact_sol; % This was the old f_pde calculation, now handled above
invTx = pinv(Tx);
invTy = pinv(Ty);
% to save offline time, we first load the 1D matrix into GPU
% then generate the 2D tensor for computation
switch dirichlet.device
    case 'gpu'
        fprintf('GPU computation: starting to load matrices/data \\n')
        Device = gpuDevice(dirichlet.device_id);
        Tx = gpuArray(Tx);
        Ty = gpuArray(Ty);
        lambda_x_gpu = gpuArray(lambda_xn); % Renaming for clarity if lambda_x is used later on CPU
        lambda_y_gpu = gpuArray(lambda_yn); % Renaming for clarity
        % ex and ey (weights) were used on CPU to calculate f_solver.
        % If needed on GPU for other purposes, they can be transferred.
        % For now, only f_solver needs to be transferred.
        f_input_for_solver_gpu = gpuArray(f_solver(dirichlet.freeNodesx,dirichlet.freeNodesy));
        % since T_i is not unitary
        invTx = gpuArray(invTx);
        invTy = gpuArray(invTy);
    case 'cpu'
        lambda_x_gpu = lambda_xn; % Keep consistent naming for Lambda2D calculation
        lambda_y_gpu = lambda_yn;
        f_input_for_solver_gpu = f_solver(dirichlet.freeNodesx,dirichlet.freeNodesy);
end

% generate eigenvalue tensor
Lambda2D = squeeze(tensorprod(lambda_x_gpu, eyn)) + squeeze(tensorprod(exn, lambda_y_gpu)); % Incorrect
% Lambda2D_corrected = bsxfun(@plus, lambda_x_gpu, lambda_y_gpu'); % lambda_x_gpu is nx x 1, lambda_y_gpu' is 1 x ny

switch dirichlet.device
    case 'gpu'
        wait(Device);
        fprintf('GPU computation: loading finished and GPU computing started \n')
end

%% online computation
tic;


%% Eigenvector method to solve the linear system by tensor matrix multiplication
u = pagemtimes(f_input_for_solver_gpu, invTy'); % Use the modified f_input_for_solver_gpu
u = squeeze(tensorprod(invTx, u, 2, 1));
u = u ./ (Lambda2D + alpha); % Use the corrected Lambda2D_corrected
u = pagemtimes(u, Ty');
u = squeeze(tensorprod(Tx, u, 2, 1));


switch dirichlet.device
    case 'gpu'
        wait(Device);
end

time = toc;

u_all(dirichlet.freeNodesx,dirichlet.freeNodesy) = u;
err = u_all - exact_sol;
l2_err = norm(err, 'fro') * sqrt(dx * dy);
fprintf('The ell-2 norm error is %d \n', l2_err)
l_infty_err = norm(err(:), inf);
fprintf('The ell-infty norm error is %d \n', l_infty_err)



switch dirichlet.device
    case 'gpu'
        fprintf('The GPU time of solver is %d \n', time)
    case 'cpu'
        fprintf('The CPU time of solver is %d \n', time)
end

fprintf('\n')



% calculate the derivative
uxx = Dmatrixx*u_all;
uyy = u_all*Dmatrixy';

% error analysis
err_uxx = uxx - duxexact_sol;
err_uyy = uyy - duyexact_sol;
l2_err_uxx = norm(err_uxx, 'fro') * sqrt(dx * dy);
fprintf('The ell-2 norm error of uxx is %d \n', l2_err_uxx)

l_infty_err_uxx = norm(err_uxx(:), inf);
fprintf('The ell-infty norm error of uxx is %d \n', l_infty_err_uxx)
l2_err_uyy = norm(err_uyy, 'fro') * sqrt(dx * dy);
fprintf('The ell-2 norm error of uyy is %d \n', l2_err_uyy)
l_infty_err_uyy = norm(err_uyy(:), inf);
fprintf('The ell-infty norm error of uyy is %d \n', l_infty_err_uyy)

% export dat to tecplot
OUTPUT_Tecplot2D(2, coord, nx_all, ny_all, 'u,uexact,uxx,uexactxx,uyy,uexactyy\n',...
    u_all(:), exact_sol(:),uxx(:),duxexact_sol(:),uyy(:),duyexact_sol(:));
fprintf('done \n');