function [l2_err, l_infty_err] = Poisson_SEM_dirichlet(Parameter)
%% 3D Poisson Equation on CPU/GPU
% FUNCTION NAME:
%   Poisson_SEM_dirichlet
%
% DESCRIPTION:
%   Use spectral element method (SEM) to solve the Poisson equation
%   \alpha*u - u_xx -u_yy - u_zz = f
%   in domain [-Lx,Lx]*[-Ly,Ly]*[-Lz,Lz] with homogeneous Dirichlet B.C..
%
% Syntax:
%   u - (tensor) u is a 3D tensor with u(i,j,k) denoting u(x_i,y_j,z_k)
%
% ASSUMPTIONS AND LIMITATIONS:
%  (k+2)th order difference (Q^k Spectral Element), k>=2
%  if k=1, it is second order centered difference
%
% REVISION HISTORY:
%   08/02/2023 - Xiangxiong Zhang & Xinyu Liu, Purdue University
%       * Initial implementation
%

num_repeat = Parameter.num_repeat;
plotfigure = Parameter.plotfigure;
Np = Parameter.Np;
Ncell = Parameter.Ncell;

% polynomial degree
Npx = Np;
Npy = Np;
Npz = Np;

% number of cells in finite element
Ncellx = Ncell;
Ncelly = Ncell;
Ncellz = Ncell;

switch Parameter.bc
    case 'dirichlet'
        nx = Ncellx * Npx - 1;
        ny = Ncelly * Npy - 1;
        nz = Ncellz * Npz - 1;
        % total number of unknowns in one direction
    case 'periodic'
        nx = Ncellx * Npx;
        ny = Ncelly * Npy;
        nz = Ncellz * Npz;
        % total number of unknowns in one direction
    case 'neumann'
        nx = Ncellx * Npx + 1;
        ny = Ncelly * Npy + 1;
        nz = Ncellz * Npz + 1;
        % total number of unknowns in one direction
end

Parameter.Npx = Npx;
Parameter.Npy = Npy;
Parameter.Npz = Npz;
Parameter.Ncellx = Ncellx;
Parameter.Ncelly = Ncelly;
Parameter.Ncellz = Ncellz;
Parameter.nx = nx;
Parameter.ny = ny;
Parameter.nz = nz;
coeffx = pi;
coeffy = 2 * pi;
coeffz = 3 * pi;
alpha = 1;


Parameter.minx = 0;
Parameter.maxx = 1;
Parameter.miny = 0;
Parameter.maxy = 1;
Parameter.minz = 0;
Parameter.maxz = 1;


%% Qk finite element with (k+1)-point Gauss-Lobatto quadrature
fprintf('This is a code solving 3D Poison on a grid of size %d by %d by %d \n', nx, ny, nz)

fprintf('Laplacian is Q%d spectral element method \n', Np)

if (Np < 2)
    fprintf('It is also classical second order discrete Laplacian \n')
else
    fprintf('It is also a %d-th order accurate finite difference scheme \n', Np + 2)
end

[dx, x, ex, Tx, ~, lambda_x] = cal_matrix('x', Parameter);
[dy, y, ey, Ty, ~, lambda_y] = cal_matrix('y', Parameter);
[dz, z, ez, Tz, ~, lambda_z] = cal_matrix('z', Parameter);
u1x = sin(coeffx * x);
u1y = sin(coeffy * y);
u1z = sin(coeffz * z);
u2x = x - x .^ 3;
u2y = y .^ 2 - y .^ 4;
u2z = 1 - z .^ 2;
du2x = 6 * x;
du2y = 12 * y .^ 2 - 2;
du2z = 2;
exact_sol = squeeze(tensorprod(u1x * u1y', u1z)) + squeeze(tensorprod(u2x * u2y', u2z));
% generate f as a 3D tensor [nx, ny, nz]
f = (coeffx ^ 2 + coeffy ^ 2 + coeffz ^ 2) * squeeze(tensorprod(u1x * u1y', u1z)) + ...
    squeeze(tensorprod(du2x * u2y', u2z)) + squeeze(tensorprod(u2x * du2y', u2z)) + ...
    squeeze(tensorprod(u2x * u2y', du2z)) + alpha * exact_sol;
invTx = pinv(Tx);
invTy = pinv(Ty);
invTz = pinv(Tz);
% to save offline time, we first load the 1D matrix into GPU
% then generate the 3D tensor for computation
switch Parameter.device
    case 'gpu'
        fprintf('GPU computation: starting to load matrices/data \n')
        Device = gpuDevice(Parameter.device_id);
        Tx = gpuArray(Tx);
        Ty = gpuArray(Ty);
        Tz = gpuArray(Tz);
        lambda_x = gpuArray(lambda_x);
        lambda_y = gpuArray(lambda_y);
        lambda_z = gpuArray(lambda_z);
        ex = gpuArray(ex);
        ey = gpuArray(ey);
        ez = gpuArray(ez);
        f = gpuArray(f);
        % since T_i is not unitary
        invTx = gpuArray(invTx);
        invTy = gpuArray(invTx);
        invTz = gpuArray(invTx);
end

% generate eigenvalue tensor
Lambda3D = squeeze(tensorprod(lambda_x, ey * ez')) + squeeze(tensorprod(ex, lambda_y * ez')) ...
    + squeeze(tensorprod(ex, ey * lambda_z'));

switch Parameter.device
    case 'gpu'
        wait(Device);
        fprintf('GPU computation: loading finished and GPU computing started \n')
end

%% online computation
tic;

for iter = 1:num_repeat
    %% Eigenvector method to solve the linear system by tensor matrix multiplication
    u = tensorprod(f, invTz', 3, 1);
    u = pagemtimes(u, invTy');
    u = squeeze(tensorprod(invTx, u, 2, 1));
    u = u ./ (Lambda3D + alpha);
    u = tensorprod(u, Tz', 3, 1);
    u = pagemtimes(u, Ty');
    u = squeeze(tensorprod(Tx, u, 2, 1));
end

switch Parameter.device
    case 'gpu'
        wait(Device);
end

time = toc;

err = u - exact_sol;
l2_err = norm(err, 'fro') * sqrt(dx * dy * dz);
fprintf('The ell-2 norm error is %d \n', l2_err)
l_infty_err = norm(err(:), inf);
fprintf('The ell-infty norm error is %d \n', l_infty_err)

switch Parameter.device
    case 'gpu'
        fprintf('The GPU time of %d solvers is %d \n', num_repeat, time)
    case 'cpu'
        fprintf('The CPU time of %d solvers is %d \n', num_repeat, time)
end

fprintf('\n')

if (plotfigure)
    [X, Y, Z] = meshgrid(y, x, z);

    FigHandle = figure;
    set(FigHandle, 'Position', [100, 100, 1449, 300]);
    set(0, 'DefaultTextFontSize', 12, 'DefaultAxesFontSize', 12)

    subplot(1, 2, 1)
    slice(X, Y, Z, exact_sol, [1], 1, -3);
    shading flat
    xlabel('Y')
    ylabel('X')
    zlabel('Z')
    view(3);
    axis tight;
    axis equal;
    colorbar
    title('Exact solution')
    subplot(1, 2, 2)
    slice(X, Y, Z, u, [1], 1, -3);
    shading flat
    xlabel('Y')
    ylabel('X')
    zlabel('Z')

    view(3);
    axis tight;
    axis equal;
    colorbar
    title('Numerical solution')
end

% export dat to tecplot
coordx = repmat(x, ny * nz, 1);
coordy1 = repmat(y', nx, nz);
coordy = coordy1(:);
coordz1 = repmat(z, 1, nx * ny)';
coordz = coordz1(:);
coord(:, 1) = coordx;
coord(:, 2) = coordy;
coord(:, 3) = coordz;
OUTPUT_Tecplot2(1, coord, nx, ny, nz, 'U,V\n',u(:), exact_sol(:))

end
