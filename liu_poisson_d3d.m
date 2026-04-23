% liu Poisson_SEM_dirichlet
% Solve the 3D Helmholtz equation with non-homogeneous Dirichlet
%  boundary conditions using a spectral element method (SEM).
%
% PDE: -Δu + α u = f in Ω = [0,1]^3
% Dirichlet faces (x = 0, 1, y = 0, 1 and z = 0, 1): u = g_D
clc
clear
addpath(genpath("D:\semMatlab"))

currentTime = datetime('now', 'Format', 'yyyyMMdd_HHmm');
pathname = ['time' char(currentTime)];
copyfile("*.m", pathname)

Para.plotfigure = 1; % 1: yes, 0: no
Para.Np = 5; % polynomial degree

Para.bc = 'dirichlet'; % 'dirichlet' 'periodic'  'neumann'

Para.basis = 'SEM'; %'FFT' 'SEM'
if Para.basis == 'FFT'
    Para.length = 1;
else
    Para.minx = 0;
    Para.maxx = 1;
    Para.miny = 0;
    Para.maxy = 2;
    Para.minz = 0;
    Para.maxz = 3;

    Para.Ncell = 10; % number of cells in finite element
    % number of cells in finite element
    Para.Ncellx = 10;
    Para.Ncelly = 20;
    Para.Ncellz = 30;

end


if gpuDeviceCount('available') < 1
    Para.device = 'cpu';
else
    Para.device = 'gpu';
    Para.device_id = 1;
end % ID=1,2,3,...


plotfigure = Para.plotfigure;
Np = Para.Np;

% polynomial degree
Npx = Np;
Npy = Np;
Npz = Np;
Para.Npx = Npx;
Para.Npy = Npy;
Para.Npz = Npz;

nx_all = Para.Ncellx * Npx + 1;
ny_all = Para.Ncelly * Npy + 1;
nz_all = Para.Ncellz * Npz + 1;
Para.nx_all = nx_all;
Para.ny_all = ny_all;
Para.nz_all = nz_all;
switch Para.bc
    case 'dirichlet'
        nx = Para.Ncellx * Npx - 1;
        ny = Para.Ncelly * Npy - 1;
        nz = Para.Ncellz * Npz - 1;
        bcNodesx = [1, nx_all];
        bcNodesy = [1, ny_all];
        bcNodesz = [1, nz_all];
        % total number of unknowns in one direction
    case 'periodic'
        nx = Para.Ncellx * Npx;
        ny = Para.Ncelly * Npy;
        nz = Para.Ncellz * Npz;
        % total number of unknowns in one direction
    case 'neumann'
        nx = Para.Ncellx * Npx + 1;
        ny = Para.Ncelly * Npy + 1;
        nz = Para.Ncellz * Npz + 1;
        bcNodesx = [];
        bcNodesy = [];
        bcNodesz = [];
        % total number of unknowns in one direction
end

freeNodesx = 1:nx_all;
Para.freeNodesx = freeNodesx(~ismember(freeNodesx, bcNodesx));
Para.bcNodesx = bcNodesx;

freeNodesy = 1:ny_all;
Para.freeNodesy = freeNodesy(~ismember(freeNodesy, bcNodesy));
Para.bcNodesy = bcNodesy;

freeNodesz = 1:nz_all;
Para.freeNodesz = freeNodesz(~ismember(freeNodesz, bcNodesz));
Para.bcNodesz = bcNodesz;

Para.nx = nx;
Para.ny = ny;
Para.nz = nz;
coeffx = pi;
coeffy = pi; % 2 *
coeffz = pi; % 3 *
alpha = 1;
% the domain is [Lminx, Lmaxx] x [Lminy, Lmaxy] x [Lminz, Lmaxz]

%% Qk finite element with (k+1)-point Gauss-Lobatto quadrature
fprintf('This is a code solving 3D Poison on a grid of size %d by %d by %d \n', nx, ny, nz)
fprintf('Laplacian is Q%d spectral element method \n', Np)

if (Np < 2)
    fprintf('It is also classical second order discrete Laplacian \n')
else
    fprintf('It is also a %d-th order accurate finite difference scheme \n', Np + 2)
end

[dx, x, Tx, Hx, lambda_x, Dmatrixx, ex] = cal_matrix2('x', Para);
[dy, y, Ty, Hy, lambda_y,Dmatrixy,ey] = cal_matrix2('y', Para);
[dz, z, Tz, Hz, lambda_z,Dmatrixz,ez] = cal_matrix2('z', Para);
Dmatrixx = full(Dmatrixx);
Dmatrixy = full(Dmatrixy);
Dmatrixz = full(Dmatrixz);



% generate the coordinates of the grid points
[X, Y, Z] = meshgrid(y, x, z);
% coordx = repmat(x, ny * nz, 1);
% coordy1 = repmat(y', nx, nz);
% coordy = coordy1(:);
% coordz1 = repmat(z, 1, nx * ny)';
% coordz = coordz1(:);
coord(:, 1) = X(:);
coord(:, 2) = Y(:);
coord(:, 3) = Z(:);


% exact values
u1x = sin(coeffx * x);
u1y = sin(coeffy * y);
u1z = sin(coeffz * z);
exact_sol = squeeze(tensorprod(u1x * u1y', u1z));

% derivative of exact values
dux = coeffx * cos(coeffx * x);
duy = coeffy * cos(coeffy * y);
duz = coeffz * cos(coeffz * z);
duxexact_sol = squeeze(tensorprod(dux * u1y', u1z));
duyexact_sol = squeeze(tensorprod(u1x * duy', u1z));
duzexact_sol = squeeze(tensorprod(u1x * u1y', duz));

u_all = exact_sol*0;
u_all(Para.bcNodesx,Para.bcNodesy,Para.bcNodesz) = ...
    exact_sol(Para.bcNodesx,Para.bcNodesy,Para.bcNodesz);
% generate f as a 3D tensor [nx, ny, nz]
f = (alpha + 3 * pi*pi) * exact_sol;
invTx = pinv(Tx);
invTy = pinv(Ty);
invTz = pinv(Tz);
% to save offline time, we first load the 1D matrix into GPU
% then generate the 3D tensor for computation
switch Para.device
    case 'gpu'
        fprintf('GPU computation: starting to load matrices/data \n')
        Device = gpuDevice(Para.device_id);
        Tx = gpuArray(Tx);
        Ty = gpuArray(Ty);
        Tz = gpuArray(Tz);
        lambda_x = gpuArray(lambda_x);
        lambda_y = gpuArray(lambda_y);
        lambda_z = gpuArray(lambda_z);
        ex = gpuArray(ex);
        ey = gpuArray(ey);
        ez = gpuArray(ez);
        f = gpuArray(f(Para.freeNodesx,Para.freeNodesy,Para.freeNodesz));
        % since T_i is not unitary
        invTx = gpuArray(invTx);
        invTy = gpuArray(invTy);
        invTz = gpuArray(invTz);
end

% generate eigenvalue tensor
Lambda3D = squeeze(tensorprod(lambda_x, ey * ez')) + squeeze(tensorprod(ex, lambda_y * ez')) ...
    + squeeze(tensorprod(ex, ey * lambda_z'));

switch Para.device
    case 'gpu'
        wait(Device);
        fprintf('GPU computation: loading finished and GPU computing started \n')
end

%% online computation
tic;

%% Eigenvector method to solve the linear system by tensor matrix multiplication
u = tensorprod(f, invTz', 3, 1);
u = pagemtimes(u, invTy');
u = squeeze(tensorprod(invTx, u, 2, 1));
u = u ./ (Lambda3D + alpha);
u = tensorprod(u, Tz', 3, 1);
u = pagemtimes(u, Ty');
u = squeeze(tensorprod(Tx, u, 2, 1));


switch Para.device
    case 'gpu'
        wait(Device);
end

time = toc;

u_all(Para.freeNodesx,Para.freeNodesy,Para.freeNodesz) = u;
err = u_all - exact_sol;
l2_err = norm(err, 'fro') * sqrt(dx * dy * dz);
fprintf('The ell-2 norm error is %d \n', l2_err)
l_infty_err = norm(err(:), inf);
fprintf('The ell-infty norm error is %d \n', l_infty_err)

switch Para.device
    case 'gpu'
        fprintf('The GPU time of solver is %d \n', time)
    case 'cpu'
        fprintf('The CPU time of solver is %d \n', time)
end

fprintf('\n')

% calculate the derivative
% For x derivative - matrix multiplication along rows
ux = pagemtimes(Dmatrixx, u_all);
% ux = tensorprod(Dmatrixx, u_all, 2, 1); % d(un)/dx
uy = pagemtimes(u_all, Dmatrixy');  % d(un)/dy

% For x derivative - matrix multiplication along rows
% ux = zeros(size(u_all));
% for k = 1:nz_all
    % ux(:,:,k) = Dmatrixx * u_all(:,:,k);
% end

% For y derivative - matrix multiplication along columns
% uy = zeros(size(u_all));
% for k = 1:nz_all
    % uy(:,:,k) = u_all(:,:,k) * Dmatrixy';
% end

% For z derivative - matrix multiplication along z-direction
% uz = zeros(size(u_all));
% for i = 1:nx_all
%     for j = 1:ny_all
%         temp = reshape(u_all(i,j,:), [nz_all, 1]);
%         uz(i,j,:) = Dmatrixz * temp;
% 
%     end
% end
% uz = tensorprod(u_all, Dmatrixz', 3, 2); % d(un)/dz
    uz = derivative_z(Dmatrixz,u_all,nx_all, ny_all, nz_all);% d(wn)/dz


errorux = ux - duxexact_sol;
erroruy = uy - duyexact_sol;
erroruz = uz - duzexact_sol;
fprintf('The ell-2 norm error of ux is %d \n', norm(errorux, 'fro') * sqrt(dx * dy * dz))
fprintf('The ell-2 norm error of uy is %d \n', norm(erroruy, 'fro') * sqrt(dx * dy * dz))
fprintf('The ell-2 norm error of uz is %d \n', norm(erroruz, 'fro') * sqrt(dx * dy * dz))

% export dat to tecplot
variableName = 'u,uexact,dux,duxexact,duy,duyexact,duz,duzexact\n';
OUTPUT_Tecplot2(3, coord, nx_all, ny_all, nz_all, variableName, ...
u_all(:), exact_sol(:), ux(:), duxexact_sol(:), uy(:), duyexact_sol(:), uz(:), duzexact_sol(:));
fprintf('done \n');