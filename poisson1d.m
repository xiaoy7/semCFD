
% liu Poisson_SEM_dirichlet
clc
clear
addpath(genpath("D:\semMatlab"))

currentTime = datetime('now', 'Format', 'yyyyMMdd_HHmm');
pathname = ['time' char(currentTime)];
copyfile("*.m", pathname)


Parameter.plotfigure = 1; % 1: yes, 0: no
Parameter.Np = 4; % polynomial degree

Parameter.bc = 'dirichlet'; % 'dirichlet' 'periodic'  'neumann'

Parameter.basis = 'SEM'; %'FFT' 'SEM'
if Parameter.basis == 'FFT'
    Parameter.length = 1;
else
    % the domain is [Lminx, Lmaxx] x [Lminy, Lmaxy]
    Parameter.minx = 0;
    Parameter.maxx = 1;

    Parameter.Ncell = 10; % number of cells in finite element
    % number of cells in finite element
    Parameter.Ncellx = 10;
end


if gpuDeviceCount('available') < 1
    Parameter.device = 'cpu';
else
    Parameter.device = 'gpu';
    Parameter.device_id = 1;
end % ID=1,2,3,...

% polynomial degree
Parameter.Npx = Parameter.Np;
nx_all = Parameter.Ncellx * Parameter.Npx + 1;
Parameter.nx_all = nx_all;
switch Parameter.bc
    case 'dirichlet'
        nx = Parameter.Ncellx * Parameter.Npx - 1;
        bcNodesx = [1, nx_all];
        % total number of unknowns in one direction
    case 'periodic'
        nx = Parameter.Ncellx * Parameter.Npx;
        bcNodesx = 1;
        % total number of unknowns in one direction
    case 'neumann'
        nx = Parameter.Ncellx * Parameter.Npx + 1;
        bcNodesx = [];
        % total number of unknowns in one direction
end
Parameter.nx = nx;
freeNodesx = 1:nx_all;
Parameter.freeNodesx = freeNodesx(~ismember(freeNodesx, bcNodesx));
Parameter.bcNodesx = bcNodesx;


%% Qk finite element with (k+1)-point Gauss-Lobatto quadrature
fprintf('Laplacian is Q%d spectral element method \n', Parameter.Np)

if (Parameter.Np < 2)
    fprintf('It is also classical second order discrete Laplacian \n')
else
    fprintf('It is also a %d-th order accurate finite difference scheme \n', Parameter.Np + 2)
end

[dx, x, Tx, Hx,lambda_x,Dmatrixx] = cal_matrix2('x', Parameter);
coeffx = pi;
alpha = 1;
exact_sol = sin(coeffx * x);
dexact_sol = coeffx * cos(coeffx * x);
u_all = exact_sol;

% generate f as a 1D tensor [nx]
f = (alpha + pi*pi) * exact_sol;
f(bcNodesx) = exact_sol(bcNodesx);
f(Parameter.freeNodesx) = f(Parameter.freeNodesx) ...
- Hx(Parameter.freeNodesx,Parameter.bcNodesx) * exact_sol(bcNodesx);

invTx = pinv(Tx);
% to save offline time, we first load the 1D matrix into GPU
% then generate the 1D tensor for computation
switch Parameter.device
    case 'gpu'
        fprintf('GPU computation: starting to load matrices/data \n')
        Device = gpuDevice(Parameter.device_id);
        Tx = gpuArray(Tx);
        lambda_x = gpuArray(lambda_x);
        % ex = gpuArray(ex);
        f = gpuArray(f);
        % f = gpuArray(f(Parameter.freeNodesx));
        % since T_i is not unitary
        invTx = gpuArray(invTx);
end

% generate eigenvalue tensor
Lambda1D = lambda_x;

switch Parameter.device
    case 'gpu'
        wait(Device);
        fprintf('GPU computation: loading finished and GPU computing started \n')
end

%% online computation
tic;

%% Eigenvector method to solve the linear system by tensor matrix multiplication
u = squeeze(tensorprod(invTx, f(Parameter.freeNodesx), 2, 1));
u = u ./ (Lambda1D + alpha);
u = squeeze(tensorprod(Tx, u, 2, 1));


switch Parameter.device
    case 'gpu'
        wait(Device);
end

time = toc;
u_all(Parameter.freeNodesx) = u;
err = u_all - exact_sol;
l2_err = norm(err, 'fro') * sqrt(dx);
fprintf('The ell-2 norm error is %d \n', l2_err)
l_infty_err = norm(err(:), inf);
fprintf('The ell-infty norm error is %d \n', l_infty_err)

switch Parameter.device
    case 'gpu'
        fprintf('The GPU time of solver is %d \n', time)
    case 'cpu'
        fprintf('The CPU time of solver is %d \n', time)
end

fprintf('\n')


figure;
p = plot(x, u_all, 'b-', x, exact_sol, 'r*');
p(1).LineWidth = 2;
p(2).MarkerSize = 6;
legend('Numerical Solution', 'Exact Solution')
title('1D Helmholtz Equation with Non-Zero Dirichlet BC')
xlabel('x')
ylabel('u(x)')
grid on;