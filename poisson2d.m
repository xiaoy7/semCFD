% liu Poisson_SEM_dirichlet
clc
clear
addpath(genpath("D:\semMatlab"))

currentLocation = pwd;
cd ..
fileLocation = pwd;
currentTime = datetime('now', 'Format', 'yyyyMMdd_HHmm_ss');% 获取当前日期和时间
pathname = [fileLocation, '\time' char(currentTime)];
cd(currentLocation)
copyfile("*.m", pathname)

para.Np = 5; % polynomial degree

para.bc = 'dirichlet'; % 'dirichlet' 'periodic'  'neumann'

para.basis = 'SEM'; %'FFT' 'SEM'
if para.basis == 'FFT'
    para.length = 1;
else
    para.minx = 0;
    para.maxx = 1;
    para.miny = 0;
    para.maxy = 1.5;
    para.Ncellx = 10;
    para.Ncelly = 15;

end


if gpuDeviceCount('available') < 1
    para.device = 'cpu';
else
    para.device = 'gpu';
    para.device_id = 1;
end % ID=1,2,3,...

Np = para.Np;

% polynomial degree
Npx = Np;
Npy = Np;
para.Npx = Npx;
para.Npy = Npy;
para.nx_all = para.Ncellx * Npx + 1;
para.ny_all = para.Ncelly * Npy + 1;
switch para.bc
    case 'dirichlet'
        nx = para.Ncellx * Npx - 1;
        ny = para.Ncelly * Npy - 1;
        bcNodesx = [1, para.nx_all];
        bcNodesy = [1, para.ny_all];

        % total number of unknowns in one direction
    case 'periodic'
        nx = para.Ncellx * Npx;
        ny = para.Ncelly * Npy;
        % total number of unknowns in one direction
    case 'neumann'
        nx = para.Ncellx * Npx + 1;
        ny = para.Ncelly * Npy + 1;
        bcNodesx = [];
        bcNodesy = [];
        % total number of unknowns in one direction
end

freeNodesx = 1:para.nx_all;
para.freeNodesx = freeNodesx(~ismember(freeNodesx, bcNodesx));
para.bcNodesx = bcNodesx;

freeNodesy = 1:para.ny_all;
para.freeNodesy = freeNodesy(~ismember(freeNodesy, bcNodesy));
para.bcNodesy = bcNodesy;


para.nx = nx;
para.ny = ny;
coeffx = pi;
coeffy = pi; % 2 *
alpha = 1;
% the domain is [Lminx, Lmaxx] x [Lminy, Lmaxy]

%% Qk finite element with (k+1)-point Gauss-Lobatto quadrature
fprintf('This is a code solving 3D Poison on a grid of size %d by %d \n', nx, ny)

fprintf('Laplacian is Q%d spectral element method \n', Np)

if (Np < 2)
    fprintf('It is also classical second order discrete Laplacian \n')
else
    fprintf('It is also a %d-th order accurate finite difference scheme \n', Np + 2)
end

[dx, x, Tx, Hx, lambda_x, Dmatrixx, ex] = cal_matrix2('x', para);
[dy, y, Ty, Hy, lambda_y, Dmatrixy, ey] = cal_matrix2('y', para);

[coordY, coordX] = meshgrid(y, x); % Grid coordinates for output/plotting

%% exact values
u1x = sin(coeffx * x);
u1y = sin(coeffy * y);
exact_sol = squeeze(u1x * u1y');

%% derivative of exact values
du1x = coeffx * cos(coeffx * x);
du1y = coeffy * cos(coeffy * y);
duxexact_sol = squeeze(du1x * u1y');
duyexact_sol = squeeze(u1x * du1y');

%% initial parameters

u_all = exact_sol*0;

% generate f as a 2D tensor [nx, ny]
f = (alpha + 2 * pi*pi) * exact_sol;
for i = 1:2
    u_all(para.bcNodesx(i),:) = exact_sol(para.bcNodesx(i),:);
    u_all(:,para.bcNodesy(i)) = exact_sol(:,para.bcNodesy(i));
    f(:,para.bcNodesy(i)) = f(:,para.bcNodesy(i));
end
% f(Parameter.bcNodesx,Parameter.bcNodesy) = exact_sol(Parameter.bcNodesx,Parameter.bcNodesy);
% === Correct f for non-zero Dirichlet BCs ===
f(para.freeNodesx,:) = f(para.freeNodesx,:) ...
    - Hx(para.freeNodesx,para.bcNodesx) * exact_sol(bcNodesx,:);

f(:,para.freeNodesy) = f(:,para.freeNodesy) - ...
    exact_sol(:,bcNodesy)*Hy(para.freeNodesy,para.bcNodesy)';



invTx = pinv(Tx);
invTy = pinv(Ty);
% to save offline time, we first load the 1D matrix into GPU
% then generate the 2D tensor for computation
switch para.device
    case 'gpu'
        fprintf('GPU computation: starting to load matrices/data \n')
        Device = gpuDevice(para.device_id);
        Tx = gpuArray(Tx);
        Ty = gpuArray(Ty);
        lambda_x = gpuArray(lambda_x);
        lambda_y = gpuArray(lambda_y);
        ex = gpuArray(ex);
        ey = gpuArray(ey);
        f = gpuArray(f(para.freeNodesx,para.freeNodesy));
        % since T_i is not unitary
        invTx = gpuArray(invTx);
        invTy = gpuArray(invTy);
end

% generate eigenvalue tensor
Lambda2D = squeeze(tensorprod(lambda_x, ey)) + squeeze(tensorprod(ex, lambda_y));

switch para.device
    case 'gpu'
        wait(Device);
        fprintf('GPU computation: loading finished and GPU computing started \n')
end

%% online computation
tic;


%% Eigenvector method to solve the linear system by tensor matrix multiplication
u = pagemtimes(f, invTy');
u = squeeze(tensorprod(invTx, u, 2, 1));
u = u ./ (Lambda2D + alpha);
u = pagemtimes(u, Ty');
u = squeeze(tensorprod(Tx, u, 2, 1));


switch para.device
    case 'gpu'
        wait(Device);
end

time = toc;

u_all(para.freeNodesx,para.freeNodesy) = u;
err = u_all - exact_sol;
l2_err = norm(err, 'fro') * sqrt(dx * dy);
fprintf('The ell-2 norm error is %d \n', l2_err)
l_infty_err = norm(err(:), inf);
fprintf('The ell-infty norm error is %d \n', l_infty_err)



switch para.device
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
% OUTPUT_Tecplot2D(2, coord, nx_all, ny_all, 'u,uexact,uxx,uexactxx,uyy,uexactyy\n',...
%     u_all(:), exact_sol(:),uxx(:),duxexact_sol(:),uyy(:),duyexact_sol(:));
varName = 'u,uexact,uxx,uexactxx,uyy,uexactyy\n';
 OUTPUT_Tecplot2D4(2,pathname, para.nx_all, para.ny_all, varName, ...
            coordX(:),  coordY(:), u_all(:), exact_sol(:),uxx(:),duxexact_sol(:),uyy(:),duyexact_sol(:));

fprintf('done \n');