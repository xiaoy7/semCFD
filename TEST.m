% xiao yao v2025.4.26
% sovling the navier stokes equation based on spectral element method
% with 2D tensor product and projection method
% bubble rising 152.621765 seconds
% M0 = 1e-7; % 5e-7 1e-6;
clc
clear
addpath(genpath("D:\semMatlab")) % Ensure this path is correct

currentLocation = pwd;
cd ..
fileLocation = pwd;
currentTime = datetime('now', 'Format', 'yyyyMMdd_HHmm_ss');% 获取当前日期和时间
pathname = [fileLocation, '\time' char(currentTime)];
cd(currentLocation)
copyfile("*.m", pathname)

stage = 1;
fprintf('=== %d Program Starts ===\n', stage);
stage = stage + 1;

%% set parameters
delta = 2e-4;    % Time step
steps = 100; % Max number of steps (increased for potential convergence)
freOut = 20;
% varName = 'U,V,P,phi,psi,rho,mu,ftx,fty\n'; % export data
 variables_grid={'X','Y'}; % export grid
    velName={'u','v','p','phi','psi','rho','mu'}; %,'ftx','fty','ftz'
 
    filename_grid = fullfile(pathname, ['grid','.plt']);

LL1 = 1; % the characteristic length

Cx = 0.5;     % Center of circle in x direction
Cy = 0.5;       % Center of circle in y direction
radius = 0.25; % Radius of circle

Cn = 0.01;
eta = Cn * LL1;  % Smoothing parameter for tanh
Lrho = 1000;     % density
Grho = 1;
Lmu = 10;
Gmu = 0.1; %  dynamic viscosity
nium = 0.1; %mu/rho
rho0 = min(Lrho,Grho);

para_d.Np = 4; % polynomial degree
Np = para_d.Np;


para_d.basis = 'SEM'; %'FFT' 'SEM'
para_d.minx = 0;
para_d.maxx = 1*LL1;
para_d.miny = 0;
para_d.maxy = 2*LL1; % Square domain


% number of cells in finite element
para_d.Ncellx = 3;
para_d.Ncelly = 6;

gravity = -0.98;
tension1 = 1.96;
M0 = 1e-7; % 5e-7 1e-6;
gamma1 = M0; % mobility
gamma0 = 1.5;
bigs = 1.1 * eta ^ 2 * sqrt(4 * gamma0 / (M0 * delta));
aa = 1 - gamma0 / (M0 * delta) * 4 * eta^ 4 / bigs^ 2;
alpha = -bigs / (2 * eta ^ 2) * (1 + sqrt(aa));
lambda = 3 * eta / 2 /sqrt(2); % mixing energy density tension1 *

if gpuDeviceCount('available') < 1
    para_d.device = 'cpu';
else
    para_d.device = 'gpu';
    para_d.device_id = 1; % Choose appropriate GPU ID
end


% polynomial degree
Npx = Np;
Npy = Np;
para_d.Npx = Npx;
para_d.Npy = Npy;
para_d.nx_all = para_d.Ncellx * Npx + 1;
para_d.ny_all = para_d.Ncelly * Npy + 1;


para_d.bc = 'dirichlet'; % Boundary condition type for velocity
% Parameterv = Parameteru; % Redundant
para_n = para_d; % Use same domain/discretization parameters initially
para_n.bc = 'neumann'; % Boundary condition type for pressure correction

% Calculate free/boundary nodes based on BC type
para_d = parameter_bc2d(para_d); % dirichlet
para_n = parameter_bc2d(para_n); % Neumann

%% Qk finite element with (k+1)-point Gauss-Lobatto quadrature
fprintf('Laplacian is Q%d spectral element method \n', Np)

% Get matrices for Velocity (Dirichlet BCs applied internally by cal_matrix2 for INTERIOR solve)
[dx, x, Txd, ~, lambda_xd, Dmatrixx, ex] = cal_matrix2('x', para_d); % Hx not needed directly
[dy, y, Tyd, ~, lambda_yd, Dmatrixy, ey] = cal_matrix2('y', para_d); % Hy not needed directly
DmatrixyT = Dmatrixy'; % Transpose for convenience

% Get matrices for Pressure (Neumann BCs - use cal_matrix2_1 or similar)
% Ensure cal_matrix2_1 correctly handles Neumann for pressure
[dxn, exn, Txn, lambda_xn] = cal_matrix2_1('x', para_n,x);
[dyn, eyn, Tyn, lambda_yn] = cal_matrix2_1('y', para_n,y);

[coordY, coordX] = meshgrid(y, x); % Grid coordinates for output/plotting
 coords = [coordX(:), coordY(:)];
wx = assemble_sem_weights(para_d.Npx, para_d.Ncellx, para_d.minx, para_d.maxx);
wy = assemble_sem_weights(para_d.Npy, para_d.Ncelly, para_d.miny, para_d.maxy);
mass_diag = wx * wy';

 