% xiao yao v2025.4.26
% sovling the navier stokes equation based on spectral element method
% with 2D tensor product and projection method

xy = 1;%0:继续计算, 1:重新开始
xyxy = 1; %0:继续计算, 1:重新开始
if xy == 1
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
RE = 2000; % Reynolds number
nu = 1/RE; % Kinematic viscosity
dT = 1e-3; % Time step
steps = 5000; % Max number of steps (increased for potential convergence)
tol = 1e-7; % Convergence tolerance (tightened slightly)
alpha_helmholtz = 1 / dT; % Coefficient for Helmholtz equation (implicit time term)
varName = 'U,V,PRE\n'; % export data

para_u.Np = 4; % polynomial degree
Np = para_u.Np;

para_u.basis = 'SEM'; %'FFT' 'SEM'
if para_u.basis == 'FFT'
    para_u.length = 1; % Assuming square domain if FFT
    para_u.minx = -para_u.length;
    para_u.maxx = para_u.length;
    para_u.miny = -para_u.length;
    para_u.maxy = para_u.length;
    para_u.Ncellx = 1;
    para_u.Ncelly = 1; % FFT implies single domain usually
else
    para_u.minx = 0;
    para_u.maxx = 1;  % Unit square domain
    para_u.miny = 0;
    para_u.maxy = 1;  % Unit square domain

    para_u.Ncell = 10; % number of cells in finite element (default)
    % number of cells in finite element
    para_u.Ncellx = 50;
    para_u.Ncelly = 50;

end

if gpuDeviceCount('available') < 1
    para_u.device = 'cpu';
    Device = [];
else
    para_u.device = 'gpu';
    para_u.device_id = 1; % Choose appropriate GPU ID
    Device = gpuDevice(para_u.device_id);
end

% polynomial degree
Npx = Np;
Npy = Np;
para_u.Npx = Npx;
para_u.Npy = Npy;
para_u.nx_all = para_u.Ncellx * Npx + 1;
para_u.ny_all = para_u.Ncelly * Npy + 1;

para_u.bc = 'dirichlet'; % Boundary condition type for velocity
% Parameterv = Parameteru; % Redundant
para_p = para_u; % Use same domain/discretization parameters initially
para_p.bc = 'neumann'; % Boundary condition type for pressure correction

% Calculate free/boundary nodes based on BC type
para_u = parameter_bc2d(para_u);
para_p = parameter_bc2d(para_p); % Pressure uses Neumann

% the domain is [Lminx, Lmaxx] x [Lminy, Lmaxy]

%% Qk finite element with (k+1)-point Gauss-Lobatto quadrature
fprintf('Laplacian is Q%d spectral element method \n', Np)

% Get matrices for Velocity (Dirichlet BCs applied internally by cal_matrix2 for INTERIOR solve)
[dx, x, Tx, ~, lambda_xd, Dmatrixx, ex] = cal_matrix2('x', para_u); % Hx not needed directly
[dy, y, Ty, ~, lambda_yd, Dmatrixy, ey] = cal_matrix2('y', para_u); % Hy not needed directly
DmatrixyT = Dmatrixy'; % Transpose for convenience

% Get matrices for Pressure (Neumann BCs - use cal_matrix2_1 or similar)
% Ensure cal_matrix2_1 correctly handles Neumann for pressure
[dxn, exn, Txn, lambda_xn] = cal_matrix2_1('x', para_p,x);
[dyn, eyn, Tyn, lambda_yn] = cal_matrix2_1('y', para_p,y);

[coordY, coordX] = meshgrid(y, x); % Grid coordinates for output/plotting
gravity_potential = - coordY; % Hydrostatic pressure profile balancing gravity

% immersed boundary method (IBM) setup
ibm = ibm_setup2d(coordX, coordY, dT, para_u);

%% initial parameters
fprintf('=== %d initial parameter\n', stage);
stage = stage + 1;

invTx = pinv(Tx); % Inverse transforms for INTERIOR velocity nodes
invTy = pinv(Ty);
invTxn = pinv(Txn); % Inverse transforms for pressure nodes (Neumann)
invTyn = pinv(Tyn);

% Build spectral eigenvalue tensors for Helmholtz and Poisson solves
lambda_xd_col = lambda_xd(:);
lambda_yd_col = lambda_yd(:);
helmholtz_u = reshape(lambda_xd_col, [], 1) + reshape(lambda_yd_col, 1, []);
helmholtz_u = alpha_helmholtz + nu * helmholtz_u;

lambda_xn_col = lambda_xn(:);
lambda_yn_col = lambda_yn(:);
poisson_p = reshape(lambda_xn_col, [], 1) + reshape(lambda_yn_col, 1, []);
% Regularize constant pressure mode for pure Neumann pressure
poisson_p(1, 1) = 1;

% Initialize velocity and pressure fields
[un, vn, pre] = deal(zeros(para_u.nx_all, para_u.ny_all));

% Set initial boundary conditions for velocity
% Lid-driven cavity: u = 1 on top boundary, no-slip elsewhere
lid_velocity = 1;  % Set lid velocity to 1 for Re=100
un(:, para_u.bcNodesy(2)) = lid_velocity; % Set top boundary u=1
un(:, para_u.bcNodesy(1)) = 0; % bottom boundary u=0
un(para_u.bcNodesx, :) = 0; % left/right boundaries u=0
vn(:, para_u.bcNodesy) = 0; % top/bottom v=0
vn(para_u.bcNodesx, :) = 0; % left/right v=0

% Precompute boundary contribution for the u Helmholtz RHS
u_boundary = zeros(para_u.nx_all, para_u.ny_all);
u_boundary(:, para_u.bcNodesy(2)) = lid_velocity;

laplacian_u_boundary_x = Dmatrixx * (Dmatrixx * u_boundary);
laplacian_u_boundary_y = (u_boundary * DmatrixyT) * DmatrixyT;
laplacian_u_boundary = laplacian_u_boundary_x + laplacian_u_boundary_y;
fubc_contribution = -nu * laplacian_u_boundary(para_u.freeNodesx, para_u.freeNodesy);

un1 = un; % Initialize next step velocity
vn1 = vn;

% Move data and transform operators to GPU if requested
if strcmp(para_u.device, 'gpu')
    fprintf('GPU computation: starting to load matrices/data\n');
    Tx = gpuArray(Tx);
    Ty = gpuArray(Ty);
    invTx = gpuArray(invTx);
    invTy = gpuArray(invTy);
    Txn = gpuArray(Txn);
    Tyn = gpuArray(Tyn);
    invTxn = gpuArray(invTxn);
    invTyn = gpuArray(invTyn);
    Dmatrixx = gpuArray(Dmatrixx);
    DmatrixyT = gpuArray(DmatrixyT);
    helmholtz_u = gpuArray(helmholtz_u);
    poisson_p = gpuArray(poisson_p);
    fubc_contribution = gpuArray(fubc_contribution);
    un = gpuArray(un);
    vn = gpuArray(vn);
    pre = gpuArray(pre);
    un1 = gpuArray(un1);
    vn1 = gpuArray(vn1);
    u_boundary = gpuArray(u_boundary);
    ibm.mask = gpuArray(ibm.mask);
    ibm.target_u = gpuArray(ibm.target_u);
    ibm.target_v = gpuArray(ibm.target_v);
    wait(Device);
end

if strcmp(para_u.device, 'gpu')
    OUTPUT_Tecplot2D4(0, pathname, para_u.ny_all, para_u.nx_all, varName, ...
        coordX(:), coordY(:), gather(un(:)), gather(vn(:)), gather(pre(:)));
else
    OUTPUT_Tecplot2D4(0, pathname, para_u.ny_all, para_u.nx_all, varName, ...
        coordX(:), coordY(:), un(:), vn(:), pre(:));
end

Iter1 = 1;
else
    % load flow

    steps = 2 * steps;
    Iter1 = Iter + 1;


    stage = 1;
    fprintf('=== %d Inital old old old === \n',stage);
    stage = stage + 1;
    if xyxy == 0
        fprintf('=== %d load old file === \n',stage);
        stage = stage + 1;
        load flow

        A = dlmread('rere.dat');
        Un1 = A(:,3);
        Vn1 = A(:,4);
        phin1 = A(:,5);
        Umid = Un1;
        Vmid = Vn1;

    end
end

%% online computation
tic;

fprintf('=== %d start time stepping\n', stage);
stage = stage + 1;
for Iter = Iter1:steps

    %% Step 1: Calculate intermediate velocity RHS (convection + previous diffusion/pressure)
    % Calculate convective terms (e.g., using Adams-Bashforth 1st order)
    DUnx = Dmatrixx * un;
    DUny = un * DmatrixyT;
    DVnx = Dmatrixx * vn;
    DVny = vn * DmatrixyT;

    convection_u = un .* DUnx + vn .* DUny;
    convection_v = un .* DVnx + vn .* DVny;

    % Intermediate velocity RHS (explicit convection, implicit time term)
    U_star = un / dT - convection_u;
    V_star = vn / dT - convection_v;  % Removed gravity term (-1) for lid-driven cavity
    % Note: Diffusion term is handled implicitly in the Helmholtz solve (Step 3)

    % Step 1.5: IBM forcing (direct forcing / Brinkman-style penalization)
    if ibm.enabled
        ibm = ibm_update_rigid_body2d(ibm, Iter, dT);
        [ibm_force_u, ibm_force_v] = ibm_direct_forcing2d(un, vn, ibm, dT);
        U_star = U_star + ibm_force_u;
        V_star = V_star + ibm_force_v;
    end

    %% Step 2: Pressure correction (Poisson equation)
    % Calculate divergence of intermediate velocity RHS (using full matrices)
    Div_U_star = Dmatrixx * U_star;
    Div_V_star = V_star * DmatrixyT;

    % RHS for pressure Poisson equation (Neumann BCs assumed for pressure)
    % We need the divergence evaluated at the pressure nodes (which might be different if staggered)
    % Assuming pressure nodes are the same as velocity nodes for now
    FP = -(Div_U_star + Div_V_star);

    % Solve Poisson equation for pressure correction 'pre' using Neumann matrices/nodes
    pre_spec = pagemtimes(FP, invTyn'); % Use pressure nodes/matrices
    pre_spec = squeeze(tensorprod(invTxn, pre_spec, 2, 1));
    pre_spec(1, 1) = 0; % Remove arbitrary constant mode for Neumann pressure
    pre_spec = pre_spec ./ poisson_p; % Solve in spectral space
    pre = pagemtimes(pre_spec, Tyn');
    pre = squeeze(tensorprod(Txn, pre, 2, 1));


    % if strcmp(para_u.device, 'gpu')
    %     wait(Device);
    % end

    %% Step 3: Velocity correction (Helmholtz equation)
    % Calculate pressure gradient (full domain, using full matrices)
    GradPre_x = Dmatrixx * pre;
    GradPre_y = pre * DmatrixyT;

    % RHS for Helmholtz equation (full domain initially)
    FU = U_star - GradPre_x;
    FV = V_star - GradPre_y;

    % Adjust RHS for INTERIOR nodes using pre-calculated boundary contributions
    FU_interior = FU(para_u.freeNodesx, para_u.freeNodesy) - fubc_contribution;
    FV_interior = FV(para_u.freeNodesx, para_u.freeNodesy);% - fvbc_contribution; % fvbc is zero here

    % Solve Helmholtz equation for u^(n+1) (INTERIOR nodes)
    uv_spec = pagemtimes(FU_interior, invTy'); % Use INTERIOR velocity matrices
    uv_spec = squeeze(tensorprod(invTx, uv_spec, 2, 1));
    uv_spec = uv_spec ./ helmholtz_u; % Solve in spectral space
    uv_phys = pagemtimes(uv_spec, Ty');
    uv_phys = squeeze(tensorprod(Tx, uv_phys, 2, 1));
    un1(para_u.freeNodesx, para_u.freeNodesy) = uv_phys; % Update INTERIOR u

    % if strcmp(para_u.device, 'gpu')
    %     wait(Device);
    % end

    % Solve Helmholtz equation for v^(n+1) (INTERIOR nodes)
    uv_spec = pagemtimes(FV_interior, invTy'); % Use INTERIOR velocity matrices
    uv_spec = squeeze(tensorprod(invTx, uv_spec, 2, 1));
    uv_spec = uv_spec ./ helmholtz_u; % Solve in spectral space
    uv_phys = pagemtimes(uv_spec, Ty');
    uv_phys = squeeze(tensorprod(Tx, uv_phys, 2, 1));
    vn1(para_u.freeNodesx, para_u.freeNodesy) = uv_phys; % Update INTERIOR v

    % if strcmp(para_u.device, 'gpu')
    %     wait(Device);
    % end

    % Re-enforce lid-driven cavity boundary conditions after the solve
    un1(:, para_u.bcNodesy(2)) = lid_velocity;
    un1(:, para_u.bcNodesy(1)) = 0;
    un1(para_u.bcNodesx, :) = 0;
    vn1(:, para_u.bcNodesy) = 0;
    vn1(para_u.bcNodesx, :) = 0;

    if ibm.enabled
        % Sharp re-enforcement of no-slip at immersed solid nodes
        un1 = (1 - ibm.mask) .* un1 + ibm.mask .* ibm.target_u;
        vn1 = (1 - ibm.mask) .* vn1 + ibm.mask .* ibm.target_v;
    end

    %% Error analysis and convergence check (based on INTERIOR nodes)
    error_u = norm(un1 - un, 'fro');
    error_v = norm(vn1 - vn, 'fro');
    total_rms_error = sqrt(error_u^2 + error_v^2);


    if rem(Iter, 10) == 0 % Print less frequently
        fprintf('Iter = %d, RMS = %e\n', Iter, total_rms_error);
    end

    % Output data periodically (e.g., every 100 steps)
    if rem(Iter, 100) == 0
    % if total_rms_error <= tol
        if strcmp(para_u.device,'gpu')
            un_out=gather(un1);
            vn_out=gather(vn1);
            pre_out=gather(pre);
        else
            un_out=un1;
            vn_out=vn1;
            pre_out=pre;
        end

        OUTPUT_Tecplot2D5(Iter,pathname, para_u.nx_all, para_u.ny_all,  varName, ...
            coordX(:), coordY(:), un_out(:), vn_out(:), pre_out(:));

        % Iter
        % break
    end

 
    % Update velocity for next iteration
    un = un1;
    vn = vn1;

end


time = toc;
fprintf('Total computation time: %f seconds\n', time);

fprintf('=== %d Program Ends ===\n', stage);
