% xiao yao v2025.4.26
% sovling the navier stokes equation based on spectral element method
% with 2D tensor product and projection method
% bubble rising 152.621765 seconds
% M0 = 1e-7; % 5e-7 1e-6;
% rt 3000 256
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
delta = 1e-2;  % Time step
steps = 50000; % Max number of steps (increased for potential convergence)
freOut = 1000;

variables_grid={'X','Y'}; % export grid
velName={'u','v','p','phi','psi','rho','mu'}; %,'ftx','fty','ftz'
parpool(size(velName,2));
filename_grid = fullfile(pathname, ['grid','.plt']);


LL1 = 100; % the characteristic length
Cx = 0.25*LL1;       % Center of circle in x direction
Cy = 0.25*LL1;       % Center of circle in y direction
radius = 0.5*LL1; % Radius of circle

Cn = 0.01;
eta = Cn * LL1;  % Smoothing parameter for tanh
Lrho = 1;     % density
Grho = 3;
Lmu = 300/256; %0.1;
Gmu = 300/256; %0.1;  % dynamic viscosity
nium = 300/256; %0.1; % mu/rho
rho0 = min(Lrho,Grho);

para_d.Np = 4; % polynomial degree
Np = para_d.Np;

para_d.basis = 'SEM'; %'FFT' 'SEM'
para_d.minx = 0;
para_d.maxx = 0.5*LL1;
para_d.miny = -2*LL1;
para_d.maxy = 2*LL1; % Square domain

% number of cells in finite element
para_d.Ncellx = 20;
para_d.Ncelly = 160;

gravity = -0.01;
tension1 = 0;
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
wx = assemble_sem_weights(para_n.Npx, para_n.Ncellx, para_n.minx, para_n.maxx);
wy = assemble_sem_weights(para_n.Npy, para_n.Ncelly, para_n.miny, para_n.maxy);
mass_diag = wx * wy';


%% initial parameters
fprintf('=== %d initial parameter\n', stage);
stage = stage + 1;

invTx = pinv(Txd); % Inverse transforms for INTERIOR velocity nodes
invTy = pinv(Tyd);
invTxn = pinv(Txn); % Inverse transforms for pressure nodes (Neumann)
invTyn = pinv(Tyn);


% Initialize velocity and pressure fields
[Un, Vn, zeross,Pn1,Pn,Pn_1] = deal(zeros(para_d.nx_all, para_d.ny_all));

Un1 = Un; % Initialize next step velocity
Vn1 = Vn;
Un_1 = Un;
Vn_1 = Vn;

% Initialize phase field (Cahn-Hilliard) with a circle
phin1 = initialPhi2d(coordX,coordY,Cx,Cy,radius,eta,LL1);

% phin1 = -tanh((sqrt((coordX - Cx) .^ 2 + (coordY - Cy) .^ 2) - radius) ./ (sqrt(2) * eta));
phin = phin1; % Previous step phase field
phin_1 = phin1; % Previous step phase field for time-stepping

title='';%无标题
zone_title='';%无标题
time=0;%非定常时间
IJK=[para_d.nx_all, para_d.ny_all];
%创建文件头
plt_Head(filename_grid,title,variables_grid,'GRID')
%创建zone（point）格式
plt_Zone(filename_grid,zone_title,IJK,time,coords)


clear coords
% Move data to GPU if specified
switch para_d.device
    case 'gpu'
        fprintf('GPU computation: starting to load matrices/data \n')
        Device = gpuDevice(para_d.device_id);
        % Velocity matrices (interior)
        Txd = gpuArray(Txd);
        Tyd = gpuArray(Tyd);
        lambda_xd = gpuArray(lambda_xd);
        lambda_yd = gpuArray(lambda_yd);
        ex = gpuArray(ex);
        ey = gpuArray(ey);
        invTx = gpuArray(invTx);
        invTy = gpuArray(invTy);
        % Pressure matrices (Neumann)
        Txn = gpuArray(Txn);
        Tyn = gpuArray(Tyn);
        lambda_xn = gpuArray(lambda_xn);
        lambda_yn = gpuArray(lambda_yn);
        exn = gpuArray(exn);
        eyn = gpuArray(eyn);
        invTxn = gpuArray(invTxn);
        invTyn = gpuArray(invTyn);
        % Differentiation matrices (assuming full)
        Dmatrixx = gpuArray(Dmatrixx);
        DmatrixyT = gpuArray(DmatrixyT); % Note: Dmatrixy is transposed before GPU transfer
        % Fields
        Un = gpuArray(Un);
        Vn = gpuArray(Vn);
        Un1 = gpuArray(Un1);
        Vn1 = gpuArray(Vn1);
end
Dmatrixx2 = Dmatrixx * Dmatrixx;
DmatrixyT2 = DmatrixyT * DmatrixyT;
% generate eigenvalue tensor for Helmholtz solve
% poisson_d = squeeze(tensorprod(lambda_xd, ey)) + squeeze(tensorprod(ex, lambda_yd));
poisson_d = bsxfun(@plus, lambda_xd, lambda_yd');
helmholtz_u = 1.5/delta + nium * poisson_d;

poisson_dn = bsxfun(@plus, lambda_xn, lambda_yn');
% poisson_dn = squeeze(tensorprod(lambda_xd, ey)) + squeeze(tensorprod(exn, lambda_yn));
helmholtz_v = 1.5/delta + nium * poisson_dn;

% generate eigenvalue tensor for Neumann nodes
poisson_n = bsxfun(@plus, lambda_xn, lambda_yn');
% poisson_n = squeeze(tensorprod(lambda_xn, eyn)) + squeeze(tensorprod(exn, lambda_yn));
poisson_p = poisson_n;

helmholtzPhi = alpha - poisson_n; % For phase field (Cahn-Hilliard)
helmholtzPsi = (alpha + bigs / eta ^ 2) + poisson_n; % For phase field (Cahn-Hilliard)

switch para_d.device
    case 'gpu'
        wait(Device);
        fprintf('GPU computation: loading finished and GPU computing started \n')
end

%% online computation
tic;


fprintf('=== %d start time stepping\n', stage);
stage = stage + 1;
for Iter = 1:steps

    %% Step 1: Calculate intermediate velocity RHS (convection + previous diffusion/pressure)
    % Calculate convective terms (e.g., using Adams-Bashforth 1st order)
    uStar = 2 * Un - Un_1; %2 * u_n - u_n-1
    vStar = 2 * Vn - Vn_1; %2 * v_n - v_n-1
    pStar = 2 * Pn - Pn_1;
    phiStar = 2 * phin - phin_1;

    uCap = 2 * Un - 0.5 * Un_1;
    vCap = 2 * Vn - 0.5 * Vn_1;
    phiCap = 2 * phin - 0.5 * phin_1;

    %% phase field
    Q1 = (uStar .* (Dmatrixx * phiStar) + vStar .* (phiStar * DmatrixyT) - phiCap / delta) / M0;
    Q2 = (phiStar .^ 2 - 1 - bigs) / eta ^ 2 .* phiStar;
    % Fpsi = Q1 + poisson_n .* Q2; % type 1
    lapQ = calLaplace(Q2,Dmatrixx2,DmatrixyT2);
    Fpsi = Q1 - lapQ; % type 2

    % Proper 2D spectral transform for psi
    psi_hat = invTxn * Fpsi * invTyn';
    psi_hat = psi_hat ./ helmholtzPsi;
    psi = Txn * psi_hat * Tyn';

    % Proper 2D spectral transform for phi
    phi_hat = invTxn * psi * invTyn';
    phi_hat = phi_hat ./ helmholtzPhi;
    phin1 = Txn * phi_hat * Tyn';

    Dphix = Dmatrixx * phin1;
    Dphiy = phin1 * DmatrixyT;


    lapPhi = calLaplace(phin1,Dmatrixx2,DmatrixyT2);
    psi1 = 1/eta^2 * (phin1.^3 - phin1) - lapPhi;
    % ft1 = tension1 * lambda * psi1 .* Dphix ;
    % ft2 = tension1 * lambda * psi1 .* Dphiy;


    %% navier stokes
    phin1(phin1 >  1) =  1;
    phin1(phin1 < -1) = -1;
    rho = (Lrho + Grho)/2 + phin1 * (Grho - Lrho)/2;
    mu = (Lmu + Gmu)/2 + phin1 * (Gmu - Lmu)/2;
    Dmux = (Gmu - Lmu)/2 * Dphix;
    Dmuy = (Gmu - Lmu)/2 * Dphiy;

    DuStarx = Dmatrixx * uStar;
    DuStary = uStar * DmatrixyT;
    DvStarx = Dmatrixx * vStar;
    DvStary = vStar * DmatrixyT;
    DpStarx = Dmatrixx * pStar;
    DpStary = pStar * DmatrixyT;

    lapU = calLaplace(uStar, Dmatrixx2, DmatrixyT2);
    lapV = calLaplace(vStar, Dmatrixx2, DmatrixyT2);


    miuRho = mu ./ rho;
    Dustarx = 2 * Dmux .* DuStarx + Dmuy .* (DvStarx + DuStary); % diffusion term in x
    Dustary = Dmux .* (DvStarx + DuStary) + 2 * Dmuy .* DvStary; % diffusion term in y

    uv31x = uCap ./ delta + (1/rho0 - 1./rho) .* DpStarx - uStar .* DuStarx  ...
        - vStar .* DuStary + (miuRho - nium) .* lapU + Dustarx ./ rho;% + ft1 ./ rho;
    uv31y = vCap ./ delta + (1/rho0 - 1./rho) .* DpStary - uStar .* DvStarx...
        - vStar .* DvStary + (miuRho - nium) .* lapV + Dustary ./ rho + gravity;% + ft2 ./ rho;

    %% Step 2: Pressure correction (Poisson equation)
    % Calculate divergence of intermediate velocity RHS (using full matrices)
    FP = -rho0 * (Dmatrixx * uv31x + uv31y * DmatrixyT);

    midx = rho0 *(uv31x - uStar / delta + nium * lapU);
    midy = rho0 *(uv31y - vStar / delta + nium * lapV);
    Fx = zeros(para_n.nx_all, para_n.ny_all);
    Fy = zeros(para_n.nx_all, para_n.ny_all);
    Fx(1, :) = -midx(1, :) .* wy';
    Fx(end, :) = midx(end, :) .* wy';
    Fy(:, 1) = -midy(:, 1) .* wx;
    Fy(:, end) = midy(:, end) .* wx;
    Fg = Fx + Fy;

    f_solver = FP + Fg ./ mass_diag;

    % Solve Poisson equation for pressure correction 'pre' using Neumann matrices/nodes
    pre_spec = pagemtimes(f_solver, invTyn'); % Use pressure nodes/matrices
    pre_spec = squeeze(tensorprod(invTxn, pre_spec, 2, 1));
    pre_spec = pre_spec ./ poisson_p; % Solve in spectral space
    Pn1 = pagemtimes(pre_spec, Tyn');
    Pn1 = squeeze(tensorprod(Txn, Pn1, 2, 1));
    Pn1 = Pn1 - Pn1(1,1);


    if strcmp(para_d.device, 'gpu')
        wait(Device);
    end

    %% Step 3: Velocity correction (Helmholtz equation)
    % Calculate pressure gradient (full domain, using full matrices)
    GradPre_x = Dmatrixx * Pn1;
    GradPre_y = Pn1 * DmatrixyT;

    % RHS for Helmholtz equation (full domain initially)
    FU = uv31x - GradPre_x ./ rho0;
    FV = uv31y - GradPre_y ./ rho0;

    % Adjust RHS for INTERIOR nodes using pre-calculated boundary contributions
    FU_interior = FU(para_d.freeNodesx, para_d.freeNodesy);% - fubc_contribution;
    % FV_interior = FV(para_d.freeNodesx, :);% - fvbc_contribution; % fvbc is zero here

    % Solve Helmholtz equation for u^(n+1) (INTERIOR nodes)
    uv_spec = pagemtimes(FU_interior, invTy'); % Use INTERIOR velocity matrices
    uv_spec = squeeze(tensorprod(invTx, uv_spec, 2, 1));
    uv_spec = uv_spec ./ helmholtz_u; % Solve in spectral space
    uv_phys = pagemtimes(uv_spec, Tyd');
    uv_phys = squeeze(tensorprod(Txd, uv_phys, 2, 1));
    Un1(para_d.freeNodesx, para_d.freeNodesy) = uv_phys; % Update INTERIOR u

    if strcmp(para_d.device, 'gpu')
        wait(Device);
    end

    % Solve Helmholtz equation for v^(n+1) (INTERIOR nodes)
    % uv_spec = pagemtimes(FV_interior, invTyn'); % Use INTERIOR velocity matrices
    % uv_spec = squeeze(tensorprod(invTx, uv_spec, 2, 1));
    % uv_spec = uv_spec ./ helmholtz_v; % Solve in spectral space
    % uv_phys = pagemtimes(uv_spec, Tyn');
    % uv_phys = squeeze(tensorprod(Tx, uv_phys, 2, 1));
    % Vn1(para_d.freeNodesx, :) = uv_phys; % Update INTERIOR v

    v_hat = invTxn * FV * invTyn';
    v_hat = v_hat ./ helmholtz_v;
    Vn1 = Txn * v_hat * Tyn';
    % Vn1(para_d.freeNodesx, :) = v_free;

    if strcmp(para_d.device, 'gpu')
        wait(Device);
    end

    %% Error analysis and convergence check (based on INTERIOR nodes)
    error_u = norm(Un1 - Un, 'fro');
    error_v = norm(Vn1 - Vn, 'fro');
    total_rms_error = sqrt(error_u^2 + error_v^2);


    if rem(Iter, 10) == 0 % Print less frequently
        fprintf('Iter = %d, error_u = %e\n', Iter, total_rms_error);
        % Output data periodically (e.g., every 100 steps)
        if rem(Iter, freOut) == 0
            saveTec(Iter,pathname,IJK,velName,Un1, Vn1, Pn1,...
                phin1, psi, rho, mu)
        end
    end

    if total_rms_error > 1 || isnan(total_rms_error)
        fprintf('divergence at %d, error_u = %e\n', Iter, total_rms_error);
        break
    end

    % Update velocity for next iteration
    Un_1 = Un;
    Vn_1 = Vn;
    Pn_1 = Pn;

    Un = Un1;
    Vn = Vn1;
    Pn = Pn1;

    phin_1 = phin;
    phin = phin1;

end


time = toc;
fprintf('Total computation time: %f seconds\n', time);

fprintf('=== %d Program Ends ===\n', stage);

