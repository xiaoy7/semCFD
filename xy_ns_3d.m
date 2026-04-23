% xiao yao v2025.4.26
% sovling the 3D navier stokes equation based on spectral element method
% with 3D tensor product and projection method for Lid-Driven Cavity

clc
clear
addpath(genpath("D:\semMatlab")) % Ensure this path is correct

currentTime = datetime('now', 'Format', 'yyyyMMdd_HHmm');
pathname = ['time' char(currentTime)];
copyfile("*.m", pathname)

stage = 1;
fprintf('=== %d Program Starts ===\n', stage);
stage = stage + 1;


messages = "sem 3d re1000";
%% set parameters
RE = 1000; % Reynolds number (adjust as needed, e.g., 100, 400, 1000)
nu = 1/RE; % Kinematic viscosity
dT = 1e-4; % Time step
steps = 90000; % Max number of steps (adjust as needed)
tol = 1e-5; % Convergence tolerance
alpha_helmholtz = 1 / dT; % Coefficient for Helmholtz equation (implicit time term)
varName = 'U,V,W,PRE\n'; % export data


Para_d.Np = 4; % polynomial degree
Np = Para_d.Np;


Para_d.basis = 'SEM'; %'FFT' 'SEM'
if Para_d.basis == 'FFT'
    % FFT setup not typical for lid-driven cavity, assuming SEM
    error('FFT basis not configured for this problem setup.');
else
    Para_d.minx = 0; Para_d.maxx = 1;
    Para_d.miny = 0; Para_d.maxy = 1;
    Para_d.minz = 0; Para_d.maxz = 1; % Add Z dimension (unit cube)

    % number of cells in finite element
    Para_d.Ncellx = 50;
    Para_d.Ncelly = 50;
    Para_d.Ncellz = 50; % Add Z dimension cells
end


if gpuDeviceCount('available') < 1
    Para_d.device = 'cpu';
else
    Para_d.device = 'gpu';
    Para_d.device_id = 1; % Choose appropriate GPU ID
end

% polynomial degree
Npx = Np;
Npy = Np;
Npz = Np; % Add Z dimension degree
Para_d.Npx = Npx;
Para_d.Npy = Npy;
Para_d.Npz = Npz;
Para_d.nx_all = Para_d.Ncellx * Npx + 1;
Para_d.ny_all = Para_d.Ncelly * Npy + 1;
Para_d.nz_all = Para_d.Ncellz * Npz + 1; % Add Z dimension size


Para_d.bc = 'dirichlet'; % Boundary condition type for velocity
Para_n = Para_d; % Use same domain/discretization parameters initially
Para_n.bc = 'neumann'; % Boundary condition type for pressure correction

% Calculate free/boundary nodes based on BC type for all dimensions
% *** CRITICAL: Ensure parameter_bc handles 3D ***
Para_d = parameter_bc(Para_d); % Use the 3D version
Para_n = parameter_bc(Para_n); % Use the 3D version


%% Qk finite element with (k+1)-point Gauss-Lobatto quadrature
fprintf('Laplacian is Q%d spectral element method \n', Np)

if (Np < 2)
    fprintf('It is also classical second order discrete Laplacian \n')
else
    fprintf('It is also a %d-th order accurate finite difference scheme \n', Np + 2)
end

% Get matrices for Velocity (Dirichlet BCs applied internally by cal_matrix2 for INTERIOR solve)
% *** CRITICAL: Ensure cal_matrix2 returns FULL Dmatrix needed for BC contribution ***
[dx, x, Tx, ~, lambda_x, Dmatrixx, ex] = cal_matrix2('x', Para_d);
[dy, y, Ty, ~, lambda_y, Dmatrixy, ey] = cal_matrix2('y', Para_d);
[dz, z, Tz, ~, lambda_z, Dmatrixz, ez] = cal_matrix2('z', Para_d); % Add Z dimension matrices
Dmatrixx = full(Dmatrixx);
Dmatrixy = full(Dmatrixy);
Dmatrixz = full(Dmatrixz);
DmatrixyT = Dmatrixy'; % Transpose for convenience

% Get matrices for Pressure (Neumann BCs - use cal_matrix2_1 or similar)
% *** CRITICAL: Ensure cal_matrix2_1 returns FULL Dmatrix if needed for gradients ***
[~, exp, Txp, lambda_xp] = cal_matrix2_1('x', Para_n, x); % Assuming Dmatrix not needed directly from here
[~, eyp, Typ, lambda_yp] = cal_matrix2_1('y', Para_n, y);
[~, ezp, Tzp, lambda_zp] = cal_matrix2_1('z', Para_n, z); % Add Z dimension pressure matrices


[coordY, coordX, coordZ] = meshgrid(y, x, z); % 3D Grid coordinates (Note order for meshgrid)

coord(:, 1) = coordX(:);
coord(:, 2) = coordY(:);
coord(:, 3) = coordZ(:);
%% initial parameters
fprintf('=== %d initial parameter\n', stage);
stage = stage + 1;

invTx = pinv(Tx); % Inverse transforms for INTERIOR velocity nodes
invTy = pinv(Ty);
invTz = pinv(Tz);
invTxp = pinv(Txp); % Inverse transforms for pressure nodes (Neumann)
invTyp = pinv(Typ);
invTzp = pinv(Tzp);


% Initialize velocity and pressure fields (3D)
[un, vn, wn, pre] = deal(zeros(Para_d.nx_all, Para_d.ny_all, Para_d.nz_all));

% Set initial boundary conditions for velocity
% Lid-driven cavity u=1 at z=maxz (top face)
% Assuming bcNodesz(2) corresponds to the top boundary (z=maxz)
lid_velocity = 1;
un(:, :, Para_d.bcNodesz(2)) = lid_velocity; % Set top face u=1
% Other boundaries (u=0, v=0, w=0) are implicitly handled by solving for interior

un1 = un; % Initialize next step velocity
vn1 = vn;
wn1 = wn;
fprintf('Warning: Initial OUTPUT_Tecplot3D function call is commented out.\n');


% --- Correctly calculate boundary contribution for RHS adjustment ---
% This term accounts for -(nu * Laplacian(u_boundary)) contribution to interior nodes
% where u_boundary contains only the non-homogeneous boundary values.
u_boundary = zeros(Para_d.nx_all, Para_d.ny_all, Para_d.nz_all);
u_boundary(:, :, Para_d.bcNodesz(2)) = lid_velocity; % Set only the non-zero BC part

% Calculate Laplacian of boundary values using FULL differentiation matrices
% *** Ensure Dmatrixx, Dmatrixy, Dmatrixz ARE the full matrices ***
% Apply Dmatrixx along the first dimension (rows) of u_boundary
du_dx = tensorprod(Dmatrixx, u_boundary, 2, 1);
laplacian_u_boundary_x = tensorprod(Dmatrixx, du_dx, 2, 1); % Apply Dmatrixx again

laplacian_u_boundary_y = pagemtimes(u_boundary, DmatrixyT); % Apply Dmatrixy along the second dimension (columns)
laplacian_u_boundary_y = pagemtimes(laplacian_u_boundary_y, DmatrixyT); % d2u/dy2

% Apply Dmatrixz along the third dimension
laplacian_u_boundary_z = derivative_z(Dmatrixz,u_boundary,Para_d.nx_all,...
    Para_d.ny_all, Para_d.nz_all);% du/dz
laplacian_u_boundary_z = derivative_z(Dmatrixz,laplacian_u_boundary_z,...
    Para_d.nx_all, Para_d.ny_all, Para_d.nz_all);% d2u/dz2

laplacian_u_boundary = laplacian_u_boundary_x + laplacian_u_boundary_y + laplacian_u_boundary_z;

% The contribution to subtract from the intermediate RHS FU (interior nodes)
fubc_contribution = -nu * laplacian_u_boundary(Para_d.freeNodesx, ...
    Para_d.freeNodesy, Para_d.freeNodesz);
% v and w have zero Dirichlet BCs, so contributions are zero
% fvbc_contribution = zeros(size(fubc_contribution));
% fwbc_contribution = zeros(size(fubc_contribution));
% generate eigenvalue tensor for Helmholtz solve (velocity - INTERIOR nodes)
% Ensure lambda vectors are column vectors before reshaping
lambda_x_col = lambda_x(:);
lambda_y_col = lambda_y(:);
lambda_z_col = lambda_z(:);
helmholtz_u = reshape(lambda_x_col, [], 1, 1) + reshape(lambda_y_col, 1, [], 1)...
+ reshape(lambda_z_col, 1, 1, []);
helmholtz_u = alpha_helmholtz + nu * helmholtz_u; % Include alpha and viscosity

% generate eigenvalue tensor for Pressure Poisson (Neumann nodes)
% Ensure lambda vectors are column vectors before reshaping
lambda_xp_col = lambda_xp(:);
lambda_yp_col = lambda_yp(:);
lambda_zp_col = lambda_zp(:);
poisson_p = reshape(lambda_xp_col, [], 1, 1) + reshape(lambda_yp_col, 1, [], 1) ...
    + reshape(lambda_zp_col, 1, 1, []);
% Avoid division by zero for the zero eigenvalue (constant pressure mode)
% Lambda3Dp_poisson(Lambda3Dp_poisson < 1e-12) = 1e-12; % Regularization

% Move data to GPU if specified
switch Para_d.device
    case 'gpu'
        fprintf('GPU computation: starting to load matrices/data \n')
        Device = gpuDevice(Para_d.device_id);
        % Velocity matrices (interior)
        Tx = gpuArray(Tx);
        Ty = gpuArray(Ty);
        Tz = gpuArray(Tz);
        helmholtz_u = gpuArray(helmholtz_u);

        invTx = gpuArray(invTx);
        invTy = gpuArray(invTy);
        invTz = gpuArray(invTz);
        % Pressure matrices (Neumann)
        Txp = gpuArray(Txp);
        Typ = gpuArray(Typ);
        Tzp = gpuArray(Tzp);
        poisson_p = gpuArray(poisson_p);

        invTxp = gpuArray(invTxp);
        invTyp = gpuArray(invTyp);
        invTzp = gpuArray(invTzp);
        % Differentiation matrices (assuming full)
        Dmatrixx = gpuArray(Dmatrixx);
        DmatrixyT = gpuArray(DmatrixyT);
        Dmatrixz = gpuArray(Dmatrixz);
        % Fields
        un = gpuArray(un);
        vn = gpuArray(vn);
        wn = gpuArray(wn);
        un1 = gpuArray(un1);
        vn1 = gpuArray(vn1);
        wn1 = gpuArray(wn1);
        pre = gpuArray(pre);
        % Boundary contribution
        fubc_contribution = gpuArray(fubc_contribution);
        % fvbc_contribution = gpuArray(fvbc_contribution);
        % fwbc_contribution = gpuArray(fwbc_contribution); % Add W
end


switch Para_d.device
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
    % Need full derivatives using Dmatrix
    DUnx = pagemtimes(Dmatrixx, un); % d(un)/dx
    DVnx = pagemtimes(Dmatrixx, vn); % d(vn)/dx
    DWnx = pagemtimes(Dmatrixx, wn); % d(wn)/dx
    DUny = pagemtimes(un, DmatrixyT);      % d(un)/dy
    DVny = pagemtimes(vn, DmatrixyT);      % d(vn)/dy
    DWny = pagemtimes(wn, DmatrixyT);      % d(wn)/dy

    DUnz = derivative_z(Dmatrixz,un,Para_d.nx_all, Para_d.ny_all, Para_d.nz_all);% d(un)/dz
    DVnz = derivative_z(Dmatrixz,vn,Para_d.nx_all, Para_d.ny_all, Para_d.nz_all);% d(vn)/dz
    DWnz = derivative_z(Dmatrixz,wn,Para_d.nx_all, Para_d.ny_all, Para_d.nz_all);% d(wn)/dz

    convection_u = un .* DUnx + vn .* DUny + wn .* DUnz;
    convection_v = un .* DVnx + vn .* DVny + wn .* DVnz;
    convection_w = un .* DWnx + vn .* DWny + wn .* DWnz; % Add W convection

    % Intermediate velocity RHS (explicit convection, implicit time term)
    U_star_rhs = un / dT - convection_u;
    V_star_rhs = vn / dT - convection_v;
    W_star_rhs = wn / dT - convection_w; % Add W RHS
    % Note: Diffusion term is handled implicitly in the Helmholtz solve (Step 3)

    %% Step 2: Pressure correction (Poisson equation)
    % Calculate divergence of intermediate velocity RHS (using full matrices)
    Div_U_star_rhs_x = pagemtimes(Dmatrixx, U_star_rhs); % d(U*)/dx
    Div_V_star_rhs_y = pagemtimes(V_star_rhs, DmatrixyT);      % d(V*)/dy
    Div_W_star_rhs_z = derivative_z(Dmatrixz,W_star_rhs,Para_d.nx_all, ...
        Para_d.ny_all, Para_d.nz_all);% d(W*)/dz


    % RHS for pressure Poisson equation (Neumann BCs assumed for pressure)
    FP = -(Div_U_star_rhs_x + Div_V_star_rhs_y + Div_W_star_rhs_z); % Add W term

    % Solve Poisson equation for pressure correction 'pre' using Neumann matrices/nodes
    % Apply transforms in 3D
    pre_spec = tensorprod(FP(Para_n.freeNodesx, Para_n.freeNodesy,...
        Para_n.freeNodesz), invTzp', 3, 1); % Transform Z
    pre_spec = pagemtimes(pre_spec, invTyp'); % Transform Y
    pre_spec = squeeze(tensorprod(invTxp, pre_spec, 2, 1)); % Transform X

    pre_spec = pre_spec ./ poisson_p; % Solve in spectral space

    % Apply inverse transforms in 3D
    pre_phys = tensorprod(pre_spec, Tzp', 3, 1); % Inverse Transform Z
    pre_phys = pagemtimes(pre_phys, Typ'); % Inverse Transform Y
    pre_phys = squeeze(tensorprod(Txp, pre_phys, 2, 1)); % Inverse Transform X

    % Update pressure field (handle boundary nodes if necessary, depends on Neumann impl.)
    pre(Para_n.freeNodesx, Para_n.freeNodesy, Para_n.freeNodesz) = pre_phys;
    % Optional: Set mean pressure to zero if needed for pure Neumann problem
    % pre = pre - mean(pre(Parameterp.freeNodesx, Parameterp.freeNodesy, Parameterp.freeNodesz), 'all');


    if strcmp(Para_d.device, 'gpu')
        wait(Device);
    end

    %% Step 3: Velocity correction (Helmholtz equation)
    % Calculate pressure gradient (full domain, using full matrices)
    GradPre_x = pagemtimes(Dmatrixx, pre); % d(pre)/dx
    GradPre_y = pagemtimes(pre, DmatrixyT);% d(pre)/dy
    GradPre_z = derivative_z(Dmatrixz,pre,Para_d.nx_all, Para_d.ny_all, Para_d.nz_all);% d(pre)/dz

    % RHS for Helmholtz equation (full domain initially)
    FU = U_star_rhs - GradPre_x;
    FV = V_star_rhs - GradPre_y;
    FW = W_star_rhs - GradPre_z; % Add W RHS

    % Adjust RHS for INTERIOR nodes using pre-calculated boundary contributions
    FU_interior = FU(Para_d.freeNodesx, Para_d.freeNodesy, Para_d.freeNodesz) - fubc_contribution;
    FV_interior = FV(Para_d.freeNodesx, Para_d.freeNodesy, Para_d.freeNodesz);% - fvbc_contribution;
    FW_interior = FW(Para_d.freeNodesx, Para_d.freeNodesy, Para_d.freeNodesz);% - fwbc_contribution; % Add W

    % --- Solve Helmholtz for u^(n+1) (INTERIOR nodes) ---
    uvw_spec = tensorprod(FU_interior, invTz', 3, 1); % Transform Z
    uvw_spec = pagemtimes(uvw_spec, invTy'); % Transform Y
    uvw_spec = squeeze(tensorprod(invTx, uvw_spec, 2, 1)); % Transform X

    uvw_spec = uvw_spec ./ helmholtz_u; % Solve in spectral space

    uvw_phys = tensorprod(uvw_spec, Tz', 3, 1); % Inverse Transform Z
    uvw_phys = pagemtimes(uvw_phys, Ty'); % Inverse Transform Y
    uvw_phys = squeeze(tensorprod(Tx, uvw_phys, 2, 1)); % Inverse Transform X
    un1(Para_d.freeNodesx, Para_d.freeNodesy, Para_d.freeNodesz) = uvw_phys; % Update INTERIOR u

    if strcmp(Para_d.device, 'gpu'); wait(Device); end

    % --- Solve Helmholtz for v^(n+1) (INTERIOR nodes) ---
    uvw_spec = tensorprod(FV_interior, invTz', 3, 1); % Transform Z
    uvw_spec = pagemtimes(uvw_spec, invTy'); % Transform Y
    uvw_spec = squeeze(tensorprod(invTx, uvw_spec, 2, 1)); % Transform X

    uvw_spec = uvw_spec ./ helmholtz_u; % Solve in spectral space

    uvw_phys = tensorprod(uvw_spec, Tz', 3, 1); % Inverse Transform Z
    uvw_phys = pagemtimes(uvw_phys, Ty'); % Inverse Transform Y
    uvw_phys = squeeze(tensorprod(Tx, uvw_phys, 2, 1)); % Inverse Transform X
    vn1(Para_d.freeNodesx, Para_d.freeNodesy, Para_d.freeNodesz) = uvw_phys; % Update INTERIOR v

    if strcmp(Para_d.device, 'gpu'); wait(Device); end

    % --- Solve Helmholtz for w^(n+1) (INTERIOR nodes) ---
    uvw_spec = tensorprod(FW_interior, invTz', 3, 1); % Transform Z
    uvw_spec = pagemtimes(uvw_spec, invTy'); % Transform Y
    uvw_spec = squeeze(tensorprod(invTx, uvw_spec, 2, 1)); % Transform X

    uvw_spec = uvw_spec ./ helmholtz_u; % Solve in spectral space

    uvw_phys = tensorprod(uvw_spec, Tz', 3, 1); % Inverse Transform Z
    uvw_phys = pagemtimes(uvw_phys, Ty'); % Inverse Transform Y
    uvw_phys = squeeze(tensorprod(Tx, uvw_phys, 2, 1)); % Inverse Transform X
    wn1(Para_d.freeNodesx, Para_d.freeNodesy, Para_d.freeNodesz) = uvw_phys; % Update INTERIOR w

    if strcmp(Para_d.device, 'gpu'); wait(Device); end

    % Boundary values of un1, vn1, wn1 remain fixed as they were not part of the solve
    % Re-assert boundary conditions explicitly for clarity
    % Top face (lid, z=maxz)
    % un1(:, :, Parameteru.bcNodesz(2)) = lid_velocity;
    % vn1(:, :, Parameteru.bcNodesz(2)) = 0;
    % wn1(:, :, Parameteru.bcNodesz(2)) = 0;
    % % Bottom face (z=minz)
    % un1(:, :, Parameteru.bcNodesz(1)) = 0;
    % vn1(:, :, Parameteru.bcNodesz(1)) = 0;
    % wn1(:, :, Parameteru.bcNodesz(1)) = 0;
    % % Side faces (x-boundaries)
    % un1(Parameteru.bcNodesx(1), :, :) = 0;
    % vn1(Parameteru.bcNodesx(1), :, :) = 0;
    % wn1(Parameteru.bcNodesx(1), :, :) = 0;
    % un1(Parameteru.bcNodesx(2), :, :) = 0;
    % vn1(Parameteru.bcNodesx(2), :, :) = 0;
    % wn1(Parameteru.bcNodesx(2), :, :) = 0;
    % % Side faces (y-boundaries)
    % un1(:, Parameteru.bcNodesy(1), :) = 0;
    % vn1(:, Parameteru.bcNodesy(1), :) = 0;
    % wn1(:, Parameteru.bcNodesy(1), :) = 0;
    % un1(:, Parameteru.bcNodesy(2), :) = 0;
    % vn1(:, Parameteru.bcNodesy(2), :) = 0;
    % wn1(:, Parameteru.bcNodesy(2), :) = 0;


    %% Error analysis and convergence check
    error_u = norm(un1(:) - un(:));
    % error_v = norm(vn1(:) - vn(:));
    % error_w = norm(wn1(:) - wn(:));

    % num_interior_nodes = numel(Parameteru.freeNodesx) * numel(Parameteru.freeNodesy) * numel(Parameteru.freeNodesz);
    % Avoid division by zero if num_interior_nodes is zero (shouldn't happen in 3D)
    % if num_interior_nodes > 0
    %     total_rms_error = sqrt(error_u^2 + error_v^2 + error_w^2) / sqrt(num_interior_nodes); % Include W
    % else
    %     total_rms_error = sqrt(error_u^2 + error_v^2 + error_w^2);
    % end

    if isnan(error_u) || error_u > 100
        fprintf('Iter = %d, error_u = %e\n', Iter, error_u);
        break
    end
    if rem(Iter, 1) == 0 % Print less frequently
        fprintf('Iter = %d, error_u = %e\n', Iter, error_u);
    end

    % Output data periodically (e.g., every 100 steps)
    % if rem(Iter, 100) == 0
    %
    %     fprintf('Iter = %d, RMS Vel Change = %e\n', Iter, error_u);
    %     OUTPUT_Tecplot2(Iter, coord, Parameteru.nx_all, Parameteru.ny_all, ...
    %         Parameteru.nz_all, varName, ...
    %         un(:), vn(:), wn(:), pre(:));
    % end


    if error_u <= tol
        fprintf('Convergence reached at iteration %d\n', Iter);
        fprintf('error_u = %e\n', error_u);
        OUTPUT_Tecplot2(Iter, coord, Para_d.nx_all, Para_d.ny_all, ...
            Para_d.nz_all, varName, un(:), vn(:), wn(:), pre(:));
        break;
    end

    % Update velocity for next iteration
    un = un1;
    vn = vn1;
    wn = wn1; % Add W update

end



time = toc;
fprintf('Total computation time: %f seconds\n', time);

fprintf('=== %d Program Ends ===\n', stage);
sending_to_emil2(Iter,steps,messages)
save flow.mat

% --- Helper function assumed to exist ---
% function Parameter = parameter_bc(Parameter) % Must handle 3D
% function [d, x, T, H, lambda, Dmatrix, e] = cal_matrix2(direction, Parameter) % Assumes Dmatrix is FULL
% function [d, e, T, lambda, Dmatrix, ex] = cal_matrix2_1(direction, Parameter, x) % Assumes Dmatrix is FULL
% function OUTPUT_Tecplot3D(Iter, nx, ny, nz, varName, X, Y, Z, U, V, W, P) % Example signature