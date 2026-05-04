% Sovling cahn hilliard equation based on spectral element method
% with 2D tensor product and projection method
% Xiao Yao v2025.8.3


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
% set parameters

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
eta = Cn * LL1;  % Smoothing parameter for tanh



M0 = 5e-5;
gamma1 = M0; % mobility
gamma0 = 1.5;
bigs = 1.1 * eta ^ 2 * sqrt(4 * gamma0 / (gamma1 * delta));
aa = 1 - gamma0 / (gamma1 * delta) * 4 * eta^ 4 / bigs^ 2;
alpha = -bigs / (2 * eta ^ 2) * (1 + sqrt(aa));
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
neumann = dirichlet; % Use same domain/discretization parameters initially
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
[dxp, exp, Txp, lambda_xn] = cal_matrix2_1('x', neumann,x);
[dyp, eyp, Typ, lambda_yn] = cal_matrix2_1('y', neumann,y);

[coordY, coordX] = meshgrid(y, x); % Grid coordinates for output/plotting
%%
% initial parameters
fprintf('=== %d initial parameter\n', stage); %[output:20a42081]
stage = stage + 1;

invTx = pinv(Tx);   % Inverse transforms for INTERIOR velocity nodes
invTy = pinv(Ty);
invTxp = pinv(Txp); % Inverse transforms for pressure nodes (Neumann)
invTyp = pinv(Typ);


% Initialize velocity and pressure fields
[pre] = deal(zeros(dirichlet.nx_all, dirichlet.ny_all));

[un, vn] = deal(0.1 * ones(dirichlet.nx_all, dirichlet.ny_all));

un1 = un; % Initialize next step velocity
vn1 = vn;
un_1 = un; % Previous step velocity (for time-stepping)
vn_1 = vn;
pn = pre; % Initialize pressure field
pn_1 = pre; % Previous step pressure (for time-stepping)

% use the tanh function to initialize the phase field

% Initialize phase field (Cahn-Hilliard) with a circle
phin1 = -tanh((sqrt((coordX - Cx) .^ 2 + (coordY - Cy) .^ 2) - radius) ./ (sqrt(2) * eta));
phin = phin1; % Previous step phase field
phin_1 = phin1; % Previous step phase field for time-stepping


OUTPUT_Tecplot2D4(0,pathname, dirichlet.ny_all, dirichlet.nx_all, varName, ...
    coordX(:), coordY(:),  un(:), vn(:), pre(:),phin1(:),phin1(:));


% Move data to GPU if specified
switch dirichlet.device %[output:group:9b467188]
    case 'gpu'
        fprintf('GPU computation: starting to load matrices/data \n') %[output:9f97c27c]
        Device = gpuDevice(dirichlet.device_id);
        % Velocity matrices (interior)
        Tx = gpuArray(Tx);
        Ty = gpuArray(Ty);
        lambda_xd = gpuArray(lambda_xd);
        lambda_yd = gpuArray(lambda_yd);
        ex = gpuArray(ex);
        ey = gpuArray(ey);
        invTx = gpuArray(invTx);
        invTy = gpuArray(invTy);
        % Pressure matrices (Neumann)
        Txp = gpuArray(Txp);
        Typ = gpuArray(Typ);
        lambda_xn = gpuArray(lambda_xn);
        lambda_yn = gpuArray(lambda_yn);
        exp = gpuArray(exp);
        eyp = gpuArray(eyp);
        invTxp = gpuArray(invTxp);
        invTyp = gpuArray(invTyp);
        % Differentiation matrices (assuming full)
        Dmatrixx = gpuArray(Dmatrixx);
        DmatrixyT = gpuArray(DmatrixyT); % Note: Dmatrixy is transposed before GPU transfer
        % Fields
        un = gpuArray(un);
        vn = gpuArray(vn);
        un1 = gpuArray(un1);
        vn1 = gpuArray(vn1);
        pre = gpuArray(pre);
        phin1 = gpuArray(phin1);
        phin = gpuArray(phin);
        phin_1 = gpuArray(phin_1);
        % Boundary contribution

        % fvbc_contribution = gpuArray(fvbc_contribution); % Add if needed
end
Dmatrixx2 = Dmatrixx * Dmatrixx;
DmatrixyT2 = DmatrixyT * DmatrixyT;
nabla2 = Dmatrixx2 + DmatrixyT2;


% generate eigenvalue tensor for Helmholtz solve (velocity - INTERIOR nodes)

% Correct eigenvalue tensor construction for Laplacian (tensor product grid)
laplace2d = bsxfun(@plus, lambda_xd, lambda_yd');
laplace2n = bsxfun(@plus, lambda_xn, lambda_yn');
% Avoid division by zero for the zero eigenvalue (constant pressure mode)
laplace2n(laplace2n < 1e-12) = 1e-12; % Regularization

helmholtzPhi = alpha - laplace2n; % For phase field (Cahn-Hilliard)
helmholtzPsi = (alpha + bigs / eta ^ 2) + laplace2n; % For phase field (Cahn-Hilliard)


switch dirichlet.device %[output:group:0c101d2a]
    case 'gpu'
        wait(Device);
        fprintf('GPU computation: loading finished and GPU computing started \n') %[output:60f3a9b1]
end %[output:group:0c101d2a]
%%
% online computation
tic;


fprintf('=== %d start time stepping\n', stage); %[output:9f2c7bf4]
stage = stage + 1;
for Iter = 1:steps

    %%
    % Step 1: Calculate intermediate velocity RHS (convection + previous diffusion/pressure)
    %  Calculate convective terms (e.g., using Adams-Bashforth 1st order)
    uStar = 2 * un - un_1; %2 * u_n - u_n-1
    vStar = 2 * vn - vn_1; %2 * v_n - v_n-1
    pStar = 2 * pn - pn_1;
    phiStar = 2 * phin - phin_1;

    uCap = 2 * un - 0.5 * un_1;
    vCap = 2 * vn - 0.5 * vn_1;
    phiCap = 2 * phin - 0.5 * phin_1;



    %% phase field
    %  $\\Delta \\psi -\\left(\\alpha +\\frac{S}{\\eta^2 }\\right)\\psi =Q$


    Q1 = (uStar .* (Dmatrixx * phiStar) + vStar .* (phiStar * DmatrixyT) - phiCap / delta) / gamma1;
    Q2 = (phiStar .^ 2 - 1 - bigs) / eta ^ 2 .* phiStar;
    % Fpsi = Q1 - laplace2n .* Q2; % type 1
    Fpsi = Q1 - Dmatrixx2 * Q2 - Q2 * DmatrixyT2; % type 2
    
    % Proper 2D spectral transform for psi
    psi_hat = invTxp * Fpsi * invTyp';
    psi_hat = psi_hat ./ helmholtzPsi;
    psi = Txp * psi_hat * Typ';

    % Proper 2D spectral transform for phi
    phi_hat = invTxp * psi * invTyp';
    phi_hat = phi_hat ./ helmholtzPhi;
    phin1 = Txp * phi_hat * Typ';




    if strcmp(dirichlet.device, 'gpu')
        wait(Device);
    end



    %%
    % Error analysis and convergence check (based on INTERIOR nodes)
    error_phi = norm(phin1 - phin, 'fro');
    if error_phi > 10
        fprintf('divergence at Iter = %d, norm2 = %e\n', Iter, error_phi); %[output:7c2df20d]
        if strcmp(dirichlet.device,'gpu')
            un_out=gather(un1);
            vn_out=gather(vn1);
            pre_out=gather(pre);
            phi_out = gather(phin1);
            psi_out = gather(psi);
        else
            un_out=un1;
            vn_out=vn1;
            pre_out=pre;
            phi_out = phin1;
            psi_out = psi;
        end
        OUTPUT_Tecplot2D4(Iter,pathname, dirichlet.ny_all, dirichlet.nx_all, varName, ...
            coordX(:), coordY(:),  un_out(:), vn_out(:), pre_out(:),phi_out(:),psi_out(:));
        break
    end



    if rem(Iter, frePrint) == 0 % Print less frequently
        fprintf('Iter = %d, norm2 = %e\n', Iter, error_phi);
    end

    % Output data periodically (e.g., every 100 steps)
    if rem(Iter, freOut) == 0
        if strcmp(dirichlet.device,'gpu')
            un_out=gather(un1);
            vn_out=gather(vn1);
            pre_out=gather(pre);
            phi_out = gather(phin1);
            psi_out = gather(psi);
        else
            un_out=un1;
            vn_out=vn1;
            pre_out=pre;
            phi_out = phin1;
            psi_out = psi;
        end

        OUTPUT_Tecplot2D4(Iter, pathname, dirichlet.ny_all, dirichlet.nx_all, varName, ...
            coordX(:), coordY(:),  un_out(:), vn_out(:), pre_out(:),phi_out(:),psi_out(:));
    end


    % Update velocity for next iteration
    un_1 = un;
    vn_1 = vn;
    phin_1 = phin;

    un = un1;
    vn = vn1;
    phin = phin1;

    if(mod(Iter,freMat) == 0 )
        filename = ['flow',num2str(Iter)];
        cd(pathname)
        save(filename);
        cd(currentLocation)
    end
end %[output:group:8d63b58c]
cd(pathname)
save flow
cd(currentLocation)


time = toc;
fprintf('Total computation time: %f seconds\n', time);

fprintf('=== %d Program Ends ===\n', stage);


