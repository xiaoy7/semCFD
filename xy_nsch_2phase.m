% Coupled Cahn–Hilliard / Navier–Stokes solver for two-phase flow
% Spectral-element discretisation in tensor-product SEM basis
% Combines functionality from TEST.m (two-phase CH + refined projection)
% and xy_ns_2d.m (velocity BC handling / SEM infrastructure)

clc;
clear;
addpath(genpath("D:\semMatlab")); % Ensure this path exists on local machine

% -------------------------------------------------------------------------
% I/O setup
currentLocation = pwd;
cd ..;
fileLocation = pwd;
currentTime = datetime('now', 'Format', 'yyyyMMdd_HHmm_ss');
pathname = [fileLocation, '\time' char(currentTime)];
cd(currentLocation);
copyfile("*.m", pathname);

stage = 1;
fprintf('=== %d Coupled NS-CH Solver Starts ===\n', stage);
stage = stage + 1;

% -------------------------------------------------------------------------
% Numerical parameters

frePrint = 10;
freOut = frePrint * 10;
freMat = freOut * 5;

delta = 1e-4;          % Time step
steps = 2000;          % Maximum iterations
tol = 5e-5;            % Convergence tolerance
varName = 'U,V,PRE,phi,psi,rho,mu\n';

% SEM discretisation
domain.Np = 4;
domain.basis = 'SEM';
domain.minx = -0.5;
domain.maxx = 0.5;
domain.miny = 0.0;
domain.maxy = 1.0;
domain.Ncellx = 30;
domain.Ncelly = 30;

% Physical parameters
LL1 = 1.0;
Cn = 0.015;
eta = Cn * LL1;
radius = 0.15;
Cx = 0.0;
Cy = 0.5;

rhoh = 1000;
rhol = 1;
miuh = 10;
miul = 0.1;
rho0 = min(rhol, rhoh);
nium = 0.1;

tension = 1.0;
gravity = 0.0;
lambda = 3 * eta * tension / (2 * sqrt(2));

M0 = 5e-5;
gamma1 = M0;
gamma0 = 1.5;
bigs = 1.1 * eta^2 * sqrt(4 * gamma0 / (gamma1 * delta));
aa = 1 - gamma0 / (gamma1 * delta) * 4 * eta^4 / bigs^2;
alphaCH = -bigs / (2 * eta^2) * (1 + sqrt(aa));
if aa < 0
    warning('Alpha requirement not met – adjust parameters.');
end

% GPU selection
if gpuDeviceCount('available') < 1
    domain.device = 'cpu';
else
    domain.device = 'gpu';
    domain.device_id = 1;
end

% SEM derived sizes
domain.Npx = domain.Np;
domain.Npy = domain.Np;
domain.nx_all = domain.Ncellx * domain.Np + 1;
domain.ny_all = domain.Ncelly * domain.Np + 1;

domain.bc = 'dirichlet';
neumann = domain;
neumann.bc = 'neumann';

domain = parameter_bc2d(domain);
neumann = parameter_bc2d(neumann);

fprintf('Laplacian discretised with Q%d SEM\n', domain.Np);
if domain.Np < 2
    fprintf('Equivalent to 2nd order centered stencil.\n');
else
    fprintf('Spatial accuracy corresponds to order %d FD.\n', domain.Np + 2);
end

% -------------------------------------------------------------------------
% Assemble SEM operators
[dx_d, x_d, Tx_d, ~, lambda_x_d, Dmatrixx_d, ~] = cal_matrix2('x', domain);
[dy_d, y_d, Ty_d, ~, lambda_y_d, Dmatrixy_d, ~] = cal_matrix2('y', domain);
Dmatrixy_dT = Dmatrixy_d';

[~, x_n, Tx_n, ~, lambda_x_n, Dmatrixx_n, ~] = cal_matrix2('x', neumann);
[~, y_n, Ty_n, ~, lambda_y_n, Dmatrixy_n, ~] = cal_matrix2('y', neumann);
Dmatrixy_nT = Dmatrixy_n';

[~, ~, Txp, lambda_xp] = cal_matrix2_1('x', neumann, x_n);
[~, ~, Typ, lambda_yp] = cal_matrix2_1('y', neumann, y_n);

[coordY, coordX] = meshgrid(y_d, x_d);
wx = assemble_sem_weights(domain.Npx, domain.Ncellx, domain.minx, domain.maxx);
wy = assemble_sem_weights(domain.Npy, domain.Ncelly, domain.miny, domain.maxy);
wx = wx(:);
wy = wy(:);

invTx = pinv(Tx_d);
invTy = pinv(Ty_d);
invTxp = pinv(Txp);
invTyp = pinv(Typ);

laplace2d = bsxfun(@plus, lambda_x_d, lambda_y_d');
laplace2n = bsxfun(@plus, lambda_x_n, lambda_y_n');
laplace2n(laplace2n < 1e-12) = 1e-12;

helmholtzPhi = alphaCH - laplace2n;
helmholtzPsi = (alphaCH + bigs / eta^2) + laplace2n;
poisson_p = laplace2n;
helmholtz_u = 3 / (2 * nium * delta) + laplace2d;
helmholtz_v = helmholtz_u;

Dmatrixx2_n = Dmatrixx_n * Dmatrixx_n;
Dmatrixy_nT2 = Dmatrixy_nT * Dmatrixy_nT;

% -------------------------------------------------------------------------
% Boundary data (optional lid-driven top wall)

u_bc = zeros(domain.nx_all, domain.ny_all);
v_bc = zeros(domain.nx_all, domain.ny_all);
lid_velocity = 0.0;
u_bc(:, domain.bcNodesy(2)) = lid_velocity;

% -------------------------------------------------------------------------
% Initial fields
[pre, un, vn, un1, vn1, un_1, vn_1, pn, pn_1] = deal(zeros(domain.nx_all, domain.ny_all));
un = u_bc;
un1 = un;
un_1 = un;

phin = -tanh((sqrt((coordX - Cx).^2 + (coordY - Cy).^2) - radius) ./ (sqrt(2) * eta));
phin1 = phin;
phin_1 = phin;

psi = zeros(domain.nx_all, domain.ny_all);

OUTPUT_Tecplot2D4(0, pathname, domain.ny_all, domain.nx_all, varName, ...
    coordX(:), coordY(:), un(:), vn(:), pre(:), phin(:), psi(:), rhoh * ones(numel(phin), 1), miuh * ones(numel(phin), 1));

if strcmp(domain.device, 'gpu')
    fprintf('GPU computation: transferring matrices.\n');
    Device = gpuDevice(domain.device_id);
    Tx_d = gpuArray(Tx_d); Ty_d = gpuArray(Ty_d);
    Tx_n = gpuArray(Tx_n); Ty_n = gpuArray(Ty_n);
    Txp = gpuArray(Txp);   Typ = gpuArray(Typ);
    invTx = gpuArray(invTx); invTy = gpuArray(invTy);
    invTxp = gpuArray(invTxp); invTyp = gpuArray(invTyp);
    Dmatrixx_d = gpuArray(Dmatrixx_d); Dmatrixy_dT = gpuArray(Dmatrixy_dT);
    Dmatrixx_n = gpuArray(Dmatrixx_n); Dmatrixy_nT = gpuArray(Dmatrixy_nT);
    Dmatrixx2_n = gpuArray(Dmatrixx2_n); Dmatrixy_nT2 = gpuArray(Dmatrixy_nT2);
    helmholtz_u = gpuArray(helmholtz_u); helmholtz_v = gpuArray(helmholtz_v);
    helmholtzPhi = gpuArray(helmholtzPhi); helmholtzPsi = gpuArray(helmholtzPsi);
    poisson_p = gpuArray(poisson_p);
    laplace2n = gpuArray(laplace2n);
    wx = gpuArray(wx); wy = gpuArray(wy);
    un = gpuArray(un); vn = gpuArray(vn);
    un1 = gpuArray(un1); vn1 = gpuArray(vn1);
    un_1 = gpuArray(un_1); vn_1 = gpuArray(vn_1);
    pn = gpuArray(pn); pn_1 = gpuArray(pn_1);
    pre = gpuArray(pre);
    phin = gpuArray(phin); phin1 = gpuArray(phin1); phin_1 = gpuArray(phin_1);
    psi = gpuArray(psi);
    u_bc = gpuArray(u_bc); v_bc = gpuArray(v_bc);
    coordX = gpuArray(coordX); coordY = gpuArray(coordY);
end

Reconstrct = @(p1, p2, phi) (p1 + p2) / 2 + (p1 - p2) .* phi / 2;

fprintf('=== %d Time stepping starts ===\n', stage);
stage = stage + 1;

for Iter = 1:steps
    uStar = 2 * un - un_1;
    vStar = 2 * vn - vn_1;
    pStar = 2 * pn - pn_1;
    phiStar = 2 * phin - phin_1;

    uCap = 2 * un - 0.5 * un_1;
    vCap = 2 * vn - 0.5 * vn_1;
    phiCap = 2 * phin - 0.5 * phin_1;

    % ----------------------- Phase field update --------------------------
    Q1 = (uStar .* (Dmatrixx_n * phiStar) + vStar .* (phiStar * Dmatrixy_nT) - phiCap / delta) / gamma1;
    Q2 = (phiStar.^2 - 1 - bigs) / eta^2 .* phiStar;
    lapQ2 = Dmatrixx2_n * Q2 + Q2 * Dmatrixy_nT2;
    Fpsi = Q1 - lapQ2;

    psi_hat = invTxp * Fpsi * invTyp';
    psi_hat = psi_hat ./ helmholtzPsi;
    psi = Txp * psi_hat * Typ';

    phi_hat = invTxp * psi * invTyp';
    phi_hat = phi_hat ./ helmholtzPhi;
    phin1 = Txp * phi_hat * Typ';

    phin1 = max(min(phin1, 1), -1);
    rho = Reconstrct(rhoh, rhol, phin1);
    mu = Reconstrct(miuh, miul, phin1);

    Dphix = Dmatrixx_n * phin1;
    Dphiy = phin1 * Dmatrixy_nT;
    ftx = lambda * psi .* Dphix;
    fty = lambda * psi .* Dphiy;

    Dmux = 0.5 * (miuh - miul) * Dphix;
    Dmuy = 0.5 * (miuh - miul) * Dphiy;

    % ------------------ Momentum RHS (Helmholtz) ------------------------
    DuStarx = Dmatrixx_n * uStar;
    DuStary = uStar * Dmatrixy_nT;
    DvStarx = Dmatrixx_n * vStar;
    DvStary = vStar * Dmatrixy_nT;

    vor0 = Dmatrixx_n * vn_1 - un_1 * Dmatrixy_nT;
    vor1 = Dmatrixx_n * vn - un * Dmatrixy_nT;
    vor = 2 * vor1 - vor0;

    miuRho = mu ./ rho;
    gradMuRhox = Dmatrixx_n * miuRho;
    gradMuRhoy = miuRho * Dmatrixy_nT;

    Dustarx = 2 * Dmux .* DuStarx + Dmuy .* (DvStarx + DuStary);
    Dustary = Dmux .* (DvStarx + DuStary) + 2 * Dmuy .* DvStary;

    Gx_core = uCap ./ delta + ftx ./ rho ...
        - uStar .* DuStarx - vStar .* DuStary + Dustarx ./ rho ...
        - (Dmatrixx_n * pStar) ./ rho;

    Gy_core = gravity + vCap ./ delta + fty ./ rho ...
        - uStar .* DvStarx - vStar .* DvStary + Dustary ./ rho ...
        - (pStar * Dmatrixy_nT) ./ rho;

    crossGradVelocity_x = gradMuRhoy .* vor;
    crossGradVelocity_y = -gradMuRhox .* vor;

    muRhoDiff = miuRho - nium;
    muRhoDiffOmega = muRhoDiff .* vor;
    corr_div_x = (muRhoDiffOmega * Dmatrixy_nT) / nium;
    corr_div_y = -(Dmatrixx_n * muRhoDiffOmega) / nium;

    FU = (Gx_core + crossGradVelocity_x) / nium + corr_div_x;
    FV = (Gy_core + crossGradVelocity_y) / nium + corr_div_y;

    % bc_fu_bottom = muRhoDiffOmega(:, 1) / (nium * wx(1));
    % bc_fu_bottom([1 end]) = bc_fu_bottom([1 end]) * 0.5;
    % FU(:, 1) = FU(:, 1) + bc_fu_bottom;
    % 
    % bc_fu_top = -muRhoDiffOmega(:, end) / (nium * wx(end));
    % bc_fu_top([1 end]) = bc_fu_top([1 end]) * 0.5;
    % FU(:, end) = FU(:, end) + bc_fu_top;
    % 
    % bc_fv_left = -muRhoDiffOmega(1, :) / (nium * wy(1));
    % bc_fv_left([1 end]) = bc_fv_left([1 end]) * 0.5;
    % FV(1, :) = FV(1, :) + bc_fv_left;
    % 
    % bc_fv_right = muRhoDiffOmega(end, :) / (nium * wy(end));
    % bc_fv_right([1 end]) = bc_fv_right([1 end]) * 0.5;
    % FV(end, :) = FV(end, :) + bc_fv_right;

    FU_interior = FU(domain.freeNodesx, domain.freeNodesy);
    FV_interior = FV(domain.freeNodesx, domain.freeNodesy);

    u_hat = invTx * FU_interior * invTy';
    u_hat = u_hat ./ helmholtz_u;
    u_phys = Tx_d * u_hat * Ty_d';

    v_hat = invTx * FV_interior * invTy';
    v_hat = v_hat ./ helmholtz_v;
    v_phys = Tx_d * v_hat * Ty_d';

    u_tilde = zeros(size(un));
    v_tilde = zeros(size(vn));
    u_tilde(domain.freeNodesx, domain.freeNodesy) = u_phys;
    v_tilde(domain.freeNodesx, domain.freeNodesy) = v_phys;
    u_tilde(domain.bcNodesx, :) = u_bc(domain.bcNodesx, :);
    u_tilde(:, domain.bcNodesy) = u_bc(:, domain.bcNodesy);
    v_tilde(domain.bcNodesx, :) = v_bc(domain.bcNodesx, :);
    v_tilde(:, domain.bcNodesy) = v_bc(:, domain.bcNodesy);

    % --------------------- Pressure projection --------------------------
    div_tilde = Dmatrixx_n * u_tilde + v_tilde * Dmatrixy_nT;
    rhs_p = (gamma0 * rho0 / delta) * div_tilde;
    rhs_p = rhs_p - mean(rhs_p(:));

    phi_hat = invTxp * rhs_p * invTyp';
    phi_hat = phi_hat ./ poisson_p;
    phi = Txp * phi_hat * Typ';
    phi = phi - mean(phi(:));

    pn1 = pStar + phi;
    pre = pn1;

    gradPhi_x = Dmatrixx_n * phi;
    gradPhi_y = phi * Dmatrixy_nT;
    un1 = u_tilde - (delta / gamma0) * (gradPhi_x ./ rho);
    vn1 = v_tilde - (delta / gamma0) * (gradPhi_y ./ rho);
    un1(domain.bcNodesx, :) = u_bc(domain.bcNodesx, :);
    un1(:, domain.bcNodesy) = u_bc(:, domain.bcNodesy);
    vn1(domain.bcNodesx, :) = v_bc(domain.bcNodesx, :);
    vn1(:, domain.bcNodesy) = v_bc(:, domain.bcNodesy);

    if strcmp(domain.device, 'gpu')
        wait(Device);
    end

    % ------------------------ Diagnostics --------------------------------
    vel_change = norm(un1 - un, 'fro') + norm(vn1 - vn, 'fro');
    if vel_change > 10
        fprintf("divergence at 'Iter = %d\n", Iter)
          OUTPUT_Tecplot2D4(Iter, pathname, domain.ny_all, domain.nx_all, varName, ...
                coordX(:), coordY(:), un1(:), vn1(:), pre(:), phin1(:), psi(:), rho(:), mu(:));
        break
    end


    if rem(Iter, frePrint) == 0
        fprintf('Iter = %d, velocity diff = %.4e\n', Iter, vel_change);
    end

    if vel_change < tol
        fprintf('Converged at iter %d (Δu = %.3e)\n', Iter, vel_change);
        break;
    end

    if rem(Iter, freOut) == 0
        if strcmp(domain.device, 'gpu')
            OUTPUT_Tecplot2D4(Iter, pathname, domain.ny_all, domain.nx_all, varName, ...
                gather(coordX(:)), gather(coordY(:)), gather(un1(:)), gather(vn1(:)), ...
                gather(pre(:)), gather(phin1(:)), gather(psi(:)), gather(rho(:)), gather(mu(:)));
        else
            OUTPUT_Tecplot2D4(Iter, pathname, domain.ny_all, domain.nx_all, varName, ...
                coordX(:), coordY(:), un1(:), vn1(:), pre(:), phin1(:), psi(:), rho(:), mu(:));
        end
    end

    % ------------------------ Advance states -----------------------------
    un_1 = un;
    vn_1 = vn;
    pn_1 = pn;
    phin_1 = phin;

    un = un1;
    vn = vn1;
    pn = pn1;
    phin = phin1;

    if rem(Iter, freMat) == 0
        filename = sprintf('flow%d', Iter);
        cd(pathname);
        save(filename, 'Iter');
        cd(currentLocation);
    end
end

cd(pathname);
save flow;
cd(currentLocation);

fprintf('Total wall-clock time: %.2f s\n', toc);
fprintf('=== %d Coupled NS-CH Solver Ends ===\n', stage);
