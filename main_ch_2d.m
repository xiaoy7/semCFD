% MAIN_CH_2D Solve the 2D Cahn-Hilliard equation by tensor-product SEM.
%   This driver follows the compact style used by main_ch_convergence_2d.m.
%   The default case evolves a circular interface on [0,1]^2 and writes
%   Tecplot snapshots to a sibling time-stamped run directory.
clc
clear

repoDir = fileparts(mfilename('fullpath'));
addpath(genpath(repoDir));

cfg.Np = 4;
cfg.Ncellx = 20;
cfg.Ncelly = 20;
cfg.domain = [0, 1, 0, 1];
cfg.dt = 1e-3;
cfg.steps = 2000;
cfg.frePrint = 10;
cfg.freOut = 100;
cfg.freMat = 2000;
cfg.divergenceTol = 10;

cfg.lengthScale = 1;
cfg.circleCenter = [0.5, 0.5];
cfg.circleRadius = 0.2;
cfg.Cn = 0.02;
cfg.eta = cfg.Cn * cfg.lengthScale;
cfg.mobility = 5e-5;
cfg.gamma0 = 1.5;

cfg.bigs = 1.1 * cfg.eta^2 * sqrt(4 * cfg.gamma0 / (cfg.mobility * cfg.dt));
stabilizerArg = 1 - cfg.gamma0 / (cfg.mobility * cfg.dt) * 4 * cfg.eta^4 / cfg.bigs^2;
cfg.alpha = -cfg.bigs / (2 * cfg.eta^2) * (1 + sqrt(stabilizerArg));

runDir = create_run_directory(repoDir);

fprintf('=== Cahn-Hilliard 2D SEM run ===\n');
fprintf('Q%d SEM, Ncell=(%d,%d), dt=%.3e, steps=%d\n', ...
    cfg.Np, cfg.Ncellx, cfg.Ncelly, cfg.dt, cfg.steps);
fprintf('epsilon=%g, M=%g, output=%s\n', cfg.eta, cfg.mobility, runDir);

para = build_sem_parameters(cfg, 'dirichlet');
paraN = build_sem_parameters(cfg, 'neumann');
para = parameter_bc2d(para);
paraN = parameter_bc2d(paraN);

[~, x, Tx, ~, lambdaXd, DmatrixX] = cal_matrix2('x', para);
[~, y, Ty, ~, lambdaYd, DmatrixY] = cal_matrix2('y', para);
[~, exn, Txn, lambdaXn] = cal_matrix2_1('x', paraN, x);
[~, eyn, Tyn, lambdaYn] = cal_matrix2_1('y', paraN, y);

[coordY, coordX] = meshgrid(y, x);
DmatrixYT = DmatrixY';
DmatrixX2 = DmatrixX * DmatrixX;
DmatrixYT2 = DmatrixYT * DmatrixYT;

invTx = pinv(Tx);
invTy = pinv(Ty);
invTxn = pinv(Txn);
invTyn = pinv(Tyn);

poissonD = bsxfun(@plus, lambdaXd, lambdaYd');
poissonN = bsxfun(@plus, lambdaXn, lambdaYn');
poissonN(abs(poissonN) < 1e-12) = 1e-12;

semPsi.Tx = Txn;
semPsi.Ty = Tyn;
semPsi.invTx = invTxn;
semPsi.invTy = invTyn;
semPsi.lambda2 = poissonN;

semPhi = semPsi;
semPhi.lambda2 = -poissonN;

varName = 'U,V,PRE,phi,psi\n';
pressure = zeros(para.nx_all, para.ny_all);
[un, vn] = deal(0.1 * ones(para.nx_all, para.ny_all));
un1 = un;
vn1 = vn;
unOld = un;
vnOld = vn;

radius = sqrt((coordX - cfg.circleCenter(1)).^2 + (coordY - cfg.circleCenter(2)).^2);
phi = -tanh((radius - cfg.circleRadius) ./ (sqrt(2) * cfg.eta));
phiOld = phi;
phiNew = phi;
psi = phi;

OUTPUT_Tecplot2D4(0, runDir, para.ny_all, para.nx_all, varName, ...
    coordX(:), coordY(:), un(:), vn(:), pressure(:), phi(:), psi(:));

device = select_device();
if strcmp(device.type, 'gpu')
    fprintf('GPU computation: loading matrices and fields\n');
    [Tx, Ty, Txn, Tyn, invTx, invTy, invTxn, invTyn, lambdaXd, lambdaYd, ...
        lambdaXn, lambdaYn, exn, eyn, DmatrixX, DmatrixYT, DmatrixX2, ...
        DmatrixYT2, poissonD, poissonN, semPsi, semPhi, ...
        un, vn, un1, vn1, unOld, vnOld, pressure, phi, phiOld, phiNew, psi] = ...
        move_to_gpu(Tx, Ty, Txn, Tyn, invTx, invTy, invTxn, invTyn, ...
        lambdaXd, lambdaYd, lambdaXn, lambdaYn, exn, eyn, DmatrixX, ...
        DmatrixYT, DmatrixX2, DmatrixYT2, poissonD, poissonN, ...
        semPsi, semPhi, un, vn, un1, vn1, unOld, vnOld, ...
        pressure, phi, phiOld, phiNew, psi);
    wait(device.handle);
    fprintf('GPU computation: loading finished\n');
end

tic;
fprintf('=== start time stepping ===\n');
for iter = 1:cfg.steps
    uStar = 2 * un - unOld;
    vStar = 2 * vn - vnOld;
    phiStar = 2 * phi - phiOld;
    phiCap = 2 * phi - 0.5 * phiOld;

    advPhi = uStar .* (DmatrixX * phiStar) + vStar .* (phiStar * DmatrixYT);
    q1 = (advPhi - phiCap / cfg.dt) / cfg.mobility;
    q2 = (phiStar.^2 - 1 - cfg.bigs) / cfg.eta^2 .* phiStar;
    fPsi = q1 - DmatrixX2 * q2 - q2 * DmatrixYT2;

    psi = solve_linear(semPsi, fPsi, cfg.alpha + cfg.bigs / cfg.eta^2, 1);
    phiNew = solve_linear(semPhi, psi, cfg.alpha, 1);

    if strcmp(device.type, 'gpu')
        wait(device.handle);
    end

    errorPhi = norm(phiNew - phi, 'fro');
    if rem(iter, cfg.frePrint) == 0
        fprintf('Iter = %d, norm2 = %e\n', iter, errorPhi);
    end

    if errorPhi > cfg.divergenceTol || isnan(errorPhi)
        fprintf('divergence at Iter = %d, norm2 = %e\n', iter, errorPhi);
        write_snapshot(iter, runDir, para, varName, coordX, coordY, ...
            un1, vn1, pressure, phiNew, psi, device.type);
        break
    end

    if rem(iter, cfg.freOut) == 0
        write_snapshot(iter, runDir, para, varName, coordX, coordY, ...
            un1, vn1, pressure, phiNew, psi, device.type);
    end

    unOld = un;
    vnOld = vn;
    phiOld = phi;
    un = un1;
    vn = vn1;
    phi = phiNew;

    if rem(iter, cfg.freMat) == 0
        save(fullfile(runDir, sprintf('flow%d.mat', iter)));
    end
end

save(fullfile(runDir, 'flow.mat'));

fprintf('Total computation time: %f seconds\n', toc);
fprintf('=== Program Ends ===\n');

