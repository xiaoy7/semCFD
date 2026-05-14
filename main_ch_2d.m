% MAIN_CH_2D Solve the 2D Cahn-Hilliard equation by tensor-product SEM.
%   The default case evolves a circular interface on [0,1]^2 and writes
%   Tecplot snapshots to a sibling time-stamped run directory.
clc
clear

repoDir = fileparts(mfilename('fullpath'));
addpath(genpath(repoDir));

cfg = default_ch_config();
runDir = create_run_directory(repoDir);

fprintf('=== Cahn-Hilliard 2D SEM run ===\n');
fprintf('Q%d SEM, Ncell=(%d,%d), dt=%.3e, steps=%d\n', ...
    cfg.Np, cfg.Ncellx, cfg.Ncelly, cfg.dt, cfg.steps);
fprintf('epsilon=%g, M=%g, output=%s\n', cfg.eta, cfg.mobility, runDir);

results = run_ch_case(cfg, runDir);


function cfg = default_ch_config()
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
stabilizerArg = 1 - cfg.gamma0 / (cfg.mobility * cfg.dt) ...
    * 4 * cfg.eta^4 / cfg.bigs^2;
cfg.alpha = -cfg.bigs / (2 * cfg.eta^2) * (1 + sqrt(stabilizerArg));
end


function results = run_ch_case(cfg, runDir)
[sem, para, coordX, coordY] = setup_ch_sem(cfg);
state = initial_ch_state(cfg, para, coordX, coordY);
varName = 'U,V,PRE,phi,psi\n';

write_snapshot(0, runDir, para, varName, coordX, coordY, ...
    state.un, state.vn, state.pressure, state.phi, state.psi, 'cpu');

device = select_device();
if strcmp(device.type, 'gpu')
    fprintf('GPU computation: loading matrices and fields\n');
    [sem, state] = move_ch_to_gpu(sem, state);
    wait(device.handle);
    fprintf('GPU computation: loading finished\n');
end

tic;
fprintf('=== start time stepping ===\n');
[state, stopIter] = advance_ch(cfg, sem, para, state, runDir, varName, ...
    coordX, coordY, device);
elapsedTime = toc;

stateForSave = gather_struct(state);
save(fullfile(runDir, 'flow.mat'), 'cfg', 'stateForSave', ...
    'stopIter', 'elapsedTime');

fprintf('Total computation time: %f seconds\n', elapsedTime);
fprintf('=== Program Ends ===\n');

results.config = cfg;
results.runDir = runDir;
results.stopIter = stopIter;
results.elapsedTime = elapsedTime;
end


function [sem, para, coordX, coordY] = setup_ch_sem(cfg)
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
poissonN = bsxfun(@plus, lambdaXn, lambdaYn');
poissonN(abs(poissonN) < 1e-12) = 1e-12;

sem.Tx = Tx;
sem.Ty = Ty;
sem.Txn = Txn;
sem.Tyn = Tyn;
sem.invTx = pinv(Tx);
sem.invTy = pinv(Ty);
sem.invTxn = pinv(Txn);
sem.invTyn = pinv(Tyn);
sem.lambdaXd = lambdaXd;
sem.lambdaYd = lambdaYd;
sem.lambdaXn = lambdaXn;
sem.lambdaYn = lambdaYn;
sem.exn = exn;
sem.eyn = eyn;
sem.Dx = DmatrixX;
sem.DyT = DmatrixYT;
sem.Dxx = DmatrixX * DmatrixX;
sem.DyyT = DmatrixYT * DmatrixYT;
sem.poissonD = bsxfun(@plus, lambdaXd, lambdaYd');
sem.poissonN = poissonN;

sem.psi.Tx = Txn;
sem.psi.Ty = Tyn;
sem.psi.invTx = sem.invTxn;
sem.psi.invTy = sem.invTyn;
sem.psi.lambda2 = poissonN;

sem.phi = sem.psi;
sem.phi.lambda2 = -poissonN;
end


function state = initial_ch_state(cfg, para, coordX, coordY)
state.pressure = zeros(para.nx_all, para.ny_all);
[state.un, state.vn] = deal(0.1 * ones(para.nx_all, para.ny_all));
state.un1 = state.un;
state.vn1 = state.vn;
state.unOld = state.un;
state.vnOld = state.vn;

radius = sqrt((coordX - cfg.circleCenter(1)).^2 ...
    + (coordY - cfg.circleCenter(2)).^2);
state.phi = -tanh((radius - cfg.circleRadius) ./ (sqrt(2) * cfg.eta));
state.phiOld = state.phi;
state.phiNew = state.phi;
state.psi = state.phi;
end


function [sem, state] = move_ch_to_gpu(sem, state)
fieldNames = fieldnames(sem);
for k = 1:numel(fieldNames)
    name = fieldNames{k};
    if isstruct(sem.(name))
        sem.(name) = move_struct_to_gpu(sem.(name));
    else
        sem.(name) = gpuArray(sem.(name));
    end
end
state = move_struct_to_gpu(state);
end


function out = move_struct_to_gpu(in)
out = in;
fieldNames = fieldnames(in);
for k = 1:numel(fieldNames)
    name = fieldNames{k};
    out.(name) = gpuArray(in.(name));
end
end


function [state, stopIter] = advance_ch(cfg, sem, para, state, runDir, ...
    varName, coordX, coordY, device)
stopIter = cfg.steps;

for iter = 1:cfg.steps
    state = step_ch(cfg, sem, state);

    if strcmp(device.type, 'gpu')
        wait(device.handle);
    end

    errorPhi = norm(state.phiNew - state.phi, 'fro');
    if rem(iter, cfg.frePrint) == 0
        fprintf('Iter = %d, norm2 = %e\n', iter, errorPhi);
    end

    if errorPhi > cfg.divergenceTol || isnan(errorPhi)
        fprintf('divergence at Iter = %d, norm2 = %e\n', iter, errorPhi);
        write_snapshot(iter, runDir, para, varName, coordX, coordY, ...
            state.un1, state.vn1, state.pressure, state.phiNew, ...
            state.psi, device.type);
        stopIter = iter;
        break
    end

    if rem(iter, cfg.freOut) == 0
        write_snapshot(iter, runDir, para, varName, coordX, coordY, ...
            state.un1, state.vn1, state.pressure, state.phiNew, ...
            state.psi, device.type);
    end

    state = accept_ch_step(state);

    if rem(iter, cfg.freMat) == 0
        stateForSave = gather_struct(state);
        save(fullfile(runDir, sprintf('flow%d.mat', iter)), 'cfg', ...
            'stateForSave', 'iter');
    end
end
end


function state = step_ch(cfg, sem, state)
uStar = 2 * state.un - state.unOld;
vStar = 2 * state.vn - state.vnOld;
phiStar = 2 * state.phi - state.phiOld;
phiCap = 2 * state.phi - 0.5 * state.phiOld;

advPhi = uStar .* (sem.Dx * phiStar) + vStar .* (phiStar * sem.DyT);
q1 = (advPhi - phiCap / cfg.dt) / cfg.mobility;
q2 = (phiStar.^2 - 1 - cfg.bigs) / cfg.eta^2 .* phiStar;
fPsi = q1 - sem.Dxx * q2 - q2 * sem.DyyT;

state.psi = solve_linear(sem.psi, fPsi, ...
    cfg.alpha + cfg.bigs / cfg.eta^2, 1);
state.phiNew = solve_linear(sem.phi, state.psi, cfg.alpha, 1);
end


function state = accept_ch_step(state)
state.unOld = state.un;
state.vnOld = state.vn;
state.phiOld = state.phi;
state.un = state.un1;
state.vn = state.vn1;
state.phi = state.phiNew;
end


function out = gather_struct(in)
out = in;
fieldNames = fieldnames(in);
for k = 1:numel(fieldNames)
    name = fieldNames{k};
    if isstruct(in.(name))
        out.(name) = gather_struct(in.(name));
    elseif isa(in.(name), 'gpuArray')
        out.(name) = gather(in.(name));
    else
        out.(name) = in.(name);
    end
end
end
