% MAIN_NS_2D Solve the 2D Navier-Stokes/Cahn-Hilliard system by SEM.
%   The default case follows the original two-phase projection-method driver
%   and writes Tecplot files to a sibling time-stamped run directory.
clc
clear

repoDir = fileparts(mfilename('fullpath'));
addpath(genpath(repoDir));

cfg = default_ns_config();
runDir = create_run_directory(repoDir);

fprintf('=== Navier-Stokes 2D SEM run ===\n');
fprintf('Q%d SEM, Ncell=(%d,%d), dt=%.3e, steps=%d\n', ...
    cfg.Np, cfg.Ncellx, cfg.Ncelly, cfg.dt, cfg.steps);
fprintf('domain=[%g,%g]x[%g,%g], output=%s\n', cfg.domain, runDir);

results = run_ns_case(cfg, runDir);


function cfg = default_ns_config()
cfg.dt = 1e-2;
cfg.steps = 50000;
cfg.frePrint = 10;
cfg.freOut = 1000;
cfg.divergenceTol = 1;

cfg.lengthScale = 100;
cfg.circleCenter = [0.25, 0.25] * cfg.lengthScale;
cfg.circleRadius = 0.5 * cfg.lengthScale;
cfg.Cn = 0.01;
cfg.eta = cfg.Cn * cfg.lengthScale;

cfg.liquidDensity = 1;
cfg.gasDensity = 3;
cfg.liquidViscosity = 300 / 256;
cfg.gasViscosity = 300 / 256;
cfg.nium = 300 / 256;
cfg.rho0 = min(cfg.liquidDensity, cfg.gasDensity);

cfg.gravity = -0.01;
cfg.tension = 0;
cfg.mobility = 1e-7;
cfg.gamma0 = 1.5;
cfg.bigs = 1.1 * cfg.eta^2 * sqrt(4 * cfg.gamma0 ...
    / (cfg.mobility * cfg.dt));
stabilizerArg = 1 - cfg.gamma0 / (cfg.mobility * cfg.dt) ...
    * 4 * cfg.eta^4 / cfg.bigs^2;
cfg.alpha = -cfg.bigs / (2 * cfg.eta^2) * (1 + sqrt(stabilizerArg));
cfg.lambda = 3 * cfg.eta / (2 * sqrt(2));

cfg.Np = 4;
cfg.Ncellx = 20;
cfg.Ncelly = 160;
cfg.domain = [0, 0.5 * cfg.lengthScale, ...
    -2 * cfg.lengthScale, 2 * cfg.lengthScale];
end


function results = run_ns_case(cfg, runDir)
[sem, paraD, paraN, coordX, coordY] = setup_ns_sem(cfg);
state = initial_ns_state(cfg, paraD, coordX, coordY);
write_grid(runDir, paraD, coordX, coordY);

device = select_device();
if strcmp(device.type, 'gpu')
    fprintf('GPU computation: loading matrices and fields\n');
    [sem, state] = move_ns_to_gpu(sem, state);
    wait(device.handle);
    fprintf('GPU computation: loading finished\n');
end

tic;
fprintf('=== start time stepping ===\n');
[state, stopIter] = advance_ns(cfg, sem, paraD, paraN, state, ...
    runDir, device);
elapsedTime = toc;

fprintf('Total computation time: %f seconds\n', elapsedTime);
fprintf('=== Program Ends ===\n');

results.config = cfg;
results.runDir = runDir;
results.stopIter = stopIter;
results.elapsedTime = elapsedTime;
results.state = gather_struct(state);
end


function [sem, paraD, paraN, coordX, coordY] = setup_ns_sem(cfg)
paraD = build_sem_parameters(cfg, 'dirichlet');
paraN = build_sem_parameters(cfg, 'neumann');
paraD = parameter_bc2d(paraD);
paraN = parameter_bc2d(paraN);

fprintf('Laplacian is Q%d spectral element method \n', cfg.Np);

[~, x, Txd, ~, lambdaXd, DmatrixX] = cal_matrix2('x', paraD);
[~, y, Tyd, ~, lambdaYd, DmatrixY] = cal_matrix2('y', paraD);
[~, ~, Txn, lambdaXn] = cal_matrix2_1('x', paraN, x);
[~, ~, Tyn, lambdaYn] = cal_matrix2_1('y', paraN, y);

[coordY, coordX] = meshgrid(y, x);
DmatrixYT = DmatrixY';
poissonD = bsxfun(@plus, lambdaXd, lambdaYd');
poissonN = bsxfun(@plus, lambdaXn, lambdaYn');

sem.Txd = Txd;
sem.Tyd = Tyd;
sem.Txn = Txn;
sem.Tyn = Tyn;
sem.invTx = pinv(Txd);
sem.invTy = pinv(Tyd);
sem.invTxn = pinv(Txn);
sem.invTyn = pinv(Tyn);
sem.Dx = DmatrixX;
sem.DyT = DmatrixYT;
sem.Dxx = DmatrixX * DmatrixX;
sem.DyyT = DmatrixYT * DmatrixYT;
sem.poissonD = poissonD;
sem.poissonN = poissonN;
sem.poissonP = poissonN;
sem.helmholtzU = 1.5 / cfg.dt + cfg.nium * poissonD;
sem.helmholtzV = 1.5 / cfg.dt + cfg.nium * poissonN;
sem.helmholtzPhi = cfg.alpha - poissonN;
sem.helmholtzPsi = cfg.alpha + cfg.bigs / cfg.eta^2 + poissonN;
sem.wx = assemble_sem_weights(paraN.Npx, paraN.Ncellx, ...
    paraN.minx, paraN.maxx);
sem.wy = assemble_sem_weights(paraN.Npy, paraN.Ncelly, ...
    paraN.miny, paraN.maxy);
sem.massDiag = sem.wx * sem.wy';
end


function state = initial_ns_state(cfg, para, coordX, coordY)
zeroField = zeros(para.nx_all, para.ny_all);
state.u = zeroField;
state.v = zeroField;
state.p = zeroField;
state.uNew = zeroField;
state.vNew = zeroField;
state.pNew = zeroField;
state.uOld = zeroField;
state.vOld = zeroField;
state.pOld = zeroField;

state.phiNew = initialPhi2d(coordX, coordY, cfg.circleCenter(1), ...
    cfg.circleCenter(2), cfg.circleRadius, cfg.eta, cfg.lengthScale);
state.phi = state.phiNew;
state.phiOld = state.phiNew;
state.psi = zeroField;
state.rho = zeroField;
state.mu = zeroField;
end


function write_grid(runDir, para, coordX, coordY)
filenameGrid = fullfile(runDir, 'grid.plt');
coords = [coordX(:), coordY(:)];
plt_Head(filenameGrid, '', {'X', 'Y'}, 'GRID');
plt_Zone(filenameGrid, '', [para.nx_all, para.ny_all], 0, coords);
end


function [sem, state] = move_ns_to_gpu(sem, state)
sem = move_struct_to_gpu(sem);
state = move_struct_to_gpu(state);
end


function out = move_struct_to_gpu(in)
out = in;
fieldNames = fieldnames(in);
for k = 1:numel(fieldNames)
    name = fieldNames{k};
    if isstruct(in.(name))
        out.(name) = move_struct_to_gpu(in.(name));
    else
        out.(name) = gpuArray(in.(name));
    end
end
end


function [state, stopIter] = advance_ns(cfg, sem, paraD, paraN, state, ...
    runDir, device)
stopIter = cfg.steps;
ijk = [paraD.nx_all, paraD.ny_all];
varNames = {'u', 'v', 'p', 'phi', 'psi', 'rho', 'mu'};

for iter = 1:cfg.steps
    state = step_ns_ch(cfg, sem, paraD, paraN, state);

    if strcmp(device.type, 'gpu')
        wait(device.handle);
    end

    errorU = norm(state.uNew - state.u, 'fro');
    errorV = norm(state.vNew - state.v, 'fro');
    totalError = gather(sqrt(errorU^2 + errorV^2));

    if rem(iter, cfg.frePrint) == 0
        fprintf('Iter = %d, error_u = %e\n', iter, totalError);
        if rem(iter, cfg.freOut) == 0
            saveTec(iter, runDir, ijk, varNames, state.uNew, state.vNew, ...
                state.pNew, state.phiNew, state.psi, state.rho, state.mu);
        end
    end

    if totalError > cfg.divergenceTol || isnan(totalError)
        fprintf('divergence at %d, error_u = %e\n', iter, totalError);
        stopIter = iter;
        break
    end

    state = accept_ns_step(state);
end
end


function state = step_ns_ch(cfg, sem, paraD, paraN, state)
uStar = 2 * state.u - state.uOld;
vStar = 2 * state.v - state.vOld;
pStar = 2 * state.p - state.pOld;
phiStar = 2 * state.phi - state.phiOld;

uCap = 2 * state.u - 0.5 * state.uOld;
vCap = 2 * state.v - 0.5 * state.vOld;
phiCap = 2 * state.phi - 0.5 * state.phiOld;

q1 = (uStar .* (sem.Dx * phiStar) + vStar .* (phiStar * sem.DyT) ...
    - phiCap / cfg.dt) / cfg.mobility;
q2 = (phiStar.^2 - 1 - cfg.bigs) / cfg.eta^2 .* phiStar;
fPsi = q1 - calLaplace(q2, sem.Dxx, sem.DyyT);

psiHat = sem.invTxn * fPsi * sem.invTyn';
psiHat = psiHat ./ sem.helmholtzPsi;
state.psi = sem.Txn * psiHat * sem.Tyn';

phiHat = sem.invTxn * state.psi * sem.invTyn';
phiHat = phiHat ./ sem.helmholtzPhi;
state.phiNew = sem.Txn * phiHat * sem.Tyn';

dPhiX = sem.Dx * state.phiNew;
dPhiY = state.phiNew * sem.DyT;

state.phiNew(state.phiNew > 1) = 1;
state.phiNew(state.phiNew < -1) = -1;
state.rho = (cfg.liquidDensity + cfg.gasDensity) / 2 ...
    + state.phiNew * (cfg.gasDensity - cfg.liquidDensity) / 2;
state.mu = (cfg.liquidViscosity + cfg.gasViscosity) / 2 ...
    + state.phiNew * (cfg.gasViscosity - cfg.liquidViscosity) / 2;

dMuX = (cfg.gasViscosity - cfg.liquidViscosity) / 2 * dPhiX;
dMuY = (cfg.gasViscosity - cfg.liquidViscosity) / 2 * dPhiY;

dUStarX = sem.Dx * uStar;
dUStarY = uStar * sem.DyT;
dVStarX = sem.Dx * vStar;
dVStarY = vStar * sem.DyT;
dPStarX = sem.Dx * pStar;
dPStarY = pStar * sem.DyT;

lapU = calLaplace(uStar, sem.Dxx, sem.DyyT);
lapV = calLaplace(vStar, sem.Dxx, sem.DyyT);
viscosityOverDensity = state.mu ./ state.rho;
diffusionX = 2 * dMuX .* dUStarX + dMuY .* (dVStarX + dUStarY);
diffusionY = dMuX .* (dVStarX + dUStarY) + 2 * dMuY .* dVStarY;

uv31x = uCap ./ cfg.dt + (1 / cfg.rho0 - 1 ./ state.rho) .* dPStarX ...
    - uStar .* dUStarX - vStar .* dUStarY ...
    + (viscosityOverDensity - cfg.nium) .* lapU + diffusionX ./ state.rho;
uv31y = vCap ./ cfg.dt + (1 / cfg.rho0 - 1 ./ state.rho) .* dPStarY ...
    - uStar .* dVStarX - vStar .* dVStarY ...
    + (viscosityOverDensity - cfg.nium) .* lapV + diffusionY ./ state.rho ...
    + cfg.gravity;

fp = -cfg.rho0 * (sem.Dx * uv31x + uv31y * sem.DyT);
midX = cfg.rho0 * (uv31x - uStar / cfg.dt + cfg.nium * lapU);
midY = cfg.rho0 * (uv31y - vStar / cfg.dt + cfg.nium * lapV);

fx = zeros(paraN.nx_all, paraN.ny_all, 'like', state.u);
fy = zeros(paraN.nx_all, paraN.ny_all, 'like', state.u);
fx(1, :) = -midX(1, :) .* sem.wy';
fx(end, :) = midX(end, :) .* sem.wy';
fy(:, 1) = -midY(:, 1) .* sem.wx;
fy(:, end) = midY(:, end) .* sem.wx;
fSolver = fp + (fx + fy) ./ sem.massDiag;

pressureHat = sem.invTxn * fSolver * sem.invTyn';
pressureHat = pressureHat ./ sem.poissonP;
state.pNew = sem.Txn * pressureHat * sem.Tyn';
state.pNew = state.pNew - state.pNew(1, 1);

gradPressureX = sem.Dx * state.pNew;
gradPressureY = state.pNew * sem.DyT;
fu = uv31x - gradPressureX ./ cfg.rho0;
fv = uv31y - gradPressureY ./ cfg.rho0;

fuInterior = fu(paraD.freeNodesx, paraD.freeNodesy);
uHat = sem.invTx * fuInterior * sem.invTy';
uHat = uHat ./ sem.helmholtzU;
uInterior = sem.Txd * uHat * sem.Tyd';
state.uNew(paraD.freeNodesx, paraD.freeNodesy) = uInterior;

vHat = sem.invTxn * fv * sem.invTyn';
vHat = vHat ./ sem.helmholtzV;
state.vNew = sem.Txn * vHat * sem.Tyn';
end


function state = accept_ns_step(state)
state.uOld = state.u;
state.vOld = state.v;
state.pOld = state.p;

state.u = state.uNew;
state.v = state.vNew;
state.p = state.pNew;

state.phiOld = state.phi;
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
