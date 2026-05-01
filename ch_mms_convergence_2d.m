function results = ch_mms_convergence_2d()
%CH_MMS_CONVERGENCE_2D Manufactured-solution convergence test for 2D CH.
%   This driver verifies the SEM phase-field discretization on [-1,1]^2 with
%
%       phi_exact = sin(pi*x)*sin(pi*y)*cos(t).
%
%   The test solves
%
%       phi_t = M*laplacian(mu) + g,
%       mu    = (phi^3 - phi)/epsilon^2 - laplacian(phi),
%
%   with homogeneous Dirichlet data for phi and mu. For this manufactured
%   solution both fields vanish on the boundary.

repoDir = fileparts(mfilename('fullpath'));
addpath(genpath(repoDir));

cfg.Np = 4;
cfg.epsilon = 0.1;
cfg.mobility = 1e-4;
cfg.domain = [-1, 1, -1, 1];
cfg.Tfinal = 2e-2;

spatialCells = [32; 64; 128; 256;512;1024];
spatialDt = 5e-5;

temporalCells = 18;
temporalDt = [2e-3; 1e-3; 5e-4; 2.5e-4];

fprintf('=== Cahn-Hilliard MMS convergence test ===\n');
fprintf('phi = sin(pi*x) sin(pi*y) cos(t), epsilon=%g, M=%g, Q%d SEM\n', ...
    cfg.epsilon, cfg.mobility, cfg.Np);

fprintf('\nSpatial refinement: fixed dt = %.3e\n', spatialDt);
spatial = run_spatial_sweep(cfg, spatialCells, spatialDt);
disp(spatial);

fprintf('\nTemporal refinement: fixed Ncell = %d\n', temporalCells);
temporal = run_temporal_sweep(cfg, temporalCells, temporalDt);
disp(temporal);

results.config = cfg;
results.spatial = spatial;
results.temporal = temporal;
end

function tbl = run_spatial_sweep(cfg, cells, dt)
ncase = numel(cells);
h = zeros(ncase, 1);
l2 = zeros(ncase, 1);
linf = zeros(ncase, 1);

for k = 1:ncase
    sem = setup_sem_dirichlet(cfg, cells(k));
    [l2(k), linf(k)] = solve_mms_case(cfg, sem, dt);
    h(k) = (cfg.domain(2) - cfg.domain(1)) / cells(k);
    fprintf('  Ncell=%3d, h=%.4e, L2=%.6e, Linf=%.6e\n', ...
        cells(k), h(k), l2(k), linf(k));
end

rateL2 = convergence_rate(l2, h);
rateLinf = convergence_rate(linf, h);
tbl = table(cells, h, repmat(dt, ncase, 1), l2, rateL2, linf, rateLinf, ...
    'VariableNames', {'Ncell', 'h', 'dt', 'L2', 'rateL2', 'Linf', 'rateLinf'});
end

function tbl = run_temporal_sweep(cfg, ncell, dts)
ncase = numel(dts);
l2 = zeros(ncase, 1);
linf = zeros(ncase, 1);
sem = setup_sem_dirichlet(cfg, ncell);

for k = 1:ncase
    [l2(k), linf(k)] = solve_mms_case(cfg, sem, dts(k));
    fprintf('  dt=%.4e, L2=%.6e, Linf=%.6e\n', dts(k), l2(k), linf(k));
end

rateL2 = convergence_rate(l2, dts);
rateLinf = convergence_rate(linf, dts);
tbl = table(repmat(ncell, ncase, 1), dts, l2, rateL2, linf, rateLinf, ...
    'VariableNames', {'Ncell', 'dt', 'L2', 'rateL2', 'Linf', 'rateLinf'});
end

function sem = setup_sem_dirichlet(cfg, ncell)
param.Np = cfg.Np;
param.Npx = cfg.Np;
param.Npy = cfg.Np;
param.basis = 'SEM';
param.minx = cfg.domain(1);
param.maxx = cfg.domain(2);
param.miny = cfg.domain(3);
param.maxy = cfg.domain(4);
param.Ncellx = ncell;
param.Ncelly = ncell;
param.nx_all = ncell * cfg.Np + 1;
param.ny_all = ncell * cfg.Np + 1;
param.bc = 'dirichlet';
param = parameter_bc2d(param);

[~, x, Tx, ~, lambda_x, ~, ~] = cal_matrix2('x', param);
[~, y, Ty, ~, lambda_y, ~, ~] = cal_matrix2('y', param);

sem.x = x(param.freeNodesx);
sem.y = y(param.freeNodesy);
sem.Tx = Tx;
sem.Ty = Ty;
sem.invTx = pinv(Tx);
sem.invTy = pinv(Ty);
sem.lambda = bsxfun(@plus, lambda_x, lambda_y');
[sem.X, sem.Y] = ndgrid(sem.x, sem.y);

wx = assemble_sem_weights(param.Npx, param.Ncellx, param.minx, param.maxx);
wy = assemble_sem_weights(param.Npy, param.Ncelly, param.miny, param.maxy);
sem.mass = wx(param.freeNodesx) * wy(param.freeNodesy)';
end

function [l2err, linferr] = solve_mms_case(cfg, sem, dt)
steps = round(cfg.Tfinal / dt);
dt = cfg.Tfinal / steps;

phi_nm1 = manufactured_ch_2d(sem.x, sem.y, 0, cfg.epsilon, cfg.mobility);

t = dt;
g = manufactured_ch_2d_source(sem, t, cfg);
rhs = phi_nm1 / dt - cfg.mobility / cfg.epsilon^2 * apply_positive_lap(sem, nonlinear(phi_nm1)) + g;
phi_n = solve_linear(sem, rhs, 1 / dt, cfg.mobility);

for n = 1:steps-1
    t = (n + 1) * dt;
    phi_star = 2 * phi_n - phi_nm1;
    g = manufactured_ch_2d_source(sem, t, cfg);

    rhs = (4 * phi_n - phi_nm1) / (2 * dt) ...
        - cfg.mobility / cfg.epsilon^2 * apply_positive_lap(sem, nonlinear(phi_star)) ...
        + g;
    phi_np1 = solve_linear(sem, rhs, 3 / (2 * dt), cfg.mobility);

    phi_nm1 = phi_n;
    phi_n = phi_np1;
end

phi_exact = manufactured_ch_2d(sem.x, sem.y, cfg.Tfinal, cfg.epsilon, cfg.mobility);
err = phi_n - phi_exact;
l2err = sqrt(sum(sem.mass .* err.^2, 'all'));
linferr = max(abs(err), [], 'all');
end

function f = nonlinear(phi)
f = phi.^3 - phi;
end

function g = manufactured_ch_2d_source(sem, t, cfg)
[~, ~, g] = manufactured_ch_2d(sem.x, sem.y, t, cfg.epsilon, cfg.mobility);
end

function Lu = apply_positive_lap(sem, u)
uhat = sem.invTx * u * sem.invTy';
Lu = sem.Tx * (sem.lambda .* uhat) * sem.Ty';
end

function u = solve_linear(sem, rhs, massCoef, mobility)
rhsHat = sem.invTx * rhs * sem.invTy';
uHat = rhsHat ./ (massCoef + mobility * sem.lambda.^2);
u = sem.Tx * uHat * sem.Ty';
end

function rate = convergence_rate(err, scale)
rate = nan(size(err));
for k = 2:numel(err)
    rate(k) = log(err(k - 1) / err(k)) / log(scale(k - 1) / scale(k));
end
end
