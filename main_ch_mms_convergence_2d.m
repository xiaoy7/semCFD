% CH_MMS_CONVERGENCE_2D Manufactured-solution convergence test for 2D CH.
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

spatialCells = [4;8;16;32; 64; 128]; % 256;512;1024
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
 




