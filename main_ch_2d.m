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

