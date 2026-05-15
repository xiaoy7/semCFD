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

