% MAIN_NS_3D Solve the 3D lid-driven-cavity Navier-Stokes problem by SEM.
%   This driver uses a tensor-product spectral element projection method and
%   follows the compact configuration/runner style used by the 2D drivers.
clc
clear

repoDir = fileparts(mfilename('fullpath'));
addpath(genpath(repoDir));

cfg = default_ns3d_config();
runDir = create_run_directory(repoDir);

fprintf('=== Navier-Stokes 3D SEM run ===\n');
fprintf('Q%d SEM, Ncell=(%d,%d,%d), dt=%.3e, steps=%d\n', ...
    cfg.Np, cfg.Ncellx, cfg.Ncelly, cfg.Ncellz, cfg.dt, cfg.steps);
fprintf('Re=%g, nu=%g, output=%s\n', cfg.reynolds, cfg.nu, runDir);

results = run_ns3d_case(cfg, runDir);
