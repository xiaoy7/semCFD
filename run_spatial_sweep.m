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


