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

