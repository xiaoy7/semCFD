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
