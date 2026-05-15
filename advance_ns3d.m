function [state, stopIter, converged] = advance_ns3d(cfg, sem, paraD, ...
    paraN, state, runDir, coord, device)
stopIter = cfg.steps;
converged = false;

for iter = 1:cfg.steps
    state = step_ns3d(cfg, sem, paraD, paraN, state);

    if strcmp(device.type, 'gpu')
        wait(device.handle);
    end

    errorU = gather_ns3d_value(norm(state.uNew(:) - state.u(:)));
    if isnan(errorU) || errorU > cfg.divergenceTol
        fprintf('Iter = %d, error_u = %e\n', iter, errorU);
        stopIter = iter;
        break
    end

    if rem(iter, cfg.frePrint) == 0
        fprintf('Iter = %d, error_u = %e\n', iter, errorU);
    end

    if cfg.freOut > 0 && rem(iter, cfg.freOut) == 0
        write_ns3d_snapshot(iter, runDir, coord, paraD, state);
    end

    if errorU <= cfg.tolerance
        fprintf('Convergence reached at iteration %d\n', iter);
        fprintf('error_u = %e\n', errorU);
        write_ns3d_snapshot(iter, runDir, coord, paraD, state);
        stopIter = iter;
        converged = true;
        break
    end

    state = accept_ns3d_step(state);
end
end
