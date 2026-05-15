function results = run_ns3d_case(cfg, runDir)
[sem, paraD, paraN, coord] = setup_ns3d_sem(cfg);
state = initial_ns3d_state(cfg, paraD);

device = select_device();
if strcmp(device.type, 'gpu')
    fprintf('GPU computation: loading matrices and fields\n');
    [sem, state] = move_ns3d_to_gpu(sem, state);
    wait(device.handle);
    fprintf('GPU computation: loading finished\n');
end

tic;
fprintf('=== start time stepping ===\n');
[state, stopIter, converged] = advance_ns3d(cfg, sem, paraD, paraN, ...
    state, runDir, coord, device);
elapsedTime = toc;

stateForSave = gather_ns3d_struct(state);
save(fullfile(runDir, 'flow.mat'), 'cfg', 'stateForSave', ...
    'stopIter', 'converged', 'elapsedTime');

fprintf('Total computation time: %f seconds\n', elapsedTime);
fprintf('=== Program Ends ===\n');

results.config = cfg;
results.runDir = runDir;
results.stopIter = stopIter;
results.converged = converged;
results.elapsedTime = elapsedTime;
end
