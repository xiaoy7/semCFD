function state = step_ns3d(cfg, sem, paraD, paraN, state)
[dUx, dUy, dUz] = gradient_ns3d(sem, paraD, state.u);
[dVx, dVy, dVz] = gradient_ns3d(sem, paraD, state.v);
[dWx, dWy, dWz] = gradient_ns3d(sem, paraD, state.w);

convectionU = state.u .* dUx + state.v .* dUy + state.w .* dUz;
convectionV = state.u .* dVx + state.v .* dVy + state.w .* dVz;
convectionW = state.u .* dWx + state.v .* dWy + state.w .* dWz;

uRhs = state.u / cfg.dt - convectionU;
vRhs = state.v / cfg.dt - convectionV;
wRhs = state.w / cfg.dt - convectionW;

fp = -(pagemtimes(sem.Dx, uRhs) + pagemtimes(vRhs, sem.DyT) ...
    + derivative_z(sem.Dz, wRhs, paraN.nx_all, paraN.ny_all, paraN.nz_all));
state.p = solve_poisson_ns3d(sem, paraN, fp);

[gradPx, gradPy, gradPz] = gradient_ns3d(sem, paraD, state.p);
fu = uRhs - gradPx;
fv = vRhs - gradPy;
fw = wRhs - gradPz;

fuInterior = fu(paraD.freeNodesx, paraD.freeNodesy, paraD.freeNodesz) ...
    - sem.uBoundaryContribution;
fvInterior = fv(paraD.freeNodesx, paraD.freeNodesy, paraD.freeNodesz);
fwInterior = fw(paraD.freeNodesx, paraD.freeNodesy, paraD.freeNodesz);

state.uNew(paraD.freeNodesx, paraD.freeNodesy, paraD.freeNodesz) = ...
    solve_helmholtz_ns3d(sem, fuInterior);
state.vNew(paraD.freeNodesx, paraD.freeNodesy, paraD.freeNodesz) = ...
    solve_helmholtz_ns3d(sem, fvInterior);
state.wNew(paraD.freeNodesx, paraD.freeNodesy, paraD.freeNodesz) = ...
    solve_helmholtz_ns3d(sem, fwInterior);

state.uNew(:, :, paraD.bcNodesz(2)) = cfg.lidVelocity;
state.vNew(:, :, paraD.bcNodesz(2)) = 0;
state.wNew(:, :, paraD.bcNodesz(2)) = 0;
end
