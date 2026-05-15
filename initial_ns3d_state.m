function state = initial_ns3d_state(cfg, para)
zeroField = zeros(para.nx_all, para.ny_all, para.nz_all);
state.u = zeroField;
state.v = zeroField;
state.w = zeroField;
state.p = zeroField;
state.uNew = zeroField;
state.vNew = zeroField;
state.wNew = zeroField;

state.u(:, :, para.bcNodesz(2)) = cfg.lidVelocity;
state.uNew(:, :, para.bcNodesz(2)) = cfg.lidVelocity;
end
