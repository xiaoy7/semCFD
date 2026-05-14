function ibm = setup_ibm(cfg, coordX, coordY, para, runDir)
if ~cfg.useIBM
    ibm.enabled = false;
    return
end

ibm = ibm_setup2d(coordX, coordY, cfg.dt, para);
if ibm.enabled && isfield(ibm, 'mask')
    write_immersed_body_nodes(runDir, coordX, coordY, ibm.mask);
end
end
