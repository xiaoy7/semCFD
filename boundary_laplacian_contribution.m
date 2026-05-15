function contribution = boundary_laplacian_contribution(cfg, sem, para)
uBoundary = zeros(para.nx_all, para.ny_all, para.nz_all);
uBoundary(:, :, para.bcNodesz(2)) = cfg.lidVelocity;
lapBoundary = calLaplace3D(uBoundary, sem.Dx, sem.DyT, sem.Dz, para);
contribution = -cfg.nu * lapBoundary(para.freeNodesx, ...
    para.freeNodesy, para.freeNodesz);
end
