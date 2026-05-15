function [sem, paraD, paraN, coord] = setup_ns3d_sem(cfg)
paraD = build_ns3d_parameters(cfg, 'dirichlet');
paraN = build_ns3d_parameters(cfg, 'neumann');
paraD = parameter_bc(paraD);
paraN = parameter_bc(paraN);

fprintf('Laplacian is Q%d spectral element method \n', cfg.Np);
if cfg.Np < 2
    fprintf('It is also classical second order discrete Laplacian \n');
else
    fprintf('It is also a %d-th order accurate finite difference scheme \n', ...
        cfg.Np + 2);
end

[~, x, Tx, ~, lambdaX, DmatrixX] = cal_matrix2('x', paraD);
[~, y, Ty, ~, lambdaY, DmatrixY] = cal_matrix2('y', paraD);
[~, z, Tz, ~, lambdaZ, DmatrixZ] = cal_matrix2('z', paraD);
DmatrixX = full(DmatrixX);
DmatrixY = full(DmatrixY);
DmatrixZ = full(DmatrixZ);
DmatrixYT = DmatrixY';

[~, ~, Txp, lambdaXp] = cal_matrix2_1('x', paraN, x);
[~, ~, Typ, lambdaYp] = cal_matrix2_1('y', paraN, y);
[~, ~, Tzp, lambdaZp] = cal_matrix2_1('z', paraN, z);

[coordY, coordX, coordZ] = meshgrid(y, x, z);
coord = [coordX(:), coordY(:), coordZ(:)];

sem.Tx = Tx;
sem.Ty = Ty;
sem.Tz = Tz;
sem.Txp = Txp;
sem.Typ = Typ;
sem.Tzp = Tzp;
sem.invTx = pinv(Tx);
sem.invTy = pinv(Ty);
sem.invTz = pinv(Tz);
sem.invTxp = pinv(Txp);
sem.invTyp = pinv(Typ);
sem.invTzp = pinv(Tzp);
sem.Dx = DmatrixX;
sem.DyT = DmatrixYT;
sem.Dz = DmatrixZ;

sem.helmholtzU = cfg.alphaHelmholtz + cfg.nu ...
    * eigenvalue_tensor3d(lambdaX, lambdaY, lambdaZ);
sem.poissonP = eigenvalue_tensor3d(lambdaXp, lambdaYp, lambdaZp);
sem.poissonP(abs(sem.poissonP) < 1e-12) = 1;
sem.uBoundaryContribution = boundary_laplacian_contribution(cfg, sem, paraD);
end
