function sem = setup_sem_dirichlet(cfg, ncell)
param.Np = cfg.Np;
param.Npx = cfg.Np;
param.Npy = cfg.Np;
param.basis = 'SEM';
param.minx = cfg.domain(1);
param.maxx = cfg.domain(2);
param.miny = cfg.domain(3);
param.maxy = cfg.domain(4);
param.Ncellx = ncell;
param.Ncelly = ncell;
param.nx_all = ncell * cfg.Np + 1;
param.ny_all = ncell * cfg.Np + 1;
param.bc = 'dirichlet';
param = parameter_bc2d(param);

[~, x, Tx, ~, lambda_x, ~, ~] = cal_matrix2('x', param);
[~, y, Ty, ~, lambda_y, ~, ~] = cal_matrix2('y', param);

sem.x = x(param.freeNodesx);
sem.y = y(param.freeNodesy);
sem.Tx = Tx;
sem.Ty = Ty;
sem.invTx = pinv(Tx);
sem.invTy = pinv(Ty);
sem.lambda = bsxfun(@plus, lambda_x, lambda_y');
[sem.X, sem.Y] = ndgrid(sem.x, sem.y);

wx = assemble_sem_weights(param.Npx, param.Ncellx, param.minx, param.maxx);
wy = assemble_sem_weights(param.Npy, param.Ncelly, param.miny, param.maxy);
sem.mass = wx(param.freeNodesx) * wy(param.freeNodesy)';
end

