function y = applyPoissonPreconditioner(b, Tx, Ty, Tz, TxInv, TyInv, TzInv, Lambda3D)
    Nx = size(Tx, 1);
    b = reshape(b, Nx, Nx, Nx);
    u = tensorprod(b, TzInv', 3, 1);
    u = pagemtimes(u, TyInv');
    u = squeeze(tensorprod(TxInv, u, 2, 1));
    u = u ./ Lambda3D;
    u = tensorprod(u, Tz', 3, 1);
    u = pagemtimes(u, Ty');
    u = squeeze(tensorprod(Tx, u, 2, 1));
    y = u(:);
end

