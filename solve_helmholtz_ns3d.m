function u = solve_helmholtz_ns3d(sem, rhs)
uHat = tensorprod(rhs, sem.invTz', 3, 1);
uHat = pagemtimes(uHat, sem.invTy');
uHat = squeeze(tensorprod(sem.invTx, uHat, 2, 1));
uHat = uHat ./ sem.helmholtzU;

u = tensorprod(uHat, sem.Tz', 3, 1);
u = pagemtimes(u, sem.Ty');
u = squeeze(tensorprod(sem.Tx, u, 2, 1));
end
