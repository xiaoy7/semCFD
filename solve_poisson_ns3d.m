function p = solve_poisson_ns3d(sem, para, rhs)
rhsInterior = rhs(para.freeNodesx, para.freeNodesy, para.freeNodesz);
pHat = tensorprod(rhsInterior, sem.invTzp', 3, 1);
pHat = pagemtimes(pHat, sem.invTyp');
pHat = squeeze(tensorprod(sem.invTxp, pHat, 2, 1));
pHat = pHat ./ sem.poissonP;

pInterior = tensorprod(pHat, sem.Tzp', 3, 1);
pInterior = pagemtimes(pInterior, sem.Typ');
pInterior = squeeze(tensorprod(sem.Txp, pInterior, 2, 1));
pInterior = pInterior - pInterior(1, 1, 1);

p = zeros(para.nx_all, para.ny_all, para.nz_all, 'like', rhs);
p(para.freeNodesx, para.freeNodesy, para.freeNodesz) = pInterior;
end
