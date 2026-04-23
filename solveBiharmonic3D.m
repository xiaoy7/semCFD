function U = solveBiharmonic3D(RHS, C, Lambda3D, Tx,Ty,Tz, TxInv,TyInv,TzInv)
% solve (I + C Δ^2) U = RHS  in eigen-space:
% (1 + C * Λ^2) * Uhat = RHShat

% forward transform to eigen-space
tmp   = tensorprod(RHS, TzInv', 3, 1);
tmp   = pagemtimes(tmp, TyInv');
RHSh  = squeeze(tensorprod(TxInv, tmp, 2, 1));  % hat

denom = 1 + C .* (Lambda3D.^2);
Uhat  = RHSh ./ denom;

% back
tmp = tensorprod(Uhat, Tz', 3, 1);
tmp = pagemtimes(tmp, Ty');
U   = squeeze(tensorprod(Tx, tmp, 2, 1));
end
