function U = solveCHLinear3D(RHS, C1, C2, Lambda3D, Tx,Ty,Tz, TxInv,TyInv,TzInv)
% Solve (I + C1*Δ + C2*Δ^2) U = RHS.
% In spectral space (per mode with λ >= 0 for −Δ):
%    Δ  ->  −λ
%    Δ^2->  +λ^2
% So denom = 1 + C1*(-λ) + C2*(λ^2) = 1 - C1*λ + C2*λ^2
% BUT with the stabilized derivation we used:
% LHS was (I + C1*Δ + C2*Δ^2) => denom = 1 + C1*(-λ) + C2*(λ^2)

% Forward transform
tmp  = tensorprod(RHS, TzInv', 3, 1);
tmp  = pagemtimes(tmp, TyInv');
RHSh = squeeze(tensorprod(TxInv, tmp, 2, 1));

denom = 1 + (-C1)*Lambda3D + C2*(Lambda3D.^2);  % 1 - C1*λ + C2*λ^2
Uhat  = RHSh ./ denom;

% Back transform
tmp = tensorprod(Uhat, Tz', 3, 1);
tmp = pagemtimes(tmp, Ty');
U   = squeeze(tensorprod(Tx, tmp, 2, 1));
end
