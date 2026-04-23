function LapU = applyLap3D(U, Tx,Ty,Tz, TxInv,TyInv,TzInv, Lambda3D)
% LapU = Δ U

tmp  = tensorprod(U, TzInv', 3, 1);
tmp  = pagemtimes(tmp, TyInv');
Uhat = squeeze(tensorprod(TxInv, tmp, 2, 1));

Uhat = Uhat .* (-Lambda3D);  % Δ eigenvalue is -λ

tmp  = tensorprod(Uhat, Tz', 3, 1);
tmp  = pagemtimes(tmp, Ty');
LapU = squeeze(tensorprod(Tx, tmp, 2, 1));
end
