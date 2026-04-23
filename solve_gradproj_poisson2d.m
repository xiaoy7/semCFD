function P = solve_gradproj_poisson2d(R, Tx,Ty, TxInv,TyInv, Lambda3D)
% R: (nx,ny,nz) RHS; Lambda3D(i,j,k)=λx(i)+λy(j)+λz(k)
% Tx,Ty,Tz: 1D eigenvector transforms; *Inv are their inverses (or pinv)
% Solve K P = R with Neumann kernel (zero-mean gauge)

% Forward transform
 
tmp  = pagemtimes(R, TyInv');
Rhat = squeeze(tensorprod(TxInv, tmp, 2, 1));  % (nx,ny,nz)

% Project out nullspace (λ=0 mode). Typically at index 1 for each axis.
% Zero the (1,1) component of RHS; set P̂(1,1)=0
Rhat(1,1) = 0;

% Divide by eigenvalue sum (avoid div-by-zero at (1,1))
P = Rhat ./ max(Lambda3D, eps);

% Back transform
tmp = pagemtimes(P, Ty');
P   = squeeze(tensorprod(Tx, tmp, 2, 1));
end
