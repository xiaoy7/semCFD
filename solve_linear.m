function u = solve_linear(sem, rhs, massCoef, mobility)
rhsHat = sem.invTx * rhs * sem.invTy';
if isfield(sem, 'lambda2')
    lambda2 = sem.lambda2;
else
    lambda2 = sem.lambda.^2;
end
uHat = rhsHat ./ (massCoef + mobility * lambda2);
u = sem.Tx * uHat * sem.Ty';
end
