function u = solve_linear(sem, rhs, massCoef, mobility)
rhsHat = sem.invTx * rhs * sem.invTy';
uHat = rhsHat ./ (massCoef + mobility * sem.lambda.^2);
u = sem.Tx * uHat * sem.Ty';
end
