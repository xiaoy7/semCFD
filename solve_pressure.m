function pressure = solve_pressure(rhs, Tx, Ty, invTx, invTy, poisson)
pressureHat = invTx * rhs * invTy';
pressureHat(1, 1) = 0;
pressureHat = pressureHat ./ poisson;
pressure = Tx * pressureHat * Ty';
end