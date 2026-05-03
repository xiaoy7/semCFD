function Lu = apply_positive_lap(sem, u)
uhat = sem.invTx * u * sem.invTy';
Lu = sem.Tx * (sem.lambda .* uhat) * sem.Ty';
end
