function y = applySchrodingerOperator(u, Tx, Ty, Tz, TxInv, TyInv, TzInv, eigx, eigy, eigz, alpha, V)
    Nx = size(Tx, 1);
    u = reshape(u, Nx, Nx, Nx);
    % Apply Laplacian
    L = del3D(u, eigx, eigy, eigz);
    y = alpha * u - L + V .* u;
    y = y(:);
end

