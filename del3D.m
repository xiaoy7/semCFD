function Lu = del3D(u, eigx, eigy, eigz)
    % Laplacian using diagonalized SEM
    [Nx, ~, ~] = size(u);
    U = u;
    % Transform to eigenbasis
    U = tensorprod(U, eigz, 3, 1);
    U = pagemtimes(U, eigy);
    U = squeeze(tensorprod(eigx, U, 2, 1));
    Lu = U;
end
