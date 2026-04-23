
function [D, r, w] = LegendreD(N)
    % LegendreD compute D = differentiation matrix, r&w are Legendre Gauss-Lobatto
    % quadrature points and weights
    Np = N + 1;
    r = JacobiGL(0, 0, N);
    % weight is equal to 2/(N*(N+1))*1/(L_N(r))^2 where L_N is the Legendre
    % polynomial; JacobiP is the normalized Jacobi polynomial, i.e.,
    % JacobiP(0,0,r)=L_N(r)*sqrt((2*N+1)/2)
    w = (2 * N + 1) / (N ^ 2 + N) ./ (JacobiP(r, 0, 0, N)) .^ 2;

    % Let l_i(r) be the Lagrange interpolation polynomial
    % Next compute the derivative of l_j(r) evaluated at r_i. We have
    % D_{ij}=l'_j(r_i)=omega_j/omega_i/(r_i-r_j), for i not equal to j.
    % see page 81 in D. Kopriva's book on spectral method.

    Distance = r * ones(1, N + 1) - ones(N + 1, 1) * r' + eye(N + 1);
    omega = prod(Distance, 2);
    D = diag(omega) * (1 ./ Distance) * diag(1 ./ omega); % off diagonal entries
    % compute the diag entries by Negative Sum Tricks, see p.81 in Kopriva's book
    D(1:Np + 1:end) = 0;
    D(1:Np + 1:end) = -sum(D, 2);
end
