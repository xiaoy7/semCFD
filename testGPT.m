clc
clear


% Parameters
Np = 5;                   % Polynomial degree
Ncell = 20;               % Number of elements per dimension
L = 1;                    % Domain [-L, L]^3
alpha = 1;
beta = 10;
Nx = Ncell * Np + 1;

% Setup SEM matrices in x, y, z
[x, ex, Tx, eigx] = SEGenerator1D('x', L, struct('Np', Np, 'Ncellx', Ncell, 'nx', Nx));
[y, ey, Ty, eigy] = SEGenerator1D('y', L, struct('Np', Np, 'Ncelly', Ncell, 'ny', Nx));
[z, ez, Tz, eigz] = SEGenerator1D('z', L, struct('Np', Np, 'Ncellz', Ncell, 'nz', Nx));

TxInv = pinv(Tx); TyInv = pinv(Ty); TzInv = pinv(Tz);

% Generate meshgrid
[X, Y, Z] = ndgrid(x, y, z);
V = beta * sin(pi * X / 4).^2 .* sin(pi * Y / 4).^2 .* sin(pi * Z / 4).^2;

% Define exact solution and compute RHS
u_exact = cos(pi*X/16).*cos(pi*Y/16).*cos(pi*Z/16);
Lu = alpha * u_exact - del3D(u_exact, eigx, eigy, eigz) + V .* u_exact;
f = Lu;

% Flatten for PCG
rhs = f(:);
u0 = zeros(size(rhs));

% Preconditioner (Poisson inverse)
Lambda3D = eigx + permute(eigy, [2 1]) + permute(eigz, [3 1 2]);
Lambda3D = alpha + 0.5 * beta + Lambda3D;
invLaplacian = @(b) applyPoissonPreconditioner(b, Tx, Ty, Tz, TxInv, TyInv, TzInv, Lambda3D);

% Operator function for A * u = f
Afun = @(u) applySchrodingerOperator(u, Tx, Ty, Tz, TxInv, TyInv, TzInv, eigx, eigy, eigz, alpha, V);

% Solve using PCG
tol = 1e-10;
maxit = 50;
[u_sol, flag, relres] = pcg(Afun, rhs, tol, maxit, invLaplacian);

% Reshape solution
u_sol = reshape(u_sol, Nx, Nx, Nx);
fprintf("PCG finished with flag %d and relative residual %e\n", flag, relres);
