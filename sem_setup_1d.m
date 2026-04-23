function [x, w, M, S, D, T, Tinv, lambdaH] = sem_setup_1d(k, Nc, L)
% sem_setup_1d  Build 1D SEM operators for each dimension and perform eigen-decomposition.
% Returns cell arrays for x{1:3}, w{1:3}, M{1:3}, S{1:3}, D{1:3}, T{1:3}, Tinv{1:3}, lambdaH{1:3}.
%
% k  : polynomial degree per element (Qk, using k+1 GLL nodes)
% Nc : [Nx, Ny, Nz] number of cells
% L  : [Lx, Ly, Lz] half-domain sizes (domain is [-Lx, Lx], etc.)
%
% Collocation SEM with Gauss–Lobatto–Legendre nodes in each element.

dims = 3;
x = cell(1,dims); w = cell(1,dims);
M = cell(1,dims); S = cell(1,dims); D = cell(1,dims);
T = cell(1,dims); Tinv = cell(1,dims); lambdaH = cell(1,dims);

% GLL nodes/weights on [-1,1]
[xgll, wgll] = gll(k+1);

for d=1:dims
    Ne   = Nc(d);
    Ld   = L(d);
    edges = linspace(-Ld, Ld, Ne+1);
    nd    = Ne*k + 1;
    xd    = zeros(nd,1);
    wd    = zeros(nd,1);
    Md    = zeros(nd);
    Sd    = zeros(nd);
    Dd    = zeros(nd);
    row = 1;
    for e = 1:Ne
        a = edges(e); b = edges(e+1);
        jac = (b-a)/2;  cen = (a+b)/2;
        xe  = cen + jac*xgll;            % physical nodes
        we  = jac*wgll;                  % physical weights

        % Lagrange basis and derivative at GLL nodes (reference)
        [Le, dLe_dxi] = lagrange_basis_and_deriv(xgll, xgll);
        dLe_dx = (1/jac)*dLe_dxi;        % d/dx = (1/jac) d/dxi

        % local mass and stiffness (reference integrals scaled)
        Me = (Le'*diag(wgll)*Le) * jac;     % scales with jac
        Se = (dLe_dxi'*diag(wgll)*dLe_dxi) / jac; % scales with 1/jac

        De = dLe_dx;                     % collocation derivative

        idx = row:(row+k);
        xd(idx) = xe;
        wd(idx) = wd(idx) + we;          % accumulate (shared nodes)
        Md(idx,idx) = Md(idx,idx) + Me;
        Sd(idx,idx) = Sd(idx,idx) + Se;
        Dd(idx,idx) = Dd(idx,idx) + De;
        row = row + k; % shared endpoint
    end

    x{d} = xd; w{d} = wd; M{d} = Md; S{d} = Sd; D{d} = Dd;

    % Robust symmetric generalized eig: S v = λ M v
    % Let A = M^{-1/2} S M^{-1/2} = Q Λ Q^T, then H = M^{-1}S = T Λ T^{-1} with T=M^{-1/2} Q
    [Q, Lambda] = eig( (Md^(-1/2)) * Sd * (Md^(-1/2)) );
    T{d}        = (Md^(-1/2)) * Q;
    Tinv{d}     = Q' * (Md^(1/2));
    lambdaH{d}  = diag(Lambda); % eigenvalues of H = M^{-1} S
end
end

%% -------- utilities --------
function [x, w] = gll(n)
% Return n Gauss–Lobatto–Legendre nodes x and weights w on [-1,1].
if n<2, error('gll requires n>=2'); end
x = zeros(n,1); w = zeros(n,1);
x(1) = -1; x(end) = 1;
% initial guesses (Chebyshev–Lobatto)
for k=2:n-1
    x0 = -cos(pi*(k-1)/(n-1));
    x(k) = newton_legendre_deriv_root(x0, n-1);
end
% weights: w_i = 2 / [ (n-1)n * (P_{n-1}(x_i))^2 ]
for i=1:n
    P = legendreP_eval(n-1, x(i));
    w(i) = 2 / ( (n-1)*n * (P*P) );
end
end

function xi = newton_legendre_deriv_root(x0, m)
% find root of P'_m(x) via Newton
xi = x0;
for it=1:50
    [Pm, dPm, ddPm] = legendre_eval_all(m, xi);
    dx = -dPm / ddPm;
    xi = xi + dx;
    if abs(dx) < 1e-14, break; end
end
end

function [P, dP, ddP] = legendre_eval_all(m, x)
% evaluate P_m(x), P'_m(x), P''_m(x)
P0 = 1; P1 = x;
if m==0, P=P0; dP=0; ddP=0; return; end
if m==1, P=P1; dP=1; ddP=0; return; end
Pkm2 = P0; Pkm1 = P1;
for k=2:m
    Pk = ( (2*k-1)*x*Pkm1 - (k-1)*Pkm2 ) / k;
    Pkm2 = Pkm1; Pkm1 = Pk;
end
P = Pkm1;

% derivative via identity: (1-x^2) P'_m = -m x P_m + m P_{m-1}
Pmm2 = 1; Pmm1 = x;
if m==1
    Pm1 = 1;
else
    for k=2:m-1
        Pk = ( (2*k-1)*x*Pmm1 - (k-1)*Pmm2 ) / k;
        Pmm2 = Pmm1; Pmm1 = Pk;
    end
    if m==1, Pm1=1; else, Pm1 = Pmm1; end
end
dP = ( -m*x*P + m*Pm1 ) / max(1 - x*x, eps);

% second derivative (finite-diff approx to keep code compact)
h = 1e-8;
dPp = (legendreP_eval(m, x+h) - legendreP_eval(m, x-h)) / (2*h);
ddP = (dPp - dP) / h;
end

function P = legendreP_eval(m, x)
% Legendre polynomial P_m(x)
if m==0, P=1; return; end
if m==1, P=x; return; end
Pkm2 = 1; Pkm1 = x;
for k=2:m
    Pk = ( (2*k-1)*x*Pkm1 - (k-1)*Pkm2 ) / k;
    Pkm2 = Pkm1; Pkm1 = Pk;
end
P = Pkm1;
end

function [L, dL] = lagrange_basis_and_deriv(xnodes, xeval)
% Build Lagrange basis L(i,j) = l_j(x_i), i over xeval, j over xnodes (square if same set)
n = numel(xnodes);
L  = zeros(n);
dL = zeros(n);
% precompute denominators
denom = ones(n,1);
for j=1:n
    for m=1:n
        if m~=j, denom(j) = denom(j) * (xnodes(j)-xnodes(m)); end
    end
end
for i=1:n
    xi = xeval(i);
    for j=1:n
        % basis value at xi
        num = 1.0;
        for m=1:n
            if m~=j, num = num * (xi - xnodes(m)); end
        end
        L(i,j) = num / denom(j);
        % derivative sum
        s = 0.0;
        for t=1:n
            if t~=j
                num2 = 1.0;
                for m=1:n
                    if m~=j && m~=t, num2 = num2*(xi - xnodes(m)); end
                end
                s = s + num2;
            end
        end
        dL(i,j) = s / denom(j);
    end
end
end
