function [x, w, D] = gll_nodes_weights_and_D(N)
% Return Gauss–Lobatto–Legendre nodes x in [-1,1], weights w, and spectral derivative matrix D.
% N = number of GLL nodes (N >= 2).
%
% Implementation:
% 1) GLL nodes are roots of (1 - x^2) P'_n(x) plus endpoints (-1,1).
% 2) Weights: w_i = 2/(N(N-1)[P_{N-1}(x_i)]^2).
% 3) D: standard spectral collocation differentiation matrix on GLL nodes.

assert(N>=2);

% Compute nodes using eigenvalue approach or Newton on L'(x).
% Here: use known routine for Legendre and Jacobi polynomials to build.
% We'll use a simple Newton iteration for interior points with good initial guesses.

% Initial guesses (Chebyshev-Lobatto)
k  = (0:N-1)';
x0 = -cos(pi * k/(N-1));

% Newton iteration for interior points to solve (1 - x^2) P'_{N-1}(x) = 0  (i.e., P'_{N-1}(x) = 0)
x  = x0;
for it = 1:100
    [P, dP] = legendre_P_and_dP(N-1, x);
    % GLL condition: derivative of P_{N-1} vanishes in interior
    f  = dP;
    % derivative of f: second derivative of P_{N-1} -> use three-term if needed
    % approximate by finite diff of dP wrt x via recurrence derivative
    % but a stable trick: derivative of dP from identity:
    % (1 - x^2) P'_{n} = -n x P_{n} + n P_{n-1}
    % differentiate: -2x P'_n + (1 - x^2) P''_n = -n P_n - n x P'_n + n P'_{n-1}
    % We approximate f' numerically here for robustness
    h = 1e-12;
    [~, dPph] = legendre_P_and_dP(N-1, x+h);
    fprime = (dPph - dP)/h;

    dx = -f ./ fprime;
    x  = x + dx;
    if max(abs(dx)) < 1e-14, break; end
end

% Force endpoints exactly
x(1)   = -1.0;
x(end) =  1.0;

% Legendre at nodes, and weights
[Pn_1, ~] = legendre_P_and_dP(N-1, x);
w = zeros(N,1);
w(2:N-1) = 2 ./ ( (N-1)*N * (Pn_1(2:N-1)).^2 );
w(1)     = 2 / ( (N-1)*N );   % at x=-1, P_{N-1}(-1)=(-1)^{N-1}
w(end)   = w(1);              % symmetry

% Derivative matrix (standard spectral formula)
D = zeros(N,N);
c = ones(N,1);
c(1)   = 2;  c(end) = 2;
for i=1:N
    for j=1:N
        if i~=j
            D(i,j) = ( c(i)/c(j) ) * ( (-1)^(i+j) / (x(i)-x(j)) );
        end
    end
end
D(1,1)     = -(N-1)*N/4;
D(N,N)     =  (N-1)*N/4;
for i=2:N-1
    D(i,i) = - x(i) / (1 - x(i)^2);
end
end

function [P, dP] = legendre_P_and_dP(n, x)
% Evaluate P_n(x) and P_n'(x) for all x (vector) via recurrence
x = x(:);
P  = zeros(size(x));
dP = zeros(size(x));
if n==0
    P(:) = 1; dP(:) = 0; return;
elseif n==1
    P = x;    dP(:) = 1;
    return;
end
P0 = ones(size(x));
P1 = x;
for k = 2:n
    Pk = ( (2*k-1).*x.*P1 - (k-1).*P0 ) / k;
    P0 = P1; P1 = Pk;
end
P = P1;
% Derivative using identity: (1 - x^2) P'_n = -n x P_n + n P_{n-1}
dP = (-n.*x.*P + n.*P0) ./ max(1e-30, (1 - x.^2));
end
