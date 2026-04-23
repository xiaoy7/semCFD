function [x,w] = LegendreGL(N)
% LegendreGL: Gauss-Lobatto nodes x and weights w on [-1,1] for degree N.
% x and w are (N+1)x1. Endpoints included. Uses simple Newton refinement.

if N<1
    x = [-1; 1];
    w = [1; 1]*2;  % trivial
    return
end

x  = cos(pi*(0:N)'/N);  % Chebyshev-Lobatto initial guess
P0 = zeros(N+1,1); P1 = zeros(N+1,1);

% Newton iterations to find exact LGL by solving (1 - x^2) P'_N(x) = 0
for it = 1:100
    [P0,P1] = legendrePn(N, x);
    % P0 = P_N(x), P1 = P'_N(x)
    dx = -( (1 - x.^2).*P1 ) ./ (-2*x.*P1 + (1 - x.^2).*legendrePnPrime(N, x, P0, P1));
    x  = x + dx;
    if max(abs(dx)) < 1e-14, break; end
end

% Ensure endpoints exactly -1 and 1
x(1)   = -1; x(end) = 1;

% Weights: w_i = 2 / (N*(N+1) [P_N(x_i)]^2 )
[P0,~] = legendrePn(N, x);
w = 2./(N*(N+1)*(P0.^2));
end

function [P, dP] = legendrePn(N, x)
% Evaluate P_N(x) and P'_N(x) via recurrence
x = x(:);
P0 = ones(size(x));
if N==0
    P  = P0; dP = zeros(size(x));
    return
end
P1 = x;
for k=2:N
    Pk = ( (2*k-1).*x.*P1 - (k-1)*P0 )/k;
    P0 = P1; P1 = Pk;
end
P = P1;

% Derivative using relation: (1-x^2)P'_N = -N x P_N + N P_{N-1}
if N==0
    dP = zeros(size(x));
else
    % compute P_{N-1}
    Pm1 = ones(size(x));
    if N>1
        Pm2 = ones(size(x)); Pm1 = x;
        for k=2:N-1
            Pk = ( (2*k-1).*x.*Pm1 - (k-1)*Pm2 )/k;
            Pm2 = Pm1; Pm1 = Pk;
        end
    end
    dP = ( -N*x.*P + N*Pm1 ) ./ max(1 - x.^2, eps);
    % fix at endpoints using analytic derivative limits
    dP(1)   = 0.5*N*(N+1);   % limit as x->-1 (with sign)
    dP(end) = 0.5*N*(N+1)*(-1)^(N+1);
end
end

function d2 = legendrePnPrime(~, x, P, dP) %#ok<INUSD>
% Cheap approximation to second derivative wrt x used in Newton denom.
% Finite-diff surrogate near machine precision; good enough for refinement.
h  = 1e-12;
[~, dPph] = legendrePn(numel(P)-1, min(x+h, 1));
d2 = (dPph - dP)/h;
% clamp ends
d2(1)   = d2(2);
d2(end) = d2(end-1);
end
