function [x, e, T, eigL, W] = SEGenerator1D(dirTag, L, Param)
% SEGenerator1D
% Build 1D SEM (Q^Np) over [-L, L] with Ncell uniform elements.
% Returns:
%   x     : global GLL points (column)
%   e     : vector of ones(size(x))  (convenience)
%   T     : eigenvector matrix that diagonalizes H = M^{-1} S
%   eigL  : eigenvalues of (−Δ) in 1D (nonnegative)
%   W     : global quadrature weights (lumped) matching x
%
% Strategy:
% - Build local GLL nodes/weights of degree Np in reference [-1,1]
% - Map to each element; assemble global x and W
% - Assemble diagonal (lumped) M and sparse S for 1D Neumann Laplacian
% - Robust eigen-decomp with M^{-1/2} S M^{-1/2} (symmetric)

Np    = Param.Np;
switch lower(dirTag)
    case 'x', Ncell = Param.Ncellx;
    case 'y', Ncell = Param.Ncelly;
    case 'z', Ncell = Param.Ncellz;
    otherwise, error('dirTag must be x/y/z');
end
Ne    = Np*Ncell + 1;  % total global nodes

% Local GLL nodes/weights on [-1,1]
[xl, wl] = LegendreGL(Np);   % (Np+1)x1 each
% Element size
h  = (2*L)/Ncell;

% Build global grid & weights with exact node gluing
x  = zeros(Ne,1);
W  = zeros(Ne,1);
% map k-th element: ξ in [-1,1] -> x = x_c + (h/2)*ξ, with centers from -L + h/2, ...
for eID = 1:Ncell
    a = -L + (eID-1)*h; b = a + h;  % element bounds
    xe = ((b-a)/2)*xl + (a+b)/2;    % mapped nodes
    we = ((b-a)/2)*wl;              % mapped weights
    i0 = (eID-1)*Np + 1;
    x( i0:i0+Np ) = xe;
    W( i0:i0+Np ) = W( i0:i0+Np ) + we;  % accumulate weights at shared nodes
end

% Assemble 1D stiffness S and mass M (lumped) with GLL quadrature
% Use Lagrange basis at GLL points. For Neumann b.c., natural in weak form.
% Build local derivative matrix Dξ on reference, then scale to physical.
[Dref, wref] = refDiffMatrixGLL(xl, wl);  %#ok<ASGLU>  % Dref(i,j) = dℓ_j/dξ at node i
S = sparse(Ne,Ne);
M = spdiags(W, 0, Ne, Ne);   % lumped mass (diagonal with global weights)

% Loop elements to assemble S
for eID = 1:Ncell
    i0  = (eID-1)*Np + 1;
    ind = (i0:i0+Np).';
    % Phys mapping ξ->x: x = (h/2) ξ + x_c ; dx/dξ = h/2
    J  = h/2;
    % Grad wrt x: d/dx = (2/h) d/dξ
    % Local stiffness: S_e(i,j) = ∫ (dℓ_i/dx)(dℓ_j/dx) dx
    % = ∫ ( (2/h) dℓ_i/dξ) ( (2/h) dℓ_j/dξ ) (h/2) dξ = (2/h) ∫(dℓ_i/dξ dℓ_j/dξ) dξ
    % Numerically by GLL quadrature:
    De = Dref;   % at local nodes
    We = wl;
    Se = (2/h) * (De' * diag(We) * De);  % (Np+1)x(Np+1)
    % Scatter-add
    S(ind,ind) = S(ind,ind) + Se;
end

% Eigen-decomposition of H = M^{-1}S via symmetric form:
% S1 = M^{-1/2} S M^{-1/2}  (symmetric)
Minvhalf = spdiags(1./sqrt(max(W,eps)), 0, Ne, Ne);
S1       = Minvhalf * S * Minvhalf;
S1       = (S1 + S1.')/2;  % symmetry clean-up
% Solve S1 * Q = Q * Λ
[Q, Lambda] = eig(full(S1), 'vector');   % small 1D so full() ok
% T = M^{-1/2} Q ;  H = T Λ T^{-1}
T    = Minvhalf * Q;
eigL = max(Lambda, 0);  % guard tiny negatives

e = ones(Ne,1);
end

function [D,w] = refDiffMatrixGLL(xl, wl)
% Derivative matrix D at GLL nodes on [-1,1] for Lagrange basis.
% D(i,j) = dℓ_j/dξ evaluated at ξ = xl(i)
% Classic formula using barycentric weights.
N = numel(xl)-1;
w  = wl(:);
D  = zeros(N+1,N+1);
% Barycentric weights for GLL nodes:
c = ones(N+1,1);
c(1) = 2; c(end) = 2;
c = c .* ((-1).^(0:N)).';
for i=1:N+1
    for j=1:N+1
        if i~=j
            D(i,j) = c(j)/(c(i)*(xl(i)-xl(j)));
        end
    end
end
D(1,1)   = -sum(D(1,2:end));
D(end,end)= -sum(D(end,1:end-1));
for i=2:N
    D(i,i) = -sum(D(i,[1:i-1,i+1:end]));
end
end
