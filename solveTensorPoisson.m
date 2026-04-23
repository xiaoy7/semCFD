

%% ------------------------------------------------------------------------
function p = solveTensorPoisson(rhs, Tx, Ty, invTx, invTy, lambdaSum, wx, wy)
%SOLVETENSORPOISSON Solve the Neumann Poisson system via tensor-product factorisation.
%   rhs      : weighted divergence evaluated at Gauss-Lobatto nodes.
%   Tx, Ty   : modal-to-nodal transforms (eigenvectors of the 1D operators).
%   invTx    : pseudo-inverse (nodal-to-modal transform) in x.
%   invTy    : pseudo-inverse in y.
%   lambdaSum: tensor sum of 1D Laplacian eigenvalues (lambda_x + lambda_y).
%   wx, wy   : 1D quadrature weights.
%
%   Follows the Kronecker-product formulation in README §3.2: the RHS is
%   converted to modal space, divided entrywise by lambda_x+lambda_y (with the zero mode removed), and transformed back to nodes with zero mean enforced.

    useGPU = isa(Tx, 'gpuArray');
    if useGPU
        rhs = gpuArray(rhs);
        wx = gpuArray(wx);
        wy = gpuArray(wy);
        lambdaSum = gpuArray(lambdaSum);
    end

    denom = sum(wx) * sum(wy);
    rhsMean = (wx.' * rhs * wy) / denom;
    rhs = rhs - rhsMean;

    rhsModal = invTx * rhs * invTy';
    divisor = lambdaSum;
    divisor(1, 1) = inf; % remove the null mode explicitly
    solModal = rhsModal ./ divisor;
    solModal(1, 1) = 0;

    p = Tx * solModal * Ty';
    pMean = (wx.' * p * wy) / denom;
    p = p - pMean;

    if useGPU
        p = gpuArray(p);
    end
end

%% ------------------------------------------------------------------------



