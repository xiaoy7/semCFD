classdef poisson_tensor_solver
% Fast tensor-product solvers using SEM eigen-bases
methods(Static)
    function U = helmholtz_solve(F, alpha, beta, T, Tinv, Lambda3D)
        % Solve (alpha I - beta Δ) U = F  on periodic box
        % alpha>0; beta>=0. Using eigen-basis of H=M^{-1}S (per dimension)
        % Transform F -> modal
        U = tensorprod(F, Tinv{3}', 3, 1);
        U = pagemtimes(U, Tinv{2}');
        U = squeeze(tensorprod(Tinv{1}, U, 2, 1));
        % divide by diagonal (alpha + beta*(λx+λy+λz))
        denom = alpha + beta*(Lambda3D);
        U = U ./ denom;
        % back_transform
        U = tensorprod(U, T{3}', 3, 1);
        U = pagemtimes(U, T{2}');
        U = squeeze(tensorprod(T{1}, U, 2, 1));
    end

    function U = poisson_solve(F, T, Tinv, Lambda3D)
        % Solve Δ U = F (periodic) with zero-mean constraint
        % Transform
        U = tensorprod(F, Tinv{3}', 3, 1);
        U = pagemtimes(U, Tinv{2}');
        U = squeeze(tensorprod(Tinv{1}, U, 2, 1));
        % divide by (λx+λy+λz); handle zero mode by setting it to zero
        denom = (Lambda3D);
        tiny = 1e-14;
        mask = abs(denom) > tiny;
        U = U .* mask ./ (denom + (~mask));   % zero out nullspace
        % back_transform
        U = tensorprod(U, T{3}', 3, 1);
        U = pagemtimes(U, T{2}');
        U = squeeze(tensorprod(T{1}, U, 2, 1));
    end
end
end
