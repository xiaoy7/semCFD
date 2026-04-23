classdef sem_tensor_ops
    methods(Static)
        function [ux, uy, uz] = grad(U, D1D)
            % Gradient via tensor contractions with 1D differentiation matrices
            % ux = dU/dx, etc.  Assumes tensor grid with collocation at GLL nodes.
            % Along x: D_x * U (mode-1)
            ux = squeeze(tensorprod(D1D{1}, U, 2, 1));
            % Along y: U * D_y^T (mode-2)
            uy = pagemtimes(U, D1D{2}');
            % Along z: multiply on mode-3
            uz = tensorprod(U, D1D{3}', 3, 1);
        end

        function d = div(u, v, w, D1D)
            % Divergence of vector field (u,v,w) using SEM collocation derivatives
            [ux,~,~] = sem_tensor_ops.grad(u, D1D);
            [~,vy,~] = sem_tensor_ops.grad(v, D1D);
            [~,~,wz] = sem_tensor_ops.grad(w, D1D);
            d = ux + vy + wz;
        end

        function LapU = laplacian(U, lambdaH, T, Tinv)
            % Optional: -Δ U via eigen-basis: (λx+λy+λz) * modal(U)
            Uhat = tensorprod(U, Tinv{3}', 3, 1);
            Uhat = pagemtimes(Uhat, Tinv{2}');
            Uhat = squeeze(tensorprod(Tinv{1}, Uhat, 2, 1));
            lam = sem_tensor_ops.lambda3D(lambdaH);
            Uhat = Uhat .* lam;
            LapU = tensorprod(Uhat, T{3}', 3, 1);
            LapU = pagemtimes(LapU, T{2}');
            LapU = squeeze(tensorprod(T{1}, LapU, 2, 1));
        end

        function Lam3D = lambda3D(lambdaH)
            % Build Λ3D(i,j,k) = λx(i) + λy(j) + λz(k) with broadcasting-safe ndgrid.
            lamx = lambdaH{1}(:);   % Nx x 1
            lamy = lambdaH{2}(:);   % Ny x 1
            lamz = lambdaH{3}(:);   % Nz x 1

            % Use ndgrid so sizes are exactly (Nx, Ny, Nz)
            [LX, LY, LZ] = ndgrid(lamx, lamy, lamz);

            Lam3D = LX + LY + LZ;   % Nx x Ny x Nz
        end

    end
end
