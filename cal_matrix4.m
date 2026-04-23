function [varargout] = cal_matrix4(direction, Parameter,mesh)
% input:
% n: number of grid points
% the domain is [-L, L]

switch mesh.basis
    case 'FFT' % fast fourier transform

        switch direction
            case 'x'
                n = mesh.nx;
            case 'y'
                n = mesh.ny;
            case 'z'
                n = mesh.nz;
        end

        %% The computational grid
        L = mesh.length;
        h = 2 * L / n;
        x = (-L:h:L)';
        x = x(1:end - 1);
        %% generate the discrete Laplacian
        ex = ones(n, 1);
        K = spdiags([-ex 2 * ex -ex], [-1 0 1], n, n);
        K(1, n) = -1; K(n, 1) = -1;
        K = K / h ^ 2;
        K = full(K);
        T = dftmtx(n);
        lambda = 2 - 2 * cos(2 * pi / n * (0:n - 1));
        lambda = lambda' / h ^ 2;

        varargout{1} = h;
        varargout{2} = x;
        varargout{3} = ex;
        varargout{4} = T;
        varargout{5} = K;
        varargout{6} = lambda;
    case 'SEM' % spectral element method
        %% generate matrices and grids for P^N spectral element CG with
        % Neumann, Dirichlet, Periodic b.c. for -u''=f
        % outputs:
        % M*(dx/2) is the mass matrix;
        % S/(dx/2) is the stiffness matrix
        % H/(dx/2)^2=Mass^{-1}*Stiffness

        % Compute basic Legendre Gauss Lobatto grid, weights and differentiation
        % matrix D where D_{ij}=l'_j(r_i) and l_j(r) is the Lagrange polynomial.
        switch direction
            case 'x'
                N = mesh.Npx;
                Ncell = mesh.Ncellx;
                n = mesh.nx;
                % Domain is [Left, Right]
                Left = mesh.minx;
                Right = mesh.maxx;
                freeNodes = Parameter.freeNodesx;
            case 'y'
                N = mesh.Npy;
                Ncell = mesh.Ncelly;
                n = mesh.ny;
                % Domain is [Left, Right]
                Left = mesh.miny;
                Right = mesh.maxy;
                freeNodes = Parameter.freeNodesy;
            case 'z'
                N = mesh.Npz;
                Ncell = mesh.Ncellz;
                n = mesh.nz;
                % Domain is [Left, Right]
                Left = mesh.minz;
                Right = mesh.maxz;
                freeNodes = Parameter.freeNodesz;
        end
        % Generate the mesh with Ncell intervals and each interval has GL points
        [D, r, w] = LegendreD(N);


        Length = Right - Left;
        dx = Length / Ncell;

        for j = 1:Ncell
            cell_left = Left + dx * (j - 1);
            local_points = cell_left + dx / 2 + r * dx / 2;

            if (j == 1)
                x = local_points;
            else
                x = [x; local_points(2:end)];
            end

        end

        S_local = D' * diag(w) * D; % local stiffness matrix for each element
        S = [];
        M = [];

        for j = 1:Ncell
            S = blkdiag(S_local, S); % global stiffness and lumped mass matrices if treating cells sperately
            M = blkdiag(diag(w), M);
        end

        % Next step: "glue" the cells
        Np = N + 1; % number of points in each cell
        Glue = sparse(zeros(Ncell * Np - Ncell + 1, Ncell * Np));

        for j = 1:Ncell
            rowstart_index = (j - 1) * Np + 2 - j;
            rowend_index = rowstart_index + Np - 1;
            colstart_index = (j - 1) * Np + 1;
            colend_index = colstart_index + Np - 1;
            Glue(rowstart_index:rowend_index, colstart_index:colend_index) = speye(Np);
        end

        switch Parameter.bc
            case 'periodic'
                Glue(1, end) = 1;
                Glue(end, 1) = 1;
        end

        S = Glue * S * Glue';
        M = Glue * M * Glue';
        H = diag(1 ./ diag(M)) * S;

        switch Parameter.bc
            case 'dirichlet'
                S = S(freeNodes, freeNodes);
                M = M(freeNodes, freeNodes);
                H = diag(1 ./ diag(M)) * S;
                ex = ones(n, 1);
            case 'periodic'
                S = S(1:end - 1, 1:end - 1);
                M = M(1:end - 1, 1:end - 1);
                H = H(1:end - 1, 1:end - 1);
                x = x(1:end - 1);
                ex = ones(n, 1);
            case 'neumann'
                ex = ones(n, 1);
        end

        % the the discrete Laplacian is (H kron I+ I kron H)*vec(U)
        % which is (T kron T)([eigenvalues]*)(T^{-1} kron T^{-1})vec(U)

        % for stability, solve a symmetric eigenvalue problem numerically:
        % H=M^{-1}S=M^{-1/2}S1 M^{1/2}=M^{-1/2}T lambda T^{-1} M^{1/2}=V lambda V^{-1}
        M_half_inv = diag(1 ./ sqrt(diag(M)));
        S1 = M_half_inv * S * M_half_inv;
        S1 = (S1 + S1') / 2;
        [U, d] = eig(S1, 'vector');
        [lambda, index_sort] = sort(d);
        T = U(:, index_sort);
        h = dx / 2;
        lambda = lambda / h ^ 2;
        % after this step, T is the eigenvector of H
        T = M_half_inv * T;
        H = full(H / h ^ 2);

        % S1 = sparse(S1 / h ^ 2);
        % M = sparse(M);
        % S = S / h;
        % M = full(M * h);

        varargout{2} = x;
        varargout{3} = ex;
        varargout{4} = T;
        varargout{5} = H;
        varargout{6} = lambda;

end
