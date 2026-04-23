function [varargout] = cal_matrix2_1(direction, Parameter,x)
% input:
% n: number of grid points
% the domain is [-L, L]


switch Parameter.basis
    case 'FFT' % fast fourier transform

        switch direction
            case 'x'
                n = Parameter.nx;
            case 'y'
                n = Parameter.ny;
            case 'z'
                n = Parameter.nz;
        end

        %% The computational grid
        L = Parameter.length;
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
                N = Parameter.Npx;
                Ncell = Parameter.Ncellx;
                n = Parameter.nx;
                % Domain is [Left, Right]
                Left = Parameter.minx;
                Right = Parameter.maxx;
                freeNodes = Parameter.freeNodesx;
            case 'y'
                N = Parameter.Npy;
                Ncell = Parameter.Ncelly;
                n = Parameter.ny;
                % Domain is [Left, Right]
                Left = Parameter.miny;
                Right = Parameter.maxy;
                freeNodes = Parameter.freeNodesy;
            case 'z'
                N = Parameter.Npz;
                Ncell = Parameter.Ncellz;
                n = Parameter.nz;
                % Domain is [Left, Right]
                Left = Parameter.minz;
                Right = Parameter.maxz;
                freeNodes = Parameter.freeNodesz;
        end

        % Generate the mesh with Ncell intervals and each interval has GL points
        [D, ~, w] = LegendreD(N);

        Length = Right - Left;
        dlx = Length / Ncell;

        S_local = D' * diag(w) * D; % local stiffness matrix for each element
        S = [];
        M = [];

        for j = 1:Ncell
            % global stiffness and lumped mass matrices if treating cells sperately
            S = blkdiag(S_local, S); % blkdiag 自动创建分块对角矩阵
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
  
        switch Parameter.bc
            case 'dirichlet'
                S = S(freeNodes, freeNodes);
                M = M(freeNodes, freeNodes);
                ex = ones(n, 1);
            case 'periodic'
                S = S(1:end - 1, 1:end - 1);
                M = M(1:end - 1, 1:end - 1);
                x = x(1:end - 1);
                ex = ones(n, 1);
            case 'neumann'
                ex = ones(n, 1);
        end

        % the the discrete Laplacian is (H kron I+ I kron H)*vec(U)
        % which is (T kron T)([eigenvalues]*)(T^{-1} kron T^{-1})vec(U)

        % for stability, solve a symmetric eigenvalue problem numerically:
        % H = M^{-1}S = M^{-1/2} S1 M^{1/2} = M^{-1/2}T lambda T^{-1} M^{1/2}=V lambda V^{-1}
        M_half_inv = diag(1 ./ sqrt(diag(M)));
        S1 = M_half_inv * S * M_half_inv;
        S1 = (S1 + S1') / 2;
        [U, d] = eig(S1, 'vector');
        [lambda, index_sort] = sort(d);
        T = U(:, index_sort);
        h = dlx / 2;
        lambda = lambda / h ^ 2;
        % after this step, T is the eigenvector of H
        T = M_half_inv * T;

        % S1 = sparse(S1 / h ^ 2);
        % M = sparse(M);
        % S = S / h;
        % M = full(M * h);

        varargout{1} = h;
        varargout{2} = ex;
        varargout{3} = T;
        varargout{4} = lambda;
 
 
end
