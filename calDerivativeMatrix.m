function  DMx = calDerivativeMatrix(N, Ncell, dx, D,nx_all)
%*************************************************************************
% Generate the global differentiation matrix DMx
%*************************************************************************

% Input:
% N: polynomial degree
% Ncell: number of cells in finite element
% dx: element length
% D: differentiation matrix in one cell
% r: local points in one cell
% w: weights in one cell

% Output:
% DMx: global differentiation matrix

%*************************************************************************

Np = N + 1; % Number of points in each cell
DMx_local = D / (dx / 2); % Scale local differentiation matrix

% Initialize global differentiation matrix
DMx = sparse(nx_all, nx_all);

for j = 1:Ncell
    % Determine the row and column indices for the current cell
    row_start = (j - 1) * Np + 1 - (j - 1);
    row_end = row_start + Np - 1;
    col_start = row_start;
    col_end = row_end;

    % Assemble the local differentiation matrix into the global matrix
    DMx(row_start:row_end, col_start:col_end) = DMx_local;
end

for j = N:nx_all-1
    DMx(j, j) = 0;
    if mod(j, N) == 1
        DMx(j, :) = DMx(j, :)/2;
    end
end



end

