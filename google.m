%% Problem Setup and Analysis of a 2D Poisson Equation
%
% This script demonstrates the solution of a 2D Poisson equation with
% non-zero Neumann boundary conditions on a unit square domain [0,1]x[0,1].
% The method utilizes the spectral element method with Gauss-Lobatto-Legendre
% points and the Kronecker sum for efficient matrix assembly and solution.
%
% The problem is defined as:
%   -nabla^2 u(x,y) = f(x,y)   in Omega = [0,1]x[0,1]
%
% With boundary conditions:
%   u(x,y) = u_exact(x,y)     on the left (x=0) and bottom (y=0) boundaries (Dirichlet)
%   du/dn = g(x,y)            on the right (x=1) and top (y=1) boundaries (Neumann)

clc; close all; clear all;

%% 1. Define the Analytical Solution and Source Term
% We choose an analytical solution to easily verify the accuracy of our method.
%
% Let the exact solution be:
u_exact = @(x,y) x .* y .* exp(x) .* exp(y);
%
% From this, the source term f is derived as:
%   f = -nabla^2 u = - (d^2u/dx^2 + d^2u/dy^2)
f_source = @(x,y) -exp(x+y) .* (y.*(2+x) + x.*(2+y));

%% 2. Define the Neumann Boundary Condition
% The Neumann condition g is the normal derivative of the exact solution
% on the right and top boundaries.
%
% Right boundary (x=1):
%   normal vector n = (1, 0)
%   du/dn = du/dx = d(x*exp(x))/dx * (y*exp(y)) = (exp(x)+x*exp(x))*y*exp(y)
%   g_right = @(y) (exp(1)+1*exp(1))*y*exp(y) = 2*exp(1)*y.*exp(y)
g_right = @(y) 2 * exp(1) * y .* exp(y);
%
% Top boundary (y=1):
%   normal vector n = (0, 1)
%   du/dn = du/dy = d(y*exp(y))/dy * (x*exp(x)) = (exp(y)+y*exp(y))*x*exp(x)
%   g_top = @(x) (exp(1)+1*exp(1))*x*exp(x) = 2*exp(1)*x.*exp(x)
g_top = @(x) 2 * exp(1) * x .* exp(x);

%% 3. Discretization and Matrix Assembly
% We use N Gauss-Lobatto-Legendre (GLL) points to create an (N-1)-degree
% polynomial approximation.
N = 10; % Number of GLL points (N-1 is the polynomial degree)

% Compute GLL points and weights using the Legendre-Gauss-Lobatto rule
[x_gll, w_gll] = lgl_nodes(N);
[y_gll, ~] = lgl_nodes(N);

% Map GLL nodes from [-1, 1] to the domain [0, 1]
x_nodes = 0.5 * (x_gll + 1);
y_nodes = 0.5 * (y_gll + 1);

% Create 1D mass and stiffness matrices
[Ax, Bx] = build_1d_matrices(N, x_nodes, w_gll);
Ay = Ax; % Same for y-dimension on a square domain
By = Bx;

% Construct the full 2D stiffness matrix using the Kronecker sum
A = kron(Ay, Bx) + kron(By, Ax);

% The number of degrees of freedom (DOF)
n_dof = N*N;

%% 4. Right-Hand Side Vector Assembly
% The right-hand side vector b is composed of two parts:
% 1. The source term f, which is a volume integral.
% 2. The Neumann boundary condition g, which is a boundary integral.
%
% Contribution from the source term f:
[X, Y] = meshgrid(x_nodes, y_nodes);
F = f_source(X,Y);
B_2D = kron(By, Bx);
b_f = B_2D * F(:);

% Contribution from the Neumann boundary condition g:
b_g = zeros(n_dof, 1);

% Right boundary (x=1):
%   The nodes are at x_nodes(end) and all y_nodes.
%   The corresponding 2D DOFs are indices (N*i) for i = 1 to N.
%   The integral is on the boundary, so we use 1D GLL weights.
for i = 1:N
    global_idx = (i-1)*N + N;
    b_g(global_idx) = b_g(global_idx) + g_right(y_nodes(i)) * w_gll(i) * 0.5;
end
% Top boundary (y=1):
%   The nodes are at y_nodes(end) and all x_nodes.
%   The corresponding 2D DOFs are indices (N*(N-1) + i) for i = 1 to N.
%   The integral is on the boundary, so we use 1D GLL weights.
for i = 1:N
    global_idx = (N-1)*N + i;
    b_g(global_idx) = b_g(global_idx) + g_top(x_nodes(i)) * w_gll(i) * 0.5;
end

% Avoid double counting at the top-right corner for boundary integrals
corner_idx = N*N; % (x=1, y=1)
% Subtract half of one edge's corner contribution (average the two)
b_g(corner_idx) = b_g(corner_idx) - 0.5 * ( g_right(1) * w_gll(end) * 0.5 );

% Total right-hand side vector
b = b_f + b_g;

%% 5. Impose Dirichlet Boundary Conditions
% The rows/columns corresponding to Dirichlet boundary nodes must be
% modified to enforce the zero-valued condition. Our exact solution is zero
% on the left (x=0) and bottom (y=0) boundaries.
%
% Find the indices of the Dirichlet nodes (x=0 or y=0)
dirichlet_indices = [];
for i = 1:N
    % Left boundary (y_nodes(i) and x_nodes(1))
    dirichlet_indices = [dirichlet_indices, (i-1)*N + 1];
    % Bottom boundary (y_nodes(1) and x_nodes(i))
    dirichlet_indices = [dirichlet_indices, (1-1)*N + i];
end
% Remove duplicates (e.g., the corner (0,0))
dirichlet_indices = unique(dirichlet_indices);

% For each Dirichlet node, set the corresponding row in A to a diagonal entry
% of 1 and all other entries to 0. Set the RHS entry to the known value (0).
for i = 1:length(dirichlet_indices)
    idx = dirichlet_indices(i);
    A(idx, :) = 0;
    A(idx, idx) = 1;
    b(idx) = u_exact(x_nodes(mod(idx-1,N)+1), y_nodes(floor((idx-1)/N)+1));
end

%% 6. Solve the Linear System
% We solve the system A*u_num = b for the numerical solution u_num.
u_num = A \ b;

% Reshape the solution vector to a 2D matrix for plotting
U_num_2d = reshape(u_num, [N, N]);
U_exact_2d = u_exact(X,Y);

%% 7. Error Analysis and Visualization

% Compute the L2 norm of the error
error_L2 = norm(u_num - U_exact_2d(:)) / norm(U_exact_2d(:));
fprintf('Relative L2-norm of the error: %e\n', error_L2);

% Plot the results
figure;
subplot(1,3,1);
surf(X,Y,U_exact_2d);
title('Exact Solution');
xlabel('x'); ylabel('y'); zlabel('u');
shading interp; colormap('jet');
view(3);

subplot(1,3,2);
surf(X,Y,U_num_2d);
title('Numerical Solution');
xlabel('x'); ylabel('y'); zlabel('u');
shading interp; colormap('jet');
view(3);

subplot(1,3,3);
surf(X,Y, abs(U_exact_2d - U_num_2d));
title('Absolute Error');
xlabel('x'); ylabel('y'); zlabel('error');
shading interp; colormap('jet');
view(3);

% Sub-functions to build 1D matrices
function [A_1d, B_1d] = build_1d_matrices(N, nodes, weights)
    % N = number of GLL points
    % nodes = 1D GLL node coordinates in [0,1]
    % weights = 1D GLL quadrature weights

    A_1d = zeros(N, N); % Stiffness matrix
    B_1d = zeros(N, N); % Mass matrix
    
    % The derivative matrix D maps a function's values at GLL nodes to the
    % derivatives at those nodes.
    D = lagrange_derivative_matrix(nodes);

    % The mass matrix is diagonal with GLL weights scaled by 0.5 for domain [0,1]
    B_1d = 0.5 * diag(weights);

    % The stiffness matrix is computed as (D^T * B * D)
    A_1d = D' * B_1d * D;
end

% Sub-function to compute GLL nodes and weights
function [x, w] = lgl_nodes(N)
    % Computes N Gauss-Lobatto-Legendre nodes and weights on [-1, 1].
    % Interior nodes are the roots of d/dx P_{N-1}(x) and endpoints at ±1.
    if N == 1
        x = 0; w = 2;
        return;
    end

    x = zeros(N, 1);
    x(1) = -1;
    x(N) = 1;

    % Initial guesses: Chebyshev-Lobatto nodes
    x_init = cos(pi*(0:N-1)/(N-1))';
    % Find roots of derivative of P_{N-1}
    for i = 2:N-1
        x(i) = newton_method(@(val) legendreP_derivative(N-1, val), x_init(i));
    end

    % Compute LGL weights: w_i = 2 / (N(N-1) [P_{N-1}(x_i)]^2)
    Pnm1 = legendreP(N-1, x);
    w = 2 ./ (N*(N-1) * (Pnm1.^2));

    % Ensure exact endpoints
    x(1) = -1; x(N) = 1;
end

% Sub-function to compute the Lagrange derivative matrix
function D = lagrange_derivative_matrix(x_nodes)
    % Barycentric differentiation matrix on arbitrary distinct nodes
    N = length(x_nodes);
    w = ones(N,1);
    for j = 1:N
        for k = [1:j-1, j+1:N]
            w(j) = w(j) * (x_nodes(j) - x_nodes(k));
        end
    end
    w = 1 ./ w; % barycentric weights
    D = zeros(N, N);
    for i = 1:N
        for j = 1:N
            if i ~= j
                D(i,j) = w(j) / (w(i) * (x_nodes(j) - x_nodes(i)));
            end
        end
    end
    % Diagonal entries as negative row sums
    D(1:N+1:end) = -sum(D, 2);
end

function dP = legendreP_derivative(n, x)
    % Derivative of Legendre polynomial P_n using recurrence
    % (1 - x^2) P_n'(x) = -n x P_n(x) + n P_{n-1}(x)
    if n == 0
        dP = 0*x;
        return;
    end
    Pn = legendreP(n, x);
    Pnm1 = legendreP(n-1, x);
    dP = (n ./ (1 - x.^2)) .* (Pnm1 - x .* Pn);
    % Handle endpoints to avoid Inf due to (1-x^2) in denominator
    at_end = abs(1 - abs(x)) < 1e-12;
    if any(at_end)
        % Use limits at x = ±1: P_n'(±1) = n(n+1)/2 * (±1)^{n-1}
        dP(at_end) = n*(n+1)/2 * (sign(x(at_end)).^(n-1));
    end
end

% Sub-function for Newton's method
function root = newton_method(f, x0)
    max_iter = 100;
    tol = 1e-12;
    h = 1e-6; % Step size for finite difference derivative

    x_curr = x0;
    for i = 1:max_iter
        % Finite difference approximation of the derivative
        f_prime = (f(x_curr + h) - f(x_curr - h)) / (2*h);
        x_next = x_curr - f(x_curr) / f_prime;

        if abs(x_next - x_curr) < tol
            root = x_next;
            return;
        end
        x_curr = x_next;
    end
    root = x_curr; % Return the best guess if tolerance not met
end

% Helper function for Legendre Polynomials. This is often part of
% MATLAB's built-in functions, but included for completeness.
function P = legendreP(n, x)
    if n == 0
        P = 1;
    elseif n == 1
        P = x;
    else
        P_prev = 1;
        P_curr = x;
        for k = 2:n
            P_next = ((2*k-1)*x.*P_curr - (k-1)*P_prev) / k;
            P_prev = P_curr;
            P_curr = P_next;
        end
        P = P_curr;
    end
end
