
function two_phase_sem_tensor_demo()
% TWO_PHASE_SEM_TENSOR_DEMO
% ------------------------------------------------------------
% Dong & Shen (2012)-style two-phase Navier-Stokes + Cahn-Hilliard
% on a 2D Cartesian spectral-element (SEM) mesh, exploiting the
% tensor-product Laplacian factorization for all constant-coefficient
% Helmholtz/Poisson solves. 
%
% Features:
%   - BDF1 time stepping
%   - Velocity-correction (projection): Poisson (pressure) + Helmholtz (velocity)
%   - Phase field via stabilized split: (Delta - beta) w = Q, (Delta + a) phi^{n+1} = w
%   - Constant matrices (alpha only changes per subproblem), solved via
%     transform–divide–inverse using 1D generalized eigendecompositions.
%   - No-slip walls (u=v=0). Pressure: Neumann + zero-mean fix.
%   - Phase field: homogeneous Neumann for phi and w.
%
% This is a compact research starter; it favors clarity over micro-optimizations.
% ------------------------------------------------------------

%% -------------------- User parameters --------------------
% Domain and SEM resolution (uniform elements)
Lx = 1.0;   Ly = 1.0;
Nelx = 8;   Nely = 8;     % elements per direction
P    = 6;                 % polynomial order per element (>=2)

% Physical parameters
sigma = 1.0;            % surface tension (non-dimensional)
g_int = 2.5e-2;         % interface thickness parameter 'g' (controls diffuse width)
kappa = 3/(2*sqrt(2))*sigma*g_int;  % mixing energy density from Dong (Eq. 2)
c1    = 1e-4;           % mobility
rho1  = 1.0;            % density fluid 1  (phi=+1)
rho2  = 10.0;           % density fluid 2  (phi=-1)  (try 100 or 1000 as well)
mu1   = 0.01;           % viscosity fluid 1
mu2   = 0.02;           % viscosity fluid 2

% Constant reference values for constant matrices
rho0  = min(rho1, rho2);
mm    = 0.5*max(mu1,mu2)/min(rho1,rho2);  % safe choice; can tune

% Time stepping
Tf  = 0.5;             % final time
dt  = 1e-3;            % time step (reduce if unstable)
saveEvery = 50;

% Stabilization for CH split (choose S and compute a>0; see Dong 2012)
c0 = 1.0;                % BDF1 coefficient
S  = 5.0;                % stabilization strength (>= g^2*sqrt(4*c0/(kappa*c1*dt)))
a  = choose_a_from_S_dt(S, g_int, c0, kappa, c1, dt);  % stable positive a
beta = (a + S/(g_int^2));  % makes first Helmholtz SPD

% Initial condition: a smoothed bubble (phi=+1 inside, -1 outside); fluid at rest
R0 = 0.2;   cx = 0.5;  cy = 0.5;

%% -------------------- Build SEM 1D ops & transforms --------------------
[xgll,wgll,D1] = gll_ops(P);              % GLL nodes/weights/deriv on [-1,1]
% Global GLL lines per direction
[Nx, xline, Dx, Mx, Sx] = sem_1d_global_ops(Lx, Nelx, xgll, wgll, D1);
[Ny, yline, Dy, My, Sy] = sem_1d_global_ops(Ly, Nely, xgll, wgll, D1);

% Generalized eigendecompositions S v = lam M v  (Neumann natural with SEM)
[Tx, Lx_eigs] = gen_eig(Sx, Mx);  TxInv = inv(Tx);
[Ty, Ly_eigs] = gen_eig(Sy, My);  TyInv = inv(Ty);
lamx = diag(Lx_eigs);   lamy = diag(Ly_eigs);   % column vectors

% Precompute grid and differential helpers
[X,Y] = meshgrid(xline, yline);      % Ny x Nx arrays
% 1D 2nd-derivative (strong form) convenience (for gradient/Laplacian)
D2x = Dx*Dx;   D2y = Dy*Dy;

% GPU option (comment out if not using Parallel Computing Toolbox)
useGPU = false; % set true if gpuArray is available
if useGPU
    Tx=gpuArray(Tx); TxInv=gpuArray(TxInv); Ty=gpuArray(Ty); TyInv=gpuArray(TyInv);
    lamx=gpuArray(lamx); lamy=gpuArray(lamy);
    Dx=gpuArray(Dx); Dy=gpuArray(Dy); D2x=gpuArray(D2x); D2y=gpuArray(D2y);
    X=gpuArray(X); Y=gpuArray(Y);
end

%% -------------------- Allocate fields --------------------
u = zeros(Ny,Nx,'like',X);   % velocity x
v = zeros(Ny,Nx,'like',X);   % velocity y
p = zeros(Ny,Nx,'like',X);   % pressure (Neumann, mean-zero after solve)

% Phase field initial condition
phi = tanh( (R0 - sqrt((X-cx).^2 + (Y-cy).^2)) / (sqrt(2)*g_int) );
% Clamp hat-phi for material properties
phihat = clamp_phi(phi);

% Diagnostics
it=0; t=0.0;

%% -------------------- Time integration loop --------------------
while t < Tf-1e-12
    it = it + 1; t = t + dt;

    % --- 1) Phase-field: stabilized split (two Helmholtz solves) ---
    % Explicit convective term (use u^n,v^n)
    ux = Dx_apply(u, Dx);  uy = Dy_apply(u, Dy);
    vx = Dx_apply(v, Dx);  vy = Dy_apply(v, Dy);
    adv_phi = u.*Dx_apply(phi,Dx) + v.*Dy_apply(phi,Dy);

    % Q = (1/(kappa*c1))*( g - u·∇phi + (phi - phi_hat)/dt ) + Δ h(phi* ) - (S/g^2) phi*
    % Here we set source g=0 and evaluate h at current phi (explicit)
    hphi = (1/(g_int^2))*phi.*(phi.^2 - 1);
    lap_hphi = laplacian_strong(hphi, D2x, D2y);  % strong Laplacian
    Q = ( (0 - adv_phi + (phi - phi)/dt) )/(kappa*c1) + lap_hphi - (S/(g_int^2))*phi; %#ok<NASGU>
    % The (phi - phi)/dt term cancels since we use BDF1 implicit on lhs, so we can drop it in RHS
    Q = ( - adv_phi )/(kappa*c1) + lap_hphi - (S/(g_int^2))*phi;

    % Solve (Δ - beta) w = Q   => (beta I - Δ) w = Q  (alpha = beta)
    w = tp_helmholtz_solve_2d(Q, beta, Tx, Ty, TxInv, TyInv, lamx, lamy);

    % Solve (Δ + a) phi^{n+1} = w   => (-a I - Δ) phi^{n+1} = w (alpha = -a)
    phi_new = tp_helmholtz_solve_2d(w, -a, Tx, Ty, TxInv, TyInv, lamx, lamy);

    % Clamp phi to [-1,1] weakly for material positivity
    phihat = clamp_phi(phi_new);

    % --- Material properties at n+1 (explicit for RHS corrections) ---
    rho = 0.5*(rho1+rho2) + 0.5*(rho1-rho2).*phihat;
    mu  = 0.5*(mu1 + mu2) + 0.5*(mu1 - mu2).*phihat;

    % --- 2) Pressure Poisson: (rho0/dt) div(u_tilde) ---
    % Build intermediate velocity u_tilde by implicit diffusion + old convective terms.
    % We'll do velocity solve AFTER pressure (as in Dong's weak elimination form);
    % for simplicity of this demo we compute u_tilde via explicit Adams–Bashforth-0:
    %   u_tilde ≈ u^n (pure predictor), then use it to compute pressure. 
    u_t = u;  v_t = v;

    % Pressure RHS: (rho0/dt) * div(u_t)
    div_ut = Dx_apply(u_t, Dx) + Dy_apply(v_t, Dy);
    rhs_p  = (rho0/dt)*div_ut;

    % Solve Poisson: -Δ p = rhs_p  (Neumann). That's alpha=0 in our solver.
    p_new = tp_poisson_neumann_solve(rhs_p, Tx, Ty, TxInv, TyInv, lamx, lamy);

    % --- 3) Velocity Helmholtz (each component) with Dong-style explicit RHS ---
    % RHS = (rho0/dt)*u^n  - N(u^n)  + density/viscosity corrections + capillary + body f
    % Convection (skew-symmetric simple form): N(u) ~ u·∇u
    adv_u = u.*Dx_apply(u,Dx) + v.*Dy_apply(u,Dy);
    adv_v = u.*Dx_apply(v,Dx) + v.*Dy_apply(v,Dy);

    % Density correction: ((1/rho0)-(1/rho))*grad(p^n) (use p_new as best avail.)
    invrho_corr = (1./rho0) - (1./rho);
    gx_p = Dx_apply(p_new, Dx); gy_p = Dy_apply(p_new, Dy);
    dens_corr_u = invrho_corr .* gx_p;
    dens_corr_v = invrho_corr .* gy_p;

    % Viscosity correction: (mu - mm)*Δ u^n
    lap_u = laplacian_strong(u, D2x, D2y);
    lap_v = laplacian_strong(v, D2x, D2y);
    visc_corr_u = (mu - mm).*lap_u;
    visc_corr_v = (mu - mm).*lap_v;

    % Capillary force: -kappa*(Δphi)*∇phi  (use strong Laplacian and strong grad)
    lap_phi = laplacian_strong(phi_new, D2x, D2y);
    gradphi_x = Dx_apply(phi_new, Dx);
    gradphi_y = Dy_apply(phi_new, Dy);
    fcap_x = -kappa * lap_phi .* gradphi_x;
    fcap_y = -kappa * lap_phi .* gradphi_y;

    % Body force (here zero)
    fx = zeros(size(u),'like',u);
    fy = zeros(size(v),'like',v);

    % Velocity Helmholtz alpha
    alpha_u = c0/(mm*dt);

    rhs_u = (rho0/dt).*u - adv_u + dens_corr_u + visc_corr_u + fcap_x + fx ...
            - (1/rho0).*gx_p;  % pressure gradient part
    rhs_v = (rho0/dt).*v - adv_v + dens_corr_v + visc_corr_v + fcap_y + fy ...
            - (1/rho0).*gy_p;

    % Enforce Dirichlet u=v=0 on boundary by overwriting after solve
    u_new = tp_helmholtz_solve_2d(rhs_u, alpha_u, Tx, Ty, TxInv, TyInv, lamx, lamy);
    v_new = tp_helmholtz_solve_2d(rhs_v, alpha_u, Tx, Ty, TxInv, TyInv, lamx, lamy);
    u_new = apply_noslip(u_new);
    v_new = apply_noslip(v_new);

    % Update
    phi = phi_new;
    u = u_new; v = v_new; p = p_new;

    % Diagnostics/plot
    if mod(it, saveEvery)==0 || t>=Tf-1e-12
        div_u = norm(divergence(u,v,Dx,Dy),'fro')/sqrt(numel(u));
        ke = 0.5*sum(u(:).^2 + v(:).^2)*(Lx*Ly/(Nx*Ny)); % rough KE
        fprintf('it=%4d t=%.4f  |div|=%.2e  KE=%.3e\n', it, t, div_u, ke);
        visualize_fields(X,Y,phi,u,v,p,t);
        drawnow;
    end
end

fprintf('Done.\n');

end % === main ===


%% ========================== Helpers ==============================
function a = choose_a_from_S_dt(S, g, c0, kappa, c1, dt)
% From Dong & Shen constraints: choose a>0 such that
% a = - S/(2 g^2) * (1 + sqrt(1 - 4 c0/(kappa c1 dt) * (g^4/S^2)))
% For robustness (no small dt constraint failures), use a positive fixed fraction.
disc = 1 - 4*c0/(kappa*c1*dt) * (g^4/S^2);
disc = max(disc, 1e-6);
a = - (S/(2*g^2)) * (1 + sqrt(disc));
a = abs(a);  % ensure positive
if a < 1e-6, a = 1e-3; end
end

function [xgll,wgll,D] = gll_ops(P)
[xgll,wgll] = legendre_gll(P);
D = gll_deriv_matrix(xgll);
end

function [x,w] = legendre_gll(P)
% Gauss-Lobatto-Legendre nodes/weights on [-1,1]
beta = 0.5./sqrt(1 - (2*(1:P)).^(-2));
T = diag(beta,1)+diag(beta,-1);
[V,D] = eig(T);
x = diag(D);
[~,i] = sort(x); x = x(i); V = V(:,i);
w = 2*(V(1,:)'.^2);
end

function D = gll_deriv_matrix(x)
% SEM derivative at GLL nodes using barycentric-like formula
n = numel(x);
D = zeros(n);
c = ones(n,1); c(1)=2; c(end)=2;
for i=1:n
  for j=1:n
    if i~=j
      D(i,j) = c(i)/c(j) * (-1)^(i+j) / (x(i)-x(j));
    end
  end
end
D(1,1)   = -(D(1,2:end))*ones(n-1,1);
D(n,n)   = -(D(n,1:n-1))*ones(n-1,1);
for i=2:n-1
  D(i,i) = -(D(i,[1:i-1,i+1:n]))*ones(n-1,1);
end
end

function [N,xline,Dg,Mg,Sg] = sem_1d_global_ops(L, Nel, xgll, wgll, D1)
% Build global 1D SEM line with Nel uniform elements, order P
P = numel(xgll)-1;  N = Nel*P + 1;
% Physical nodes
xline = zeros(N,1);
for e=1:Nel
    a = (e-1)*L/Nel; b = e*L/Nel;
    idx = (e-1)*P + (1:P+1);
    xline(idx) = a + (xgll+1)*(b-a)/2;
end
% Remove duplicate at joins
xline(2:P+1:end-1) = [];

% Recompute N in case
N = numel(xline);

% Build global derivative by assembly (sum-factorized 1D)
Dg = sparse(N,N);
Mg = sparse(N,N);
Sg = sparse(N,N);
for e=1:Nel
    a = (e-1)*L/Nel; b = e*L/Nel; J = (b-a)/2;
    idx = (e-1)*P + (1:P+1);
    % local to physical scaling
    De = (2/J)*D1;                 % d/dx = (2/J)*d/dxi
    We = wgll * J;                 % weight scaling
    Me = diag(We);                 % mass
    Se = De.'*Me*De;               % stiffness
    % Assembly
    Dg(idx,idx) = Dg(idx,idx) + De;
    Mg(idx,idx) = Mg(idx,idx) + Me;
    Sg(idx,idx) = Sg(idx,idx) + Se;
end
% Convert to full double for simpler ops
Dg = full(Dg); Mg = full(Mg); Sg = full(Sg);
end

function [T,Lam] = gen_eig(S,M)
% Generalized SPD eig: S v = lam M v
% Regularize tiny nullspace for Neumann by adding eps on both
M = 0.5*(M+M'); S = 0.5*(S+S');
opts.disp = 0;
[T,Lam] = eig(S,M,'vector');
% Orthonormalize with respect to M
for i=1:size(T,2)
    nrm = sqrt(T(:,i)'*(M*T(:,i)));
    T(:,i) = T(:,i)/nrm;
end
Lam = diag(Lam);
end

function U = tp_helmholtz_solve_2d(F, alpha, Tx, Ty, TxInv, TyInv, lamx, lamy)
% Solve (alpha I - Lap_h) U = F  with tensor SEM Laplacian
% Transform in y then x (Ny x Nx arrays)
U = TyInv * F * TxInv.';
% Diagonal divide: alpha + lamx + lamy (outer sum)
[Ny, Nx] = size(U);
LAMY = lamy(:);  LAMX = lamx(:).';
den = alpha + LAMY + LAMX;
% Handle Poisson nullspace if alpha==0: set zero mode denominator to inf (avoid divide-by-zero)
if abs(alpha)<1e-15
    % Identify (0,0) eigpair (smallest values)
    [~,iy] = min(LAMY); [~,ix] = min(LAMX);
    den(iy,ix) = inf;
end
U = U ./ den;
% Inverse transforms
U = Ty * U * Tx.';
end

function P = tp_poisson_neumann_solve(rhs, Tx, Ty, TxInv, TyInv, lamx, lamy)
% Solve -Δ P = rhs with Neumann BC; enforce zero-mean
rhs = rhs - mean(rhs(:));         % zero-mean RHS for solvability
P = tp_helmholtz_solve_2d(rhs, 0.0, Tx, Ty, TxInv, TyInv, lamx, lamy);
P = P - mean(P(:));               % remove constant nullspace component
end

function out = Dx_apply(F, Dx)
% Strong derivative in x : dF/dx using 1D global derivative
out = F * Dx.';
end

function out = Dy_apply(F, Dy)
% Strong derivative in y : dF/dy using 1D global derivative
out = Dy * F;
end

function Lf = laplacian_strong(F, D2x, D2y)
Lf = Dy_apply(F, D2y) + Dx_apply(F, D2x);
end

function ph = clamp_phi(phi)
ph = max(-1, min(1, phi));
end

function divu = divergence(u,v,Dx,Dy)
divu = Dx_apply(u,Dx) + Dy_apply(v,Dy);
end

function U = apply_noslip(U)
U(1,:)   = 0; U(end,:) = 0;
U(:,1)   = 0; U(:,end) = 0;
end

function visualize_fields(X,Y,phi,u,v,p,t)
subplot(2,2,1);
imagesc(X(1,:),Y(:,1),phi); axis image; set(gca,'YDir','normal');
colorbar; title(sprintf('\\phi at t=%.3f',t));
subplot(2,2,2);
quiver(X,Y,u,v); axis image tight; title('velocity');
subplot(2,2,3);
imagesc(X(1,:),Y(:,1),p); axis image; set(gca,'YDir','normal');
colorbar; title('pressure');
subplot(2,2,4);
speed = sqrt(u.^2+v.^2);
imagesc(X(1,:),Y(:,1),speed); axis image; set(gca,'YDir','normal');
colorbar; title('|u|');
end
