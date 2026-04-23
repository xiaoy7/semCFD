function twophase_sem2d()
% Two-phase NS–CH with spectral-element-style (GLL) discretization on a rectangle.
% - Gauss–Lobatto–Legendre (GLL) nodes, 1D derivative matrix D, weights w
% - 2D Mass M, Stiffness Kx, Ky, Laplacian L via Kronecker
% - Cahn–Hilliard: stabilized linear semi-implicit with bi-Laplacian factorization
% - Navier–Stokes: projection method (explicit convection, implicit viscous Helmholtz)
%
% Author: (you)
% -------------------------------------------------------------------------

%% -------------------- Parameters --------------------
% Domain and grid
Lx = 1.0; Ly = 1.0;        % domain size
Nx = 65;  Ny = 65;         % GLL points per direction (odd typical)
assert(mod(Nx,2)==1 && mod(Ny,2)==1,'Use odd GLL counts (include endpoints).');

% Time
dt   = 1e-3;                % time step
Tend = 0.5;                 % final time
nstep = round(Tend/dt);

% Cahn-Hilliard params
epsilon = 3e-3;             % interface thickness epsilon
mob     = 1e-2;             % mobility m
Sstab   = 4.0;              % stabilization S (>= max|f''| near interfaces). Take ~O(1..5)

% Fluid (matched density/viscosity demo)
rho = 1.0;                  % density
nu  = 1e-3;                 % kinematic viscosity (mu/rho)
muVis = rho*nu;

% Visualization cadence
plotEvery = 50;

% Seed for reproducibility of initial condition
rng(1);

%% -------------------- GLL nodes, weights, D --------------------
[xg, wx, D1x] = gll_nodes_weights_and_D(Nx);
[yg, wy, D1y] = gll_nodes_weights_and_D(Ny);

% Map from [-1,1] to [0,Lx] and [0,Ly]
x = (xg+1)*0.5*Lx;   y = (yg+1)*0.5*Ly;
wx = wx * 0.5*Lx;    wy = wy * 0.5*Ly;

% Quadrature/mass in 1D and stiffness in 1D
Mx = diag(wx); My = diag(wy);
% Symmetric stiffness in 1D: K = D^T W D  (weak form)
Kx = D1x' * Mx * D1x;
Ky = D1y' * My * D1y;

% 2D Kronecker operators (vectorized unknown ordering: (x-fast, y-slow) or vice versa)
Ix = speye(Nx); Iy = speye(Ny);
M  = kron(My, Mx);               % 2D mass
KX = kron(My, Kx);               % stiffness in x
KY = kron(Ky, Mx);               % stiffness in y
L  = KX + KY;                    % Laplacian (weak form)

% Gradient operators (weak form application uses D and M blocks)
Dx = kron(Iy,  D1x);             % geometric gradient along x on point values
Dy = kron(D1y, Ix );

% For Helmholtz solves we use: (K + alpha*M) * u = rhs

%% -------------------- Initial conditions --------------------
% Phase field: small random perturbation around +/− states
[X,Y] = ndgrid(x,y);
phi0 = 0.01*randn(Nx,Ny);                    % random noise
% Add larger structure if desired (two blobs)
phi0 = phi0 + 0.2*(tanh((0.25 - (X-0.3).^2 - (Y-0.5).^2)/ (2*epsilon)) ...
                - tanh((0.25 - (X-0.7).^2 - (Y-0.5).^2)/ (2*epsilon)));

% Velocity: start at rest
ux  = zeros(Nx,Ny);
uy  = zeros(Nx,Ny);
p   = zeros(Nx,Ny);

% Vectorize helpers
vec  = @(A) A(:);
matx = @(v) reshape(v, Nx, Ny);

% Assemble global vectors
phi = vec(phi0);
uxv = vec(ux);  uyv = vec(uy);   pv = vec(p);

% Precompute factorizations for constant operators
% CH bi-Laplacian split: choose alphas so that operator is (L + a2 M)(L + a1 M)
% We choose a1, a2 to approximate ( (S/epsilon)*L + (1/dt)*M/mob/epsilon ) balance.
a1 =  Sstab/epsilon;     % both positive -> Helmholtz
a2 =  Sstab/epsilon + 1/sqrt(dt+eps);  % a bit larger to make split well-conditioned

H1 = L + a1*M;  % Helmholtz #1
H2 = L + a2*M;  % Helmholtz #2
% Cholesky or backslash will work; for speed one can use ichol+pcg
% Here we use backslash for clarity.

% Viscous Helmholtz for velocity: (M/dt + nu K) per component
Avel = (1/dt)*M + nu*L;

% Pressure Poisson (projection): K p = (rho/dt) div(u*)
% We will enforce mean-zero pressure (gauge) by fixing one DoF.
Kp   = L;

% Index to fix pressure gauge
pFix = 1;

%% -------------------- Time stepping --------------------
fprintf('Starting NS–CH (GLL-%dx%d), dt=%.2e, steps=%d\n',Nx,Ny,dt,nstep);

for n = 1:nstep

    % ---------- 1) Cahn–Hilliard step (stabilized linear, split into two Helmholtz) ----------
    % CH equations (semi-implicit):
    %  (phi^{n+1}-phi^n)/dt + u^n · grad phi^n = m * Lap( mu^{n+1} )
    %  mu^{n+1} = -epsilon * Lap phi^{n+1} + (1/epsilon)( f'(phi^n) + S*(phi^{n+1}-phi^n) )
    %
    % Eliminating mu^{n+1} produces bi-Laplacian in phi^{n+1}.
    % We split (L + a2 M)(L + a1 M) phi^{n+1} ≈ RHS  (two Helmholtz solves)
    %
    % Build RHS pieces:
    phin = phi;
    phin_mat = matx(phin);

    % Convection term u·grad(phi) at time n (skew-symmetrized discrete form)
    ux_mat = matx(uxv);  uy_mat = matx(uyv);
    phix = matx( Dx * phin );        % weak gradient application is approximated pointwise here
    phiy = matx( Dy * phin );
    conv_phi = vec( ux_mat .* phix + uy_mat .* phiy );  % pointwise convective flux

    % Nonlinearity derivative f'(phi) = phi^3 - phi  (double-well)
    fp = phin.^3 - phin;

    % Build the target RHS for (L + a2 M)(L + a1 M) phi^{n+1} = RHS
    % Derivation (condensed):
    % Let mu^{n+1} ≈ -eps L phi^{n+1} + (1/eps)( fp^n + S*(phi^{n+1}-phi^n) )
    % CH: (phi^{n+1}-phi^n)/dt + u^n·∇phi^n = m L mu^{n+1}
    % => (1/dt)M(phi^{n+1}) + m*(-eps) L L phi^{n+1} + m*(S/eps) L M phi^{n+1}
    %    = (1/dt)M phi^n - M(conv_phi) + m*(1/eps) L (M fp^n) + m*(S/eps) L M phi^n
    %
    % We fit the LHS with split: (L + a2 M)(L + a1 M)phi^{n+1} ~ c0*L*L + c1*L*M + c2*M*M
    % Choose a1,a2 positive; we absorb constants into RHS to approximate target.
    % Here we use a pragmatic consistent choice and put all scaling on RHS:

    RHS_ch = (1/dt) * (M * phin) ...
             - M * conv_phi ...
             + (mob/epsilon) * (L * (M * fp)) ...
             + (mob*Sstab/epsilon) * (L * (M * phin));

    % Solve: (L + a1 M) w = RHS_ch
    w = H1 \ RHS_ch;

    % Solve: (L + a2 M) phi^{n+1} = w
    phi = H2 \ w;

    % Optional: clip |phi| to <= 1 to keep numerical stability (mild limiter)
    phi = max(min(phi, 1.2), -1.2);

    % Compute chemical potential mu^{n+1} for forcing
    fp_new = phi.^3 - phi;
    muCH = -epsilon * (L * phi) + (1/epsilon) * (M * fp_new);

    % ---------- 2) Velocity predictor with viscous implicit (Helmholtz) ----------
    un = uxv; vn = uyv;

    % Convection term for velocity (skew-symmetric): N(u) = 0.5*( (u·∇)u + ∇·(u⊗u) )
    % Approximate pointwise:
    ux_x = Dx * un;  ux_y = Dy * un;   % derivatives of u_x
    uy_x = Dx * vn;  uy_y = Dy * vn;   % derivatives of u_y
    ux_mat = matx(un); uy_mat = matx(vn);
    Nu_x = vec( ux_mat .* matx(ux_x) + uy_mat .* matx(ux_y) );  % u·∇ u_x
    Nu_y = vec( ux_mat .* matx(uy_x) + uy_mat .* matx(uy_y) );  % u·∇ u_y

    % Capillary force F = -phi * grad(mu)
    mux = Dx * muCH;  muy = Dy * muCH;
    Fx = - (phi .* mux);
    Fy = - (phi .* muy);

    % RHS for Helmholtz (component-wise): (M/dt + nu L) u* = M*(u^n/dt - N(u^n)) + M*(F/rho)
    RHSux = M*(un/dt - Nu_x) + M*(Fx/rho);
    RHSuy = M*(vn/dt - Nu_y) + M*(Fy/rho);

    ustarx = Avel \ RHSux;
    ustary = Avel \ RHSuy;

    % ---------- 3) Pressure projection ----------
    % Solve Poisson: L p^{n+1} = (rho/dt) * div(u*)
    divu = Dx * ustarx + Dy * ustary;
    RHS_p = (rho/dt) * (M * divu);

    % Enforce mean-zero pressure by fixing one dof
    % Build Kp_mod and RHS_mod
    Kp_mod = Kp;
    RHS_mod = RHS_p;
    Kp_mod(pFix,:) = 0; Kp_mod(:,pFix) = 0; Kp_mod(pFix,pFix) = 1;
    RHS_mod(pFix) = 0;

    p_new = Kp_mod \ RHS_mod;

    % Correct velocities: u^{n+1} = u* - (dt/rho) grad p^{n+1}
    gradpx = Dx * p_new;  gradpy = Dy * p_new;
    uxv = ustarx - (dt/rho) * gradpx;
    uyv = ustary - (dt/rho) * gradpy;
    pv  = p_new;

    % ---------- 4) Simple diagnostics & plot ----------
    if mod(n,plotEvery)==0 || n==1 || n==nstep
        phi_img = matx(phi);
        uxm = matx(uxv); uym = matx(uyv);
        speed = sqrt(uxm.^2 + uym.^2);

        subplot(1,2,1);
        imagesc(x, y, phi_img'); axis image xy; colorbar
        title(sprintf('\\phi at step %d, t=%.3f', n, n*dt));

        subplot(1,2,2);
        imagesc(x, y, speed'); axis image xy; colorbar
        title('||u||');

        drawnow;
        fprintf('step %5d / %5d, t=%.3f, |phi|_inf=%.3f, max|u|=%.3e\n',...
            n, nstep, n*dt, max(abs(phi)), max(speed(:)));
    end
end

fprintf('Done.\n');
end
