% nse3d_sem_gpu.m
% 3D incompressible Navier–Stokes on a periodic box using a projection method
% High-order spectral-element (Qk) on a Cartesian tensor grid
% GPU-accelerated via MATLAB's tensorprod and pagemtimes
%
% u_t + (u·∇)u = -∇p + ν Δu + f,   ∇·u = 0,   periodic on [-Lx,Lx]×[-Ly,Ly]×[-Lz,Lz]
% Semi-implicit projection:
%   (I - dt*ν Δ) u* = u^n - dt * N(u^n) + dt * f^n
%   Δ p^{n+1} = (1/dt) ∇·u*
%   u^{n+1} = u* - dt ∇ p^{n+1}
%
% Default test: Taylor–Green vortex, forcing-free.
% MATLAB R2023a+ recommended.

clear; clc;

%% ---------------- Parameters ----------------
k   = 5;                    % Qk polynomial degree (>=3 recommended)
Nc  = [12, 12, 12];         % number of SEM cells in x,y,z
L   = [pi, pi, pi];         % half-domain sizes (box is [-Lx,Lx] etc.)
nu  = 1e-3;                 % kinematic viscosity
dt  = 1e-3;                 % time step
nt  = 200;                  % number of time steps
saveEvery = 50;             % output cadence
useGPU = gpuDeviceCount("available")>0;

fprintf('3D NSE (projection) with Q%d SEM, cells %dx%dx%d (periodic)\n', k, Nc);
fprintf('Domain: [-%g,%g]×[-%g,%g]×[-%g,%g], nu=%g, dt=%g, steps=%d\n', L(1),L(1),L(2),L(2),L(3),L(3),nu,dt,nt);
if useGPU, g = gpuDevice; fprintf('Using GPU: %s\n', g.Name); else, fprintf('Using CPU\n'); end

%% ---------------- 1D SEM setup ----------------
[x1D, w1D, M1D, S1D, D1D, T1D, T1D_inv, lambdaH] = sem_setup_1d(k, Nc, L);
nx = numel(x1D{1}); ny = numel(x1D{2}); nz = numel(x1D{3});

% Move 1D ops to device if available
for d=1:3
    x1D{d}     = toDev(x1D{d},useGPU);
    w1D{d}     = toDev(w1D{d},useGPU);
    M1D{d}     = toDev(M1D{d},useGPU);
    S1D{d}     = toDev(S1D{d},useGPU);
    D1D{d}     = toDev(D1D{d},useGPU);
    T1D{d}     = toDev(T1D{d},useGPU);
    T1D_inv{d} = toDev(T1D_inv{d},useGPU);
    lambdaH{d} = toDev(lambdaH{d},useGPU);
end

% Λ3D for -Δ eigenvalues: λx + λy + λz
Lambda3D = sem_tensor_ops.lambda3D(lambdaH);

%% ---------------- Initial condition: Taylor–Green vortex ----------------
[X,Y,Z] = ndgrid(x1D{1}, x1D{2}, x1D{3});  % collocation grid (tensor)
% TG vortex on [-pi,pi]^3
u =  sin(X).*cos(Y).*cos(Z);
v = -cos(X).*sin(Y).*cos(Z);
w =  0*X;

u = toDev(u,useGPU); v = toDev(v,useGPU); w = toDev(w,useGPU);

%% ---------------- Time stepping ----------------
fprintf('Starting time integration...\n');
for n=1:nt
    % gradients
    [ux,uy,uz] = sem_tensor_ops.grad(u, D1D);
    [vx,vy,vz] = sem_tensor_ops.grad(v, D1D);
    [wx,wy,wz] = sem_tensor_ops.grad(w, D1D);

    % Nonlinear advective term N(u) = (u·∇)u
    Nu = u.*ux + v.*uy + w.*uz;
    Nv = u.*vx + v.*vy + w.*vz;
    Nw = u.*wx + v.*wy + w.*wz;

    % Helmholtz step: (I - dt*nu Δ) u* = u^n - dt*N(u^n) + dt*f  (f=0 here)
    rhs_u = u - dt*Nu;
    rhs_v = v - dt*Nv;
    rhs_w = w - dt*Nw;

    alpha = 1.0;  % (alpha I - dt*nu Δ) with alpha=1
    helm = @(F) poisson_tensor_solver.helmholtz_solve(F, alpha, dt*nu, T1D, T1D_inv, Lambda3D);

    u_star = helm(rhs_u);
    v_star = helm(rhs_v);
    w_star = helm(rhs_w);

    % Pressure Poisson: Δ p^{n+1} = (1/dt) ∇·u*
    divustar = sem_tensor_ops.div(u_star, v_star, w_star, D1D);
    rhs_p = (1/dt) * divustar;

    % Solve Poisson with periodic BC (remove mean via zeroing λ=0 mode)
    p = poisson_tensor_solver.poisson_solve(rhs_p, T1D, T1D_inv, Lambda3D);

    % Velocity correction: u^{n+1} = u* - dt ∇p
    [px,py,pz] = sem_tensor_ops.grad(p, D1D);
    u = u_star - dt*px;
    v = v_star - dt*py;
    w = w_star - dt*pz;

    % Diagnostics
    if mod(n, saveEvery)==0 || n==1 || n==nt
        divu_new = gather(semtensor_norm( sem_tensor_ops.div(u,v,w,D1D) ));
        kinE = 0.5*gather( mean( (u.^2 + v.^2 + w.^2), 'all' ) );
        fprintf('step %4d | div=%8.2e | E=%10.6e\n', n, divu_new, kinE);
    end
end
fprintf('Done.\n');

%% ---------------- local helpers ----------------
function B = toDev(A,useGPU)
if useGPU, B = gpuArray(A); else, B = A; end
end

function val = semtensor_norm(A)
val = max(abs(A(:)));
end
