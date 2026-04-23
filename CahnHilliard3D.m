function CahnHilliard3D()
% CahnHilliard3D
% 3D Cahn–Hilliard (Neumann b.c.) on [-1,1]^3 with Q5 spectral elements.
% SBDF2 in time (first step BE); explicit F' with linear stabilization.
% Tensor-product eigen (SEM), GPU if available.
%
% Fixes:
% - Linear stabilization term S*(phi^{n+1}-phi^n) inside mu to avoid blow-up
% - Mass conservation by weighted average correction each step
% - Energy calc hardened against divide-by-zero/Inf
%
% Run: CahnHilliard3D

%% ------------------ user parameters ------------------
Np      = 5;               % Q^Np polynomial degree (Q5)
Ncell   = 10;              % elements per side -> (Np*Ncell+1) points/side

epsilon = 0.02;
mMob    = 0.02;
dt      = 1.0e-3;
Tfinal  = 10.0;
numSteps= round(Tfinal/dt);

% STABILIZATION: S = (gamma/epsilon)  (typical gamma in [1,3])
stab_gamma = 2.0;          % increase to 3.0 if you still see growth
Sstab      = stab_gamma/epsilon;

plotEnergy      = true;
plotIsosurfaces = false;   % true plots iso(0) snapshots (heavier)
isoTimes        = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 10.0];

useGPU = gpuDeviceCount("available") > 0;
gpuID  = 1;

Lx = 1; Ly = 1; Lz = 1;
%% -----------------------------------------------------

fprintf('=== 3D Cahn–Hilliard (SBDF2, stabilized) ===\n');
fprintf('Q%d SEM, %d elems/side -> %d pts/side\n', Np, Ncell, Np*Ncell+1);
fprintf('eps=%.3g, m=%.3g, dt=%.3g, steps=%d, gamma=%.2f (S=%.3g)\n', ...
    epsilon, mMob, dt, numSteps, stab_gamma, Sstab);

if useGPU
    dev = gpuDevice(gpuID);
    fprintf('Using GPU: %s (ID=%d)\n', dev.Name, gpuID);
else
    fprintf('Using CPU.\n');
end

%% 1D SEM setup
Param.Np     = Np;
Param.Ncellx = Ncell; Param.Ncelly = Ncell; Param.Ncellz = Ncell;
Param.nx = Np*Param.Ncellx + 1;
Param.ny = Np*Param.Ncelly + 1;
Param.nz = Np*Param.Ncelly + 1;

[x, ex, Tx, eigx, Wx] = SEGenerator1D('x', Lx, Param);
[y, ey, Ty, eigy, Wy] = SEGenerator1D('y', Ly, Param);
[z, ez, Tz, eigz, Wz] = SEGenerator1D('z', Lz, Param);

TxInv = pinv(Tx);  TyInv = pinv(Ty);  TzInv = pinv(Tz);
Lambda3D = makeLambda3D(eigx, eigy, eigz);

if useGPU
    Tx     = gpuArray(Tx);     Ty     = gpuArray(Ty);     Tz     = gpuArray(Tz);
    TxInv  = gpuArray(TxInv);  TyInv  = gpuArray(TyInv);  TzInv  = gpuArray(TzInv);
    eigx   = gpuArray(eigx);   eigy   = gpuArray(eigy);   eigz   = gpuArray(eigz);
    ex     = gpuArray(ex);     ey     = gpuArray(ey);     ez     = gpuArray(ez);
    x      = gpuArray(x);      y      = gpuArray(y);      z      = gpuArray(z);
    Wx     = gpuArray(Wx);     Wy     = gpuArray(Wy);     Wz     = gpuArray(Wz);
    Lambda3D = gpuArray(Lambda3D);
end

nx = numel(x); ny = numel(y); nz = numel(z);
fprintf('Total DoFs: %d x %d x %d = %.3g\n', nx,ny,nz, nx*ny*nz);

%% Initial condition: two droplets
[X,Y,Z] = ndgrid(x, y, z);
R   = 0.35;
x1  = [0,0, 0.37];
x2  = [0,0,-0.37];
dist1 = sqrt((X-x1(1)).^2 + (Y-x1(2)).^2 + (Z-x1(3)).^2);
dist2 = sqrt((X-x2(1)).^2 + (Y-x2(2)).^2 + (Z-x2(3)).^2);
phi0   = 1 - tanh((dist1 - R)/(sqrt(2)*epsilon)) ...
           - tanh((dist2 - R)/(sqrt(2)*epsilon));

phi_n  = phi0;
phi_nm1= phi_n;

% weighted volume and initial mass/average
Vol    = sum(Wx) * sum(Wy) * sum(Wz);
avg0   = weightedAvg(phi_n, Wx, Wy, Wz, Vol);

% energy storage
Energy = zeros(numSteps+1,1,'like',gather_scalar(phi_n));
Energy(1) = energyCH(phi_n, epsilon, Wx, Wy, Wz);

tt = (0:numSteps)'*dt;
isoNext = 1;

ticTotal = tic;
fprintf('Time marching...\n');
for n = 0:(numSteps-1)
    % coefficients
    if n==0
        a  = 1.0;
        ia = 1.0;                    % 1/a
        C2 = (mMob*epsilon*dt)/a;    % coeff for Δ^2
        C1 = (mMob*Sstab*dt)/a;      % coeff for Δ
        phi_hat = phi_n;             % BE startup
        phi_bar = phi_n;
    else
        a  = 1.5;
        ia = 2/3;                    % 1/a
        C2 = (mMob*epsilon*dt)/a;
        C1 = (mMob*Sstab*dt)/a;
        phi_hat = 2*phi_n - 0.5*phi_nm1;
        phi_bar = 2*phi_n -      phi_nm1;
    end

    % explicit nonlinear forcing: Δ( (1/eps) * F'(phi_bar) )
    Fprime = phi_bar.^3 - phi_bar;
    DF     = (1/epsilon) * Fprime;
    LapDF  = applyLap3D(DF, Tx,Ty,Tz, TxInv,TyInv,TzInv, Lambda3D);
    LapPhi = applyLap3D(phi_n, Tx,Ty,Tz, TxInv,TyInv,TzInv, Lambda3D);

    % RHS = (1/a)*phi_hat + (m*dt/a) * [ Δ(1/eps F') + S*Δ(phi^n) ]
    RHS = ia*phi_hat + ia*mMob*dt*( LapDF + Sstab*LapPhi );

    % Solve: (I + C1*Δ + C2*Δ^2) phi_{n+1} = RHS  (spectral diagonal)
    phi_np1 = solveCHLinear3D(RHS, C1, C2, Lambda3D, Tx,Ty,Tz, TxInv,TyInv,TzInv);

    % mass correction (preserve ∫phi)
    phi_np1 = massCorrect(phi_np1, Wx, Wy, Wz, Vol, avg0);

    % simple safety clamp (rarely needed, avoids catastrophic blow-up if any)
    % comment out if undesired:
    phi_np1 = min(max(phi_np1, -3), 3);

    % rotate levels
    phi_nm1 = phi_n;
    phi_n   = phi_np1;

    % energy + logs
    Ei = energyCH(phi_n, epsilon, Wx, Wy, Wz);
    Energy(n+2) = Ei;

    if any(~isfinite(gather(Ei))) || any(~isfinite(gather(phi_n(1))))
        fprintf('NaN/Inf detected at step %d (t=%.3f). Try larger gamma or smaller dt.\n', n+1, (n+1)*dt);
        break;
    end

    if (n+1)==ceil(0.1*numSteps) || mod(n+1,1000)==0 || (n+1)==numSteps
        fprintf(' step %6d / %6d  t=%.3f  E=%.6e\n', n+1, numSteps, (n+1)*dt, gather_scalar(Ei));
    end

    if plotIsosurfaces && isoNext <= numel(isoTimes)
        if (n+1)*dt >= isoTimes(isoNext) - 1e-12
            showIsosurface(phi_n, x, y, z, isoTimes(isoNext));
            isoNext = isoNext + 1;
        end
    end
end
elapsed = toc(ticTotal);
fprintf('Done. Total wall time = %.2f s (%.2f ms/step)\n', elapsed, 1e3*elapsed/max(1,n));

exportTecplot3D('ch_final.dat', x, y, z, phi_n, ...
    'Title','Cahn-Hilliard 3D', 'VarName','PHI', 'Time', Tfinal, 'Append', false);


if plotEnergy
    figure('Name','Energy vs time');
    semilogy(tt(1:n+2), gather(Energy(1:n+2)), 'LineWidth', 1.5);
    grid on; xlabel('time'); ylabel('Energy(\phi)'); title('Energy decay (stabilized SBDF2)');
end

end

%% ------------ helpers ------------
function L3D = makeLambda3D(eigx, eigy, eigz)
L3D = reshape(eigx, [],1,1) + reshape(eigy, 1,[],1) + reshape(eigz, 1,1,[]);
end

function s = gather_scalar(x)
try
    y = gather(x);
catch
    y = x;
end
s = y(1);
end

function showIsosurface(phi, x, y, z, tt)
try
    Ph = gather(phi); X = gather(x); Y = gather(y); Z = gather(z);
    [Xg,Yg,Zg] = ndgrid(X,Y,Z);
    figure('Name',sprintf('Isosurface t=%.3f',tt));
    p = patch(isosurface(Xg,Yg,Zg,Ph,0.0));
    isonormals(Xg,Yg,Zg,Ph,p);
    p.FaceColor = [0.3 0.6 0.9]; p.EdgeColor = 'none';
    daspect([1 1 1]); view(3); camlight; lighting gouraud;
    xlabel('x'); ylabel('y'); zlabel('z');
    title(sprintf('\\phi=0 isosurface at t=%.3f',tt));
    drawnow;
catch
end
end

function avg = weightedAvg(phi, Wx, Wy, Wz, Vol)
% ∫phi / ∫1  using tensor weights without forming 3D W
phiWx = bsxfun(@times, phi, reshape(Wx, [],1,1));  Sx = sum(phiWx, 1);          % 1×ny×nz
SxWy  = bsxfun(@times, Sx, reshape(Wy, 1,[],1));   Sy = sum(SxWy, 2);           % 1×1×nz
SyWz  = bsxfun(@times, Sy, reshape(Wz, 1,1,[]));   tot= sum(SyWz, 3);           % scalar
avg   = gather(tot) / gather(Vol);
end

function phi = massCorrect(phi, Wx, Wy, Wz, Vol, avg0)
avg = weightedAvg(phi, Wx, Wy, Wz, Vol);
phi = phi + (avg0 - avg);   % shift constant mode (nullspace of Δ)
end
