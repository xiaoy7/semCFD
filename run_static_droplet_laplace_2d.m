% Static droplet Laplace-law verification setup (2D)
% phi(x,0) = tanh((R - |x-x_c|)/(sqrt(2)*eta))
clc; clear;

% --- Domain and droplet parameters ---
Lx = 1.0; Ly = 1.0;
Nx = 256; Ny = 256;
xc = [0.5, 0.5];
R  = 0.2;
eta = 0.01;
sigma = 1.0;

x = linspace(0, Lx, Nx);
y = linspace(0, Ly, Ny);
[Y, X] = meshgrid(y, x);

% Initial phase field (requested form)
r = sqrt((X - xc(1)).^2 + (Y - xc(2)).^2);
phi0 = tanh((R - r) ./ (sqrt(2) * eta));

% Theoretical Laplace pressure jump in 2D
kappa = 1 / R;
delta_p_theory = sigma * kappa;

fprintf('Static droplet test initialized (2D).\n');
fprintf('R = %.6f, eta = %.6f, sigma = %.6f\n', R, eta, sigma);
fprintf('Theoretical curvature kappa = %.6f\n', kappa);
fprintf('Theoretical pressure jump Delta p = sigma*kappa = %.6f\n', delta_p_theory);

% --- Monitoring helpers for an NS-CH solve ---
% 1) Pressure jump from simulation fields:
%    Delta p_num = mean(p_inside) - mean(p_outside)
%    inside: r <= R - 2*eta, outside: r >= R + 2*eta
% 2) Maximum spurious velocity:
%    u_spurious_max = max( sqrt(u.^2 + v.^2), [], 'all' )
% 3) Relative Laplace error:
%    err = abs(Delta p_num - delta_p_theory) / max(abs(delta_p_theory), eps)

% Save initialization for use by solver scripts.
save('static_droplet_init_2d.mat', 'X', 'Y', 'phi0', 'r', 'R', 'eta', 'sigma', 'kappa', 'delta_p_theory');
fprintf('Saved initialization to static_droplet_init_2d.mat\n');
