function [phi, mu, g, terms] = manufactured_ch_2d(x, y, t, epsilon, mobility, u, v)
%MANUFACTURED_CH_2D Exact 2D Cahn-Hilliard manufactured solution and source.
%   [phi, mu, g] = MANUFACTURED_CH_2D(x, y, t, epsilon, mobility)
%   evaluates
%
%       phi = sin(pi*x).*sin(pi*y).*cos(t)
%
%   and the source g for
%
%       phi_t + u*phi_x + v*phi_y = mobility*laplacian(mu) + g,
%       mu = (phi.^3 - phi)/epsilon^2 - laplacian(phi).
%
%   x and y may be vectors or arrays. If they are vectors, ndgrid is used.
%   u and v are optional scalar or array velocities and default to zero.

if nargin < 6 || isempty(u)
    u = 0;
end
if nargin < 7 || isempty(v)
    v = 0;
end

if isvector(x) && isvector(y)
    [x, y] = ndgrid(x, y);
end

sx = sin(pi * x);
sy = sin(pi * y);
cx = cos(pi * x);
cy = cos(pi * y);
ct = cos(t);
st = sin(t);

phi = sx .* sy .* ct;
phi_t = -sx .* sy .* st;
phi_x = pi * cx .* sy .* ct;
phi_y = pi * sx .* cy .* ct;

lap_phi = -2 * pi^2 * phi;
mu = (phi.^3 - phi) / epsilon^2 - lap_phi;

lap_phi3 = pi^2 * ct^3 .* ...
    (6 * sx .* sy .* (sx.^2 + sy.^2) - 18 * sx.^3 .* sy.^3);
lap_mu = (lap_phi3 - lap_phi) / epsilon^2 - 4 * pi^4 * phi;

g = phi_t + u .* phi_x + v .* phi_y - mobility * lap_mu;

if nargout > 3
    terms.phi_t = phi_t;
    terms.phi_x = phi_x;
    terms.phi_y = phi_y;
    terms.lap_phi = lap_phi;
    terms.lap_phi3 = lap_phi3;
    terms.lap_mu = lap_mu;
end
end
