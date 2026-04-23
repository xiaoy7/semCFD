function R = rhs_grad_projection3d(fx, fy, fz, Dx, Dy, Dz, Wx, Wy, Wz)
% fx,fy,fz: size (nx,ny,nz) nodal values of f components
% Dx,Dy,Dz: global derivative-eval matrices (nx×nx, etc.)
% Wx,Wy,Wz: 1D quadrature weight vectors
% Output R: (nx,ny,nz) with r_{i,j,k} entries

[nx,ny,nz] = size(fx);
Wx = Wx(:); Wy = Wy(:); Wz = Wz(:);

% Precompute weighted f along each axis
% r_x(i,j,k) = Wy(j)Wz(k) * sum_a Wx(a) Dx(a,i) fx(a,j,k)
R = zeros(nx,ny,nz, 'like', fx);

% X-part
WxD = spdiags(Wx,0,nx,nx) * Dx;        % (nx×nx), weights then D
for k = 1:nz
    for j = 1:ny
        R(:,j,k) = R(:,j,k) + (WxD.' * fx(:,j,k)) * (Wy(j)*Wz(k));
    end
end

% Y-part
WyD = spdiags(Wy,0,ny,ny) * Dy;        % (ny×ny)
for k = 1:nz
    for i = 1:nx
        R(i,:,k) = R(i,:,k) + ( (WyD.' * fy(i,:,k).').') * (Wx(i)*Wz(k));
    end
end

% Z-part
WzD = spdiags(Wz,0,nz,nz) * Dz;        % (nz×nz)
for j = 1:ny
    for i = 1:nx
        R(i,j,:) = squeeze(R(i,j,:)) + (WzD.' * squeeze(fz(i,j,:))) * (Wx(i)*Wy(j));
    end
end
end
