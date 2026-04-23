function Rb = rhs_grad_projection_boundary3d(gx, gy, gz, Dx, Dy, Dz, Wx, Wy, Wz)
% Assemble boundary term <g, grad v_h> over all six faces.

[nx,ny,nz] = size(gx);
Rb = zeros(nx,ny,nz, 'like', gx);

% === X faces ===
% x = -Lx (i=1)
for j = 1:ny
    for k = 1:nz
        Rb(1,j,k) = Rb(1,j,k) + (-gx(1,j,k)) * Dy(j,j) * Dz(k,k) * Wy(j) * Wz(k);
    end
end
% x = +Lx (i=nx)
for j = 1:ny
    for k = 1:nz
        Rb(nx,j,k) = Rb(nx,j,k) + ( gx(nx,j,k)) * Dy(j,j) * Dz(k,k) * Wy(j) * Wz(k);
    end
end

% === Y faces ===
for i = 1:nx
    for k = 1:nz
        Rb(i,1,k)  = Rb(i,1,k)  + (-gy(i,1,k)) * Dx(i,i) * Dz(k,k) * Wx(i) * Wz(k);
        Rb(i,ny,k) = Rb(i,ny,k) + ( gy(i,ny,k)) * Dx(i,i) * Dz(k,k) * Wx(i) * Wz(k);
    end
end

% === Z faces ===
for i = 1:nx
    for j = 1:ny
        Rb(i,j,1)  = Rb(i,j,1)  + (-gz(i,j,1)) * Dx(i,i) * Dy(j,j) * Wx(i) * Wy(j);
        Rb(i,j,nz) = Rb(i,j,nz) + ( gz(i,j,nz)) * Dx(i,i) * Dy(j,j) * Wx(i) * Wy(j);
    end
end
end
