function Rb = rhs_grad_projection_boundary2d(gx, gy, Dx, Dy, Wx, Wy)
% Assemble boundary term <g, grad v_h> over all six faces.

[nx,ny] = size(gx);
Rb = zeros(nx,ny, 'like', gx);

% === X faces ===
% x = -Lx (i=1)
for j = 1:ny

    Rb(1,j) = Rb(1,j) + (-gx(1,j)) * Dy(j,j) * Wy(j);
    Rb(nx,j) = Rb(nx,j) + ( gx(nx,j)) * Dy(j,j) * Wy(j) ;

end



% === Y faces ===
for i = 1:nx
    Rb(i,1)  = Rb(i,1)  + (-gy(i,1)) * Dx(i,i) * Wx(i);
    Rb(i,ny) = Rb(i,ny) + ( gy(i,ny)) * Dx(i,i) * Wx(i);

end


end
