function R = rhs_grad_projection2d(fx, fy, Dx, Dy, Wx, Wy)
% rhs_grad_projection2d  Approximate divergence of tensor fields with SEM weights.
% Supports CPU and GPU inputs.

fx_cpu = gather(fx);
fy_cpu = gather(fy);
Dx_cpu = gather(Dx);
Dy_cpu = gather(Dy);
Wx_cpu = gather(Wx);
Wy_cpu = gather(Wy);

[nx, ny] = size(fx_cpu);
R = zeros(nx, ny);

% X contribution
WxD = spdiags(Wx_cpu, 0, nx, nx) * Dx_cpu;
for j = 1:ny
    R(:, j) = R(:, j) + (WxD.' * fx_cpu(:, j));
end

% Y contribution
WyD = spdiags(Wy_cpu, 0, ny, ny) * Dy_cpu;
for i = 1:nx
    R(i, :) = R(i, :) + ((WyD.' * fy_cpu(i, :).').');
end

end
