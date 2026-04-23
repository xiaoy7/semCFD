function R = rhs_grad_projectionx(fx, Dx, Wx, Wy)
% rhs_grad_projectionx  Project x-derivative contribution with SEM weights.
% Supports CPU and GPU arrays.

 
    fx_cpu = gather(fx);
    Dx_cpu = gather(Dx);
    Wx_cpu = gather(Wx);
 
[nx, ny] = size(fx_cpu);
R = zeros(nx, ny);

WxD = spdiags(Wx_cpu, 0, nx, nx) * Dx_cpu;
for j = 1:ny
    R(:, j) = R(:, j) + (WxD.' * fx_cpu(:, j));
end
 
end
