function R = rhs_grad_projectiony(fy, Dy, Wx, Wy)
% rhs_grad_projectiony  Project y-derivative contribution with SEM weights.
% Supports CPU and GPU arrays.

 
    fy_cpu = gather(fy);
    Dy_cpu = gather(Dy);
    Wy_cpu = gather(Wy);
 

[nx, ny] = size(fy_cpu);
R = zeros(nx, ny);

WyD = spdiags(Wy_cpu, 0, ny, ny) * Dy_cpu;
for i = 1:nx
    R(i, :) = R(i, :) + ((WyD.' * fy_cpu(i, :).').');
end

 
end
