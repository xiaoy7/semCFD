function [u, v] = apply_lid_boundary(u, v, para, lidVelocity)
u(:, para.bcNodesy(2)) = lidVelocity;
u(:, para.bcNodesy(1)) = 0;
u(para.bcNodesx, :) = 0;
v(:, para.bcNodesy) = 0;
v(para.bcNodesx, :) = 0;
end


