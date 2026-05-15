function [gx, gy, gz] = gradient_ns3d(sem, para, field)
gx = pagemtimes(sem.Dx, field);
gy = pagemtimes(field, sem.DyT);
gz = derivative_z(sem.Dz, field, para.nx_all, para.ny_all, para.nz_all);
end
