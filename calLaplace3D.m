function lapU = calLaplace3D(U, Dmatrixx, DmatrixyT, Dmatrixz, para)
Ux = pagemtimes(Dmatrixx, U);
Uxx = pagemtimes(Dmatrixx, Ux);

Uy = pagemtimes(U, DmatrixyT);
Uyy = pagemtimes(Uy, DmatrixyT);

Uz = derivative_z(Dmatrixz, U, para.nx_all, para.ny_all, para.nz_all);
Uzz = derivative_z(Dmatrixz, Uz, para.nx_all, para.ny_all, para.nz_all);

lapU = Uxx + Uyy + Uzz;
end
