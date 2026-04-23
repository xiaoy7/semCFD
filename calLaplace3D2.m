function lapU = calLaplace3D2(U, Dmatrixx2, DmatrixyT2, Dmatrixz2, para)
Uxx = pagemtimes(Dmatrixx2, U);

Uyy = pagemtimes(U, DmatrixyT2);

Uzz = derivative_z(Dmatrixz2, U, para.nx_all, para.ny_all, para.nz_all);

lapU = Uxx + Uyy + Uzz;
end
