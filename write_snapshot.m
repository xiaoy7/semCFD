function write_snapshot(iter, runDir, para, varName, coordX, coordY, un, vn, pressure, phi, psi, deviceType)
if strcmp(deviceType, 'gpu')
    un = gather(un);
    vn = gather(vn);
    pressure = gather(pressure);
    phi = gather(phi);
    psi = gather(psi);
end

OUTPUT_Tecplot2D4(iter, runDir, para.ny_all, para.nx_all, varName, ...
    coordX(:), coordY(:), un(:), vn(:), pressure(:), phi(:), psi(:));
end
