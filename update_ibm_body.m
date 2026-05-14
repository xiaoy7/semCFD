function ibm = update_ibm_body(ibm, iter, dt, coordX, coordY)
try
    ibm = ibm_update_rigid_body2d(ibm, iter, dt, coordX, coordY);
catch err
    if strcmp(err.identifier, 'MATLAB:maxrhs')
        ibm = ibm_update_rigid_body2d(ibm, iter, dt);
    else
        rethrow(err);
    end
end
end

