function ibm = ibm_refresh_mask2d(ibm, coordX, coordY)
%IBM_REFRESH_MASK2D Build a smooth visualization/enforcement mask.

radius_field = sqrt((coordX - ibm.center(1)).^2 ...
                  + (coordY - ibm.center(2)).^2);
width = max(ibm.kernel_width, eps);
ibm.mask = 0.5 * (1 - tanh((radius_field - ibm.radius) / width));

end

