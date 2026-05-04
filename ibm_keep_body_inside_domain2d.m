function ibm = ibm_keep_body_inside_domain2d(ibm)
%IBM_KEEP_BODY_INSIDE_DOMAIN2D Simple inelastic wall handling for one body.

minx = ibm.x(1);
maxx = ibm.x(end);
miny = ibm.y(1);
maxy = ibm.y(end);

if ibm.center(1) - ibm.radius < minx
    ibm.center(1) = minx + ibm.radius;
    ibm.body_velocity(1) = 0;
elseif ibm.center(1) + ibm.radius > maxx
    ibm.center(1) = maxx - ibm.radius;
    ibm.body_velocity(1) = 0;
end

if ibm.center(2) - ibm.radius < miny
    ibm.center(2) = miny + ibm.radius;
    ibm.body_velocity(2) = 0;
elseif ibm.center(2) + ibm.radius > maxy
    ibm.center(2) = maxy - ibm.radius;
    ibm.body_velocity(2) = 0;
end

end
