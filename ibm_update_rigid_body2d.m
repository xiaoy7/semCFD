% Placeholder update for rigid-body motion in future FSI coupling.
% For now the immersed body is fixed, but this function centralizes
% motion updates so the fluid-solid coupling can be extended cleanly.
% =======
function ibm = ibm_update_rigid_body2d(ibm, Iter, dT, coordX, coordY)
%IBM_UPDATE_RIGID_BODY2D Advance rigid-body state and rebuild IBM markers.
% >>>>>>> main

if ~ibm.enabled
    return
end

% <<<<<<< HEAD
% Example hook for future coupling:
% ibm.center = ibm.center + dT * ibm.body_velocity;
% ibm.body_omega can be used to rotate target velocity field.

ibm.time = Iter * dT;

% end
% =======
switch lower(ibm.motion)
    case 'fixed'
        ibm.body_velocity(:) = 0;
        ibm.body_omega = 0;

    case 'prescribed'
        ibm.body_velocity = ibm.prescribed_velocity;
        ibm.body_omega = ibm.prescribed_omega;
        ibm.center = ibm.center + dT * ibm.body_velocity;

    case 'free'
        if isfield(ibm, 'hydro_force')
            acceleration = ibm.gravity + ibm.hydro_force / ibm.body_mass;
            angular_acceleration = ibm.hydro_torque / ibm.body_inertia;
        else
            acceleration = ibm.gravity;
            angular_acceleration = 0;
        end
        ibm.body_velocity = ibm.body_velocity + dT * acceleration;
        ibm.body_omega = ibm.body_omega + dT * angular_acceleration;
        ibm.center = ibm.center + dT * ibm.body_velocity;

    otherwise
        error('Unknown IBM motion mode: %s', ibm.motion);
end

ibm = ibm_keep_body_inside_domain2d(ibm);
ibm = ibm_refresh_markers2d(ibm);
ibm = ibm_refresh_mask2d(ibm, coordX, coordY);

ibm.target_u = ibm.body_velocity(1) - ibm.body_omega * (coordY - ibm.center(2));
ibm.target_v = ibm.body_velocity(2) + ibm.body_omega * (coordX - ibm.center(1));
ibm.time = Iter * dT;

end

% >>>>>>> main
