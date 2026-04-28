function ibm = ibm_update_rigid_body2d(ibm, Iter, dT)
% Placeholder update for rigid-body motion in future FSI coupling.
% For now the immersed body is fixed, but this function centralizes
% motion updates so the fluid-solid coupling can be extended cleanly.

if ~ibm.enabled
    return
end

% Example hook for future coupling:
% ibm.center = ibm.center + dT * ibm.body_velocity;
% ibm.body_omega can be used to rotate target velocity field.

ibm.time = Iter * dT;

end
