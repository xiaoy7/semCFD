function [force_u, force_v] = ibm_direct_forcing2d(u, v, ibm, dT)
% Compute direct-forcing IBM source terms for momentum equations.

if ~ibm.enabled
    force_u = zeros(size(u), 'like', u);
    force_v = zeros(size(v), 'like', v);
    return
end

penalty = ibm.penalty;
if isempty(penalty)
    penalty = 1 / max(dT, eps);
end

force_u = penalty * ibm.mask .* (ibm.target_u - u);
force_v = penalty * ibm.mask .* (ibm.target_v - v);

end
