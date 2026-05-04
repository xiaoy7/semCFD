function c = ibm_compute_beam_curvature2d(markers, left_ids, mid_ids, right_ids)
%IBM_COMPUTE_BEAM_CURVATURE2D IB2d torsional-beam reference curvature.

xp = markers(left_ids, 1);
xq = markers(mid_ids, 1);
xr = markers(right_ids, 1);
yp = markers(left_ids, 2);
yq = markers(mid_ids, 2);
yr = markers(right_ids, 2);

c = (xr - xq) .* (yq - yp) - (yr - yq) .* (xq - xp);

end

