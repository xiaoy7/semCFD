function w2 = ibm_control_volume_weights2d(x, y)
%IBM_CONTROL_VOLUME_WEIGHTS2D Nodal control-volume weights on a tensor grid.

wx = local_weights_1d(x(:));
wy = local_weights_1d(y(:));
w2 = wx * wy.';

end