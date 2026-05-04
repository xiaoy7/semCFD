function delta = ibm_delta_peskin4_1d(r, h)
%IBM_DELTA_PESKIN4_1D Peskin four-point regularized delta kernel.

q = abs(r) ./ h;
delta = zeros(size(q));

inside1 = q < 1;
inside2 = q >= 1 & q < 2;

delta(inside1) = (3 - 2 * q(inside1) ...
    + sqrt(1 + 4 * q(inside1) - 4 * q(inside1).^2)) ./ (8 * h);
delta(inside2) = (5 - 2 * q(inside2) ...
    - sqrt(-7 + 12 * q(inside2) - 4 * q(inside2).^2)) ./ (8 * h);

end

