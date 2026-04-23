function w = assemble_sem_weights(N, Ncell, Left, Right)
% assemble_sem_weights 1D SEM quadrature weights accumulated over elements
[~, ~, w_ref] = LegendreD(N);
dl = (Right - Left) / Ncell;
w = zeros(Ncell * N + 1, 1);
scale = dl / 2;
for elem = 1:Ncell
    idx = (0:N) + (elem - 1) * N + 1;
    w(idx) = w(idx) + scale * w_ref(:);
end
end
