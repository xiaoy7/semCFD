function W = makeWeights1D(x)
% Optional helper: build crude 1D trapezoid weights from node coords x.
x = gather(x(:));
n = numel(x);
W = zeros(n,1);
W(1)   = 0.5*(x(2)-x(1));
W(end) = 0.5*(x(end)-x(end-1));
for i=2:n-1
    W(i) = 0.5*(x(i+1)-x(i-1));
end
end
