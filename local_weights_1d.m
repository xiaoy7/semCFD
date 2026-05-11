function w = local_weights_1d(x)
n = numel(x);
w = zeros(n, 1);

if n == 1
    w(1) = 1;
    return
end

w(1) = 0.5 * (x(2) - x(1));
w(n) = 0.5 * (x(n) - x(n - 1));
if n > 2
    w(2:n-1) = 0.5 * (x(3:n) - x(1:n-2));
end

w = max(w, eps);

end