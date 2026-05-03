function rate = convergence_rate(err, scale)
rate = nan(size(err));
for k = 2:numel(err)
    rate(k) = log(err(k - 1) / err(k)) / log(scale(k - 1) / scale(k));
end
end
