function g = manufactured_ch_2d_source(sem, t, cfg)
[~, ~, g] = manufactured_ch_2d(sem.x, sem.y, t, cfg.epsilon, cfg.mobility);
end

