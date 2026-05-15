function cfg = default_ns3d_config()
cfg.message = "sem 3d re1000";
cfg.reynolds = 1000;
cfg.nu = 1 / cfg.reynolds;
cfg.dt = 1e-4;
cfg.steps = 90000;
cfg.tolerance = 1e-5;
cfg.divergenceTol = 100;
cfg.frePrint = 1;
cfg.freOut = 0;
cfg.alphaHelmholtz = 1 / cfg.dt;
cfg.lidVelocity = 1;

cfg.Np = 4;
cfg.Ncellx = 20;
cfg.Ncelly = 20;
cfg.Ncellz = 20;
cfg.domain = [0, 1, 0, 1, 0, 1];
end
