% xiao yao v2025.5.xx
% Solving the three-dimensional Navier–Stokes–Cahn–Hilliard system
% using a tensor-product spectral element projection scheme.
% uw 0.01 0.02 0.03 0.05
% radius = 0.15;
% delta = 4e-5 5e-5
% Cz = radius + hz + eta
% gravity = -1;
% bubble tension = 0.01 0.1
% delta = 0.01 1
% rt


xy = 1;%0:继续计算, 1:重新开始
xyxy = 1; %0:继续计算, 1:重新开始
if xy == 1
    clc
    clear
    addpath(genpath("D:\semMatlab")) % Ensure this path is correct

    currentLocation = pwd;
    cd ..;
    fileLocation = pwd;
    currentTime = datetime('now', 'Format', 'yyyyMMdd_HHmm_ss');
    pathname = [fileLocation, '\time' char(currentTime)];
    cd(currentLocation)
    copyfile("*.m", pathname)

    stage = 1;
    fprintf('=== %d Program Starts ===\n', stage);
    stage = stage + 1;

    %% Problem parameters
    delta = 5e-3;           % Time step
    steps = 1000;          % Maximum number of iterations
    freOut = 200;           % Tecplot output frequency
    frePrint = 10;          % Console print frequency
    variables_grid={'X','Y','Z'};
    velName={'u','v','w','p','phi','psi','rho','mu'}; %,'ftx','fty','ftz'
    % parpool(size(velName,2));
    filename_grid = fullfile(pathname, ['grid','.plt']);

    LL1 = 100;
    Cn = 0.01;
    eta = Cn * LL1;
    radius = LL1/2;
    Cx = LL1/4;
    Cy = LL1/4;
    Cz = LL1/4;
    hz = radius*2;
    uw = 0;
    

    Lrho = 3;  % density
    Grho = 1;
    Lmu = 0.09766;
    Gmu = 0.09766; %  dynamic viscosity
    nium = 0.09766; %mu/rho
    rho0 = min(Lrho,Grho);


    rhoplus = (Lrho + Grho)/ 2;
    rhomius = (Lrho - Grho)/ 2;
    muplus = (Lmu + Gmu)/ 2;
    mumius = (Lmu - Gmu)/ 2;

    gravity = -0.01;
    tension = 1.96;
    M0 = 1e-7; % 5e-7 1e-6;
    gamma0 = 1.5;
    bigs = 1.1 * eta ^ 2 * sqrt(4 * gamma0 / (M0 * delta));
    aa = 1 - gamma0 / (M0 * delta) * 4 * eta ^ 4 / bigs ^ 2;
    alphaCH = -bigs / (2 * eta ^ 2) * (1 + sqrt(aa));
    lambda = 3 * eta * tension / (2 * sqrt(2));

    %% Discretisation parameters
    para_d.Np = 4;
    Np = para_d.Np;

    para_d.basis = 'SEM';
    para_d.minx = -0.5*LL1; 
    para_d.maxx = 0.5*LL1;
    para_d.miny = -0.5*LL1; 
    para_d.maxy = 0.5*LL1;
    para_d.minz = -2 * LL1;
    para_d.maxz = 2 * LL1;

    para_d.Ncellx = 32;
    para_d.Ncelly = 32;
    para_d.Ncellz = 128;

    if gpuDeviceCount('available') < 1
        para_d.device = 'cpu';
    else
        para_d.device = 'gpu';
        para_d.device_id = 1;
    end

    para_d.Npx = Np;
    para_d.Npy = Np;
    para_d.Npz = Np;
    para_d.nx_all = para_d.Ncellx * Np + 1;
    para_d.ny_all = para_d.Ncelly * Np + 1;
    para_d.nz_all = para_d.Ncellz * Np + 1;

    para_d.bc = 'dirichlet';
    para_n = para_d;
    para_n.bc = 'neumann';

    para_d = parameter_bc(para_d);
    para_n = parameter_bc(para_n);

    fprintf('Laplacian is Q%d spectral element method\n', Np);

    %% Spectral element operators
    [~, x, Txd, ~, lambda_xd, Dmatrixx, ~] = cal_matrix2('x', para_d);
    [~, y, Tyd, ~, lambda_yd, Dmatrixy, ~] = cal_matrix2('y', para_d);
    [~, z, Tzd, ~, lambda_zd, Dmatrixz, ~] = cal_matrix2('z', para_d);
    
    Dmatrixx = full(Dmatrixx);
    Dmatrixy = full(Dmatrixy);
    Dmatrixz = full(Dmatrixz);
    DmatrixyT = Dmatrixy';
    clear Dmatrixy

    [~, ~, Txn, lambda_xn] = cal_matrix2_1('x', para_n, x);
    [~, ~, Tyn, lambda_yn] = cal_matrix2_1('y', para_n, y);
    [~, ~, Tzn, lambda_zn] = cal_matrix2_1('z', para_n, z);

    [coordY, coordX, coordZ] = meshgrid(y, x, z);
    coords = [coordX(:), coordY(:), coordZ(:)];
    wx = assemble_sem_weights(para_n.Npx, para_n.Ncellx, para_n.minx, para_n.maxx);
    wy = assemble_sem_weights(para_n.Npy, para_n.Ncelly, para_n.miny, para_n.maxy);
    wz = assemble_sem_weights(para_n.Npz, para_n.Ncellz, para_n.minz, para_n.maxz);
    mass_diag = reshape(wx, [], 1, 1) .* reshape(wy', 1, [], 1) .* reshape(wz', 1, 1, []);

    clear x y z
    %% Initialisation
    fprintf('=== %d Initialisation ===\n', stage);
    stage = stage + 1;

    invTxd = pinv(Txd);
    invTyd = pinv(Tyd);
    invTzd = pinv(Tzd);
    invTxn = pinv(Txn);
    invTyn = pinv(Tyn);
    invTzn = pinv(Tzn);

    [Un, Vn, Pn, Pn1, Pn_1,Fx,Fy,Fz,zeross] = deal(zeros(para_d.nx_all, para_d.ny_all, para_d.nz_all));
    Un1 = Un; Vn1 = Vn;
    Un_1 = Un; Vn_1 = Vn;

    [phin,Wn] = intialPhi(coordX,coordY,coordZ,Cx,Cy,Cz,radius,eta,hz,uw,LL1);
      saveTecPhi(0,pathname,IJK,'phi',phin(:)) 
    Wn1 = Wn;
    Wn_1 = Wn;
    title='';%无标题    
    zone_title='';%无标题 
    time=0;%非定常时间
    IJK=[para_d.nx_all, para_d.ny_all, para_d.nz_all];
    %创建文件头
    plt_Head(filename_grid,title,variables_grid,'GRID')
    %创建zone（point）格式
    plt_Zone(filename_grid,zone_title,IJK,time,coords)
    % OUTPUT_Tecplot3d2(0, pathname, coords, para_d.nx_all, para_d.ny_all, para_d.nz_all, varName, ...
    %     phin(:), zeross(:), zeross(:), zeross(:));

    phin1 = phin;
    phin_1 = phin;


    %% Eigenvalue tensors
    lambda_xd = lambda_xd(:);
    lambda_yd = lambda_yd(:);
    lambda_zd = lambda_zd(:);
    lambda_xn = lambda_xn(:);
    lambda_yn = lambda_yn(:);
    lambda_zn = lambda_zn(:);

    poisson_d = reshape(lambda_xd, [], 1, 1) + reshape(lambda_yd, 1, [], 1) ...
        + reshape(lambda_zd, 1, 1, []);
    helmholtz_uv = gamma0 / delta + nium * poisson_d;
    clear lambda_xd lambda_yd lambda_zd
    % poisson_nd = reshape(lambda_xn, [], 1, 1) + reshape(lambda_yn, 1, [], 1) ...
    % + reshape(lambda_zd, 1, 1, []);


    poisson_n = reshape(lambda_xn, [], 1, 1) + reshape(lambda_yn, 1, [], 1) ...
        + reshape(lambda_zn, 1, 1, []);

    clear lambda_xn lambda_yn lambda_zn
    helmholtz_w = gamma0 / delta + nium * poisson_n;


    poisson_p = poisson_n;

    helmholtzPhi = alphaCH - poisson_n;

    helmholtzPsi = (alphaCH + bigs / eta ^ 2) + poisson_n;


    weight_yz = reshape(wy', 1, [], 1) .* reshape(wz', 1, 1, []);
    weight_xz = reshape(wx, [], 1, 1) .* reshape(wz', 1, 1, []);
    weight_xy = reshape(wx, [], 1, 1) .* reshape(wy', 1, [], 1);

    clear wx wy wz
    %% Device transfer
    switch para_d.device
        case 'gpu'
            fprintf('GPU computation: moving data\n');
            Device = gpuDevice(para_d.device_id);
            Txd = gpuArray(Txd); Tyd = gpuArray(Tyd); Tzd = gpuArray(Tzd);
            Txn = gpuArray(Txn); Tyn = gpuArray(Tyn); Tzn = gpuArray(Tzn);
            invTxd = gpuArray(invTxd); invTyd = gpuArray(invTyd); invTzd = gpuArray(invTzd);
            invTxn = gpuArray(invTxn); invTyn = gpuArray(invTyn); invTzn = gpuArray(invTzn);
            Dmatrixx = gpuArray(Dmatrixx); DmatrixyT = gpuArray(DmatrixyT); Dmatrixz = gpuArray(Dmatrixz);
            helmholtz_w = gpuArray(helmholtz_w);

            helmholtz_uv = gpuArray(helmholtz_uv);
            helmholtzPhi = gpuArray(helmholtzPhi); helmholtzPsi = gpuArray(helmholtzPsi);
            poisson_p = gpuArray(poisson_p);
            Un = gpuArray(Un); Vn = gpuArray(Vn); Wn = gpuArray(Wn);
            Un1 = gpuArray(Un1); Vn1 = gpuArray(Vn1); Wn1 = gpuArray(Wn1);
            Un_1 = gpuArray(Un_1); Vn_1 = gpuArray(Vn_1); Wn_1 = gpuArray(Wn_1);
            Pn = gpuArray(Pn); Pn1 = gpuArray(Pn1); Pn_1 = gpuArray(Pn_1);
            phin = gpuArray(phin); phin1 = gpuArray(phin1); phin_1 = gpuArray(phin_1);
            mass_diag = gpuArray(mass_diag);
           
    end

    if strcmp(para_d.device, 'gpu')
        wait(Device);
        fprintf('GPU initialisation complete\n');
    end

    Dmatrixx2 = Dmatrixx * Dmatrixx;
    DmatrixyT2 = DmatrixyT * DmatrixyT;
    Dmatrixz2 = Dmatrixz * Dmatrixz;

    Iter1 = 1;
else
    % load flow

    steps = 2 * steps;
    Iter1 = Iter + 1;


    stage = 1;
    fprintf('=== %d Inital old old old === \n',stage);
    stage = stage + 1;
    if xyxy == 0
        fprintf('=== %d load old file === \n',stage);
        stage = stage + 1;
        load flow

        A = dlmread('rere.dat');
        Un1 = A(:,3);
        Vn1 = A(:,4);
        phin1 = A(:,5);
        Umid = Un1;
        Vmid = Vn1;

    end
end
%% Time stepping
tic;
fprintf('=== %d Time stepping ===\n', stage);
stage = stage + 1;

for Iter = Iter1:steps

    uStar = 2 * Un - Un_1;
    vStar = 2 * Vn - Vn_1;
    wStar = 2 * Wn - Wn_1;
    pStar = 2 * Pn - Pn_1;
    phiStar = 2 * phin - phin_1;

    uCap = 2 * Un - 0.5 * Un_1;
    vCap = 2 * Vn - 0.5 * Vn_1;
    wCap = 2 * Wn - 0.5 * Wn_1;
    phiCap = 2 * phin - 0.5 * phin_1;

    % ----- Phase field -----
    Dphi_x = pagemtimes(Dmatrixx, phiStar);
    Dphi_y = pagemtimes(phiStar, DmatrixyT);
    Dphi_z = derivative_z(Dmatrixz, phiStar, para_d.nx_all, para_d.ny_all, para_d.nz_all);

    Q1 = (uStar .* Dphi_x + vStar .* Dphi_y + wStar .* Dphi_z - phiCap / delta) / M0;
    Q2 = (phiStar .^ 2 - 1 - bigs) / eta ^ 2 .* phiStar;
    lapQ2 = calLaplace3D2(Q2, Dmatrixx2, DmatrixyT2, Dmatrixz2, para_d);
    Fpsi = Q1 - lapQ2;

    psi_hat = tensorprod(Fpsi, invTzn', 3, 1);
    psi_hat = pagemtimes(psi_hat, invTyn');
    psi_hat = squeeze(tensorprod(invTxn, psi_hat, 2, 1));
    psi_hat = psi_hat ./ helmholtzPsi;
    psi = tensorprod(psi_hat, Tzn', 3, 1);
    psi = pagemtimes(psi, Tyn');
    psi = squeeze(tensorprod(Txn, psi, 2, 1));

    phi_hat = tensorprod(psi, invTzn', 3, 1);
    phi_hat = pagemtimes(phi_hat, invTyn');
    phi_hat = squeeze(tensorprod(invTxn, phi_hat, 2, 1));
    phi_hat = phi_hat ./ helmholtzPhi;
    phin1 = tensorprod(phi_hat, Tzn', 3, 1);
    phin1 = pagemtimes(phin1, Tyn');
    phin1 = squeeze(tensorprod(Txn, phin1, 2, 1));

    phin1 = max(min(phin1, 1), -1);

    Dphix = pagemtimes(Dmatrixx, phin1);
    Dphiy = pagemtimes(phin1, DmatrixyT);
    Dphiz = derivative_z(Dmatrixz, phin1, para_d.nx_all, para_d.ny_all, para_d.nz_all);

    lapPhi = calLaplace3D2(phin1, Dmatrixx2, DmatrixyT2, Dmatrixz2, para_d);
    psi1 = 1 / eta ^ 2 * (phin1 .^ 3 - phin1) - lapPhi;
    % ftx = lambda * psi1 .* Dphix;
    % fty = lambda * psi1 .* Dphiy;
    % ftz = lambda * psi1 .* Dphiz;

    rho = rhoplus + phin1 * rhomius;
    mu = muplus + phin1 * muplus;
    Dmux = mumius * Dphix;
    Dmuy = mumius * Dphiy;
    Dmuz = mumius * Dphiz;

    % ----- Momentum predictor -----
    DuStarx = pagemtimes(Dmatrixx, uStar);
    DuStary = pagemtimes(uStar, DmatrixyT);
    DuStarz = derivative_z(Dmatrixz, uStar, para_d.nx_all, para_d.ny_all, para_d.nz_all);

    DvStarx = pagemtimes(Dmatrixx, vStar);
    DvStary = pagemtimes(vStar, DmatrixyT);
    DvStarz = derivative_z(Dmatrixz, vStar, para_d.nx_all, para_d.ny_all, para_d.nz_all);

    DwStarx = pagemtimes(Dmatrixx, wStar);
    DwStary = pagemtimes(wStar, DmatrixyT);
    DwStarz = derivative_z(Dmatrixz, wStar, para_d.nx_all, para_d.ny_all, para_d.nz_all);

    DpStarx = pagemtimes(Dmatrixx, pStar);
    DpStary = pagemtimes(pStar, DmatrixyT);
    DpStarz = derivative_z(Dmatrixz, pStar, para_d.nx_all, para_d.ny_all, para_d.nz_all);

    lapU = calLaplace3D2(uStar, Dmatrixx2, DmatrixyT2, Dmatrixz2, para_d);
    lapV = calLaplace3D2(vStar, Dmatrixx2, DmatrixyT2, Dmatrixz2, para_d);
    lapW = calLaplace3D2(wStar, Dmatrixx2, DmatrixyT2, Dmatrixz2, para_d);

    miuRho = mu ./ rho;

    Dustarx = 2 * Dmux .* DuStarx + Dmuy .* (DvStarx + DuStary) + Dmuz .* (DwStarx + DuStarz);
    Dustary = Dmux .* (DvStarx + DuStary) + 2 * Dmuy .* DvStary + Dmuz .* (DwStary + DvStarz);
    Dustarz = Dmux .* (DwStarx + DuStarz) + Dmuy .* (DwStary + DvStarz) + 2 * Dmuz .* DwStarz;

    adv_u = uStar .* DuStarx + vStar .* DuStary + wStar .* DuStarz;
    adv_v = uStar .* DvStarx + vStar .* DvStary + wStar .* DvStarz;
    adv_w = uStar .* DwStarx + vStar .* DwStary + wStar .* DwStarz;

    uv31x = uCap / delta + (1 / rho0 - 1 ./ rho) .* DpStarx - adv_u ...
        + (miuRho - nium) .* lapU + Dustarx ./ rho;% + ftx ./ rho;
    uv31y = vCap / delta + (1 / rho0 - 1 ./ rho) .* DpStary - adv_v ...
        + (miuRho - nium) .* lapV + Dustary ./ rho;% + fty ./ rho;
    uv31z = wCap / delta + (1 / rho0 - 1 ./ rho) .* DpStarz - adv_w + gravity...
        + (miuRho - nium) .* lapW + Dustarz ./ rho;% + ftz ./ rho ;

    % ----- Pressure correction -----
    FP = -rho0 * (pagemtimes(Dmatrixx, uv31x) ...
        + pagemtimes(uv31y, DmatrixyT) ...
        + derivative_z(Dmatrixz, uv31z, para_d.nx_all, para_d.ny_all, para_d.nz_all));

    midx = rho0 * (uv31x - uStar / delta + nium * lapU);
    midy = rho0 * (uv31y - vStar / delta + nium * lapV);
    midz = rho0 * (uv31z - wStar / delta + nium * lapW);

    Fx(:)=0;
    Fy(:)=0;
    Fz(:)=0;
    
    Fx(1, :, :) = -midx(1, :, :) .* weight_yz;
    Fx(end, :, :) = midx(end, :, :) .* weight_yz;
    Fy(:, 1, :) = -midy(:, 1, :) .* weight_xz;
    Fy(:, end, :) = midy(:, end, :) .* weight_xz;
    Fz(:, :, 1) = -midz(:, :, 1) .* weight_xy;
    Fz(:, :, end) = midz(:, :, end) .* weight_xy;

    Fg = Fx + Fy + Fz;
    f_solver = FP + Fg ./ mass_diag;

    pre_spec = tensorprod(f_solver, invTzn', 3, 1);
    pre_spec = pagemtimes(pre_spec, invTyn');
    pre_spec = squeeze(tensorprod(invTxn, pre_spec, 2, 1));
    pre_spec = pre_spec ./ poisson_p;
    Pn1 = tensorprod(pre_spec, Tzn', 3, 1);
    Pn1 = pagemtimes(Pn1, Tyn');
    Pn1 = squeeze(tensorprod(Txn, Pn1, 2, 1));
    Pn1 = Pn1 - Pn1(1, 1, 1);

    % if strcmp(para_d.device, 'gpu'); wait(Device); end

    % ----- Velocity correction -----
    GradPre_x = pagemtimes(Dmatrixx, Pn1);
    GradPre_y = pagemtimes(Pn1, DmatrixyT);
    GradPre_z = derivative_z(Dmatrixz, Pn1, para_d.nx_all, para_d.ny_all, para_d.nz_all);

    FU = uv31x - GradPre_x ./ rho0;
    FV = uv31y - GradPre_y ./ rho0;
    FW = uv31z - GradPre_z ./ rho0;

    FU_interior = FU(para_d.freeNodesx, para_d.freeNodesy, para_d.freeNodesz);
    FV_interior = FV(para_d.freeNodesx, para_d.freeNodesy, para_d.freeNodesz);
    % FW_interior = FW;

    u_spec = tensorprod(FU_interior, invTzd', 3, 1);
    u_spec = pagemtimes(u_spec, invTyd');
    u_spec = squeeze(tensorprod(invTxd, u_spec, 2, 1));
    u_spec = u_spec ./ helmholtz_uv;
    u_phys = tensorprod(u_spec, Tzd', 3, 1);
    u_phys = pagemtimes(u_phys, Tyd');
    u_phys = squeeze(tensorprod(Txd, u_phys, 2, 1));
    Un1(para_d.freeNodesx, para_d.freeNodesy, para_d.freeNodesz) = u_phys;

    if strcmp(para_d.device, 'gpu'); wait(Device); end

    v_spec = tensorprod(FV_interior, invTzd', 3, 1);
    v_spec = pagemtimes(v_spec, invTyd');
    v_spec = squeeze(tensorprod(invTxd, v_spec, 2, 1));
    v_spec = v_spec ./ helmholtz_uv;
    v_phys = tensorprod(v_spec, Tzd', 3, 1);
    v_phys = pagemtimes(v_phys, Tyd');
    v_phys = squeeze(tensorprod(Txd, v_phys, 2, 1));
    Vn1(para_d.freeNodesx, para_d.freeNodesy, para_d.freeNodesz) = v_phys;

    if strcmp(para_d.device, 'gpu'); wait(Device); end

    w_spec = tensorprod(FW, invTzn', 3, 1);
    w_spec = pagemtimes(w_spec, invTyn');
    w_spec = squeeze(tensorprod(invTxn, w_spec, 2, 1));
    w_spec = w_spec ./ helmholtz_w;
    w_phys = tensorprod(w_spec, Tzn', 3, 1);
    w_phys = pagemtimes(w_phys, Tyn');
    Wn1 = squeeze(tensorprod(Txn, w_phys, 2, 1));

    if strcmp(para_d.device, 'gpu'); wait(Device); end

    % ----- Diagnostics & Output -----
    vel_change = norm(Un1(:) - Un(:)) + norm(Vn1(:) - Vn(:)) + norm(Wn1(:) - Wn(:));

    if rem(Iter, frePrint) == 0
        fprintf('Iter = %d, Δu = %.3e\n', Iter, vel_change);
    end

    if (vel_change > 340) || isnan(vel_change)
        fprintf('Divergence detected at iteration %d\n', Iter);
        fprintf('Iter = %d, Δu = %.3e\n', Iter, vel_change);
        break;
    end

    if rem(Iter, freOut) == 0
          saveTec(Iter,pathname,IJK,velName,Un1(:), Vn1(:), Wn1(:), Pn1(:),...
             phin1(:), psi1(:), rho(:), mu(:)) %, ftx(:), fty(:), ftz(:)
    end

    % ----- Advance states -----
    Un_1 = Un;
    Vn_1 = Vn;
    Wn_1 = Wn;
    Pn_1 = Pn;
    phin_1 = phin;

    Un = Un1;
    Vn = Vn1;
    Wn = Wn1;
    Pn = Pn1;
    phin = phin1;

end

time = toc;
fprintf('Total computation time: %.2f s\n', time);
fprintf('=== %d Program Ends ===\n', stage);

% cd(pathname);
% save flow;
% cd(currentLocation);

datetime
sending_to_emil(Iter,steps,1)