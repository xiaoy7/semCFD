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

    para_d.Ncellx = 20;
    para_d.Ncelly = 20;
    para_d.Ncellz = 20;

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

    Wn1 = Wn;
    Wn_1 = Wn;

    %% write
    fprintf('=== %d write ===\n', stage);
    stage = stage + 1;
    title='';%无标题
    zone_title='';%无标题
    time=0;%非定常时间
    IJK=[para_d.nx_all, para_d.ny_all, para_d.nz_all];
    %创建文件头
    plt_Head(filename_grid,title,variables_grid,'GRID')
    fprintf('=== write head ===\n');
    %创建zone（point）格式
    plt_Zone(filename_grid,zone_title,IJK,time,coords)
    % OUTPUT_Tecplot3d2(0, pathname, coords, para_d.nx_all, para_d.ny_all, para_d.nz_all, varName, ...
    %     phin(:), zeross(:), zeross(:), zeross(:));
    fprintf('=== write Zone ===\n');
    phin1 = phin;
    phin_1 = phin;
    saveTecPhi(0,pathname,IJK,'phi',phin(:))


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
fprintf('=== %d Time stepping ===\n', stage);
stage = stage + 1;
