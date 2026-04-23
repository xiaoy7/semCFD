% liu Poisson_SEM_dirichlet
clc
clear
addpath(genpath("D:\semMatlab"))

currentTime = datetime('now', 'Format', 'yyyyMMdd_HHmm');
pathname = ['time' char(currentTime)];
copyfile("*.m", pathname)


Param.num_repeat = 1;
Param.plotfigure = 1; % 1: yes, 0: no
Param.Np = 5; % polynomial degree
Param.Ncell = 20; % number of cells in finite element
Param.bc = 'dirichlet'; % 'dirichlet' 'periodic'  'neumann'
Param.basis = 'SEM'; %'FFT' 'SEM'
if gpuDeviceCount('available') < 1
    Param.device = 'cpu';
else
    Param.device = 'gpu';
    Param.device_id = 1;
end % ID=1,2,3,...


[l2_err, l_infty_err] = Poisson_SEM_dirichlet(Param);

fprintf('done \n');