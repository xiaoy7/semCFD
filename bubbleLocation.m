% intergral of velocity
clc
clear
addpath(genpath("D:\semMatlab"))

freOut = 500;
Ncellx = 30;
Ncelly = 60;
Np = 4; % polynomial degree
NX = Np;
miny = 0;

% 读取所有nii后缀的文件
file = dir('phi*');
% 获取文件个数
len = length(file);
fprintf('total files = %d\n',len);



fname = 'grid.plt';
[a1,a2] = importdata(fname);
coordxy = a1.data;

x1=a1.textdata(3);
x2=extractBetween(x1,'=',',');
x3 = cell2mat(x2);
x = str2num(x3);
y1 = extractBetween(x1,'J=',' ');
y2 = cell2mat(y1);
y = str2num(y2);
coordx = reshape(coordxy(:,1),x,y);
coordy = reshape(coordxy(:,2),x,y);

% Compute SEM weights
wx = assemble_sem_weights(Np, Ncellx, 0, 1);
wy = assemble_sem_weights(Np, Ncelly, 0, 2);
mass_diag = wx * wy';

i = 1;
for Iter = 1:len
    IterT = Iter * freOut;
    fname = ['phi',num2str(IterT,'%d'),'.plt'];
    [a1,a2] = importdata(fname);
    phi1 = a1.data;


    value = 0; % 面积*y坐标
    value1 = 0; % 面积
    for ie = 1:Ncellx*Ncelly
        iex = mod(ie-1, Ncellx) + 1;
        iey = floor((ie-1)/Ncellx) + 1;
        startx = (iex-1)*Np + 1;
        starty = (iey-1)*Np + 1;
        for NK = 1:Np+1
            for NJ = 1:Np+1
                i_global = startx + NJ - 1;
                j_global = starty + NK - 1;
                NGJK = (j_global - 1) * x + i_global;
                y_coord = coordy(i_global, j_global);
                if phi1(NGJK) >= 0
                    value = value + y_coord * mass_diag(NGJK);
                    value1 = value1 + mass_diag(NGJK);
                end

            end
        end

    end
    a(i,1) = Iter * freOut;
    a(i,2) = value / value1;
    i = i +1;
    i
end
plot(a(:,1),a(:,2))
% save massLocation a
save massLocation.txt -ascii a