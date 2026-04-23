% intergral of velocity
clc
clear

% 读取所有nii后缀的文件
file = dir('phi*');
% 获取文件个数
len = length(file);
fprintf('total files = %d\n',len);

freOut = 1000;

i = 1;
IterT = 0;

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


coordy = reshape(coordxy(:,2),x,y);


for Iter = 1:len

    IterT = IterT + freOut;

    fname = ['phi',num2str(IterT,'%d'),'.plt'];
    [a1,a2] = importdata(fname);
    phi = a1.data;



    %% start
    phii = reshape(phi,x,y);


    phi33 = phii(1,:);
    phi34 = phii(end,:);
    for ii = 1:y-1
        if phi33(ii) * phi33(ii+1) < 0
            aa(Iter,2) = coordy(1,ii);
            break
        end
    end
    for ii = 1:y-1
        if phi34(ii) * phi34(ii+1) < 0
            aa(Iter,3) = coordy(1,ii);
            break
        end
    end

    aa(Iter,1) = IterT;

Iter
end
%
%
aa(:,1) = aa(:,1) * sqrt(0.5) / 100;
aa(:,2) = aa(:,2) / 100;
aa(:,3) = aa(:,3) / 100;

save Lmaxmin.txt -ascii aa
text(0.4,0.7,'Finished','Color','red','FontSize',26)