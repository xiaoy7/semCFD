% plot

clear
clc

[X,Y] = meshgrid(-5:.5:5); %生成长和宽都是[-5,5]的网格坐标
Z = Y.*sin(X) - X.*cos(Y);
 
figure(1);
set(gcf, 'unit', 'centimeters', 'position', [10 5 28 20]);
 
subplot(221)
mesh(X,Y,Z)
title('例1 mesh结果')
 
subplot(222)
surf(X,Y,Z)
title('例1 surf结果')
 
subplot(223)
surf(X,Y,Z)
view(0,90)
title('例1 surf俯视看x-y平面结果')
 
subplot(224)
pcolor(X,Y,Z)
title('例1 pcolor结果')