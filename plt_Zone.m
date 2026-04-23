function plt_Zone(filename,zone_title,IJK,time,Mat_Data)
%创建zone,point格式
f_id=fopen(filename,'a');
N=size(Mat_Data,1);

Dim=numel(IJK);
if Dim==1
    s=['zone I=',num2str( IJK(1) )];
elseif Dim==2
    s=['zone I=',num2str( IJK(1) ),',J=',num2str( IJK(2) )];
elseif Dim==3
    s=['zone I=',num2str( IJK(1) ),',J=',num2str( IJK(2) ),',K=',num2str( IJK(3) )];
end

%标题
if ~isempty(zone_title)
    s=[s,',t="',zone_title,'"'];
end
fprintf(f_id,'%s \r\n',s);
%格式为point
s='DATAPACKING=point';
fprintf(f_id,'%s \r\n',s);
%定义非定常时间
if ~isempty(time)
    s=['SOLUTIONTIME=',num2str(time)];
    fprintf(f_id,'%s \r\n',s);
end

%导入数据
for k=1:N
    fprintf(f_id,'%s \r\n',num2str(Mat_Data(k,:)));
end
fclose(f_id);
end
