
function plt_Head(filename,title,variables,filetype)
%创建文件头
if nargin < 4
    filetype = '';
end
f_id=fopen(filename,'a');
%名称
if ~isempty(title)
    s=['TITLE = "',title,'"'];
    fprintf(f_id,'%s \r\n',s);
end

%文件类型
if ~isempty(filetype)
    s=['FILETYPE = ',filetype];
    fprintf(f_id,'%s \r\n',s);
end

%变量
v=numel(variables);
s='VARIABLES = ';
for k=1:v
    if k~=1
        s=[s,','];
    end
    s=[s,' "',variables{k},'"'];
end
fprintf(f_id,'%s \r\n',s);
fclose(f_id);

end

