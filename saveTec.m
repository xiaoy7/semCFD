function saveTec(Iter,pathname,IJK,varList,varargin)
nFields = numel(varargin);
for k = 1:nFields
    fk = varargin{k};
    fieldData = fk(:);


    fileName= varList{k};
    filename_sol = fullfile(pathname, [fileName,num2str(Iter,'%02.f'),'.plt']);
    plt_Head3(filename_sol,'',fileName,'SOLUTION')
    plt_Zone(filename_sol,'',IJK,Iter,gather(fieldData))

end



end