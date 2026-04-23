function saveTecPhi(Iter,pathname,IJK,fileName,fk)
fieldData = fk(:);
filename_sol = fullfile(pathname, [fileName,num2str(Iter,'%02.f'),'.plt']);
plt_Head3(filename_sol,'',fileName,'SOLUTION')
plt_Zone(filename_sol,'',IJK,Iter,gather(fieldData))

end