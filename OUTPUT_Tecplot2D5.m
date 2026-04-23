function OUTPUT_Tecplot2D5(Iter,pathname, NGX,NGY,varName1,varargin)%U2,V2,PRE,phi2,rho
result = varargin{1};
result = [result varargin{2}];

varName2 = 'VARIABLES=X,Y,';
varName = [varName2,varName1];
formatSpec ='%2.15e %2.15e\n';
a = '%2.15e ';

for i = 3:size(varargin,2)
%     var1 = exchangeCood(varargin{i},NUME,ND,NX,NY,NEY,NEX);
var1 = varargin{i};
    result = [result var1];
    formatSpec = [a,formatSpec];
end
if issparse(result)
    result = full(result);
end
fname = ['Iter=',num2str(Iter,'%i'),'.dat'];
outpath = fullfile(pathname, fname);
fid = fopen(outpath,'wt');
fprintf(fid,varName);
fprintf(fid,'ZONE I=%g J=%g F=POINT\n',NGX,NGY);
fprintf(fid,formatSpec,result');
fclose(fid);
