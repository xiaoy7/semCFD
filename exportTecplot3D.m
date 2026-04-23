function exportTecplot3D(filename, x, y, z, phi, varargin)
% exportTecplot3D Write a 3D scalar field to Tecplot ASCII (.dat) format.
%
%   exportTecplot3D(filename, x, y, z, phi)
%   exportTecplot3D(..., 'Title', 'My Title', 'VarName', 'PHI',
%                         'Time', t, 'Append', true)
%
% Inputs
%   filename : output path, e.g. 'ch3d.dat'
%   x,y,z    : 1D coordinate vectors (size nx, ny, nz). These are the
%              same vectors you pass to ndgrid(x,y,z). (Monotone required.)
%   phi      : 3D array of size (nx, ny, nz), nodal values of the field.
%
% Name-Value options (all optional)
%   'Title'   : (char)  Tecplot TITLE string. Default: 'Cahn-Hilliard 3D'.
%   'VarName' : (char)  Variable name for phi.   Default: 'PHI'.
%   'Time'    : (double)Zone time for Tecplot.   Default: [] (omitted).
%   'Append'  : (logical)Append zone to existing file. Default: false.
%
% Notes
% - Writes an ORDERED ZONE with DATAPACKING=POINT:
%       VARIABLES="X","Y","Z","PHI"
%       ZONE T="t=...", I=nx, J=ny, K=nz, DATAPACKING=POINT
%       x y z phi  (nx*ny*nz lines; x fastest)
% - Works with CPU and GPU data; gathers automatically.
% - If Append=true and file doesn't exist, it creates it.

% ---------- parse ----------
p = inputParser;
p.addParameter('Title',  'Cahn-Hilliard 3D', @(s)ischar(s)||isstring(s));
p.addParameter('VarName','PHI',              @(s)ischar(s)||isstring(s));
p.addParameter('Time',   [],                 @(t)isnumeric(t)&&isscalar(t));
p.addParameter('Append', false,              @(b)islogical(b)&&isscalar(b));
p.parse(varargin{:});
ttl     = string(p.Results.Title);
vname   = string(p.Results.VarName);
tval    = p.Results.Time;
doAppend= p.Results.Append;

% ---------- gather & shape checks ----------
x   = gather(x(:));  y = gather(y(:));  z = gather(z(:));
phi = gather(phi);

nx = numel(x); ny = numel(y); nz = numel(z);
if ~isequal(size(phi), [nx, ny, nz])
    error('phi must be size [numel(x), numel(y), numel(z)]==[%d %d %d].', nx,ny,nz);
end

% ---------- open file ----------
mode = 'w';
if doAppend
    mode = 'a';
end
[fid, msg] = fopen(filename, mode);
if fid==-1
    error('Cannot open %s: %s', filename, msg);
end

cleanupObj = onCleanup(@() fclose(fid));

% ---------- header (write TITLE/VARIABLES only when not appending or file empty) ----------
writeHeader = true;
if doAppend && isfile(filename) && (ftell(fid) > 0 || fileHasContent(filename))
    writeHeader = false;
end

if writeHeader
    fprintf(fid, 'TITLE = "%s"\n', ttl);
    fprintf(fid, 'VARIABLES = "X","Y","Z","%s"\n', vname);
end

% ---------- zone header ----------
if isempty(tval)
    fprintf(fid, 'ZONE I=%d, J=%d, K=%d, DATAPACKING=POINT\n', nx, ny, nz);
else
    fprintf(fid, 'ZONE T="t=%.9g", I=%d, J=%d, K=%d, DATAPACKING=POINT, SOLUTIONTIME=%.15g\n', ...
        tval, nx, ny, nz, tval);
end

% ---------- data (x fastest, then y, then z) ----------
% We stream write in slices to keep memory modest.
for k = 1:nz
    zk = z(k);
    for j = 1:ny
        yj = y(j);
        % vector for this j,k plane:
        % X varies fastest. Phi(:,j,k) is nx-by-1
        row = [x, repmat(yj,nx,1), repmat(zk,nx,1), phi(:,j,k)];
        % Write with fprintf in one go
        fprintf(fid, '%.16g %.16g %.16g %.16g\n', row.' );
    end
end
end

function tf = fileHasContent(fname)
d = dir(fname);
tf = ~isempty(d) && d.bytes > 0;
end
