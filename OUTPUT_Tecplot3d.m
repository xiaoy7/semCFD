function OUTPUT_Tecplot3d(iter, outDir, coord, nx, ny, nz, varList, varargin)
%OUTPUT_Tecplot3d Write 3D SEM fields to Tecplot ASCII format.
%   OUTPUT_Tecplot3d(iter, outDir, coord, nx, ny, nz, varList, fields...)
%   writes Tecplot file Iter=<iter>.dat inside outDir. coord must be
%   [Npts x 3] array of physical coordinates (x,y,z). varList is a comma
%   separated list of variable names for the supplied field arrays. Each
%   field is vectorised (numel = Npts). nx, ny, nz describe the structured
%   ordering of the tensor grid (Tecplot 'POINT' format).

    narginchk(7, inf);

    if size(coord, 2) ~= 3
        error('OUTPUT_Tecplot3d:InvalidCoord', ...
              'coord must be an N-by-3 array of [x y z] coordinates.');
    end
    npts = size(coord, 1);

    nFields = numel(varargin);
    fieldData = zeros(npts, nFields);
    for k = 1:nFields
        fk = varargin{k};
        if numel(fk) ~= npts
            error('OUTPUT_Tecplot3d:SizeMismatch', ...
                  'Field %d has %d entries. Expected %d.', k, numel(fk), npts);
        end
        fieldData(:, k) = fk(:);
    end

    if isstring(varList)
        varList = strjoin(cellstr(varList), ',');
    end
    varCells = strtrim(strsplit(varList, ','));
    varCells = varCells(~cellfun('isempty', varCells));

    allVars = [{'X','Y','Z'}, varCells];

    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    fname = fullfile(outDir, sprintf('Iter=%d.dat', iter));
    fid = fopen(fname, 'wt');
    if fid < 0
        error('OUTPUT_Tecplot3d:FileOpen', 'Unable to open %s for writing.', fname);
    end

    cleaner = onCleanup(@() fclose(fid));

    fprintf(fid, 'VARIABLES=');
    for i = 1:numel(allVars)
        fprintf(fid, '"%s"', allVars{i});
        if i < numel(allVars)
            fprintf(fid, ',');
        else
            fprintf(fid, '\n');
        end
    end
    fprintf(fid, 'ZONE I=%d J=%d K=%d F=POINT\n', nx, ny, nz);

    dataMatrix = [coord, fieldData];
    ncols = size(dataMatrix, 2);
    fmtTokens = repmat({'%.15e'}, 1, ncols);
    fmt = strjoin(fmtTokens, ' ');
    fmt = [fmt, '\n'];
    fprintf(fid, fmt, dataMatrix.');
end
