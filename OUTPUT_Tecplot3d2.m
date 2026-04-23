function OUTPUT_Tecplot3d2(iter, outDir, coord, nx, ny, nz, varList, varargin)
%OUTPUT_Tecplot3d2 Split Tecplot export: each variable gets its own file including mesh.
%   OUTPUT_Tecplot3d2(iter, outDir, coord, nx, ny, nz, varList, fields...)
%   writes one Tecplot ASCII file per variable: Iter=<iter>_<var>.dat.
%   Each file contains X,Y,Z plus that single variable, so every file is
%   self-contained (no separate mesh.dat needed). coord must be [Npts x 3]
%   (x,y,z). varList is a comma separated list of variable names for the
%   supplied field arrays. Each field is vectorised (numel = Npts). nx, ny,
%   nz describe the structured ordering of the tensor grid (Tecplot POINT).

    narginchk(7, inf);

    if size(coord, 2) ~= 3
        error('OUTPUT_Tecplot3d:InvalidCoord', ...
              'coord must be an N-by-3 array of [x y z] coordinates.');
    end
    npts = size(coord, 1);

    nFields = numel(varargin);

    if isstring(varList)
        varList = strjoin(cellstr(varList), ',');
    end
    varCells = strtrim(strsplit(varList, ','));
    varCells = varCells(~cellfun('isempty', varCells));
    if numel(varCells) ~= nFields
        error('OUTPUT_Tecplot3d:VarCountMismatch', ...
              'Number of variable names (%d) must match number of fields (%d).', ...
              numel(varCells), nFields);
    end

    % Collect field data after validating sizes
    fieldData = cell(1, nFields);
    for k = 1:nFields
        fk = varargin{k};
        if numel(fk) ~= npts
            error('OUTPUT_Tecplot3d:SizeMismatch', ...
                  'Field %d has %d entries. Expected %d.', k, numel(fk), npts);
        end
        fieldData{k} = fk(:);
    end

    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    % One file per variable, each with mesh + that variable
    for k = 1:nFields
        vnameTec = strtrim(varCells{k}); % name inside Tecplot
        vnameFile = regexprep(vnameTec, '[^A-Za-z0-9._-]', '_'); % filesystem-safe
        if isempty(vnameFile)
            vnameFile = sprintf('var%d', k);
        end

        solFile = fullfile(outDir, sprintf('%s=%d.dat', vnameFile, iter));
        fidSol = fopen(solFile, 'wt');
        if fidSol < 0
            error('OUTPUT_Tecplot3d:FileOpenSol', 'Unable to open %s for writing.', solFile);
        end
        cleanerSol = onCleanup(@() fclose(fidSol)); 

        fprintf(fidSol, 'VARIABLES="X","Y","Z","%s"\n', vnameTec);
        fprintf(fidSol, 'ZONE I=%d J=%d K=%d F=POINT\n', nx, ny, nz);

        dataMatrix = [coord, fieldData{k}];
        ncols = size(dataMatrix, 2);
        fmtTokens = repmat({'%.15e'}, 1, ncols);
        fmt = strjoin(fmtTokens, ' ');
        fmt = [fmt, '\n'];
        fprintf(fidSol, fmt, dataMatrix.');
    end
end
