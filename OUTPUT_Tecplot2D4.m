function OUTPUT_Tecplot2D4(iter, outDir, nx, ny, varList, xCoord, yCoord, varargin)
%OUTPUT_Tecplot2D4 Write 2D tensor-product SEM fields to Tecplot ASCII format.
%   OUTPUT_Tecplot2D4(iter, outDir, nx, ny, varList, xCoord, yCoord, fields...)
%   writes a Tecplot file named Iter=<iter>.dat inside outDir. nx and ny are
%   the node counts in the x- and y-directions of the tensor grid. Coordinates
%   and field arrays must contain nx*ny nodal values arranged in MATLAB's
%   column-major ordering (x-index varies fastest).
%
%   varList is a comma-separated list of Tecplot variable names matching the
%   supplied field arrays.

    narginchk(7, inf);

    npts = nx * ny;
    if numel(xCoord) ~= npts || numel(yCoord) ~= npts
        error('OUTPUT_Tecplot2D4:InvalidCoordLength', ...
              'Coordinate vectors must contain exactly nx*ny = %d entries.', npts);
    end

    xCoord = xCoord(:);
    yCoord = yCoord(:);

    nFields = numel(varargin);
    fieldData = zeros(npts, nFields);
    for k = 1:nFields
        fk = varargin{k};
        if numel(fk) ~= npts
            error('OUTPUT_Tecplot2D4:SizeMismatch', ...
                  'Field %d has %d entries (expected %d).', k, numel(fk), npts);
        end
        fieldData(:, k) = fk(:);
    end

    if isstring(varList)
        varList = strjoin(cellstr(varList), ',');
    end
    varCells = strtrim(strsplit(varList, ','));
    varCells = varCells(~cellfun('isempty', varCells));

    allVars = [{'X','Y'}, varCells];

    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    fname = fullfile(outDir, sprintf('Iter=%d.dat', iter));
    fid = fopen(fname, 'wt');
    if fid < 0
        error('OUTPUT_Tecplot2D4:FileOpen', 'Unable to open %s for writing.', fname);
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

    fprintf(fid, 'ZONE I=%d J=%d F=POINT\n', nx, ny);

    dataMatrix = [xCoord, yCoord, fieldData];
    fmtTokens = repmat({'%.15e'}, 1, size(dataMatrix, 2));
    fmt = strjoin(fmtTokens, ' ');
    fmt = [fmt, '\n'];
    fprintf(fid, fmt, dataMatrix.');
end
