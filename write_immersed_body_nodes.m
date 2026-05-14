function write_immersed_body_nodes(runDir, coordX, coordY, mask)
rigidNodeIds = find(mask >= 0.5);
rigidNodes = [coordX(rigidNodeIds), coordY(rigidNodeIds)];
rigidNodesFile = fullfile(runDir, 'immersed_body_nodes.dat');
fid = fopen(rigidNodesFile, 'w');
if fid == -1
    warning('Could not open immersed body node export file: %s', rigidNodesFile);
    return
end
cleaner = onCleanup(@() fclose(fid));
fprintf(fid, 'VARIABLES = "X","Y"\n');
fprintf(fid, 'ZONE T="immersed_body_nodes", I=%d, F=POINT\n', size(rigidNodes, 1));
fprintf(fid, '%20.12e %20.12e\n', rigidNodes');
end

