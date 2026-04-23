function Parameter = calBoundary(Parameter,mesh)

switch Parameter.bc
    case 'dirichlet'
        bcNodesx = [1, mesh.nx_all];
        bcNodesy = [1, mesh.ny_all];

        % total number of unknowns in one direction
    case 'periodic'
        % nx = mesh.Ncellx * mesh.Npx;
        % ny = mesh.Ncelly * mesh.Npy;
        % total number of unknowns in one direction
    case 'neumann'
        % nx = mesh.Ncellx * mesh.Npx + 1;
        % ny = mesh.Ncelly * mesh.Npy + 1;
        bcNodesx = [];
        bcNodesy = [];
        % total number of unknowns in one direction
end


Parameter.freeNodesx = mesh.Nodesx(~ismember(mesh.Nodesx, bcNodesx));
Parameter.bcNodesx = bcNodesx;


Parameter.freeNodesy = mesh.Nodesy(~ismember(mesh.Nodesy, bcNodesy));
Parameter.bcNodesy = bcNodesy;
