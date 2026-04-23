function Parameter = parameter_bc(Parameter)
switch Parameter.bc
    case 'dirichlet'
        Parameter.nx = Parameter.Ncellx * Parameter.Npx - 1;
        Parameter.ny = Parameter.Ncelly * Parameter.Npy - 1;
        Parameter.nz = Parameter.Ncellz * Parameter.Npz - 1;
        bcNodesx = [1, Parameter.nx_all];
        bcNodesy = [1, Parameter.ny_all];
        bcNodesz = [1, Parameter.nz_all];


        freeNodesx = 1:Parameter.nx_all;
        Parameter.freeNodesx = freeNodesx(~ismember(freeNodesx, bcNodesx));
        Parameter.bcNodesx = bcNodesx;

        freeNodesy = 1:Parameter.ny_all;
        Parameter.freeNodesy = freeNodesy(~ismember(freeNodesy, bcNodesy));
        Parameter.bcNodesy = bcNodesy;

        freeNodesz = 1:Parameter.nz_all;
        Parameter.freeNodesz = freeNodesz(~ismember(freeNodesz, bcNodesz));
        Parameter.bcNodesz = bcNodesz;

        % total number of unknowns in one direction
    case 'periodic'
        Parameter.nx = Parameter.Ncellx * Npx;
        Parameter.ny = Parameter.Ncelly * Npy;
         Parameter.nz = Parameter.Ncellz * Npz;
        % total number of unknowns in one direction
    case 'neumann'
        Parameter.nx = Parameter.Ncellx * Parameter.Npx + 1;
        Parameter.ny = Parameter.Ncelly * Parameter.Npy + 1;
        Parameter.nz = Parameter.Ncellz * Parameter.Npz + 1;

        Parameter.freeNodesx = 1:Parameter.nx_all;
        Parameter.freeNodesy = 1:Parameter.ny_all;
        Parameter.freeNodesz = 1:Parameter.nz_all;
        % total number of unknowns in one direction
end