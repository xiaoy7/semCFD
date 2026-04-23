function Parameter = parameter_bc2d(Parameter)
switch Parameter.bc
    case 'dirichlet'
        Parameter.nx = Parameter.Ncellx * Parameter.Npx - 1;
        Parameter.ny = Parameter.Ncelly * Parameter.Npy - 1;
        bcNodesx = [1, Parameter.nx_all];
        bcNodesy = [1, Parameter.ny_all];


        freeNodesx = 1:Parameter.nx_all;
        Parameter.freeNodesx = freeNodesx(~ismember(freeNodesx, bcNodesx));
        Parameter.bcNodesx = bcNodesx;

        freeNodesy = 1:Parameter.ny_all;
        Parameter.freeNodesy = freeNodesy(~ismember(freeNodesy, bcNodesy));
        Parameter.bcNodesy = bcNodesy;



        % total number of unknowns in one direction
    case 'periodic'
        Parameter.nx = Parameter.Ncellx * Npx;
        Parameter.ny = Parameter.Ncelly * Npy;

        % total number of unknowns in one direction
    case 'neumann'
        Parameter.nx = Parameter.Ncellx * Parameter.Npx + 1;
        Parameter.ny = Parameter.Ncelly * Parameter.Npy + 1;


        Parameter.freeNodesx = 1:Parameter.nx_all;
        Parameter.freeNodesy = 1:Parameter.ny_all;

        % total number of unknowns in one direction
end