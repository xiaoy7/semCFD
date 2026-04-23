function [phin,Wn] = intialPhi(coordX,coordY,coordZ,Cx,Cy,Cz,radius,eta,hz,uw,LL1)


switch 4

    case 1 % single bubble
        phin = tanh((sqrt((coordX - Cx) .^ 2 + (coordY - Cy) .^ 2 + (coordZ - Cz) .^ 2) ...
            - radius) ./ (sqrt(2) * eta));
        Wn = 0*phin;

    case 2 % single bubble with a film
        phi1 = 1-tanh((sqrt((coordX - Cx) .^ 2 + (coordY - Cy) .^ 2 + (coordZ - Cz) .^ 2) ...
            - radius) ./ (sqrt(2) * eta));

        Wn = -uw * phi1;

        phi2 = 1-tanh((coordZ - hz) ./ (sqrt(2) * eta));


        phin = phi1 + phi2 -1;
    case 3 % adding a random disturbance with Gaussian distribution to the initial
        % velocities of film and drop in each control volume
        phi1 = 1-tanh((sqrt((coordX - Cx) .^ 2 + (coordY - Cy) .^ 2 + (coordZ - Cz) .^ 2) ...
            - radius) ./ (sqrt(2) * eta));

        Wn = -uw * phi1 + 0.1 * uw * phi1 .* randn(size(coordX));

        phi2 = 1-tanh((coordZ - hz) ./ (sqrt(2) * eta));

        phin = phi1 + phi2 -1;

    case 4 % rt
        % eta2 = 0.1 * LL1 * cos(2 * pi * coordX / LL1);
        % phi1 = tanh((coordY + eta2) / (sqrt(2)*eta));
        eta2 = 0.05 * LL1 * (cos(2 * pi * coordX / LL1) + cos(2 * pi * coordY / LL1));
        phin = tanh((coordZ + eta2) / (sqrt(2)*eta));
        Wn = 0*phin;

end
