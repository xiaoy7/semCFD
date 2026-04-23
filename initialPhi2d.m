function phi1 = initialPhi2d(coordX,coordY,Cx,Cy,radius,eta,LL1)

switch 2
    %%  type 1
    case 1
        phi1  = -tanh((sqrt((coordX-Cx).^2 + (coordY-Cy).^2)-radius)./(sqrt(2)*eta));
        
        
        %% type 2
    case 2
        eta2 = 0.1 * LL1 * cos(2 * pi * coordX / LL1);
        phi1 = tanh((coordY + eta2) / (sqrt(2)*eta));
        
     
        %% type 3
    case 3
        phi1 = tanh(coordY  / (sqrt(2)*eta));
    case 4
        ND = size(coord,1);
        phi1 = zeros(ND,1);
        for ii = 1:ND
            if coord(ii,1) >= 0
                phi1(ii)  = -tanh((sqrt((coord(ii,1) - Cx)^2 + (coord(ii,2)-Cy)^2)-radius)/(sqrt(2)*eta));
            else
                phi1(ii)  = -tanh((sqrt((coord(ii,1) + Cx)^2 + (coord(ii,2)-Cy)^2)-radius)/(sqrt(2)*eta));
            end
        end
        
end