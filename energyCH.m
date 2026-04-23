function E = energyCH(phi, epsilon, Wx, Wy, Wz)
% E = ∫ [ (eps/2)|∇phi|^2 + (1/(4 eps)) (phi^2 - 1)^2 ] dx
phi = gather(phi);
nx = numel(Wx); ny = numel(Wy); nz = numel(Wz);

sx = makeCoordFromWeights(Wx);  sy = makeCoordFromWeights(Wy);  sz = makeCoordFromWeights(Wz);
[DXp, DXm] = diffOps1D(sx);
[DYP, DYM] = diffOps1D(sy);
[DZP, DZM] = diffOps1D(sz);

denx = max(DXp + DXm, eps);  deny = max(DYP + DYM, eps);  denz = max(DZP + DZM, eps);

phix = (phi([2:end,end],:,:) - phi([1,1:end-1],:,:)) ./ denx;
phiy = (phi(:,[2:end,end],:) - phi(:,[1,1:end-1],:)) ./ deny;
phiz = (phi(:,:,[2:end,end]) - phi(:,:,[1,1:end-1])) ./ denz;

grad2 = phix.^2 + phiy.^2 + phiz.^2;
pot   = (phi.^2 - 1).^2;

Wx3 = reshape(gather(Wx), [nx,1,1]);
Wy3 = reshape(gather(Wy), [1,ny,1]);
Wz3 = reshape(gather(Wz), [1,1,nz]);
W3  = Wx3 .* Wy3 .* Wz3;

E = sum( (0.5*epsilon)*grad2(:).*W3(:) + (1/(4*epsilon))*pot(:).*W3(:) );
if ~isfinite(E), E = realmax; end
end

function s = makeCoordFromWeights(W)
W = gather(W(:));
s = cumsum([0; 0.5*(W(1:end-1)+W(2:end))]);
s = s - mean(s);
end

function [Dp, Dm] = diffOps1D(s)
n = numel(s);
Dp = zeros(n,1); Dm = zeros(n,1);
Dp(1)   = max(s(2)-s(1), eps);        Dm(1)   = Dp(1);
Dp(end) = max(s(end)-s(end-1), eps);  Dm(end) = Dp(end);
for i=2:n-1
    Dp(i) = max(s(i+1)-s(i), eps);
    Dm(i) = max(s(i)-s(i-1), eps);
end
end
