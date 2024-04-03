function [X,tnn,trank] = prox_utnn2(Phi, Y,rho)

% The proximal operator of the tensor nuclear norm of a 3 way tensor
%
% min_X rho*||X||_*+0.5*||X-Y||_F^2
%
% Y     -    n1*n2*n3 tensor
%
% X     -    n1*n2*n3 tensor
% tnn   -    tensor nuclear norm of X
% trank -    tensor tubal rank of X
%

[n1,n2,n3] = size(Y);
X = zeros(n1,n2,n3);
% Y = fft(Y,[],3);
Y = ttrans(Y, Phi);
tnn = 0;
trank = 0;
% % first frontal slice

for i = 1 : n3
    [U,S,V] = svd(Y(:,:,i),'econ');
    S = diag(S);
    S = max(S-rho,0);
    r = length(find(S~=0));
    S = S(1:r);
    X(:,:,i) = U(:,1:r)*diag(S)*V(:,1:r)';    
    tnn = tnn+sum(S)*2;
    trank = max(trank,r);
end

X = ttrans(X, Phi');

tnn = tnn/n3;
% X = ifft(X,[],3);
