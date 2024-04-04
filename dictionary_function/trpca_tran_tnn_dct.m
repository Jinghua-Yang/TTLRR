

% Solve the Tensor Robust Principal Component Analysis based on Tensor Nuclear Norm problem by ADMM
%
% min_{Z,E} ||Z||_*+lambda*||E||_1, s.t. X=Z+E



function [Z,E,U_rank,err] = trpca_tran_tnn_dct(X,Index,opts,Phi)
[n1,n2,n3]=size(X);

tol = 1e-8; 
max_iter = 100;
rho = 1.1;
% beta1 = beta2 = beta3 = beta = 1e-4;
beta1 = 1e0;
beta2 = 1e0;
beta3 = 1e0;
max_beta = 1e12;

DEBUG = 0;
lambda = opts.lambda;
rho = opts.rho;
% max_beta = opts.max_beta;
Phi  = dct(eye(n3));
PhiT = conj(Phi)';

%% Z,S,T n1 n2 n3
Z = zeros(n1,n2,n3);
S = Z;
T = Z;

%% E,G,M,H,N n1 n2 n3
E = zeros(n1,n2,n3);
G = E;
M = E;
M(Index) = X(Index);
H = E;
N = E;

% lambda=1/(sqrt(n3*max(n1,n2)));

% max_beta = 1e+8;
% tol = 1e-4;
% rho = 1.1;
iter = 0;
% max_iter = 500;

%%?Pre compute

while iter < max_iter
    iter = iter+1;
    
    %% update Z
    Z_pre = Z;
    temp_Z = S - T/beta1;
    [Z, ~, U_rank] = prox_tran_tnn(temp_Z, 1/beta1, Phi);
    
    %% update E
    E_pre = E;
    temp_E = G - H/beta2;
    E = prox_l1(temp_E, lambda/beta2);
     
    %% update M
    M_pre = M;
    M = S + G - N/beta3;
    M(Index) = X(Index);
    
     %% update S G
    S_pre = S;    
    G_pre = G;
    
    J1 = beta1*Z + T + beta3*M + N;
    J2 = beta2*E + H + beta3*M + N;
    temp_S = J1/beta3 - J2/(beta2+beta3);
    S = temp_S./(beta1/beta3+beta2/(beta2+beta3));
    G = J2/(beta2+beta3) - beta3*S/(beta2+beta3);
    
    %% update Lagrange multipliers T, H, N and  penalty parameter beta
    leq1 = Z - S;
    leq2 = E - G;
    leq3 = M - S - G;
    
    T = T + beta1*leq1;
    H = H + beta2*leq2;
    N = N + beta3*leq3;
    beta1 = min(beta1*rho,max_beta);
    beta2 = min(beta2*rho,max_beta);
    beta3 = min(beta3*rho,max_beta);
    
    %% check convergence
    leqm1 = max(abs(leq1(:)));
    leqm2 = max(abs(leq2(:)));
    leqm3 = max(abs(leq3(:)));
    difZ = max(abs(Z(:)-Z_pre(:)));
    difE = max(abs(E(:)-E_pre(:)));
    difM = max(abs(M(:)-M_pre(:)));
    difS = max(abs(S(:)-S_pre(:)));
    difG = max(abs(G(:)-G_pre(:)));    
    err(iter) = max([leqm1,leqm2,leqm3,difZ,difE,difM,difS,difG]);
%     if DEBUG && (iter==1 || mod(iter,20)==0)
%         sparsity=length(find(E~=0));
%         fprintf('iter = %d, obj = %.3f, err = %.8f, beta=%.2f, rankL = %d, sparsity=%d\n'...
%             , iter,Z_nuc+lambda*norm(E(:),1),err,beta,Z_rank,sparsity);
%     end
    if err(iter) < tol
        break;
    end
    
end
end