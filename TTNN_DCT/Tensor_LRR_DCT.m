function [ Z,E,U_rank,err ] = Tensor_LRR_DCT(X,A,Index,Data,max_iter,DEBUG,lambda,beta1,beta2,beta3,opts)
[n1,n2,n3]=size(X);
[~, n4, ~]=size(A);

rho = opts.rho;
max_beta = opts.max_beta;
Phi  = dct(eye(n3));
PhiT = conj(Phi)';

%% Z,S,T n4 n2 n3
Z = zeros(n4,n2,n3);
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
tol = 1e-4;
% rho = 1.1;
iter = 0;
% max_iter = 500;

%%?Pre compute
Ain = transform_inverse(A, Phi, beta1, beta2, beta3);
AH = conj_tran(A, Phi);
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
    M = transform_tprod(A, S, Phi) + G - N/beta3;
    M(Index) = Data;
    
     %% update S G
    S_pre = S;    
    G_pre = G;
    
    J1 = beta1*Z + T + transform_tprod(AH, beta3*M + N, Phi);
    J2 = beta2*E + H + beta3*M + N;
    temp_S = J1/beta3 - transform_tprod(AH, J2, Phi)/(beta2+beta3);
    S = transform_tprod(conj_tran(Ain, Phi), temp_S, Phi);
    G = J2/(beta2+beta3) - beta3*transform_tprod(A, S, Phi)/(beta2+beta3);
    
    %% update Lagrange multipliers T, H, N and  penalty parameter beta
    leq1 = Z - S;
    leq2 = E - G;
    leq3 = M - transform_tprod(A, S, Phi) - G;
    
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
