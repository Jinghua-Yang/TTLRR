function [ Z,E,U_rank,err ] = Tensor_TransLRR(Phi,X,A,Omega,max_iter,DEBUG,lambda,beta1,beta2,beta3,opts)
[n1,n2,n3] = size(X);
[~, n4, ~] = size(A);
BarOmega = ones(n1,n2,n3) - Omega;

rho = opts.rho;
max_beta = opts.max_beta;

%% Z,S,T n4 n2 n3
Z = zeros(n4,n2,n3);
S = Z;
T = Z;

%% E,G,M,H,N n1 n2 n3
E = zeros(n1,n2,n3);
G = E;
M = X.*Omega;
H = E;
N = E;

tol = 1e-4;
iter = 0;
%%?Pre compute
% Ain = transform_inverse(A, Phi, beta1, beta2, beta3);
AT = ttrans(A, Phi);
% AH = conj_tran(A, Phi);
while iter < max_iter
    iter = iter+1;
    
    %% update Z
    Z_pre = Z;
    temp_Z = S - T/beta1;
    [Z, U_rank] = prox_utnn2(Phi, temp_Z, 1/beta1);
    
    %% update E
    E_pre = E;
    temp_E = G - H/beta2;
    E = prox_l1(temp_E, lambda/beta2);
     
    %% update M
    M_pre = M;
    M = ttprod(A, S, Phi) + G - N/beta3;
%     M(Index) = Data;
    M = X.*Omega + M.*BarOmega;
    
     %% update S G
    S_pre = S;    
    G_pre = G;
    
    ZT = ttrans(Z, Phi);
    TT = ttrans(T, Phi);
    ET = ttrans(E, Phi);
    HT = ttrans(H, Phi);
    MT = ttrans(M, Phi);
    NT = ttrans(N, Phi);
    for i = 1:size(ZT, 3)
        Ai = AT(:,:,i);
        AiA = Ai'*Ai;
        Zi = ZT(:,:,i);
        Ti = TT(:,:,i);
        Ei = ET(:,:,i);
        Hi = HT(:,:,i);
        Mi = MT(:,:,i);
        Ni = NT(:,:,i);
        J1 = beta1*Zi + Ti + beta3*Ai'*Mi + Ai'*Ni;
        J2 = beta2*Ei + Hi + beta3*Mi + Ni;
        ST(:,:,i) = ((beta1/beta3)*eye(size(AiA)) + (beta2/(beta2+beta3))*AiA)\(J1/beta3 - Ai'*J2/(beta2+beta3));
        GT(:,:,i) = J2/(beta2+beta3) - (beta3/(beta2+beta3))*Ai*ST(:,:,i);       
    end
    S = ttrans(ST, Phi');
    G = ttrans(GT, Phi');

    %% update Lagrange multipliers T, H, N and  penalty parameter beta
    leq1 = Z - S;
    leq2 = E - G;
    leq3 = M - ttprod(A, S, Phi) - G;
    
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
    if err(iter) < tol
        break;
    end
    
end
end
