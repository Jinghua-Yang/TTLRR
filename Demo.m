clc
clear all
close all
addpath(genpath(cd));

EN_TTNN_DCT  = 1;
EN_TTNN_Data = 1;
methodname   = {'TTNN_DCT', 'TTNN_Data'};

load('news')
maxC = max(X(:));
X0 = X./maxC;
nway = size(X0);
name = {'news'};
[n1, n2, n3]  = size(X0);

for NR = [0.1]   
for sr = [6]
%% generat the sparse noise
STS = sptenrand([n1 n2 n3], NR);
STS = tensor(STS);
Y = X0 + STS.data;
%% generat the missing data
temp = randperm(n1*n2*n3);
kks = round(0.1*(sr)*n1*n2*n3);
mark = zeros(n1,n2,n3); 
Index = temp(1:kks);
mark(Index) = 1;

imname=[num2str(name{1}),'_miss_SR_',num2str(0.1*sr),'.mat'];
save(imname, 'mark', 'Y');
%% Preprocessing
load('news_SR_6_NR_0.1_result_UTTNN_psnr_35.16_ssim_0.9752.mat')
Xini = X;   
%% TTNN_DCT
j =  1;
if EN_TTNN_DCT
    %%%%
    fprintf('\n');
    disp(['performing ',methodname{j}, ' ... ']);
    %% construct the dictionary  
    opts.denoising_flag=0; % set the flag whether we use R-TPCA to construct the dictionary 
    % (1 denotes we use R-TPCA; 0 deonotes we do not use)
   if  opts.denoising_flag% if we use R-TPCA to construct the dictionary, we set its parameters
       [n1,n2,n3]=size(X0);
       opts.lambda = 1/(sqrt(SR*n3*max(n1,n2)));
       opts.mu = 1e-4;
       opts.tol = 1e-8;
       opts.rho = 1.2;
       opts.max_iter = 100;
       opts.DEBUG = 0; %% whether we debug the algorithm
   end    
   Phi  = dct(eye(n3));
   % run the dictionary construction algorithm
   [L0, LL, V] = dictionary_learning_data(Xini, Index, opts, Phi);
   
   %%  test 
   max_iter=500;
   DEBUG = 0; 
   opts.max_beta = 1e8;
   opts.rho = 1.2;
for  lam = [70]
   beta1 = 0.0001;
   beta2 = 0.01;
   beta3 = 0.01;
    lambda = lam/(sqrt(0.1*sr*n3*max(n1, n2)));

    tic;    
    [Z, tlrr_E, Z_rank, err_va] = Tensor_TransLRR(Phi, Y, LL, mark, max_iter,DEBUG,lambda,beta1,beta2,beta3,opts);
    Time = toc;
    X = ttprod(LL, Z, Phi);
    for i=1:1:n3
        PSNRvector(i) = psnr3(X0(:,:,i), X(:,:,i));
    end
    psnr = mean(PSNRvector);                                               
    for i=1:1:n3
        SSIMvector(i) = ssim3(X0(:,:,i)*255, X(:,:,i)*255);
    end
    ssim = mean(SSIMvector);
                                       
    display(sprintf('psnr=%.2f,ssim=%.4f,lambda=%.4f,beta1=%.4f,beta2=%.4f,beta3=%.4f',psnr, ssim, lambda, beta1, beta2,beta3))
    display(sprintf('=================================='))
     
end
end
%% TTNN_Data
j = j + 1;
if EN_TTNN_Data
    %%%%
    fprintf('\n');
    disp(['performing ',methodname{j}, ' ... ']);

    %% construct the dictionary  
    opts.denoising_flag=0; % set the flag whether we use R-TPCA to construct the dictionary 
    % (1 denotes we use R-TPCA; 0 deonotes we do not use)
   if  opts.denoising_flag% if we use R-TPCA to construct the dictionary, we set its parameters
       [n1,n2,n3]=size(X0);
       opts.lambda = 1/(sqrt(SR*n3*max(n1,n2)));
       opts.mu = 1e-4;
       opts.tol = 1e-8;
       opts.rho = 1.2;
       opts.max_iter = 100;
       opts.DEBUG = 0; %% whether we debug the algorithm
   end    

   %% U matrix
   O = tenmat(Xini, 3); % unfolding
   O = O.data;
   [U0 D0 V0] = svd(O, 'econ');
   Phi  = U0';
   % run the dictionary construction algorithm
   [L0, LL, V] = dictionary_learning_data(Xini, Index, opts, Phi);
   
   %%  test 
   max_iter=500;
   DEBUG = 0;
   opts.max_beta = 1e8;
   opts.rho = 1.2;
  for lam = [200]
   beta1 = 0.0001;
   beta2 = 0.01;
   beta3 = 0.01;
    lambda = lam/(sqrt(0.1*sr*n3*max(n1,n2)));

    tic;    
    [Z, tlrr_E, Z_rank, err_va] = Tensor_TransLRR(Phi, Y, LL, mark, max_iter,DEBUG,lambda,beta1,beta2,beta3,opts);
    Time = toc;
    X = ttprod(LL, Z, Phi);
   
    for i=1:1:n3
        PSNRvector(i) = psnr3(X0(:,:,i), X(:,:,i));
    end
    psnr = mean(PSNRvector);                                               
    for i=1:1:n3
        SSIMvector(i) = ssim3(X0(:,:,i)*255, X(:,:,i)*255);
    end
    ssim = mean(SSIMvector);
                                       
    display(sprintf('psnr=%.2f,ssim=%.4f,lambda=%.4f,beta1=%.4f,beta2=%.4f,beta3=%.4f',psnr, ssim, lambda, beta1, beta2,beta3))
    display(sprintf('=================================='))
       
  end
end
end
end

 
 
