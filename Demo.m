clc
clear all
close all
%rand('seed',213412);
addpath(genpath(cd));
% addpath('tensor_toolbox_2.6');

EN_FTTNN     = 1;
EN_TTNN_DCT  = 1;
EN_TTNN_Data = 1;
methodname   = {'FTTNN','TTNN_DCT', 'TTNN_Data'};

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

%% FTTNN
j = 1;
if EN_FTTNN
    %%%%%
    fprintf('\n');
    disp(['performing ',methodname{j}, ' ... ']);
    
    opts = [];   
%     opts.beta = 0.05;
    opts.gamma = 1.618;
    opts.MaxIte = 100;
    opts.tol = 5e-4;
%     opts.mu = 1/sqrt(max(n1,n2)*n3);
    dim = [n1, n2, n3];
    %%  FFT
for mu = [2]
for beta = [0.05]
    mu = mu/sqrt(max(n1,n2)*n3);
    opts.beta = beta;
    opts.mu = mu;
    
    tic;
    [X] = TNN(Y, mark, dim, opts); 
    Time = toc;   
    X = min(1, max(X, 0));
        
    for i=1:1:n3
        PSNRvector(i) = psnr3(X0(:,:,i), X(:,:,i));
    end
    psnr = mean(PSNRvector);
                                                     
    for i=1:1:n3
        SSIMvector(i) = ssim3(X0(:,:,i)*255, X(:,:,i)*255);
    end
    ssim = mean(SSIMvector);
    
    display(sprintf('psnr=%.2f,ssim=%.4f,mu=%.4f,beta=%.2f', psnr, ssim, mu, beta))
    display(sprintf('=================================='))
        
    imname=[num2str(name{1}),'_SR_',num2str(sr),'_result_',num2str(methodname{j}),...
        '_psnr_',num2str(psnr,'%.2f'),'_ssim_',num2str(ssim,'%.4f'),'_mu_',num2str(mu),...
        '_beta_',num2str(beta),'_Time_',num2str(Time,'%.2f'),'.mat'];
    save(imname,'X'); 
end
end
   Xini = X;   
end


%% TTNN_DCT
j = j + 1;
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
   DEBUG = 0; %% do not output the convergence behaviors at each iteration
   opts.max_beta = 1e8;
   opts.rho = 1.2;
%    lambda = 1/(sqrt(SR*n3*max(n1,n2)));
for lam = [50 55 60 65 70 75 80 100]
for beta1 = [0.0001]
for beta2 = [0.01]
for beta3 = [0.01]
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
       
    imname=[num2str(name{1}),'_SR_',num2str(sr),'_result_',num2str(methodname{j}),...
        '_psnr_',num2str(psnr,'%.2f'),'_ssim_',num2str(ssim,'%.4f'),'_lambda_',num2str(lambda),...
        '_beta1_',num2str(beta1),'_beta2_',num2str(beta2),'_beta3_',num2str(beta3),'_Time_',num2str(Time,'%.2f'),'.mat'];
    save(imname,'X', 'Time'); 
    
end
end
end
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
   DEBUG = 0; %% do not output the convergence behaviors at each iteration
   opts.max_beta = 1e8;
   opts.rho = 1.2;
%    lambda = 1/(sqrt(SR*n3*max(n1,n2)));
for lam = [50 55 60 65 70 75 80 100]
for beta1 = [0.0001]
for beta2 = [0.01]
for beta3 = [0.01]
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
       
    imname=[num2str(name{1}),'_SR_',num2str(sr),'_result_',num2str(methodname{j}),...
        '_psnr_',num2str(psnr,'%.2f'),'_ssim_',num2str(ssim,'%.4f'),'_lambda_',num2str(lambda),...
        '_beta1_',num2str(beta1),'_beta2_',num2str(beta2),'_beta3_',num2str(beta3),'_Time_',num2str(Time,'%.2f'),'.mat'];
    save(imname,'X', 'Time');     
end
end
end
end

end

end
end

 