function [LL, V] = dictionary_learning_fft(X,Index,opts)
if opts.denoising_flag==0
%% directly using raw data as dictionary
    tho=300;% sigma(i)<=tho*sigma(1)
    [ ~,~,U,V,S ] = prox_low_rank(X,tho);
    LL=tprod(U,S);
else
%% use R-TPCA to denoise data first and then use the recovered data as dictionary
    %% raw R-TPCA algorithm
    [L,~,~,~] = trpca_tran_tnn_fft(X,Index,opts,Phi);
    %% approximate L, since sometimes R-TPCA cannot produce a good dictionary
    [ U,V,S,~] = tran_tSVDs( L,Phi );
    LL=transform_tprod(U,S,Phi);
end
