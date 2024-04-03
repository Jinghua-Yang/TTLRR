function [LL, V] = dictionary_learning_dct(X,Index,opts,Phi)
if opts.denoising_flag==0
%% directly using raw data as dictionary
    tho=100;% sigma(i)<=tho*sigma(1)
%     X = fillmissing(X,'movmean',5);
    [ ~,~,U,V,S ] = prox_tran_low_rank_dct(X,tho,Phi);
    LL=transform_tprod(U,S,Phi);
else
%% use R-TPCA to denoise data first and then use the recovered data as dictionary
    %% raw R-TPCA algorithm
    [L,~,~,~] = trpca_tran_tnn_dct(X,Index,opts,Phi);
    %% approximate L, since sometimes R-TPCA cannot produce a good dictionary
    [ U,V,S,~] = tran_tSVDs( L,Phi );
    LL=transform_tprod(U,S,Phi);
end
