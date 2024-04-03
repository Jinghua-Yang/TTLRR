function Ain = tt_inverse(A,beta1, beta2, beta3)

[~,n4,n3]=size(A);
sigma1 = beta1/beta3;
sigma2 = beta2/(beta2+beta3);
A = fft(A,[],3);
Ain  = zeros(n4,n4,n3);
for i = 1 : n3
   Ain (:,:,i) =  (sigma2*A(:,:,i)'*A(:,:,i) + sigma1*eye(n4))\eye(n4);
end
% inv_a = (Abar'*Abar + eye(m*d2))\eye(m*d2);
Ain  = ifft(Ain,[],3);

end
