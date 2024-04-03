function Ain = transform_inverse(A, Phi, beta1, beta2, beta3)

[~,n4,n3]=size(A);
sigma1 = beta1/beta3;
sigma2 = beta2/(beta2+beta3);
AT = transform(A, Phi);
Ain = zeros(n4,n4,n3);
for i = 1 : n3
   Ain(:,:,i) =  (sigma1*eye(n4) + sigma2*conj(AT(:,:,i))'*AT(:,:,i))\eye(n4);
end

PhiH = conj(Phi)';
Ain  = transform(Ain, PhiH);

end