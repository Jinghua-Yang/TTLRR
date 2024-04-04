function AH = conj_tran(A, Phi)

[n1, n2, n3] = size(A);
AH = zeros(n2, n1, n3);
T = transform(A, Phi);
for i = 1:n3
    AH(:,:,i) = conj(T(:,:,i))';
end

PhiH = conj(Phi)';
AH = transform(AH, PhiH);

end