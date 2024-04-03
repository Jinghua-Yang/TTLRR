function C = transform_tprod(A, B, Phi)

[n1, ~, n3] = size(A);
[~, n4, ~] = size(B);

CT = zeros(n1, n4, n3);
AT = transform(A, Phi);
BT = transform(B, Phi);
for i = 1:n3
    CT(:,:,i) = AT(:,:,i)*BT(:,:,i);
end

PhiH = conj(Phi)';
C = transform(CT, PhiH);
end