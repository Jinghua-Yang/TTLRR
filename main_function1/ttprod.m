function C = ttprod(A, B, Phi)

[n1, ~, n3] = size(A);
[~, n4, ~] = size(B);

AT = ttrans(A, Phi);
BT = ttrans(B, Phi);

CT = zeros(n1, n4, n3);
for i = 1:n3
    CT(:,:,i) = AT(:,:,i)*BT(:,:,i);
end

C = ttrans(CT, Phi');
end