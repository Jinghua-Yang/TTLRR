function T = transform(A, Phi)

[n1, n2, n3] = size(A);
A3 = reshape(A, [n1*n2, n3]);
temp_A = Phi*A3';
T = reshape(temp_A', [n1, n2, n3]);

end