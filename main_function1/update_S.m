function S = update_S(Phi, A, J1, J2, beta1, beta2)

[n1, n2, n3] = size(J1);

AO = tenmat(A, 3);
AT = AO.data;
AT = Phi*AT;
AT = tensor(tenmat(AT, AO.rdims, AO.cdims, AO.tsize));
AT = AT.data;

J1O = tenmat(J1, 3);
J1T = J1O.data;
J1T = Phi*J1T;
J1T = tensor(tenmat(J1T, J1O.rdims, J1O.cdims, J1O.tsize));
J1T = J1T.data;

J2O = tenmat(J2, 3);
J2T = J2O.data;
J2T = Phi*J2T;
J2T = tensor(tenmat(J2T, J2O.rdims, J2O.cdims, J2O.tsize));
J2T = J2T.data;

ST = zeros(n1, n2, n3);
for i = 1:n3
    AA = AT(:,:,i)'*AT(:,:,i);
    ST(:,:,i) = (beta1*eye(size(AA)) + beta2*AA)\(J1T(:,:,i) + AT(:,:,i)'*J2T(:,:,i));
end

O = tenmat(ST, 3); %square norm
SS = O.data;
Y = Phi'*SS;
Y = tensor(tenmat(Y, O.rdims, O.cdims, O.tsize));
S = Y.data;

end