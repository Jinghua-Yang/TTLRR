function AT = ttrans(A, Phi)

AO = tenmat(A, 3);
AT = AO.data;
AT = Phi*AT;
AT = tensor(tenmat(AT, AO.rdims, AO.cdims, AO.tsize));
AT = AT.data;

end