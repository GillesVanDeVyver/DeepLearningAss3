function acc = ComputeAccuracy(x, y, ConvNet,nlen)
    MFs=MakeMFMatrices(ConvNet, nlen);
    [~,P] = ForwardPass(x,MFs,ConvNet);
    [~,I] = max(P);
    acc = sum(permute(I,[2,1])==y)/size(x,2);
end