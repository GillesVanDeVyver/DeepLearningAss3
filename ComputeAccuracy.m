function acc = ComputeAccuracy(x, y, ConvNet,nlen)
    MFs={MakeMFMatrix(ConvNet.F{1}, nlen{1}),MakeMFMatrix(ConvNet.F{2}, nlen{2})};
    [~,P] = ForwardPass(x,MFs,ConvNet.W);
    [~,I] = max(P);
    acc = sum(permute(I,[2,1])==y)/size(x,2);
end