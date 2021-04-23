function loss = ComputeLoss2(x_input, Ys_batch, ConvNet,nlen)
    MFs={MakeMFMatrix(ConvNet.F{1}, nlen{1}),MakeMFMatrix(ConvNet.F{2},nlen{2})};
    [~,P_batch] = ForwardPass(x_input,MFs,ConvNet);
    lcross = -Ys_batch'*log(P_batch);
    whos Ys_batch
    whos P_batch
    loss = sum(lcross(:))/size(x_input,2);
end

