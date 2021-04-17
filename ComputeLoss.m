function loss = ComputeLoss(x_input, Ys_batch, ConvNet,nlen)
    MFs={MakeMFMatrix(ConvNet.F{1}, nlen{1}),MakeMFMatrix(ConvNet.F{2},nlen{2})};
    [x_batch,P_batch] = ForwardPass(x_input,MFs,ConvNet.W);
    lcross = -log(dot(Ys_batch,P_batch));
    loss = sum(lcross)/size(x_input,2);
end

