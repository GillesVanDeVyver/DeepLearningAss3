function loss = ComputeLoss(x_input, Ys_batch, ConvNet,nlen)
    MFs=MakeMFMatrices(ConvNet, nlen);
    [~,P_batch] = ForwardPass(x_input,MFs,ConvNet);
    lcross = -log(dot(Ys_batch,P_batch));
    loss = sum(lcross)/size(x_input,2);
end

