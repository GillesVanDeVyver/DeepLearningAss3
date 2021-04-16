function loss = ComputeLoss(x_batch, ys_batch, MFs, W)
    [~,~,P_batch] = ForwardPass(x_batch,MFs,W);
    lcross = -log(dot(ys_batch,P_batch));
    loss = sum(lcross)/size(x_batch,2);
end