function ConvNet = MiniBatchGD(trainX,trainy,validationX,validationy, hyper_paras,ConvNet,nlen,d,K, plotTitle)
    trainx = reshape(trainX,d*nlen{1},[]);
    trainY = double(permute(trainy==1:K,[2,1]));
    validationx = reshape(validationX,d*nlen{1},[]);
    validationY = double(permute(validationy==1:K,[2,1]));
    n_batch = hyper_paras.n_batch;
    eta = hyper_paras.eta;
    n_epochs = hyper_paras.n_epochs;
    n = size(trainX,3);
    loss_train = zeros(n_epochs+1,1);
    loss_valid = zeros(n_epochs+1,1);
    loss_train(1) = ComputeLoss(trainx, trainY, ConvNet,nlen);
    loss_valid(1) = ComputeLoss(validationx, validationY, ConvNet,nlen);
    for i=1:n_epochs
        i
        shuffleInds = randperm(n);
        Xshuffle = trainX(:, :,shuffleInds);
        xshuffle = reshape(Xshuffle,d*nlen{1},[]);
        Yshuffle = trainY(:, shuffleInds);
        for j=1:n/n_batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            X_batch={Xshuffle(:,:,j_start:j_end)};
            x_batch={xshuffle(:,j_start:j_end)};
            Y_batch=Yshuffle(:,j_start:j_end);
            MFs={MakeMFMatrix(ConvNet.F{1}, nlen{1}),MakeMFMatrix(ConvNet.F{2}, nlen{2})};
            [x_batch,P_batch] = ForwardPass(x_batch{1},MFs,ConvNet.W);
            X_batch{2}=reshape(x_batch{2},hyper_paras.n1,nlen{2},n_batch);
            X_batch{3}=reshape(x_batch{3},hyper_paras.n2,nlen{3},n_batch);
            [grad_W, grad_vecF] = ComputeGradients(X_batch,x_batch, Y_batch,...
                P_batch, ConvNet.W,MFs,d,hyper_paras.k1,hyper_paras.n1,...
                hyper_paras.k2,hyper_paras.n2,nlen);
            ConvNet.W = ConvNet.W - eta*grad_W;
            grad_F1 = reshape(grad_vecF{1}, [d, hyper_paras.k1, hyper_paras.n1]);
            grad_F2 = reshape(grad_vecF{2}, [hyper_paras.n1, hyper_paras.k2, hyper_paras.n2]);
            
            ConvNet.F{1} = ConvNet.F{1} - eta*grad_F1;
            ConvNet.F{2} = ConvNet.F{2} - eta*grad_F2;
        end
        eta = eta*.9;
        loss_train(i+1) = ComputeLoss(trainx, trainY, ConvNet,nlen);
        loss_valid(i+1) = ComputeLoss(validationx, validationY, ConvNet,nlen);
    end
    loss_train
    loss_valid
    epochInds = 0:n_epochs;
    figure
    plot(epochInds,loss_train,epochInds,loss_valid)
    xlabel('epoch') 
    ylabel('loss')
    legend({'training loss','validation loss'},'Location','northeast')
    title(plotTitle)
    axis tight
    print -depsc loss_SVM_paras3
end


