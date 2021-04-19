function ConvNet = MiniBatchGD(trainX,trainy,validationX,validationy, hyper_paras,ConvNet,nlen,d,K, plotTitle,plot_bool)
    trainx = reshape(trainX,d*nlen{1},[]);
    trainY = double(permute(trainy==1:K,[2,1]));
    validationx = reshape(validationX,d*nlen{1},[]);
    validationY = double(permute(validationy==1:K,[2,1]));
    n_batch = hyper_paras.n_batch;
    eta = hyper_paras.eta;
    n_epochs = hyper_paras.n_epochs;
    n = size(trainX,3);
    if plot_bool
        loss_train = zeros(n_epochs+1,1);
        loss_valid = zeros(n_epochs+1,1);
        acc_train = zeros(n_epochs+1,1);
        acc_valid = zeros(n_epochs+1,1);
        plot_info = {loss_train,loss_valid,acc_train,acc_valid};
    else
       plot_info=0; 
    end
    plot_info{1}(1) = ComputeLoss(trainx, trainY, ConvNet,nlen);
    plot_info{2}(1) = ComputeLoss(validationx, validationY, ConvNet,nlen);
    plot_info{3}(1) = ComputeAccuracy(trainx, trainy, ConvNet,nlen);
    plot_info{4}(1) = ComputeAccuracy(validationx, validationy, ConvNet,nlen);
    % pre-compute MX matrices
    MX1s = cell(n,1);
    for j=1:n
        xj= trainX(:,:,j);
        MX1s{j}= sparse(MakeMXMatrix(xj,d,hyper_paras.k1,hyper_paras.n1,nlen{1}));
    end    
    for i=1:n_epochs
        i
        shuffleInds = randperm(n);
        Xshuffle = trainX(:,:, shuffleInds);
        Yshuffle = trainY(:, shuffleInds);
        xshuffle = reshape(Xshuffle,d*nlen{1},[]);
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
                hyper_paras.k2,hyper_paras.n2,nlen,MX1s);
            ConvNet.W = ConvNet.W - eta*grad_W;
            grad_F1 = reshape(grad_vecF{1}, [d, hyper_paras.k1, hyper_paras.n1]);
            grad_F2 = reshape(grad_vecF{2}, [hyper_paras.n1, hyper_paras.k2, hyper_paras.n2]);
            
            ConvNet.F{1} = ConvNet.F{1} - eta*grad_F1;
            ConvNet.F{2} = ConvNet.F{2} - eta*grad_F2;
        end
        eta = eta*hyper_paras.rho;
        plot_info{1}(i+1) = ComputeLoss(trainx, trainY, ConvNet,nlen);
        plot_info{2}(i+1) = ComputeLoss(validationx, validationY, ConvNet,nlen);
        plot_info{3}(i+1) = ComputeAccuracy(trainx, trainy, ConvNet,nlen);
        plot_info{4}(i+1) = ComputeAccuracy(validationx, validationy, ConvNet,nlen);
    end
    if plot_bool
        x_axis = 0:n_epochs;
        figure('Renderer', 'painters', 'Position', [10 10 1500 300])
        tiledlayout(1,2)
        nexttile
        plot(x_axis,plot_info{1},x_axis,plot_info{2})
        ylim([0 max(plot_info{2})+1])
        xlabel('epoch') 
        ylabel('loss')
        legend({'training loss','validation loss'},'Location','northeast')
        nexttile
        plot(x_axis,plot_info{3},x_axis,plot_info{4})
        ylim([0 max(plot_info{4})+1])
        xlabel('epoch') 
        ylabel('accuracy')
        legend({'training accuracy','validation accuracy'},'Location','northeast')
        sgtitle(plotTitle) 
        saveas(1,strcat(strrep(plotTitle, '.', ','),'.png'))
    end
end


