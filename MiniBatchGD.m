function ConvNet = MiniBatchGD(trainX,trainy,validationX,validationy,...
                               hyper_paras,ConvNet,nlen,d,K, plotTitle,...
                               plot_bool,MX1s,class_counts,class_starts,...
                               trainx,trainY,validationx,validationY)
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
        plot_info{1}(1) = ComputeLoss(trainx, trainY, ConvNet,nlen);
        plot_info{2}(1) = ComputeLoss(validationx, validationY, ConvNet,nlen);
        plot_info{3}(1) = ComputeAccuracy(trainx, trainy, ConvNet,nlen);
        plot_info{4}(1) = ComputeAccuracy(validationx, validationy, ConvNet,nlen);
    else
       plot_info=0; 
    end

    nb_samples = min(class_counts);
    effective_n=nb_samples*K;
    for i=1:n_epochs
        effective_trainx=zeros(d*nlen{1},effective_n);
        effective_trainX=zeros(d,nlen{1},effective_n);
        effective_trainY=zeros(K,effective_n);
        effective_trainMX1s=cell(effective_n,1);
        for class =1:K
            sample_inds = class_starts(class)-1+randsample(class_counts(class),nb_samples);
            effective_trainx(:,(class-1)*nb_samples+1:(class-1)*nb_samples+nb_samples)=trainx(:,sample_inds);
            effective_trainX(:,:,(class-1)*nb_samples+1:(class-1)*nb_samples+nb_samples)=trainX(:,:,sample_inds);
            effective_trainY(:,(class-1)*nb_samples+1:(class-1)*nb_samples+nb_samples)=trainY(:,sample_inds);
            for ind=1:nb_samples
                effective_trainMX1s{ind+(class-1)*nb_samples}=MX1s{sample_inds(ind)};
            end
        end
        shuffleInds=randperm(nb_samples*K);
        Xshuffle = effective_trainX(:,:, shuffleInds);
        Yshuffle = effective_trainY(:, shuffleInds);
        xshuffle = reshape(Xshuffle,d*nlen{1},[]);
        for j=1:effective_n/n_batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            X_batch={Xshuffle(:,:,j_start:j_end)};
            x_batch={xshuffle(:,j_start:j_end)};
            Y_batch=Yshuffle(:,j_start:j_end);
            MFs={MakeMFMatrix(ConvNet.F{1}, nlen{1}),MakeMFMatrix(ConvNet.F{2}, nlen{2})};
            [x_batch,P_batch] = ForwardPass(x_batch{1},MFs,ConvNet);
            X_batch{2}=reshape(x_batch{2},hyper_paras.n1,nlen{2},n_batch);
            X_batch{3}=reshape(x_batch{3},hyper_paras.n2,nlen{3},n_batch);
            [grad_W, grad_vecF,grad_b] = ComputeGradients(X_batch,x_batch, Y_batch,...
                P_batch, ConvNet.W,MFs,d,hyper_paras.k1,hyper_paras.n1,...
                hyper_paras.k2,hyper_paras.n2,nlen,effective_trainMX1s,K);
            ConvNet.W = ConvNet.W - eta*grad_W;
            grad_F1 = reshape(grad_vecF{1}, [d, hyper_paras.k1, hyper_paras.n1]);
            grad_F2 = reshape(grad_vecF{2}, [hyper_paras.n1, hyper_paras.k2, hyper_paras.n2]);
            
            ConvNet.F{1} = ConvNet.F{1} - eta*grad_F1;
            ConvNet.F{2} = ConvNet.F{2} - eta*grad_F2;
            ConvNet.b{1} = ConvNet.b{1} - eta*grad_b{1};
            ConvNet.b{2} = ConvNet.b{2} - eta*grad_b{2};
            ConvNet.b{3} = ConvNet.b{3} - eta*grad_b{3};
        end
        eta = eta*hyper_paras.rho;
        if plot_bool
            plot_info{1}(i+1) = ComputeLoss(trainx, trainY, ConvNet,nlen);
            plot_info{2}(i+1) = ComputeLoss(validationx, validationY, ConvNet,nlen);
            plot_info{3}(i+1) = ComputeAccuracy(trainx, trainy, ConvNet,nlen);
            plot_info{4}(i+1) = ComputeAccuracy(validationx, validationy, ConvNet,nlen);
        end
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
        ylim([0 max(plot_info{3})+0.2])
        xlabel('epoch') 
        ylabel('accuracy')
        legend({'training accuracy','validation accuracy'},'Location','northeast')
        sgtitle(plotTitle) 
        saveas(1,strcat(strrep(plotTitle, '.', ','),'.png'))
    end
end


