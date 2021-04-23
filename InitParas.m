function ConvNet = InitParas(hyper_paras,n_len,K)
    nb_layers = size(hyper_paras.ns,2); 
    for layer =1:nb_layers-1
        input_size = hyper_paras.ns(layer)*n_len(layer);
        sig= sqrt(2/input_size);
        ConvNet.F{layer} = randn(hyper_paras.ns(layer), hyper_paras.ks(layer),hyper_paras.ns(layer+1))*sig;
        ConvNet.b{layer}=zeros(n_len(layer+1)*hyper_paras.ns(layer+1),1);

    end
    input_size_final = hyper_paras.ns(nb_layers-1)*n_len(nb_layers);
    sig_final= sqrt(2/input_size_final);
    fsize=hyper_paras.ns(nb_layers-1)*n_len(nb_layers);
    ConvNet.W = randn(K, fsize)*sig_final;
    ConvNet.b{nb_layers}=zeros(K,1);

