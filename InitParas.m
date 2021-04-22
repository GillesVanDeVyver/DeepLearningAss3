function ConvNet = InitParas(n1,k1,n2,k2,n_len,d,K)
    input_size_1 = d*n_len{1};
    input_size_2 = n1*n_len{2};
    input_size_3 = n2*n_len{3};
    % https://arxiv.org/pdf/1502.01852.pdf page 4 bottom
    sig1= sqrt(1/input_size_1);
    sig2= sqrt(2/input_size_2);
    sig3= sqrt(2/input_size_3);
    ConvNet.F{1} = randn(d, k1, n1)*sig1;
    ConvNet.F{2} = randn(n1, k2, n2)*sig2;
    fsize=n2*n_len{3};
    ConvNet.W = randn(K, fsize)*sig3;
    
    ConvNet.b{1}=zeros(n_len{2}*n1,1);
    ConvNet.b{2}=zeros(n_len{3}*n2,1);
    ConvNet.b{3}=zeros(K,1);

