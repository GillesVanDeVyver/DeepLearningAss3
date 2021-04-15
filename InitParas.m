function ConvNet = InitParas(n1,k1,n2,k2,n_len,d,K)
input_size_1 = d*n_len;
n_len1 = n_len - k1 + 1;
input_size_2 = n1*n_len1;
n_len2 = n_len1 - k2 + 1;
input_size_3 = n2*n_len2;
% https://arxiv.org/pdf/1502.01852.pdf page 4 bottom
sig1= sqrt(1/input_size_1);
sig2= sqrt(2/input_size_2);
sig3= sqrt(2/input_size_3);
ConvNet.F{1} = randn(d, k1, n1)*sig1;
ConvNet.F{2} = randn(n1, k2, n2)*sig2;
fsize=n2*n_len2;
ConvNet.W = randn(K, fsize)*sig3;

