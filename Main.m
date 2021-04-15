ExtractNames();
C = unique(cell2mat(all_names));
d = numel(C);
n_len = 19;
K = 18;
nb_names = size(all_names,2);
char_to_ind = CreateCharToInd(d,C);
flat_size=d*n_len;
%[trainX,trainY,validationX,validationY] = LoadData(all_names,char_to_ind,n_len,d,nb_names,ys,flat_size);


hyper_paras = struct('n1',3,'k1',3,'n2',3,'k2', 3, 'eta',0.001,'rho',0.9);
title = strcat('n1=',string(hyper_paras.n1),',k1=',string(hyper_paras.k1),...
            ',n2=',string(hyper_paras.n2),',n2=',string(hyper_paras.n2),...
            ',k2=',string(hyper_paras.k2),',eta=',string(hyper_paras.eta),...
            ',rho=',string(hyper_paras.rho))
eta=0.001;
rho=0.9;
ConvNet = InitParas(hyper_paras.n1,hyper_paras.k1,hyper_paras.n2,hyper_paras.k2,n_len,d,K)


