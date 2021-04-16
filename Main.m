hyper_paras = struct('n1',4,'k1',5,'n2',3,'k2', 3, 'eta',0.001,'rho',0.9);
title = strcat('n1=',string(hyper_paras.n1),',k1=',string(hyper_paras.k1),...
            ',n2=',string(hyper_paras.n2),',n2=',string(hyper_paras.n2),...
            ',k2=',string(hyper_paras.k2),',eta=',string(hyper_paras.eta),...
            ',rho=',string(hyper_paras.rho)) 
ExtractNames();
C = unique(cell2mat(all_names));
d = numel(C);
nlen = 19;
nlen1=nlen-hyper_paras.k1+1;
K = 18;
nb_names = size(all_names,2);
char_to_ind = CreateCharToInd(d,C);
flat_size=d*nlen;
[trainX,trainY,validationX,validationY] = LoadData(all_names,char_to_ind,nlen,d,nb_names,ys,flat_size);
trainx = reshape(trainX,d*nlen,[]);
ConvNet = InitParas(hyper_paras.n1,hyper_paras.k1,hyper_paras.n2,hyper_paras.k2,nlen,d,K);
MF1 = MakeMFMatrix(ConvNet.F{1}, nlen);
x1=MF1*trainx;
MF2 = MakeMFMatrix(ConvNet.F{2}, nlen1);
[d, k, nf] = size(ConvNet.F{1});

X_batch=trainX(:,:,1);
x_batch=trainx(:,1);


[x_batch_1,x_batch_2,P_batch] = ForwardPass(x_batch,MF1,MF2,ConvNet.W);













%{
% debug tests

MF1 = MakeMFMatrix(ConvNet.F{1}, nlen);
MF2 = MakeMFMatrix(ConvNet.F{2}, nlen);
[d, k, nf] = size(ConvNet.F{1});
X_input=trainX(:,:,1);
x_input=trainx(:,1);
MX = MakeMXMatrix(X_input, d, k, nf,nlen);
s1 = MX * ConvNet.F{1}(:);
s2 = MF1 * x_input;
assert(isequal(s1,s2));
load('DebugInfo.mat');
MX = MakeMXMatrix(X_input, d, k, nf,nlen);
MF1 = MakeMFMatrix(F, nlen);
s1 = MX * F(:);
s2 = MF1 * x_input;
assert(isequal(s1,s2));
assert(isequal(s1,vecS));
assert(isequal(reshape(s1,4,15),S))

%}



















