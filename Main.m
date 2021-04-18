rng(400);
hyper_paras = struct('n1',4,'k1',5,'n2',3,'k2', 3, 'eta',0.001,'rho',0.9,'n_batch',100,'n_epochs',10);
plotTitle = strcat('n1=',string(hyper_paras.n1),',k1=',string(hyper_paras.k1),...
            ',n2=',string(hyper_paras.n2),',n2=',string(hyper_paras.n2),...
            ',k2=',string(hyper_paras.k2),',eta=',string(hyper_paras.eta),...
            ',rho=',string(hyper_paras.rho),',n_batch=',string(hyper_paras.n_batch),...
            ',n_epochs=',string(hyper_paras.n_epochs)) 
ExtractNames();
C = unique(cell2mat(all_names));
d = numel(C);
nlen={19,19-hyper_paras.k1+1,0};
nlen{3} = nlen{2} - hyper_paras.k2 + 1
K = 18;
nb_names = size(all_names,2);
char_to_ind = CreateCharToInd(d,C);
[trainX,trainy,validationX,validationy] = LoadData(all_names,char_to_ind,nlen{1},d,nb_names,ys);
ConvNet = InitParas(hyper_paras.n1,hyper_paras.k1,hyper_paras.n2,hyper_paras.k2,nlen,d,K);

ConvNet = MiniBatchGD(trainX,trainy,validationX,validationy, hyper_paras,ConvNet,nlen,d,K, plotTitle)

%{

Ys = double(permute(ys==1:K,[2,1]));

trainx = reshape(trainX,d*nlen{1},[]);

nb_examples = 100;

X_batch={trainX(:,:,1:nb_examples)};
x_batch={trainx(:,1:nb_examples)};
Ys_batch=Ys(:,1:nb_examples);
ys_batch=ys(1:nb_examples);

MFs={MakeMFMatrix(ConvNet.F{1}, nlen{1}),MakeMFMatrix(ConvNet.F{2}, nlen{2})};
[x_batch,P_batch] = ForwardPass(x_batch{1},MFs,ConvNet.W);

X_batch{2}=reshape(x_batch{2},hyper_paras.n1,nlen{2},nb_examples);
X_batch{3}=reshape(x_batch{3},hyper_paras.n2,nlen{3},nb_examples);
X_batch1 = X_batch{1};

MXs = {MakeMXMatrix(X_batch{1},d, hyper_paras.k1, hyper_paras.n1,nlen{1}),...
       MakeMXMatrix(X_batch{2},hyper_paras.n1, hyper_paras.k2, hyper_paras.n2,nlen{2})};
temp = X_batch{1};
loss = ComputeLoss(x_batch{1}, Ys_batch, ConvNet,nlen);
MX1s = cell(n,1);
for j=1:n
    xj= trainX(:,:,j);
    MX1s{j}= sparse(MakeMXMatrix(xj,d,hyper_paras.k1,hyper_paras.n1,nlen{1}));
end
[grad_W, grad_F] = ComputeGradients(X_batch,x_batch, Ys_batch, P_batch, ConvNet.W,...
                                    MFs,d,hyper_paras.k1,hyper_paras.n1,...
                                    hyper_paras.k2,hyper_paras.n2,nlen,MX1s);

ConvNet = MiniBatchGD(trainX,trainy,validationX,validationy, hyper_paras,ConvNet,nlen,d,K, plotTitle)
                            
%}


%{
% debug tests gradients
h= 1e-6;
eps=1e-3;
Gs = NumericalGradient(x_batch{1}, Ys_batch, ConvNet, h,nlen);

assert(testSame(grad_W,Gs{end}, eps));
assert(testSame(grad_F{1},Gs{1}(:)', eps));
assert(testSame(grad_F{2},Gs{2}(:)', eps));

% debug tests making natrices
MF=MakeMFMatrix(ConvNet.F{1}, nlen{1});

[d, k, nf] = size(ConvNet.F{1});
X_input=trainX(:,:,5);
x_input=trainx(:,5);
MX = MakeMXMatrix(X_input,d, hyper_paras.k1, hyper_paras.n1,nlen{1});
s1 = MX * ConvNet.F{1}(:);
s2 = MF * x_input;
assert(isequal(s1,s2));
load('DebugInfo.mat');
MX=MakeMXMatrix(X_input,d, hyper_paras.k1,hyper_paras.n1,nlen{1});
MF=MakeMFMatrix(F, nlen{1});

s1 = MX * F(:);
s2 = MF * x_input;

assert(isequal(s1,s2));
assert(isequal(s1,vecS));
assert(isequal(reshape(s1,4,15),S));



%}

















