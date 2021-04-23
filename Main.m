rng(400);


ExtractNames();
C = unique(cell2mat(all_names));
d = numel(C);


ns=[d,20,20,20,20,20];
ks=[5,3,3,3,3];
nb_layers = size(ns,2);
hyper_paras = struct('ns',ns,'ks',ks,'eta',0.30556,'rho',0.99913,'n_batch',105,'n_epochs',3500);
plotTitle = strcat('nb_layers=',string(size(ns,2)),',eta=',string(hyper_paras.eta),...
            ',rho=',string(hyper_paras.rho),',n batch=',string(hyper_paras.n_batch),...
            ',n epochs=',string(hyper_paras.n_epochs)) 


nlen=zeros(nb_layers,1);
nlen(1)=19;
for i=2:nb_layers
    nlen(i)=nlen(i-1) - hyper_paras.ks(i-1) + 1;
end

K = 18;
nb_names = size(all_names,2);
char_to_ind = CreateCharToInd(d,C);
[trainX,trainy,validationX,validationy] = LoadData(all_names,char_to_ind,nlen(1),d,nb_names,ys);

% pre-compute MX matrices
[MX1s,class_counts,class_starts,trainx,trainY,validationx,validationY] = Preprocess(K,trainy,trainX,validationX,validationy,hyper_paras,nlen,d);
ConvNet = InitParas(hyper_paras,nlen,K);
ConvNet = MiniBatchGD(trainX,trainy,validationX,validationy, hyper_paras,...
              ConvNet,nlen,K, plotTitle,1,MX1s,class_counts,class_starts,...
              trainx,trainY,validationx,validationY);
final_valid_loss = ComputeLoss(validationx, validationY, ConvNet,nlen)
final_valid_acc = ComputeAccuracy(validationx, validationy, ConvNet,nlen)
%confusion_matrix = createConfMatrix(validationx, validationy, ConvNet,nlen,K)
%writematrix(confusion_matrix,strcat(strrep(plotTitle, '.', ','),'confMatrix.txt'));



MFs=MakeMFMatrices(ConvNet, nlen);
names = {'van de vyver','de luca','bogacova','griffin','charlier','karl'};
%languages: Dutch,Italian,Russian,English,French,German
for i =1:size(names,2)
    CreateNameResult(names{i},char_to_ind,nlen,d,MFs,ConvNet);
end





























%{
hyper parameter search

fileID = fopen('Results_vary_eta.txt','w');
e_min=-1;
e_max=0; %best eta eta=0.31623
nb_uniform_tests = 10; 
for i = 0:nb_uniform_tests 
    l_exp = e_min + i*(e_max-e_min)/nb_uniform_tests;
    hyper_paras.eta = 10^l_exp;
        final_valid_acc = TrainAndPrint(hyper_paras,nlen,d,K,trainX,trainx,trainy,trainY,...
            validationX,validationx,validationy,validationY,fileID,MX1s,class_counts,class_starts)
end
fclose(fileID);
%}

%{
hyper parameter search

fileID = fopen('Results_vary_rho.txt','w');
rho_exp_min = -4.66;
rho_exp_max = -2.66; 
nb_uniform_tests = 30; 
for i = 0:nb_uniform_tests 
    rho_exp = rho_exp_min + i*(rho_exp_max-rho_exp_min)/nb_uniform_tests;
    hyper_paras.rho = 1-10^rho_exp;
        final_valid_acc = TrainAndPrint(hyper_paras,nlen,d,K,trainX,trainx,trainy,trainY,...
            validationX,validationx,validationy,validationY,fileID,MX1s,class_counts,class_starts)
end
fclose(fileID);
%}

%{
hyper parameter search

fileID = fopen('Results_vary_n_batch.txt','w');
n_batch_min = 50;
n_batch_max = 250; 
nb_uniform_tests = 40; 
for i = 0:nb_uniform_tests 
    hyper_paras.n_batch = n_batch_min + i*(n_batch_max-n_batch_min)/nb_uniform_tests;
        final_valid_acc = TrainAndPrint(hyper_paras,nlen,d,K,trainX,trainx,trainy,trainY,...
            validationX,validationx,validationy,validationY,fileID,MX1s,class_counts,class_starts)
end
fclose(fileID);
%}

%{
hyper parameter search

fileID = fopen('Results_unifrom_grid_search1.txt','w');
l_min=-5;
l_max=1;
rho_exp_min = -5;
rho_exp_max = -1; 
n_batch_exp_min = 1.5;
n_batch_exp_max = 2.5;
nb_uniform_tests = 3; % total nb of tests 4*3*3 = 36
for i = 0:nb_uniform_tests %eta
    l_exp = l_min + i*(l_max-l_min)/nb_uniform_tests;
    hyper_paras.eta = 10^l_exp;
    i
    for j = 1:nb_uniform_tests %n_batch 
        n_batch_exp = n_batch_exp_min + j*(n_batch_exp_max-n_batch_exp_min)/nb_uniform_tests;
        hyper_paras.n_batch = floor(10^n_batch_exp);
        j

        for l = 1:nb_uniform_tests %rho
            rho_exp = rho_exp_min + l*(rho_exp_max-rho_exp_min)/nb_uniform_tests;
            hyper_paras.rho = 1-10^rho_exp;
            final_valid_acc = TrainAndPrint(hyper_paras,nlen,d,K,trainX,trainx,trainy,trainY,...
                validationX,validationx,validationy,validationY,fileID,MX1s,class_counts,class_starts)
        end
    end
end
fclose(fileID);

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

















