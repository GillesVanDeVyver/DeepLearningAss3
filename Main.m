rng(400);

hyper_paras = struct('n1',20,'k1',5,'n2',20,'k2', 3, 'eta',0.30556,'rho',0.99913,'n_batch',105,'n_epochs',5);
plotTitle = strcat('n1=',string(hyper_paras.n1),',k1=',string(hyper_paras.k1),...
            ',n2=',string(hyper_paras.n2),...
            ',k2=',string(hyper_paras.k2),',eta=',string(hyper_paras.eta),...
            ',rho=',string(hyper_paras.rho),',n batch=',string(hyper_paras.n_batch),...
            ',n epochs=',string(hyper_paras.n_epochs)) 
ExtractNames();
C = unique(cell2mat(all_names));
d = numel(C);
nlen={19,19-hyper_paras.k1+1,0};
nlen{3} = nlen{2} - hyper_paras.k2 + 1
K = 18;
nb_names = size(all_names,2);
char_to_ind = CreateCharToInd(d,C);
[trainX,trainy,validationX,validationy] = LoadData(all_names,char_to_ind,nlen{1},d,nb_names,ys);


% pre-compute MX matrices
[MX1s,class_counts,class_starts,trainx,trainY,validationx,validationY] = Preprocess(K,trainy,trainX,validationX,validationy,hyper_paras,nlen,d);
validationx = reshape(validationX,d*nlen{1},[]);
ConvNet = InitParas(hyper_paras.n1,hyper_paras.k1,hyper_paras.n2,hyper_paras.k2,nlen,d,K);
ConvNet = MiniBatchGD(trainX,trainy,validationX,validationy, hyper_paras,...
              ConvNet,nlen,d,K, plotTitle,0,MX1s,class_counts,class_starts,...
              trainx,trainY,validationx,validationY);
%confusion_matrix = createConfMatrix(validationx, validationy, ConvNet,nlen,K)
%writematrix(confusion_matrix,strcat(strrep(plotTitle, '.', ','),'confMatrix.txt'));



MFs={MakeMFMatrix(ConvNet.F{1}, nlen{1}),MakeMFMatrix(ConvNet.F{2}, nlen{2})};
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

















