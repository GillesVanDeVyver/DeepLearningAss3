ExtractNames();
C = unique(cell2mat(all_names));
d = numel(C);
n_len = 19;
nb_classes = 18;
nb_names = size(all_names,2);
char_to_ind = CreateCharToInd(d,C);
[trainX,trainY,validationX,validationY] = LoadData(all_names,char_to_ind,n_len,d,nb_names,ys);

