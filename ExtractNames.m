data_fname = 'ascii_names.txt';
fid = fopen(data_fname,'r');
S = fscanf(fid,'%c');
fclose(fid);
names = strsplit(S, '\n');
fid = fopen('Validation_Inds.txt','r');
S = fscanf(fid,'%c');
fclose(fid);
inds = strsplit(S, ' ');
K=18;
if length(names{end}) < 1        
    names(end) = [];
end
names_map = containers.Map();
for i=1:length(names)
    nn = strsplit(names{i}, ' ');
    l = str2num(nn{end});
    if length(nn) > 2
        name = strjoin(nn(1:end-1));
    else
        name = nn{1};
    end
    name = lower(name);
  %  if strlength(name)>max_len
  %      max_len=strlength(name)
  %      name
  %  end   
  
  
    if isKey(names_map,name)
        names_map(name)=[names_map(name) l];
    else
        names_map(name)=[l];
    end
    all_names{i} = name;
end
C = unique(cell2mat(all_names));
d = numel(C);
char_to_ind = CreateCharToInd(d,C);

nb_valid_samples=length(inds)-1;
nb_unique_names=length(names_map);
trainX=zeros(d,19,nb_unique_names-nb_valid_samples);
validationX=zeros(d,19,nb_valid_samples);
trainY=zeros(K,nb_unique_names-nb_valid_samples);
validationY=zeros(K,nb_valid_samples);
whos trainY
j=1;
next_valid=str2double(inds{j});



 for i = 1:size(all_names,2)
    name=all_names{i};
    name_matrix = ConvertToMatrix(name,char_to_ind,19,d);
    if i==next_valid
        validationX(:,:,j)=name_matrix;
        val = names_map(name);
        nb_classes = size(val,2);
        vec = zeros(K,1);
        for l=1:nb_classes
            vec(val(l))=1/nb_classes;
        end
        validationY(:,j)=vec;
        j=j+1;
        next_valid=str2double(inds{j});
    else
        trainX(:,:,i-j+1)=name_matrix;
        val = names_map(name);
        nb_classes = size(val,2);
        vec = zeros(K,1);
        for l=1:nb_classes
            vec(val(l))=1/nb_classes;
        end
        trainY(:,i-j+1)=vec;
    end  
 end
 



