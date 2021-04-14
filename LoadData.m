function [trainX,trainY,validationX,validationY] = LoadData(all_names,char_to_ind,n_len,d,nb_names,ys)
    
    fid = fopen('Validation_Inds.txt','r');
    S = fscanf(fid,'%c');
    fclose(fid);
    inds = strsplit(S, ' ');
    nb_valid_samples=length(inds);
    flat_size=d*n_len;
    trainX=zeros(flat_size,nb_names-nb_valid_samples);
    validationX=zeros(flat_size,nb_valid_samples);
    trainY=zeros(nb_names-nb_valid_samples,1);
    validationY=zeros(nb_valid_samples,1);
    j=1;
    next_valid=str2double(inds{j});
    for i=1:length(all_names)
        name=all_names{i};
        name_matrix = ConvertToMatrix(name,char_to_ind,n_len,d);
        flat_name_matrix=name_matrix(:);
        if i==next_valid
            validationX(:,j)=flat_name_matrix;
            validationY(j)=ys(i);
            j=j+1;
            next_valid=str2double(inds{j});
        else
            trainX(:,i-j+1)=flat_name_matrix;
            trainY(i-j+1)=ys(i);
        end
    end

end