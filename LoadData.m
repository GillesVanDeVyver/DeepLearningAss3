function [trainX,trainY,validationX,validationY] = LoadData(all_names,char_to_ind,nlen,d,nb_names,ys,flat_size)
    
    fid = fopen('Validation_Inds.txt','r');
    S = fscanf(fid,'%c');
    fclose(fid);
    inds = strsplit(S, ' ');
    nb_valid_samples=length(inds);

    trainX=zeros(d,nlen,nb_names-nb_valid_samples);
    validationX=zeros(d,nlen,nb_valid_samples);
    trainY=zeros(nb_names-nb_valid_samples,1);
    validationY=zeros(nb_valid_samples,1);
    j=1;
    next_valid=str2double(inds{j});
    for i=1:length(all_names)
        name=all_names{i};
        name_matrix = ConvertToMatrix(name,char_to_ind,nlen,d);
        %flat_name_matrix=name_matrix(:);
        if i==next_valid
            validationX(:,:,j)=name_matrix;
            validationY(j)=ys(i);
            j=j+1;
            next_valid=str2double(inds{j});
        else
            trainX(:,:,i-j+1)=name_matrix;
            trainY(i-j+1)=ys(i);
        end
    end

end