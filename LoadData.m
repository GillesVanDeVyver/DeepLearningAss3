function [trainX,trainy,validationX,validationy] = LoadData(all_names,char_to_ind,nlen,d,nb_names,ys)
    
    fid = fopen('Validation_Inds.txt','r');
    S = fscanf(fid,'%c');
    fclose(fid);
    inds = strsplit(S, ' ');
    nb_valid_samples=length(inds)-1;
    trainX=zeros(d,nlen,nb_names-nb_valid_samples);
    validationX=zeros(d,nlen,nb_valid_samples);
    trainy=zeros(nb_names-nb_valid_samples,1);
    validationy=zeros(nb_valid_samples,1);
    j=1;
    next_valid=str2double(inds{j});
    for i=1:length(all_names)
        name=all_names{i};
        name_matrix = ConvertToMatrix(name,char_to_ind,nlen,d);
        %flat_name_matrix=name_matrix(:);
        if i==next_valid
            validationX(:,:,j)=name_matrix;
            validationy(j)=ys(i);
            j=j+1;
            next_valid=str2double(inds{j});
        else
            trainX(:,:,i-j+1)=name_matrix;
            trainy(i-j+1)=ys(i);
        end
    end

end