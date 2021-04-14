function name_as_matrix = ConvertToMatrix(name,char_to_ind,n_len,d)
    name_as_matrix=zeros(d,n_len);
    if strlength(name)> n_len
        name = extractBetween(str,1,n_len)
    end
    for i = 1:length(name)
      letter_ind=char_to_ind(name(i));
      letter_ind==1:d;
      name_as_matrix(:,i) = letter_ind==1:d;
    end

end