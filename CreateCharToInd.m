function  char_to_ind = CreateCharToInd(d,C)
    values = zeros(1,d);
    keys = cell(1,d);
    for i = 1:length(C)
      keys(i)={C(i)};
      values(i)=i;
    end
    char_to_ind = containers.Map(keys,values);
end