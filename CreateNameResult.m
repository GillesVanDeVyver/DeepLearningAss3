function CreateNameResult(name,char_to_ind,nlen,d,MFs,ConvNet)
    name_matrix = ConvertToMatrix(name,char_to_ind,nlen{1},d);
    test_x = name_matrix(:);
    [~,P] = ForwardPass(test_x,MFs,ConvNet.W);
    labels = ["Arabic","Chinese","Czech","Dutch","English","French","German","Greek","Irish","Italian","Japanese","Korean","Polish","Portuguese","Russian","Scottish","Spanish","Vietnamese"];
    fileID = fopen(strcat(name,'_result.txt'),'w');
    for i =1:18
        result = strcat(string(P(i)),'%% ',labels(i),'\n')
        fprintf(fileID,result);
    end
    fclose(fileID);
end