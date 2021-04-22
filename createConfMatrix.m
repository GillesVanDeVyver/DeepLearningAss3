function confusion_matrix = createConfMatrix(x, y, ConvNet,nlen,K)
    confusion_matrix = zeros(K,K);
    MFs={MakeMFMatrix(ConvNet.F{1}, nlen{1}),MakeMFMatrix(ConvNet.F{2}, nlen{2})};
    [~,P] = ForwardPass(x,MFs,ConvNet);
    [~,I] = max(P);
    for i=1:size(y,1)
        y(i)
        confusion_matrix(y(i),I(i))=confusion_matrix(y(i),I(i))+1;
    end
end