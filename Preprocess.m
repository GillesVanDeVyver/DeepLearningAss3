function [MX1s,class_counts,class_starts,trainx,validationx]...
          = Preprocess(K,trainX,validationX,trainY,hyper_paras,nlen,d)
    n = size(trainX,3);
    MX1s = cell(n,1);
    class_starts=ones(K,1);
    curr_class=1;
    class_counts=zeros(K,1);
    counter=0;
    last_start=1;
    for j=1:n
        if trainY(curr_class,j)==0
            class_counts(curr_class)=counter;
            last_start = counter+last_start;
            class_starts(curr_class+1)=last_start;
            curr_class=curr_class+1;
            counter=0;
        end
        counter = counter+1;
        xj= trainX(:,:,j);
        MX1s{j}= sparse(MakeMXMatrix(xj,d,hyper_paras.ks(1),hyper_paras.ns(2),nlen(1)));
    end
    class_counts(curr_class)=counter;
    trainx = reshape(trainX,d*nlen(1),[]);
    validationx = reshape(validationX,d*nlen(1),[]);
end