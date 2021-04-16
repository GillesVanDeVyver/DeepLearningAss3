function [x_batch_1,x_batch_2,P_batch] = ForwardPass(x_batch,MFs,W)
    x_batch_1 = max(MFs{1}*x_batch , 0);
    x_batch_2 = max(MFs{2}*x_batch_1 , 0);
    S_batch = W*x_batch_2;
    denom = sum(exp(S_batch),1);
    P_batch = zeros(size(W,1),size(S_batch,2));
    for i =1:size(S_batch,2)
        P_batch(:,i) = exp(S_batch(:,i))/denom(i);
    end
end