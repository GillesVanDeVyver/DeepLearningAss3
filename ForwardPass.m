function [x_batch,P_batch] = ForwardPass(x_input,MFs,ConvNet)
    x_batch{1}=x_input;
    x_batch_1 = max(MFs{1}*x_input +ConvNet.b{1}, 0);
    x_batch_2 = max(MFs{2}*x_batch_1+ConvNet.b{2} , 0);
    S_batch = ConvNet.W*x_batch_2+ConvNet.b{3};
    denom = sum(exp(S_batch),1);
    P_batch = zeros(size(ConvNet.W,1),size(S_batch,2));
    for i =1:size(S_batch,2)
        P_batch(:,i) = exp(S_batch(:,i))/denom(i);
    end
    x_batch{2}=x_batch_1;
    x_batch{3}=x_batch_2;   
end