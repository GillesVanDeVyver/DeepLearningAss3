function [x_batch,P_batch] = ForwardPass(x_input,MFs,ConvNet)
    nb_layers = size(MFs,1)+1;
    x_batch=cell(nb_layers,1);
    x_batch{1}=x_input;
    for layer = 2:nb_layers
        x_batch{layer} = max(MFs{layer-1}*x_batch{layer-1} +ConvNet.b{layer-1}, 0);
    end
    S_batch = ConvNet.W*x_batch{nb_layers}+ConvNet.b{nb_layers};
    denom = sum(exp(S_batch),1);
    P_batch = zeros(size(ConvNet.W,1),size(S_batch,2));
    for i =1:size(S_batch,2)
        P_batch(:,i) = exp(S_batch(:,i))/denom(i);
    end
end