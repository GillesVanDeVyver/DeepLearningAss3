function [grad_W, grad_F,grad_b] = ComputeGradients(X_batch,x_batch, Ys_batch, P_batch, W,MFs,hyper_paras,nlen,MX1s,K)
        nb_layers = size(hyper_paras.ns,2);
        grad_F=cell(nb_layers-1,1);
        grad_b=cell(nb_layers,1);
    for layer = 1:nb_layers-1
        grad_F{layer}=zeros(1,hyper_paras.ns(layer)*hyper_paras.ks(layer)*hyper_paras.ns(layer+1));
        grad_b{nb_layers-layer}=zeros(nlen(layer),1);
    end
    grad_b{nb_layers}=zeros(K,1);
    n = size(x_batch{nb_layers},2);
    G_batch = -(Ys_batch-P_batch);
    grad_W = 1/n*G_batch*x_batch{nb_layers}';
    grad_b{nb_layers} = 1/n*G_batch*ones(n,1);
    
    G_batch = W'*G_batch;   
    G_batch = G_batch.*(x_batch{nb_layers} > 0);
    
    for layer = nb_layers-1:-1:2
        grad_b{layer} = 1/n*G_batch*ones(n,1);
        for j=1:n
            gj = G_batch(:, j);
            xj= X_batch{2}(:,:,j);
            MX_general = MakeMXMatrixGeneral(xj, hyper_paras.ns(layer), hyper_paras.ks(layer),nlen(layer));
            Gj=reshape(gj,nlen(layer+1),hyper_paras.ns(layer+1));
            V = MX_general'*Gj;
            v=V(:)';
            grad_F{layer}=grad_F{layer}+1/n*v; 
        end
        G_batch = MFs{layer}'*G_batch;
        G_batch = G_batch.*(x_batch{layer} > 0);
    end
    %{
    for j=1:n
        gj = G_batch(:, j);
        xj= X_batch{2}(:,:,j);
        MX_general = MakeMXMatrixGeneral(xj, n1, k2,nlen{2});
        Gj=reshape(gj,nlen{3},n2);
        V = MX_general'*Gj;
        v=V(:)';
        grad_F{2}=grad_F{2}+1/n*v; 
    end
        
    G_batch = MFs{2}'*G_batch;
    G_batch = G_batch.*(x_batch{2} > 0);
    %}
    grad_b{1} = 1/n*G_batch*ones(n,1);   
    for j=1:n
        gj = G_batch(:, j);
        v = gj'* MX1s{j};
        grad_F{1}=grad_F{1}+1/n*v;
    end
end
