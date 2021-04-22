function [grad_W, grad_F,grad_b] = ComputeGradients(X_batch,x_batch, Ys_batch, P_batch, W,MFs,d,k1,n1,k2,n2,nlen,MX1s,K)
    grad_F={zeros(1,d*k1*n1),zeros(1,n1*k2*n2)};
    grad_b={zeros(nlen{2},1),zeros(nlen{1},1),zeros(K,1)};
    n = size(x_batch{3},2);
    G_batch = -(Ys_batch-P_batch);
    grad_W = 1/n*G_batch*x_batch{3}';
    grad_b{3} = 1/n*G_batch*ones(n,1);
    G_batch = W'*G_batch;   
    G_batch = G_batch.*(x_batch{3} > 0);
    grad_b{2} = 1/n*G_batch*ones(n,1);

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
    grad_b{1} = 1/n*G_batch*ones(n,1);
        
    for j=1:n
        gj = G_batch(:, j);
        v = gj'* MX1s{j};
        grad_F{1}=grad_F{1}+1/n*v;
    end
end
