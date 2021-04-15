function MX = MakeMXMatrix(x_input, d, k, nf,nlen)
    flat_dim=k*d;
    MX=zeros((nlen-k+1)*nf,flat_dim*nf);
    for i =0:(nlen-k)
        sub_block_X= x_input(:,i+1:i+k);
        vec_X=sub_block_X(:);
        for j=0:nf-1
            MX(i*nf+1+j,j*flat_dim+1:j*flat_dim+flat_dim)=vec_X;
        end
    end
end