function MX = MakeMXMatrix(X_input, size1, size2, size3,nlen)
    flat_dim=size2*size1;
    MX=zeros((nlen-size2+1)*size3,flat_dim*size3);
    for i =0:(nlen-size2)
        sub_block_X= X_input(:,i+1:i+size2);
        vec_X=sub_block_X(:);
        for j=0:size3-1
            MX(i*size3+1+j,j*flat_dim+1:j*flat_dim+flat_dim)=vec_X;
        end
    end
end