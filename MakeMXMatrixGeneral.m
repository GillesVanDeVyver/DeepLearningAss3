function MX = MakeMXMatrixGeneral(X_input, size1, size2,nlen)
    flat_dim=size2*size1;
    MX=zeros((nlen-size2+1),flat_dim);
    for i =1:nlen-size2+1
        sub_block_X= X_input(:,i:i+size2-1);
        vec_X=sub_block_X(:);
        MX(i,:)=vec_X;
    end
end