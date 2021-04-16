function MF = MakeMFMatrix(F, input_size)
    [dd, k, nf] = size(F);
    MF=zeros((input_size-k+1)*nf,input_size*dd);
    offset=1;
    VF=zeros(nf,dd*k);
    for i = 1:nf
        Fi=F(:,:,i);
        VF(i,:)= Fi(:)';
    end
    for i =1:nf:(input_size-k+1)*nf
        MF(i:i+nf-1,offset:offset+dd*k-1)=VF;
        offset=offset+dd;
    end
    
    
end

