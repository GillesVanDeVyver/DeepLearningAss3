function MFs = MakeMFMatrices(ConvNet, nlen)
    nb_layers = size(nlen,1);
    MFs = cell(nb_layers-1,1);
    for layer =1:nb_layers-1
        input_size = nlen(layer);
        [dd, k, nf] = size(ConvNet.F{layer});
        MFs{layer}=zeros((input_size-k+1)*nf,input_size*dd);
        offset=1;
        VF=zeros(nf,dd*k);
        for i = 1:nf
            Fi=ConvNet.F{layer}(:,:,i);
            VF(i,:)= Fi(:)';
        end
        for i =1:nf:(input_size-k+1)*nf
            MFs{layer}(i:i+nf-1,offset:offset+dd*k-1)=VF;
            offset=offset+dd;
        end
        
    end
    

    
    
end


