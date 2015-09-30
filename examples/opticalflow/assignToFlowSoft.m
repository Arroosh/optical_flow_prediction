function [N2] = assignToFlowSoft(N, codebook)

    size(N)
    channelNum = size(codebook, 1);
    Nv = reshape(N, channelNum, []);
    size(Nv)
    
    for i = 1:size(Nv, 2)
        NvX(i) = sum(Nv(:,i).*codebook(:,1));
        NvY(i) = sum(Nv(:,i).*codebook(:,2));
    end
        
    N2(:,:,1) = reshape(NvX, [20 20]);
    N2(:,:,2) = reshape(NvY, [20 20]);
    
    for i = 1:20
        for j = 1:20
            N2(i,j,1) = NvX((i-1)*20 + j);
            N2(i,j,2) = NvY((i-1)*20 + j);
        end
    end
    
    
end


