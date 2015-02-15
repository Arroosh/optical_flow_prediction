function [N2] = assignToFlowSoft(N, codebook)

    global paramBall;
    size(N)
    channelNum = size(codebook, 1);
    Nv = reshape(N, channelNum, []);
    size(Nv)
    
    for i = 1:size(Nv, 2)
        NvX(i) = sum(Nv(:,i).*codebook(:,1));
        NvY(i) = sum(Nv(:,i).*codebook(:,2));
    end
        
    N2(:,:,1) = reshape(NvX, [paramBall.labelDim paramBall.labelDim]);
    N2(:,:,2) = reshape(NvY, [paramBall.labelDim paramBall.labelDim]);
    
    for i = 1:paramBall.labelDim
        for j = 1:paramBall.labelDim
            N2(i,j,1) = NvX((i-1)*paramBall.labelDim + j);
            N2(i,j,2) = NvY((i-1)*paramBall.labelDim + j);
        end
    end
    
    
end


