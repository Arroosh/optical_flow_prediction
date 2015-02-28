function [] = loadDeepResults(testResult)
    
    
    theLogLosses = [];
    theLogLossesZero = [];

    theLogEntropy = [];
    theLogEntropyNonZero = [];

    theLogLossesMedian = [];        
    EPE = [];
    EPEZero = [];
    EPERand = [];
    EPENon = [];
    EPERandNon = [];
    EPEZeroNon = [];
    
    global paramBall;
        
    rootFile = [paramBall.caffeDataDir 'theResult'];
    ultraRootFile = paramBall.testVidDir;

    load([paramBall.caffeDataDir '/clusters.mat'], 'C');
    
    fid = fopen(testResult,'r');

    N = zeros((paramBall.vectorClusters*(paramBall.labelDim^2)),1);

    cnt = 0;

    while (~feof(fid))
        %str = '';
        str = fscanf(fid,'%s', 1 );
        subdir = '';
        imageFile = '';
        flag = 1;
        %keyboard;
        
        for i = 1 : numel(str)
            if(str(i) == ' ')
                break;
            end
                imageFile = [imageFile str(i)];
        end
                
        [subdir,Name,~] = fileparts([imageFile])
        if(length(subdir) == 0)
            break
        end
        
        imageSaveFile = [rootFile '/' subdir '/' Name '.jpg']
        
        mkdir([rootFile '/' subdir]);

        for i = 1 : ((paramBall.vectorClusters*(paramBall.labelDim^2)))
            
            N(i) = fscanf(fid, '%f', 1);
        end
        
    N2 = assignToFlowSoft(N, C);
    
    labeledOptFlow = zeros(paramBall.canonicalSize);
    
    labeledOptFlow(:,:,1) = imresize(N2(:,:,1), paramBall.canonicalSize);
    labeledOptFlow(:,:,2) = imresize(N2(:,:,2), paramBall.canonicalSize);
  
    N = reshape(N, [paramBall.vectorClusters paramBall.labelDim paramBall.labelDim]);
    
    G = [ultraRootFile '/' imageFile];
    G = G(1:(end-4));
    P = load([G '.mat'])
    optFlow = P.optFlow;
    tmp = []
    tmp(:,:,1) = imresize(optFlow(:,:,1), paramBall.canonicalSize);
    tmp(:,:,2) = imresize(optFlow(:,:,2), paramBall.canonicalSize);
    optFlow = tmp;
    
    theHeatMap = zeros(size(optFlow,1),size(optFlow,2),paramBall.vectorClusters);

    for k = 1:paramBall.vectorClusters
        curLayer = N(k,:,:);
        tmp = zeros(paramBall.labelDim, paramBall.labelDim);
        for i = 1:paramBall.labelDim
            for j = 1:paramBall.labelDim
                tmp(i,j) = curLayer((i-1)*paramBall.labelDim + j);
            end
        end
        theHeatMap(:,:,k) = imresize(tmp, [size(optFlow,1),size(optFlow,2)], 'nearest');
    end
    [theGT] = labelize(optFlow,C);
    theRinds = randperm(paramBall.vectorClusters);
    theRand = randi(paramBall.vectorClusters, size(optFlow,1), size(optFlow,2));
    theRand(:) = randperm(1);
    tmp = [];
    tmp(:,:,1) = reshape(C(theRand(:),1), paramBall.canonicalSize);
    tmp(:,:,2) = reshape(C(theRand(:),2), paramBall.canonicalSize);
    theRand = tmp;
    tmp = [];

        
    theZeroInds = (theGT(:) ~= paramBall.theZeroCluster);
    
    
    theProbabilities = zeros(size(optFlow,1),size(optFlow,2));
        
    theEntropies = zeros(size(optFlow,1),size(optFlow,2));
        
    EDiff = (optFlow - labeledOptFlow).^2;
    EDiff = sqrt(EDiff(:,:,1)+EDiff(:,:,2));
    
    EDiffRand = (optFlow - theRand).^2;
    EDiffRand = sqrt(EDiffRand(:,:,1)+EDiffRand(:,:,2));

            
    EDiffZero = (optFlow).^2;
    EDiffZero = sqrt(EDiffZero(:,:,1)+EDiffZero(:,:,2));
       
    EPE = [EPE;mean(EDiff(:))];
    EPEZero = [EPEZero;mean(EDiffZero(:))];
    EPERand = [EPERand;mean(EDiffRand(:))];
    EPENon = [EPENon;mean(EDiff(theZeroInds)) length(find(theZeroInds))];
    EPEZeroNon = [EPEZeroNon;mean(EDiffZero(theZeroInds)) length(find(theZeroInds))];
    EPERandNon = [EPERandNon;mean(EDiffRand(theZeroInds)) length(find(theZeroInds))];
    
    for i = 1:size(optFlow,1)
         for j = 1:size(optFlow,2)
                theVector = theGT(i,j);
                theProbabilities(i,j) = theHeatMap(i,j, theVector) + paramBall.minProb;
                distributionVector = theHeatMap(i,j, :);
                distributionVector = distributionVector(:) + paramBall.minProb;              
                theEntropies(i,j) = -sum(distributionVector.*log(distributionVector));
            end
        end
    theLogEntropy = [theLogEntropy;mean(theEntropies(:))];
    theLogEntropyNonZero = [theLogEntropyNonZero;mean(theEntropies(theZeroInds)) length(find(theZeroInds))];


    theLogLoss = mean(log(theProbabilities(:)));
        
    theLogLosses = [theLogLosses;theLogLoss];
    theLogLossesZero = [theLogLossesZero;mean(log(theProbabilities(theZeroInds))) length(find(theZeroInds))];
    
    N2 = assignToFlowSoft(N, C);
    I = imread([ultraRootFile '/' imageFile]);
    rgbOut = uint8(flowToColor(N2));   
    rgbOut = imresize(rgbOut, [size(I,1) size(I,2)]);
        
        
    clear tmp;
    tmp(:,:,1) = [I(:,:,1) rgbOut(:,:,1)];
    tmp(:,:,2) = [I(:,:,2) rgbOut(:,:,2)];
    tmp(:,:,3) = [I(:,:,3) rgbOut(:,:,3)];
        
    try
            G = [ultraRootFile '/' imageFile];
            G = G(1:(end-4));
            P = load([G '.mat'])
            rgbOut = uint8(flowToColor(P.optFlow));   
            rgbOut = imresize(rgbOut, paramBall.canonicalSize);
            tmp2(:,:,1) = [tmp(:,:,1) rgbOut(:,:,1)];
            tmp2(:,:,2) = [tmp(:,:,2) rgbOut(:,:,2)];
            tmp2(:,:,3) = [tmp(:,:,3) rgbOut(:,:,3)];
            tmp = tmp2;            
     catch
     end
        
    imwrite(tmp, imageSaveFile);
    
    theLogLossesMedian = [theLogLosses;median(log(theProbabilities(:)))];
    
    save([imageSaveFile '.mat'], 'theLogLosses', 'theLogLossesMedian', 'theLogEntropy', 'theLogEntropyNonZero',...
        'theLogLossesZero', 'EPE', 'EPEZero', 'EPENon', 'EPEZeroNon', 'EPERand', 'EPERandNon');

    fprintf('%d\n',cnt);
        
    cnt = cnt + 1;

    end
    
    fclose(fid);
 %   end

end

