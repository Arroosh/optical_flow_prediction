function [] = doCollectiveFlow(firstInd, lastInd, theJpgs, theFolder, C)

    global paramBall;

    curH = eye(3);

    format long;
    first = imread([theFolder '/' theJpgs(firstInd).name]);

    [~, Name, ~] = fileparts(theJpgs(firstInd).name);


    finOptFlow = readFlowFile([theFolder '/' Name '.flo'])*0.0;

    for i = firstInd+1:lastInd
       [~,Name,~] = fileparts(theJpgs(i).name);
       H = load([theFolder '/' Name '.txt']);
       H = reshape(H, [3 3]);
       H = H';
       I = imread([theFolder '/' theJpgs(i).name]);
       curH = curH*inv(H);

       optFlow = readFlowFile([theFolder '/' Name '.flo']);

       [optFlow, theMask] = applyH(optFlow, (curH), ones(size(I,1), size(I,2)));
	   finOptFlow = optFlow + finOptFlow;

       [J, theMask] = applyH(I, (curH), ones(size(I,1), size(I,2)));
       theMask = (~theMask);

       J(:,:,1) = J(:,:,1) + uint8(double(first(:,:,1)).*theMask);   
       J(:,:,2) = J(:,:,2) + uint8(double(first(:,:,2)).*theMask);
       J(:,:,3) = J(:,:,3) + uint8(double(first(:,:,3)).*theMask);

   %    imwrite(J, [theFolder '/' num2str(i) '.jpeg']);
    end

     optFlow = finOptFlow;
     [~, Name, ~] = fileparts(theJpgs(firstInd).name);
     %save([theFolder '/' Name '.mat'], 'optFlow');

     I = imread([theFolder '/' theJpgs(firstInd).name]);
     I = imresize(I, paramBall.canonicalSize);
     imwrite(I, [theFolder '/' Name '.jpg']);
     R = optFlow(:,:,1);
     L = optFlow(:,:,2);
     R = imresize(R, [paramBall.labelDim, paramBall.labelDim]);
     L = imresize(L, [paramBall.labelDim, paramBall.labelDim]);
            
     R = imresize(R, [size(I,1), size(I,2)]);
     L = imresize(L, [size(I,1), size(I,2)]);
     M = -1.0*imresize(R, [size(I,1), size(I,2)]);
            
     theDists = pdist2([R(:) L(:)], C);
     [~,theInds] = min(theDists');
            
     theInds = reshape(theInds, [size(I,1), size(I,2)]);
            
     theDists = pdist2([M(:) L(:)], C);
     [~,theIndsM] = min(theDists');
            
     theIndsM = reshape(theIndsM, [size(I,1), size(I,2)]);
            
     tmp = uint8(theInds);
     tmp(:,:,2) = uint8(theIndsM);
     tmp(:,:,3) = uint8(0*theIndsM);
            
     imwrite(tmp, [theFolder '/' Name '.tif'], 'Compression', 'lzw')

    % I = flowToColor(optFlow);
    % imwrite(I, [theFolder '/' Name '.bmp'])

    format shortEng;
    format compact;

end