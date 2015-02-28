addpath(genpath('/nfs/hn48/jcwalker/caffe/'));

genParamBall;

global paramBall;

theAvis = dir([paramBall.vidDir '*.avi']);

load([paramBall.caffeDataDir '/clusters.mat'], 'C');

for i = 1:length(theAvis)
    i
    if(~isLocked([paramBall.vidDir 'Lock/' theAvis(i).name '_flow_lock']))
        [~, AviName, ~] = fileparts(theAvis(i).name);
        homeFolder = [paramBall.vidDir '/' AviName '/images/'];
        theJpgs = dir([homeFolder '/*.jpg']);
        system(['rm ' homeFolder '/*.tif']);
        for j = 1:length(theJpgs)
            
            [~, Name, ~] = fileparts(theJpgs(j).name);
            I = imread([homeFolder Name '.jpg']);
            I = imresize(I, paramBall.canonicalSize);
            imwrite(I, [homeFolder Name '.jpg']);
            P = load([homeFolder Name '.mat']);
            R = P.optFlow(:,:,1);
            L = P.optFlow(:,:,2);
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
            
            imwrite(tmp, [homeFolder Name '.tif'], 'Compression', 'lzw')
        end
    end
end
