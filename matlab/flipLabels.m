addpath(genpath('/nfs/hn48/jcwalker/caffe/'));

genParamBall


    global paramBall;

    theAvis = dir([paramBall.vidDir '*.avi']);

    for i = 1:length(theAvis)
       if(~isLocked([paramBall.vidDir 'Lock/' theAvis(i).name '_flow_lock']))
        [~, AviName, ~] = fileparts(theAvis(i).name);
        homeFolder = [paramBall.vidDir '/' AviName '/images/'];
        system(['rm ' homeFolder '/*.jpg_*']);
        theJpgs = dir([homeFolder '/*.jpg']);
        i
        for j = 1:length(theJpgs)
            [~, Name, ~] = fileparts(theJpgs(j).name);
            P = load([homeFolder Name '.mat']);
            
            optFlow = flipdim(P.optFlow, 2);
            optFlow(:,:,1) = -1.0*optFlow(:,:,1);
            
            I = imread([homeFolder Name '.jpg']);
            J = flipdim(I, 2);
            imwrite(J, [homeFolder theJpgs((j)).name '_flip.jpg']);
            save([homeFolder theJpgs((j)).name '_flip.mat'], 'optFlow');
            %rgbOut = uint8(flowToColor(optFlow));
            %imwrite(rgbOut, [homeFolder theJpgs((j)).name '_flip.bmp']);           
        end
	end
    end

