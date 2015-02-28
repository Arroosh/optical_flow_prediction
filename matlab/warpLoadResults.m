addpath(genpath('/nfs/hn48/jcwalker/caffe/'));

genParamBall;


paramBall.testVidDir = '/home/jcwalker/OptFlowTest/'
paramBall.caffeDataDir = '/home/jcwalker/OptFlowResults/'

rootFile = [paramBall.caffeDataDir 'theResult'];
ultraRootFile = paramBall.testVidDir;

theTxts = dir([rootFile '/*.txt']);


for i = 1:length(theTxts)
        if(strcmp(theTxts(i).name, 'FlowResult.txt'))
            continue;
        end
    
        if(~isLocked([rootFile '/Lock/' theTxts(i).name '_lock']))
            loadDeepResults([rootFile '/' theTxts(i).name]);
        end
end

