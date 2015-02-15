function [C, bigData] = makeVectorDictionary()

global paramBall;

theAvis = dir([paramBall.vidDir '*.avi']);

bigData = cell(length(theAvis), 1);
count = 0;
for i = 1:length(theAvis)
    [~, Name, ~] = fileparts(theAvis(i).name);
    homeFolder = [paramBall.vidDir '/' Name '/images/'];
    theOptFlow = dir([homeFolder '/*.mat']);
    count = count + length(theOptFlow)
    [count i ]
    try
    theInds = randperm(size(theOptFlow));
    
    j = theInds(1);
    
    P = load([homeFolder '/' theOptFlow(j).name]);
    R = P.optFlow(:,:,1);
    L = P.optFlow(:,:,2);
    R = R(:);
    L = L(:);
    randPix = randperm(length(R));
    R = R(randPix(1:paramBall.randPixSample));
    L = L(randPix(1:paramBall.randPixSample));
    bigData{i} = [R(:) L(:)];
    catch
    end
end

bigData = cell2mat(bigData);

[idx, C] = kmeans(bigData, paramBall.vectorClusters);

save([paramBall.caffeDataDir '/clusters.mat'], 'C', 'paramBall');
save([paramBall.caffeDataDir '/bigData.mat'], 'bigData', 'paramBall');

quiver(C(:,1)*0.0, C(:,2)*0.0, C(:,1), C(:,2))

end
