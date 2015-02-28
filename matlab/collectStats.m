rootFile = [paramBall.caffeDataDir 'theResult'];

theFolders = dir([rootFile '/v_*']);

theLogLosses  =[]
theLogEntropies = [];
theLogLossesZero = [];
EPEs = [];
EPENons = [];
EPERands = [];
EPERandNons = [];
count = 0;
for i = 1:length(theFolders)
    theMats = dir([rootFile '/' theFolders(i).name '/images/*.mat']);
    i
    for j = 1:length(theMats)
            P = load([rootFile '/' theFolders(i).name '/images/' theMats(j).name]);
            theLogLosses  =[theLogLosses;P.theLogLosses];
            theLogEntropies = [theLogEntropies;P.theLogEntropy];
            theLogLossesZero = [theLogLossesZero;P.theLogLossesZero];
            EPEs = [EPEs;P.EPE];
            EPENons = [EPENons;P.EPENon];
            EPERands = [EPERands;P.EPERand];
            EPERandNons = [EPERandNons;P.EPERandNon];
    end
    
end

HH = find(theLogLossesZero(:,2) ~= 0)
size(theLogLosses)
mean(theLogLosses)
median(theLogLosses)
mean(theLogLossesZero(HH,1))
median(theLogLossesZero(HH,1))
mean(EPEs)
median(EPEs)
mean(EPENons(HH,1))
median(EPENons(HH,1))
[~,inds] = sort(theLogEntropies);
finalY = [];
for i = 101:(length(inds)-100)
	finalY = [finalY;mean(theLogLosses(inds(i:i+100)))];
end
