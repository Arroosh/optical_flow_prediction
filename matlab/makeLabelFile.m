function [] = makeLabelFile()

global paramBall;

theAvis = dir([paramBall.vidDir '*.avi']);

fid = fopen([paramBall.caffeDataDir '/label.txt'], 'w');


for i = 1:length(theAvis)
    [~, AviName, ~] = fileparts(theAvis(i).name);
    homeFolder = [paramBall.vidDir '/' AviName '/images/'];
    theJpgs = dir([homeFolder '/*.jpg']);
    i
       
    for j = 1:length(theJpgs)
        fprintf(fid, '%s ', [AviName '/images/'  theJpgs((j)).name]);
        fprintf(fid, '\n');
    end

end

fclose(fid);

end
