

listOfNames = [];
theRars = dir([paramBall.vidDir '/*.rar'])

for j = 1:length(theRars)
        [~, theFile, ~] = fileparts(theRars(j).name);

	system(['rm -rf ' paramBall.vidDir '/' theFile]);
	system(['/home/jcwalker/rar/rar x ' paramBall.vidDir '/' theFile '.rar'])

	theAvis = dir([paramBall.vidDir '/' theFile '/*.avi'])

	for i = 1:length(theAvis)
		system([paramBall.progDir '/ffmpeg -i ' '"' paramBall.vidDir '/' theFile '/' theAvis(i).name...
		'" -c:a copy -c:v copy -s:v ' num2str(paramBall.canonicalSize(2)) 'x' num2str(paramBall.canonicalSize(1)) ' ' paramBall.vidDir '/' theFile '_' num2str(i) '.avi'])
       		listOfNames = [listOfNames; {[theAvis(i).name ' ' theFile '_' num2str(i) '.avi']}];
	end
end
