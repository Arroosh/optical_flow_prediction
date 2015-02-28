function breakFile()

    global paramBall;
    
    rootFile = [paramBall.caffeDataDir 'theResult'];
    ultraRootFile = paramBall.vidDir;

    testResult = [rootFile '/FlowResult.txt'];

    load([paramBall.caffeDataDir '/clusters.mat'], 'C');
    
    fid = fopen(testResult,'r');

    N = zeros((paramBall.vectorClusters*(paramBall.labelDim^2)),1);

    cnt = 0;
    while (~feof(fid))
        %str = '';
        str = fgets(fid);
        cnt = cnt + 1;
        fop = fopen([rootFile '/' num2str(cnt) '.txt'],'w');
        fprintf(fop, '%s\n',str);
        fclose(fop)
    end
end