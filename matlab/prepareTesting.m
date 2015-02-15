function prepareTesting(iter)

    global paramBall

    fid = fopen([paramBall.caffeDataDir '/prepareTest.sh'], 'w');

    fprintf(fid, '%s', '#!/usr/bin/env sh');
    fprintf(fid, '\n');
    
    fprintf(fid, '%s', ['GLOG_logtostderr=1 ' paramBall.caffeProgDir '/build/examples/opticalflow/convert_normal.bin ' paramBall.testVidDir ' ' paramBall.caffeDataDir '/labeltest.txt ' paramBall.caffeDataDir '/opt_test_db 0 0 ' num2str(paramBall.canonicalSize(2)) ' ' num2str(paramBall.canonicalSize(1)) ' 1 0 0 99 101']);
    fprintf(fid, '\n');

    fclose(fid);

    
    fid = fopen([paramBall.caffeDataDir '/test.sh'], 'w');

       
    fprintf(fid, '%s', '#!/usr/bin/env sh');
    fprintf(fid, '\n');
    

    fprintf(fid, '%s', ['GLOG_logtostderr=1 ' paramBall.caffeProgDir '/build/examples/opticalflow/test_net_flow.bin '...
        paramBall.test_net ' ' paramBall.snapshot_prefix '_iter_' num2str(iter) '.caffemodel ' paramBall.caffeDataDir ...
        '/labeltest.txt ' paramBall.caffeDataDir '/theResult ' num2str(paramBall.vectorClusters) ' ' num2str(paramBall.labelDim) ]);
    fprintf(fid, '\n');
    fclose(fid);

end
