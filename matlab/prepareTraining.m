function prepareTraining()

    global paramBall

    fid = fopen([paramBall.caffeDataDir '/prepare.sh'], 'w');

       
    fprintf(fid, '%s', '#!/usr/bin/env sh');
    fprintf(fid, '\n');
    
    fprintf(fid, '%s', ['GLOG_logtostderr=1 ' paramBall.caffeProgDir '/build/examples/opticalflow/convert_flow.bin ' paramBall.vidDir ' ' paramBall.caffeDataDir '/label.txt ' paramBall.caffeDataDir '/opt_train_db 0 1 ' num2str(paramBall.canonicalSize(2)) ' ' num2str(paramBall.canonicalSize(1)) ' 1 0 0 99 101']);
    fprintf(fid, '\n');

    fprintf(fid, '%s', ['GLOG_logtostderr=1 ' paramBall.caffeProgDir '/build/tools/compute_image_mean.bin ' paramBall.caffeDataDir... 
        '/opt_train_db ' paramBall.caffeDataDir '/opt_train_db.binaryproto leveldb']);
    fprintf(fid, '\n');
    fclose(fid);

    
    fid = fopen([paramBall.caffeDataDir '/train.sh'], 'w');

       
    fprintf(fid, '%s', '#!/usr/bin/env sh');
    fprintf(fid, '\n');

    fprintf(fid, '%s', ['GLOG_logtostderr=1 ' paramBall.caffeProgDir '/build/tools/caffe train -solver='...
        paramBall.caffeDataDir '/train.prototxt']);
    fprintf(fid, '\n');
    fclose(fid);

    fid = fopen([paramBall.caffeDataDir '/train.prototxt'], 'w');

    fprintf(fid, '%s', ['train_net: ' '"' paramBall.train_net '"']);
    fprintf(fid, '\n');
    
    fprintf(fid, '%s', ['base_lr: ' num2str(paramBall.base_lr, '%10.6f')]);
    fprintf(fid, '\n');

    fprintf(fid, '%s', ['lr_policy: ' '"' paramBall.lr_policy '"']);
    fprintf(fid, '\n');

    fprintf(fid, '%s', ['gamma: ' num2str(paramBall.gamma)]);
    fprintf(fid, '\n');

    fprintf(fid, '%s', ['stepsize: ' num2str(paramBall.stepsize)]);
    fprintf(fid, '\n');

    fprintf(fid, '%s', ['display: ' num2str(paramBall.display)]);
    fprintf(fid, '\n');

    fprintf(fid, '%s', ['max_iter: ' num2str(paramBall.max_iter)]);
    fprintf(fid, '\n');

    fprintf(fid, '%s', ['momentum: ' num2str(paramBall.momentum)]);
    fprintf(fid, '\n');

    fprintf(fid, '%s', ['weight_decay: ' num2str(paramBall.weight_decay)]);
    fprintf(fid, '\n');

    fprintf(fid, '%s', ['snapshot: ' num2str(paramBall.snapshot)]);
    fprintf(fid, '\n');

    fprintf(fid, '%s', ['snapshot_prefix: ' '"' paramBall.snapshot_prefix '"']);
    fprintf(fid, '\n');

    fprintf(fid, '%s', ['solver_mode: '  paramBall.solver_mode]);
    fprintf(fid, '\n');

    fprintf(fid, '%s', ['device_id: '  num2str(paramBall.device_id)]);
    fprintf(fid, '\n');

    fclose(fid);

end
