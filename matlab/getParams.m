function [] = getParams(iter)

    global paramBall

    system(['GLOG_logtostderr=1 ' paramBall.caffeProgDir '/build/tools/showParameters.bin '  paramBall.test_net ' '...
           paramBall.snapshot_prefix '_iter_' num2str(iter) '.caffemodel']);

    fid = fopen([paramBall.caffeDataDir '/firstParam.txt'], 'rb');
    size = fread(fid, 1, 'int32');
    W = fread(fid, size, 'float');
    num_filters = paramBall.num_filters;
    filter_size = paramBall.filter_size;
    
    h = 4;
    num_channels = 3;
    W = W - min(W(:));
    W = W./max(W(:))*256;

    W = reshape(W, num_filters, filter_size, filter_size, num_channels);
    im = zeros(h*(filter_size+1)+1, num_filters/h*(filter_size+1)+1, num_channels)*128;
    k = 1;
    for y = 1:h
        for x = 1:num_filters/h
            startx = x*(filter_size+1)-filter_size+1;
            starty = y*(filter_size+1)-filter_size+1;
            a = W(k,:,:,:);
            im(starty:starty+filter_size-1, startx:startx+filter_size-1, :) = ...
                reshape(a, filter_size, filter_size, num_channels);
            k = k + 1;
        end
    end
    I = uint8(im);
    imwrite(I, [paramBall.caffeDataDir '/theParams.jpg']);
    fclose(fid);

end