function loadResults()

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
        str = fscanf(fid,'%s', 1 );
        subdir = '';
        imageFile = '';
        flag = 1;
        %keyboard;
        
        for i = 1 : numel(str)
            if(str(i) == ' ')
                break;
            end
                imageFile = [imageFile str(i)];
        end
                
        [subdir,Name,~] = fileparts([imageFile])
        imageSaveFile = [rootFile '/' subdir '/' Name '.jpg']
        
        mkdir([rootFile '/' subdir]);

        for i = 1 : ((paramBall.vectorClusters*(paramBall.labelDim^2)))
            i
            N(i) = fscanf(fid, '%f', 1);
        end
        
        N2 = assignToFlowSoft(N, C);
        I = imread([ultraRootFile '/' imageFile]);
        rgbOut = uint8(flowToColor(N2));   
        rgbOut = imresize(rgbOut, [size(I,1) size(I,2)]);
        
        
        clear tmp;
        tmp(:,:,1) = [I(:,:,1) rgbOut(:,:,1)];
        tmp(:,:,2) = [I(:,:,2) rgbOut(:,:,2)];
        tmp(:,:,3) = [I(:,:,3) rgbOut(:,:,3)];
        
        try
            G = [ultraRootFile '/' imageFile];
            G = G(1:(end-4));
            P = load([G '.mat'])
            rgbOut = uint8(flowToColor(P.optFlow));   
            rgbOut = imresize(rgbOut, paramBall.canonicalSize);
            tmp2(:,:,1) = [tmp(:,:,1) rgbOut(:,:,1)];
            tmp2(:,:,2) = [tmp(:,:,2) rgbOut(:,:,2)];
            tmp2(:,:,3) = [tmp(:,:,3) rgbOut(:,:,3)];
            tmp = tmp2;            
        catch
        end
        
        imwrite(tmp, imageSaveFile);
        fprintf('%d\n',cnt);
        
        tmp = [I(:,:,1) rgbOut(:,:,1)];

        cnt = cnt + 1;

    end
    
    fclose(fid);
    
end


