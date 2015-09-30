function [] = makeArrowFlowFigure(optFlow, multiple, interval, lineWidth, headMult, maxSize, magThresh, image)

    %Some of this code was adapted from the flowToColor code.
    bgrd = flowToColor(optFlow);
    image = uint8(.8*image + .2*255.0);
    theMags = ((sqrt(optFlow(:,:,1).^2 + optFlow(:,:,2).^2)));
    theMags = mat2gray(theMags);
    
    bgrd(:,:,1) = uint8((1-theMags).*double(image(:,:,1))) + uint8(theMags.*double(bgrd(:,:,1)));
    bgrd(:,:,2) = uint8((1-theMags).*double(image(:,:,2))) + uint8(theMags.*double(bgrd(:,:,2)));
    bgrd(:,:,3) = uint8((1-theMags).*double(image(:,:,3))) + uint8(theMags.*double(bgrd(:,:,3)));

    
    UNKNOWN_FLOW_THRESH = 1e9;
    UNKNOWN_FLOW = 1e10;            % 

    [XX,YY] = meshgrid(1:size(optFlow,2), 1:size(optFlow,1));
    
    u = optFlow(:,:,1);
    v = optFlow(:,:,2);

    maxu = -999;
    maxv = -999;

    minu = 999;
    minv = 999;
    maxrad = -1;

% fix unknown flow
    idxUnknown = (abs(u)> UNKNOWN_FLOW_THRESH) | (abs(v)> UNKNOWN_FLOW_THRESH) ;
    u(idxUnknown) = 0;
    v(idxUnknown) = 0;

    maxu = max(maxu, max(u(:)));
    minu = min(minu, min(u(:)));

    maxv = max(maxv, max(v(:)));
    minv = min(minv, min(v(:)));

    rad = sqrt(u.^2+v.^2);
    maxrad = max(maxrad, max(rad(:)));

    u = u/(maxrad+eps);
    v = v/(maxrad+eps);
    
    theVects(:,:,1) = XX;
    theVects(:,:,2) = YY;
    theVects(:,:,3) = u;
    theVects(:,:,4) = v;
    
    theFinalVects = [0 0 0 0];
    
    for i = 1:interval:size(optFlow,1)
        for j = 1:interval:size(optFlow,2)
            if(norm([theVects(i,j,3) theVects(i,j,4)]) < magThresh)
                continue;
            end
            theFinalVects = [theFinalVects; theVects(i,j,1) theVects(i,j,2) theVects(i,j,3) theVects(i,j,4)];
        end
    end
    imshow(bgrd);
    hold on
    handle=quiver(theFinalVects(:,1), theFinalVects(:,2),theFinalVects(:,3),theFinalVects(:,4), 'Color', ...
        'Black', 'LineWidth', lineWidth, 'AutoScale', 'on', 'ShowArrowHead', 'on', 'MaxHeadSize', maxSize,...
        'AutoScaleFactor', multiple);

    %Some of this code was adapted from this site:
    %http://stackoverflow.com/questions/22911594/matlab-how-to-fill-quiver-arrow-heads
    
    children=get(handle,'children'); % retrieve the plot-children - 
                                 % second element are the arrow tips

    XData=get(children(2),'XData'); % retrieve the coordinates of the tips
    YData=get(children(2),'YData');

    hold on
    delete(children(2))  % delete old arrow tips

    for l=1:4:length(XData)-3   % paint new arrow tips, skipping the NaN-values
        
        [a,b] = min(XData(l:l+2));
        [c,d] = max(XData(l:l+2));
        
        theWidth = c - a;
        
        XData(l-1+b) = XData(l-1+b) - headMult*theWidth;
        XData(l-1+d) = XData(l-1+d) + headMult*theWidth;
        
        [a,b] = min(YData(l:l+2));
        [c,d] = max(YData(l:l+2));
        
        theWidth = c - a;
        
        YData(l-1+b) = YData(l-1+b) - headMult*theWidth;
        YData(l-1+d) = YData(l-1+d) + headMult*theWidth;



        ArrowTips((l-1)/4+1)=fill(XData(l:l+2),YData(l:l+2),'black');
    end
end
