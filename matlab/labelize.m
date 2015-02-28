function [ourLabels] = labelize(optFlow,C)

            R = optFlow(:,:,1);
            L = optFlow(:,:,2);
            
            theDists = pdist2([R(:) L(:)], C);
            [~,theInds] = min(theDists');
            
            ourLabels = reshape(theInds, [size(optFlow,1), size(optFlow,2)]);
end