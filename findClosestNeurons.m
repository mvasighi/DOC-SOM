function closestNeurons = findClosestNeurons(valData, kernelCenters, deadNeurons)
    closestNeurons = zeros(size(valData,1), 1);
    
    for sampleIdx = 1:size(valData, 1)
        distances = sqrt(sum((valData(sampleIdx,:) - kernelCenters).^2, 2)); 
        [~, closestIdx] = min(distances);
        
        while ismember(closestIdx, deadNeurons)
            distances(closestIdx) = inf;
            [~, closestIdx] = min(distances);
        end
        
        closestNeurons(sampleIdx) = closestIdx;
    end
end
