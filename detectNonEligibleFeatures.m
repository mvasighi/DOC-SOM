function nonEligible = detectNonEligibleFeatures(Centers, h, threshold)
    numFeatures = size(Centers, 2);  
    numKernels = size(Centers, 1); 
    
    eli_f = zeros(1, numFeatures);  
    
    for i = 1:numFeatures
        diff = zeros(1, numKernels - 1);  
        for ii = 1:numKernels - 1
            pair1 = Centers(ii, i);
            pair2 = Centers(ii + 1, i);
            if pair1 ~= pair2
                diff(ii) = abs(pair1 - pair2);
            else
                diff(ii) = 1e7;  
            end
        end
        
        eli_f(i) = min(diff) / mean(h(:, i));
    end
    
    meanData = mean(eli_f);
    stdData = std(eli_f);
    zScores = (eli_f - meanData) / stdData;
    
    nonEligible = find(abs(zScores) > threshold);
    
    nonEligible = nonEligible(eli_f(nonEligible) > 1);
end
