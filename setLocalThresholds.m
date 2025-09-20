function localThreshold = setLocalThresholds(pCenters, gammaValIter, net)
    % local threshold scaling for SOM neurons
    
    numCenters   = size(pCenters, 2);        
    numFeatures  = size(pCenters{1}, 1);     
    localThreshold = zeros(numFeatures, numCenters);

    neuronPos = net.W';                       % (numCenters x numFeatures)
    lambda = 0.5;                             % scaling strength

    % Compute angular variance for each neuron
    abodVar = zeros(1, numCenters);

    for j = 1:numCenters
        mu_j = neuronPos(j, :);
        othersIdx = setdiff(1:numCenters, j);
        vecs = neuronPos(othersIdx, :) - mu_j;  % (numOthers x numFeatures)

        numOthers = size(vecs, 1);
        cosVals = [];
        for a = 1:numOthers-1
            for b = a+1:numOthers
                va = vecs(a, :);
                vb = vecs(b, :);
                na = norm(va);
                nb = norm(vb);
                if na > 0 && nb > 0
                    cosTheta = dot(va, vb) / (na * nb);
                    cosVals(end+1) = cosTheta; %#ok<AGROW>
                end
            end
        end

        % Variance of cosine values = ABOD score
        if isempty(cosVals)
            abodVar(j) = 0;
        else
            abodVar(j) = var(cosVals);
        end
    end

    % Normalize and invert so that low variance → high value
    if max(abodVar) > 0
        abodNorm = abodVar / max(abodVar);
        abodInv  = 1 - abodNorm;  % inversion
    else
        abodNorm = abodVar;
        abodInv  = abodVar;  % all zeros → inversion also zero
    end

    % Apply scaling factors to thresholds
    for j = 1:numCenters
        c_j = 1 + lambda * abodInv(j);  % low variance → higher scaling
        localThreshold(:, j) = gammaValIter .* pCenters{j}(:, j) .* c_j;
    end
end
