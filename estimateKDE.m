function [pdfValData, pVal] = estimateKDE(valData, kernelCenters, h, hitcount, b, L, numFeatures)

    numSamples = size(valData,1);
    numKernels = size(kernelCenters,1);

    pdfValData = zeros(numFeatures, numSamples);  
    pVal = cell(1, numSamples);                   

    totalHit = sum(hitcount);
    hitcount(hitcount == 0) = eps;  % ??????? ?? ????? ?? ???

    for da = 1:numSamples
        for d = 1:numFeatures
            point = valData(da,d);
            kernels = kernelCenters(:,d);
            hVec = h(:,d);
            px = 0;
            p = zeros(numKernels,1);
            for i = 1:numKernels
                a = hVec(i) * sqrt(2);
                diff = abs(point - kernels(i));
                p(i) = (hitcount(i)/totalHit) * (b/(2 * a * L)) * exp(-(diff/a)^b);
                px = px + p(i);
            end
            pdfValData(d,da) = px;
            pVal{da}(d,:) = p;
        end
    end
end
