function h = computeBandwidth(kernelCenters, adnw, adnn, coeffVal, approachType)
    numKernels = size(kernelCenters, 1);   
    numDimensions = size(adnw, 2);         

    h = zeros(numKernels, numDimensions);  
    
    for i = 1:numKernels
        for ii = 1:numDimensions
            if approachType == 1
                h(i,ii) = coeffVal * ((adnw(i,ii) * 0.5) + (adnn(i,ii) * 0.5));
            elseif approachType == 2
                h(i,ii) = coeffVal * min([adnw(i,ii), adnn(i,ii)]);
            else
                h(i,ii) = coeffVal * nthroot(prod([adnw(i,ii), adnn(i,ii)]), 2);
            end
        end
    end
end
