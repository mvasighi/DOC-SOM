function predictions = makePredictions(pdfValData, localThreshold, closestNeurons, selectedFeatures)
    numSamples = size(pdfValData, 2);
    predictions = zeros(1, numSamples);
    for t = 1:numSamples
        m = pdfValData(selectedFeatures, t) > localThreshold(selectedFeatures, closestNeurons(t));
        predictions(t) = sign(double(all(m)));
    end
end