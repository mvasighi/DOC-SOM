close all;
clear;
clc;

%% Load Data    
datasetList = dir('datasets/*.mat');
fprintf('Available datasets:\n');
for i = 1:length(datasetList)
    fprintf('%d. %s\n', i, datasetList(i).name);
end

datasetIndex = input('Select a dataset by number: ');
while datasetIndex < 1 || datasetIndex > length(datasetList)
    fprintf('Invalid dataset number. Please choose between 1 and %d.\n', length(datasetList));
    datasetIndex = input('Select a dataset by number: ');
end
load(fullfile('datasets', datasetList(datasetIndex).name));  % Assumes it loads X and class

%% Detect and Separate Classes
classLabels = unique(class);
numClasses  = numel(classLabels);

fprintf('\nAvailable class labels:\n');
for i = 1:numClasses
    fprintf('%d. Class %d\n', i, classLabels(i));
end

targetClass = input('Select the target class by number: ');
while targetClass < 1 || targetClass > numClasses
    fprintf('Invalid class number. Please choose between 1 and %d.\n', numClasses);
    targetClass = input('Select the target class by number: ');
end


iterationCounter = 0;

% Start training phase
for repeatIdx = 1:10


    % Separate into base and other classes
    baseData  = X(class == targetClass, :);
    otherData = X(class ~= targetClass, :);

    %% Network Configuration
    netConfig = setting('dbgsom');
    netConfig.epch = 100;
    netConfig.sf   = 0.4;

    %% Parameters
    sampleFraction   = 0.1;
    approachType     = 3;
    maxGMean         = 0;
    coefficientRange = 0.1:0.1:2;
    gammaVal         = 0.05;
    kernelShapeParam = 2;
    normConstant     = gamma(1/kernelShapeParam);


    % Sample and Merge Data
    sampleSize = round(sampleFraction * size(baseData, 1));
    randIdx = randperm(size(otherData, 1), sampleSize);
    mainData = [baseData; otherData(randIdx, :)];
    mainData(:, all(diff(mainData) == 0)) = []; % Remove constant columns
    mainLabels = [ones(size(baseData, 1), 1); zeros(sampleSize, 1)];

    % Dataset Info
    datasetInfo = struct('numClasses', numel(unique(class)), 'numFeatures', size(mainData, 2));
    numSamples = size(mainData, 1);

    % Normalize Data
    [mainData, ~] = prefun(mainData, 'rs');

    % Split Train/Test
    testIdx = 1:3:numSamples;
    trainIdx = setdiff(1:numSamples, testIdx);

    trainData = mainData(trainIdx, :);
    testData  = mainData(testIdx, :);

    trainLabels = mainLabels(trainIdx);
    testLabels  = mainLabels(testIdx);
    numTrainSamples = size(trainData, 1);

%% Cross-validation (10-fold)
    for coeffVal = coefficientRange

        valLabelsAll = [];
        predictionAll = [];

        iterationCounter = iterationCounter + 1;
        fprintf('[Done %d ,  Repeat index %d]\n',iterationCounter, repeatIdx)

        for fold = 1:10
            % Cross-validation split
            valIdx = fold:10:numTrainSamples;
            valData = trainData(valIdx, :);
            valLabels = trainLabels(valIdx);

            trainFoldIdx = setdiff(1:numTrainSamples, valIdx);
            trainFoldData = trainData(trainFoldIdx, :);

            valLabelsAll = [valLabelsAll; valLabels];

            % Train Network
            network = dbgsom(trainFoldData, netConfig);
            deadNodes = find(network.hitcount == 0);
            kernelCenters = network.W';

            % find Closest Neurons
            closestNeurons = findClosestNeurons(valData, kernelCenters, deadNodes);

            % Calculate Average Distances
            adnw = MeanVoronoiWeightDistance(network.W, network.winlist, trainFoldData);
            adnn = MeanNeighborDistance(network.W, network.grd, deadNodes);
            adnw = adnw + eps;
            adnn = adnn + eps;

            % Compute bandwidth h
            h = computeBandwidth(kernelCenters, adnw, adnn, coeffVal, approachType);

            % Identify Non-Eligible Features
            threshold = 2;
            nonEligible = detectNonEligibleFeatures(kernelCenters, h, threshold);

            % KDE Probability Estimation
            [pdfValData, ~] = estimateKDE(valData, kernelCenters, h, network.hitcount, kernelShapeParam, normConstant, datasetInfo.numFeatures);
            [pdfKernels, pCenters] = estimateKDE(kernelCenters, kernelCenters, h, network.hitcount, kernelShapeParam, normConstant, datasetInfo.numFeatures);

            % Set Local Thresholds
            localThreshold = setLocalThresholds(pCenters, gammaVal, network);

            % Final Decision based on thresholds
            selectedFeatures = setdiff(1:datasetInfo.numFeatures, nonEligible);
            predictions = makePredictions(pdfValData, localThreshold, closestNeurons, selectedFeatures);

            predictionAll = [predictionAll; predictions'];

        end % fold

        % Evaluate Performance
        [~, ~, ~, ~, ~, gmean] = evaluatePerformance(valLabelsAll, predictionAll);


        % Save Best
        if gmean > maxGMean
            bestCoeff = coeffVal;
            maxGMean = gmean;
        end

    end % coeff


% === Final Training Phase ===

% Train final network on the entire training set
finalNet = dbgsom(trainData, netConfig);

% Identify dead neurons in the trained network
deadNeurons = find(finalNet.hitcount == 0);

% Extract kernel centers (weights)
kernelCenters = finalNet.W';


% === Test Phase ===

% Find closest neurons in the network for each test sample
closestNeuronIdx = findClosestNeurons(testData, kernelCenters, deadNeurons);

% Calculate average Voronoi and neighbor distances using training data
voronoiDist = MeanVoronoiWeightDistance(finalNet.W, finalNet.winlist, trainData);
neighborDist = MeanNeighborDistance(finalNet.W, finalNet.grd, deadNeurons);
voronoiDist = voronoiDist + eps;
neighborDist = neighborDist + eps;


% Compute bandwidth h for KDE using optimal coefficient from training
h_test = computeBandwidth(kernelCenters, voronoiDist, neighborDist, bestCoeff, approachType);

% Detect non-eligible (redundant or irrelevant) features
excludedFeatures = detectNonEligibleFeatures(kernelCenters, h_test, 2);

% Estimate KDE for test samples
[pdf_test_samples, ~] = estimateKDE(testData, kernelCenters, h_test, finalNet.hitcount, kernelShapeParam, normConstant, datasetInfo.numFeatures);

% Estimate KDE for kernel centers (used to compute thresholds)
[pdf_kernel_centers, pCenters] = estimateKDE(kernelCenters, kernelCenters, h_test, finalNet.hitcount, kernelShapeParam, normConstant, datasetInfo.numFeatures);


% Set local thresholds for each neuron
localThreshold = setLocalThresholds(pCenters, gammaVal, finalNet);


% Select only valid features for final prediction
validFeatures = setdiff(1:datasetInfo.numFeatures, excludedFeatures);

% Generate predictions for the test set
testPredictions = makePredictions(pdf_test_samples, localThreshold, closestNeuronIdx, validFeatures);


% Evaluate performance on the test set
[precision, recall, f1, mcc, specificity, gmean] = evaluatePerformance(testLabels, testPredictions);


% Store final performance metrics
performance(repeatIdx,:) = [bestCoeff, precision, recall, f1, mcc, specificity, gmean];
    
end


% Column names
columnNames = {'', 'Precision', 'Recall', 'F1', 'MCC', 'Specificity', 'G-Mean'};
rowNames = {'Combination with highest average F-score'; 'Combination with highest average G-mean'; 'Average across all combinations'; 'Standard deviation across all combinations'};

% Calculate statistics excluding BestCoeff (i.e., columns 2 to end)
meanVals = mean(performance(:,2:end), 1);
stdVals  = std(performance(:,2:end), 0, 1);

[~, iBestF1]  = max(performance(:,4));
[~, iBestGM]  = max(performance(:,7));
maxValsBasedF1  = performance(iBestF1,2:end);
maxValsBasedGM  = performance(iBestGM,2:end);

% cellData = num2cell([maxValsBasedF1; maxValsBasedGM; meanVals; stdVals]);
cellData = [rowNames, num2cell([maxValsBasedF1; maxValsBasedGM; meanVals; stdVals])];

res = cell2table(cellData,'VariableNames',{' ','Precision', 'Recall', 'F1', 'MCC', 'Specificity', 'G-Mean'})