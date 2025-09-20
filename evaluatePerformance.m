function [precision, recall, f1, mcc, specificity, gmean] = evaluatePerformance(trueLabels, predictedLabels)

%     % ????? ????????? ?????? ?? ???????? ???? ???? classperf
%     trueClass = repmat({''}, size(trueLabels));
%     trueClass(trueLabels == 1) = {'p'};
%     trueClass(trueLabels == 0) = {'n'};
% 
%     predictClass = repmat({''}, size(predictedLabels));
%     predictClass(predictedLabels == 1) = {'p'};
%     predictClass(predictedLabels == 0) = {'n'};

    % ??????? ?? confusionmat
    [confMat, order] = confusionmat(trueLabels, predictedLabels);
    negIdx = find(order == 0);
    posIdx = find(order == 1);
    
% %   cp = classperf(trueClass, predictClass);
%     cp = confusionmat(trueClass, predictClass);
%     cp1 = confusionmat(trueLabels, predictedLabels);

    % ??????? ?????? ?? ???? ??????
    TN = confMat(negIdx, negIdx);
    FP = confMat(negIdx, posIdx);
    FN = confMat(posIdx, negIdx);
    TP = confMat(posIdx, posIdx);

    % ?????? ???????
    precision = TP / (TP + FP + eps);
    recall    = TP / (TP + FN + eps);  % sensitivity
    f1        = 2 * (precision * recall) / (precision + recall + eps);
    mcc       = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + eps);
    specificity = TN / (TN + FP + eps);
    gmean     = sqrt(recall * specificity);

end
