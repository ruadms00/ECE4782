clear;
clc;

%% Load Dataset
labels = load("totalLabel.mat").label;
total = load("totalFeature.mat").total;
featureName = load("featureName.mat").featureName1;
%% Split
% Split data into Train & Test
cv = cvpartition(size(total, 1), 'HoldOut', 0.3);
idx = cv.test;
Xtrain = total(~idx,:);
Ytrain = labels(~idx,:);
Xtest = total(idx,:);
Ytest = labels(idx,:);
save('Xtrain.mat', 'Xtrain');
save('Ytrain.mat', 'Ytrain');
save('Xtest.mat', 'Xtest');
save('Ytest.mat', 'Ytest');
%% Split - K-Fold
% Split data into K-Fold(10)

cvKF10 = cvpartition(size(Xtrain, 1), 'KFold', 21);
save('cvKF10.mat','cvKF10');


%% MRMR :)
[idxMRMR, scores] = fscmrmr(total, labels);
MRMR = [idxMRMR; scores];
save('MRMR.mat', 'MRMR');
%% MRMR - Score Analysis :)
%figure; plot([1:length(scores)], scores(idxMRMR))
title("MRMR Scores")
xlabel("Number of Features")
ylabel("Score")
for i = 1:21
    fprintf('%s\n',featureName(idxMRMR(i),:));
    fprintf('%d\n',scores(:,idxMRMR(i)));
end
%% KNN(MRMR): Find Optimal K :)
% Accuracy
knnMRMR_preds = [];
Ks = [1:2:51];
for k=Ks
    knnMRMR = fitcknn(Xtrain(:,idxMRMR(2:5)),Ytrain,'NumNeighbors',k,'Standardize',1);
    knnMRMR_pred = predict(knnMRMR, Xtest(:, idxMRMR(2:5)));
    knnMRMR_preds = [knnMRMR_preds, mean(knnMRMR_pred == Ytest)];
end
%% KNN(MRMR): Visualiztion - Accuracy :) pick 10
%figure; plot(Ks, knnMRMR_preds);
title("KNN w/ MRMR")
xlabel("k")
ylabel("Accuracy")
disp(max(knnMRMR_preds));