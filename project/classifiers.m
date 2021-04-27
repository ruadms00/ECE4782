clear;
clc;

%% Load Dataset
labels = load("totalLabel.mat").label;
total = load("totalFeature.mat").total;
featureName = load("featureName.mat").featureName1;
%% Split
% Split data into Train & Test
cv = cvpartition(size(total, 1), 'Leaveout', 'on');
idx = cv.test;
Xtrain = total(~idx,:);
Ytrain = labels(~idx,:);
Xtest = total(idx,:);
Ytest = labels(idx,:);
save('Xtrain.mat', 'Xtrain');
save('Ytrain.mat', 'Ytrain');
save('Xtest.mat', 'Xtest');
save('Ytest.mat', 'Ytest');

%% MRMR :)
[idxMRMR, scores] = fscmrmr(total, labels);
MRMR = [idxMRMR; scores];
save('MRMR.mat', 'MRMR');
%% MRMR - Score Analysis :)
figure; plot([1:length(scores)], scores(idxMRMR));
xlim([0 15]);
title("MRMR Scores")
xlabel("Number of Features")
ylim
ylabel("Score")
for i = 1:21
    %fprintf('%s\n',featureName(idxMRMR(i),:));
    %fprintf('%d\n',scores(:,idxMRMR(i)));
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
figure; plot(Ks, knnMRMR_preds);
title("KNN w/ MRMR")
xlabel("k")
ylabel("Accuracy")
disp(max(knnMRMR_preds));

%%

% %% KNN(MRMR): Result Analysis :)
% % Accuracy 
knnMRMR_best = fitcknn(Xtrain(:,idxMRMR(1:5)), Ytrain,'NumNeighbors',10,'Standardize',1, );
knnMRMR_pred_best = predict(knnMRMR_best, Xtrain(:,idxMRMR(1:5)));
figure;
disp(mean(knnMRMR_pred_best == Ytest));
cm10 = confusionchart(Ytest, knnMRMR_pred_best);
title('Confusion Matrix: KNN MRMR')

% %% Random Forest
% % 
% rfMRMR = fitensemble(Xtrain(:,idxMRMR(1:5)),Ytrain, 'Bag', 100, 'Tree', 'Type', 'classification', CVPartition',cvKF10);
% rfMRMR_pred = kfoldPredict(knnMRMR_best);
% disp(mean(rfMRMR_pred == Ytest));
% 
% %% Random Forest(MRMR): Result Analysis :)
% 
% rfMRMR_best = fitensemble(Xtrain(:,idxMRMR(1:5)),Ytrain, 'Bag', 100, 'Tree', 'Type', 'classification', CVPartition',cvKF10);
% rfMRMR_pred_best = kfoldPredict(knnMRMR_best);
% figure;
% cm10 = confusionchart(Ytest, rfMRMR_pred_best);
% title('Confusion Matrix: rf MRMR')

