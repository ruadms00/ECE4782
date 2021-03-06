% clear
% clc
% %% Load Dataset
% Xtrain = load('Xtrain.mat').Xtrain;
% Ytrain = load('Ytrain.mat').Ytrain;
% Xtest = load('Xtest.mat').Xtest;
% Ytest = load('Ytest.mat').Ytest;
% cvLO = load('cvLO.mat','cvLO').cvLO;
% 
% %% Load MRMR
% MRMR = load('MRMR.mat').MRMR;
% idxMRMR = MRMR(1,:);
% scores = MRMR(2,:);
% 
% %% K vs Num. Features 
% Ks =[50:10:150];
% rfMRMR_LO = zeros(12,length(Ks));
% AUC_LO = zeros(12,length(Ks));
% F1_LO = zeros(12,length(Ks));
% 
% for f=1:12
%     for k=1:length(Ks)
%         rfMRMR_CV = fitensemble(Xtrain(:,idxMRMR(1:12)),Ytrain, 'Bag', Ks(k), 'Tree', 'Type', 'classification', 'CVPartition', cvLO);
%         [rfMRMR_pred,scores] = kfoldPredict(rfMRMR_CV);
%         rfMRMR_LO(f,k) = mean(Ytrain == rfMRMR_pred);
%         
%         [~,~,~,AUCknn_g] = perfcurve(Ytrain,scores(:,1),1);
%         [~,~,~,AUCknn_b] = perfcurve(Ytrain,scores(:,2),2);
% 
%         AUC_LO(f,k) = (AUCknn_g + AUCknn_b)/2;
%         tpG = sum(ismember(Ytrain,2) & ismember(rfMRMR_pred,2));
%         tpB = sum(ismember(Ytrain,1) & ismember(rfMRMR_pred,1));
%         fpG = sum(ismember(rfMRMR_pred,2)) - tpG;
%         fpB = sum(ismember(rfMRMR_pred,1)) - tpB;
%         fnG = sum(ismember(Ytrain, 2)) - tpG;
%         fnB = sum(ismember(Ytrain, 1)) - tpB;
%         f1G = tpG/(tpG + (fpG+fnG)/2);
%         f1B = tpB/(tpB + (fpB+fnB)/2);
%         F1_LO(f,k) = (f1G + f1B)/2;
%     end
% end
% %% Heatmap Accuracy
% xvals_numF = num2cell(Ks);
% yvals_k = num2cell([1:12]);
% figure;
% h5 = heatmap(xvals_numF, yvals_k, rfMRMR_LO, 'Colormap', jet, 'CellLabelColor','none');
% title('Random Forest - Accuracy');
% h5.XLabel = 'K_R_F';
% h5.YLabel = 'Num of Features';
% %% Heatmap AUC
% figure;
% h2 = heatmap(xvals_numF, yvals_k, AUC_LO, 'Colormap', jet, 'CellLabelColor','none');
% title('Random Forest - AUC');
% h2.XLabel = 'K_R_F';
% h2.YLabel = 'Num of Features';
% %% Heatmap F1
% figure;
% h = heatmap(xvals_numF, yvals_k,F1_LO, 'Colormap', jet, 'CellLabelColor','none');
% title('Random Forest - F1 Score');
% h.XLabel = 'K_R_F';
% h.YLabel = 'Num of Features';
% %% Plot All Heatmaps
% figure;
% xvals_numF = num2cell(Ks);
% yvals_k = num2cell([1:12]);
% subplot(3,1,1);
% heatmap(xvals_numF, yvals_k, rfMRMR_LO,'Colormap', jet, 'CellLabelColor','none');
% title('Random Forest: Accruacy');
% subplot(3,1,2);
% heatmap(xvals_numF, yvals_k, AUC_LO,'Colormap', jet, 'CellLabelColor','none');
% title('Random Forest: AUC');
% subplot(3,1,3);
% heatmap(xvals_numF, yvals_k,F1_LO,'Colormap', jet, 'CellLabelColor','none');
% title('Random Forest: F1 Score');
%% External Validtion
rfMRMR_flat_1 = reshape(rfMRMR_LO,[],1);
rfMRMR_flat = sort(unique(rfMRMR_flat_1), 'descend');
fk = [];
rfMRMR_CV_preds = [];
%%%%%%%%%%%%%%%%%%%%Different feature num: Choose 3 features out of 7
for a=1:length(rfMRMR_flat)
    inds = find(rfMRMR_LO == rfMRMR_flat(a));
    for i = 1:length(inds)
        [f,k] = ind2sub(size(rfMRMR_LO),inds(i));
        fk = [fk; f, k];
        rfMRMR_CV_preds = [rfMRMR_CV_preds, rfMRMR_flat(a)];
    end
end
%%
rfMRMR_preds = [];
for i=1:length(fk)
    f = fk(i,1);
    k = fk(i,2);
   % disp(Ks(k))
    rfMRMR_CV = fitensemble(Xtrain(:,idxMRMR(1:f)),Ytrain, 'Bag', Ks(k), 'Tree', 'Type', 'classification');
    pred = predict(rfMRMR_CV, Xtest(:,idxMRMR(1:f)));
    rfMRMR_preds = [rfMRMR_preds, mean(pred==Ytest)]; % accuracy
    %disp(Ks(k) + "/" + mean(pred==Ytest));
end

% %% Internval vs. External
figure;
x = [0:1];
y = x;
plot(x, y,'Linewidth',1);
hold on

scatter(rfMRMR_CV_preds,rfMRMR_preds,'+','Linewidth',1);

xlabel('Internal Accuracy');
ylabel('External Accuracy');
title('Internal Validation vs. External Validation');

%% Confusion Matrix
% 12, 80
master_rf = [rfMRMR_preds.', fk];
master_rf = sortrows(master_rf,1,'descend');
bf = master_rf(1,2);
bk = Ks(master_rf(1,3));
rfMRMR = fitensemble(Xtrain(:,idxMRMR(1:9)),Ytrain, 'Bag', 150, 'Tree', 'Type', 'classification');
[pred,score] = predict(rfMRMR, Xtest(:,idxMRMR(1:9)));
acc = mean(pred==Ytest) * 100; % accuracy
figure;
cm = confusionchart(Ytest,pred);
title(['Random Forest: ', num2str(acc), '%']);
%% Internval vs. External & Confusion Matrix
figure;
subplot(2,1,1);
x = [0:1];
y = x;
plot(x, y,'Linewidth',1);
hold on

scatter(rfMRMR_CV_preds,rfMRMR_preds,'+','Linewidth',1);

xlabel('Internal');
ylabel('External');
title('Internal Validation vs. External Validation');
master_rf = [rfMRMR_preds.', fk];
master_rf = sortrows(master_rf,1,'descend');
bf = master_rf(1,2);
bk = Ks(master_rf(1,3));
rfMRMR = fitensemble(Xtrain(:,idxMRMR(1:9)),Ytrain, 'Bag', 150, 'Tree', 'Type', 'classification');
[pred,score] = predict(rfMRMR, Xtest(:,idxMRMR(1:9)));
acc = mean(pred==Ytest) * 100; % accuracy
subplot(2,1,2);
confusionchart(Ytest,pred);
title(['Random Forest: ', num2str(acc), '%'])
