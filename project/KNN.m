format long
%% Load Dataset
Xtrain = load('Xtrain.mat').Xtrain;
Ytrain = load('Ytrain.mat').Ytrain;
Xtest = load('Xtest.mat').Xtest;
Ytest = load('Ytest.mat').Ytest;
cvLO = load('cvLO.mat','cvLO').cvLO;

%% Load MRMR
MRMR = load('MRMR.mat').MRMR;
idxMRMR = MRMR(1,:);
scores = MRMR(2,:);

%% K vs Num. Features 
Ks =[1:1:12];
knnMRMR_LO = zeros(12,length(Ks));
AUC_LO = zeros(12,length(Ks));
F1_LO = zeros(12,length(Ks));

for f=1:12
    for k=1:length(Ks)
        knnMRMR_CV = fitcknn(Xtrain(:,idxMRMR(1:f)),Ytrain,'NumNeighbors',k,'Standardize',1,'CVPartition',cvLO);
        [knnMRMR_pred, scores] = kfoldPredict(knnMRMR_CV);
        knnMRMR_LO(f,k) = mean(Ytrain == knnMRMR_pred);
        [~,~,~,AUCknn_g] = perfcurve(Ytrain,scores(:,1),1);
        [~,~,~,AUCknn_b] = perfcurve(Ytrain,scores(:,2),2);

        AUC_LO(f,k) = (AUCknn_g + AUCknn_b)/2;
        tpG = sum(ismember(Ytrain,2) & ismember(knnMRMR_pred,2));
        tpB = sum(ismember(Ytrain,1) & ismember(knnMRMR_pred,1));
        fpG = sum(ismember(knnMRMR_pred,2)) - tpG;
        fpB = sum(ismember(knnMRMR_pred,1)) - tpB;
        fnG = sum(ismember(Ytrain, 2)) - tpG;
        fnB = sum(ismember(Ytrain, 1)) - tpB;
        f1G = tpG/(tpG + (fpG+fnG)/2);
        f1B = tpB/(tpB + (fpB+fnB)/2);
        F1_LO(f,k) = (f1G + f1B)/2;

    end
end
%% Heatmap Accuracy
xvals_numF = num2cell(Ks);
yvals_k = num2cell([1:12]);
figure;
h5 = heatmap(xvals_numF, yvals_k, knnMRMR_LO,'Colormap', jet, 'CellLabelColor','none');
title('Acuuracy');
h5.XLabel = 'K_N_N';
h5.YLabel = 'Num of Features';
%% Heatmap AUC
figure;
h2 = heatmap(xvals_numF, yvals_k, AUC_LO,'Colormap', jet, 'CellLabelColor','none');
title('AUC');
h2.XLabel = 'K_K_N_N';
h2.YLabel = 'Num of Features';
%% Heatmap F1
figure;
h = heatmap(xvals_numF, yvals_k,F1_LO,'Colormap', jet, 'CellLabelColor','none');
title('F1 Score');
h.XLabel = 'K_K_N_N';
h.YLabel = 'Num of Features';
%% Plot All Heatmaps
xvals_numF = num2cell(Ks);
yvals_k = num2cell([1:12]);
subplot(3,1,1);
h5 = heatmap(xvals_numF, yvals_k, knnMRMR_LO,'Colormap', jet, 'CellLabelColor','none');
title('KNN: Accruacy');
h5.XLabel = 'K_K_N_N';
h5.YLabel = 'Num of Features';
subplot(3,1,2);
h = heatmap(xvals_numF, yvals_k, AUC_LO,'Colormap', jet, 'CellLabelColor','none');
title('KNN: AUC');
h.XLabel = 'K_K_N_N';
h.YLabel = 'Num of Features';
subplot(3,1,3);
h = heatmap(xvals_numF, yvals_k,F1_LO,'Colormap', jet, 'CellLabelColor','none');
title('KNN: F1 Score');
h.XLabel = 'K_K_N_N';
h.YLabel = 'Num of Features';
%% External Validtion
knnMRMR_flat_1 = reshape(knnMRMR_LO,[],1);
knnMRMR_flat = sort(unique(knnMRMR_flat_1), 'descend');
fk = [];
knnMRMR_CV_preds = [];
for a=1:length(knnMRMR_flat)
    inds = find(knnMRMR_LO == knnMRMR_flat(a));
    for i = 1:length(inds)
        [f,k] = ind2sub(size(knnMRMR_LO),inds(i));
        fk = [fk; f, k];
        knnMRMR_CV_preds = [knnMRMR_CV_preds, knnMRMR_flat(a)];
    end
end
%%
knnMRMR_preds = [];
for i=1:length(fk)
    f = fk(i,1);
    k = fk(i,2);
    knnMRMR = fitcknn(Xtrain(:,idxMRMR(1:f)),Ytrain,'NumNeighbors',Ks(k),'Standardize',1);
    pred = predict(knnMRMR, Xtest(:,idxMRMR(1:f)));
    knnMRMR_preds = [knnMRMR_preds, mean(pred==Ytest)]; % accuracy
    
end

%% Plot Internal vs. External
figure;
x = [0:1];
y = x;
plot(x, y,'Linewidth',1);
hold on
scatter(knnMRMR_CV_preds,knnMRMR_preds,'+','Linewidth',1);
title('Internal Validation vs. External Validation');
xlabel('Internal Accuracy');
ylabel('External Accuracy');


%% Confusion Matrix
%  features: 12, k: 4
master = [knnMRMR_preds.', fk];
master = sortrows(master,1,'descend');
bf = master(1,2);
bk = Ks(master(1,3));
knnMRMR = fitcknn(Xtrain(:,idxMRMR(1:bf)),Ytrain,'NumNeighbors',2,'Standardize',1);
[pred,score,~] = predict(knnMRMR, Xtest(:,idxMRMR(1:bf)));
acc = mean(pred==Ytest) * 100 % accuracy
figure;
cm = confusionchart(Ytest,pred);
title(['KNN: ', num2str(acc), '%'])

%% Plot Internal vs. External & Confusion Matrix
% figure;
% subplot(2,1,1);
% x = [0:1];
% y = x;
% plot(x, y,'Linewidth',1);
% hold on
% scatter(knnMRMR_CV_preds,knnMRMR_preds,'+','Linewidth',1);
% title('Internal Validation vs. External Validation');
% xlabel('Internal');
% ylabel('External');
% subplot(2,1,2);
% master = [knnMRMR_preds.', fk];
% master = sortrows(master,1,'descend');
% bf = master(1,2);
% bk = Ks(master(1,3));
% knnMRMR = fitcknn(Xtrain(:,idxMRMR(1:bf)),Ytrain,'NumNeighbors',4,'Standardize',1);
% [pred,score,~] = predict(knnMRMR, Xtest(:,idxMRMR(1:bf)));
% acc = mean(pred==Ytest) * 100 % accuracy
% confusionchart(Ytest,pred);
% title(['KNN: ', num2str(acc), '%'])
