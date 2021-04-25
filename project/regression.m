clear;
clc;
%% Load Dataset
labels = load("totalLabel.mat").label;
labels = double(labels);
total = load("totalFeature.mat").total;
featureName = load("featureName.mat").featureName1;
total = [total(:,3), total(:, 6)];
m = length(labels);

%% visualize the data
histogram(labels(:),10)
% figure;
plot(total(:,1), total(:,2), 'x')

figure;
gscatter(total(:,1), total(:,2), labels)
legend('bad', 'good')
xlabel('medium')
ylabel('Efficiency')

%% linear rogistic regression
p = 0.7;
idx = randperm(m);
xtrain = total(idx(1:round(p*m)),:);
ytrain = labels(idx(1:round(p*m)),:);
xtest = total(idx(round(p*m)+1:end),:);
ytest = labels(idx(round(p*m)+1:end),:);

B = mnrfit(xtrain, ytrain);

%training set
mtrain = length(ytrain);
xtrain2 = [ones(mtrain,1) xtrain];
ztrain = xtrain2*B; 
%sigmoid
htrain = 1.0./(1.0+exp(-ztrain));

%test set
mtest = length(ytest);
xtest2 = [ones(mtest,1) xtest];
ztest = xtest2*B; 
%sigmoid
htest = 1.0./(1.0+exp(-ztest));

%histogram(htrain, 10)

%% visualize the outcome of the classification model

scatter(xtrain(:,1), xtrain(:,2), 100, htrain);
cb = colorbar();

%% visualize the decision boundary

gscatter(total(:,1),total(:,2), labels); hold on;
legend('bad', 'good')
xlabel('medium')
ylabel('Efficiency')
%when z = 0
plot(total(:,1), -(B(1)*1 + B(2)*total(:,1))/B(3)); hold off;


%% eval
%training model
ytrainpred = htrain < 0.5;
ytrainpred = ytrainpred +1;
gscatter(xtrain(:,1),xtrain(:,2), ytrainpred); 
accuracy_train = mean(double(ytrainpred == ytrain))

%test model
ytestpred = htest < 0.5;
ytestpred = ytestpred +1;
gscatter(xtest(:,1),xtest(:,2), ytestpred); hold on;
accuracy_test = mean(double(ytestpred == ytest))
plot(total(:,1), -(B(1)*1 + B(2)*total(:,1))/B(3)); 
legend('bad', 'good')
xlabel('eff')
ylabel('Sleep_Fragmentation_Index')
hold off;

