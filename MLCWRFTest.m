%% 7.1 RF Model Testing
% -- Using pre-saved tables for training RF model
% -- Loading and preprocessing data that is scaled and balanced (data2)
% -- SMOTE data should be better for predicting minority classes
% -- Data is already scaled and balanced in Python. The data is also split
% by 75/25% for train and test data

clc;
clear;

data_bal = readtable('data_train_scaled_smote.csv');
test_bal = readtable('data_test_scaled.csv');

data_train_bal = data_bal(:,1:end-1);
data_test_bal = test_bal(:,1:end-1);

lbls_train_bal = data_bal(:,end);
lbls_test_bal = test_bal(:,end);

%% 7.2 Loading best trained RF model

load RFMdlBest;

%% 7.3 Plot OOB Loss

figure;
oobErrorBaggedBest = oobError(RFMdlBest);
plot(oobErrorBaggedBest)
title('OOB Error Plot');
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';

%% 7.4 Predictor importance - plotting feature performance

pred_imp_best = RFMdlBest.OOBPermutedPredictorDeltaError;

figure;
bar(pred_imp_best);
title('Feature Plot');
ylabel('Predictor Importance Scale');
xlabel('Predictor Column Number');

% Any predictor above 1 (arbitrary) to find most valuable predictors
imp_pred_col_best_names = find(pred_imp_best>1.00)
imp_pred_col_best = pred_imp_best(find(pred_imp_best>1.00))

figure;
bar(imp_pred_col_best);
title('Most Important Feature Plot');
ylabel('Predictor Importance Scale');
xlabel('Predictor Column Number');
set(gca,'xticklabel',imp_pred_col_best_names);

%% 7.5 Calculate error and plot error measures

% computes the misclassification probability for classification trees
err_train_best = error(RFMdlBest,data_train_bal,lbls_train_bal);
figure
hold on
plot(err_train_best)
plot(oobErrorBaggedBest)
title('Error Plot');
xlabel('Number of Trees')
ylabel('Error and OOB Error')
hold off

% method excludes in-bag observations from computation of the out-of-bag error
RFMdlBest.DefaultYfit = '';
figure
hold on
plot(oobError(RFMdlBest));
title('OOB Error excl. In-Bag Samples Plot');
xlabel('Number of Trees')
ylabel('OOB Error excl. In-Bag')
hold off

% Margin for OOB
figure
hold on
plot(oobMeanMargin(RFMdlBest));
title('OOB Margin Plot');
xlabel('Number of Trees')
ylabel('OOB Margin')
hold off

%% 7.6 ROC Curves

lbls_test_bal_cfm = table2array(lbls_test_bal);

% ROC
[Yfit,Sfit] = predict(RFMdlBest,data_test_bal);
[fpr,tpr] = perfcurve(lbls_test_bal_cfm,Sfit(:,1),'1');
[fpr1,tpr1] = perfcurve(lbls_test_bal_cfm,Sfit(:,2),'2');
[fpr2,tpr2] = perfcurve(lbls_test_bal_cfm,Sfit(:,3),'3');
[fpr3,tpr3] = perfcurve(lbls_test_bal_cfm,Sfit(:,4),'4');
[fpr4,tpr4] = perfcurve(lbls_test_bal_cfm,Sfit(:,5),'5');
[fpr5,tpr5] = perfcurve(lbls_test_bal_cfm,Sfit(:,6),'6');
[fpr6,tpr6] = perfcurve(lbls_test_bal_cfm,Sfit(:,7),'7');
[fpr7,tpr7] = perfcurve(lbls_test_bal_cfm,Sfit(:,8),'8');
figure
hold on
plot(fpr,tpr)
plot(fpr1,tpr1)
plot(fpr2,tpr2)
plot(fpr3,tpr3)
plot(fpr4,tpr4)
plot(fpr5,tpr5)
plot(fpr6,tpr6)
plot(fpr7,tpr7)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
hold off

%% 7.7 Plotting confusion matrix for RF classifier

predValuesBest = predict(RFMdlBest,data_test_bal);

predValuesBest = str2double(predValuesBest);

figure
cm = confusionchart(lbls_test_bal_cfm, predValuesBest,...
    'ColumnSummary','column-normalized',...
    'RowSummary','row-normalized');

%% 7.8 Evaluating and measuring RF performance 

conf_mat_best = confusionmat(lbls_test_bal_cfm, predValuesBest); 

[accuracy, precision, recall, specificity, f1score] = conf_matrix_metrics(conf_mat_best);

accuracy = accuracy';
precision = precision';
recall = recall';
specificity = specificity';
f1score = f1score';

%% APDX - Funtion to calculate performance metrics from confusion matrix

function [accuracy, precision, recall, specificity, f1score] = conf_matrix_metrics(conf_mat)
TP = [zeros(8,1)];
FN = [zeros(8,1)];
FP = [zeros(8,1)];
TN = [zeros(8,1)];

for i = 1 : 8
    for j = 1 : 8
        if j == i
            TP(i) = conf_mat(i,j);
        elseif j ~= i
            FN(i) = FN(i) + conf_mat(i,j);
            FP(i) = FP(i) + conf_mat(j,i);
        end
    end
end

for i = 1 : 8
    TN(i) = 332 - (FN(i) + TP(i) + FP(i));
end

accuracy = [zeros(8,1)];
precision = [zeros(8,1)];
recall = [zeros(8,1)];
specificity = [zeros(8,1)];
f1score = [zeros(8,1)];

for i = 1 : 8
    accuracy(i) = (TP(i) + TN(i)) / (TP(i) + TN(i) + FP(i) + FN(i));
    precision(i) = TP(i) / (TP(i) + FP(i));
    recall(i) = TP(i) / (TP(i) + FN(i));
    specificity(i) = TN(i) / (TN(i) + FP(i));
    f1score(i) = (2 * precision(i) * recall(i)) / (recall(i) + precision(i));
end
end






