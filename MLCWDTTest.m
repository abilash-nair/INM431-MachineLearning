%% 3.1 Testing best DT Model
% -- Loading and preprocessing data that is scaled and balanced (data2)

clc;
clear;

data_bal = readtable('data_train_scaled_smote.csv');
test_bal = readtable('data_test_scaled.csv');

data_train_bal = data_bal(:,1:end-1);
data_test_bal = test_bal(:,1:end-1);

lbls_train_bal = data_bal(:,end);
lbls_test_bal = test_bal(:,end);
             
%% 3.2 Load saved model for testing

mdl_load = load('DTMdl.mat');
DTMdl = mdl_load.DTMdl;

%% 3.3 Predict data and calculate loss

% Provides array of predicted and actual labels for analysis
predLabelRTM = predict(DTMdl, data_test_bal);

predLabelRTM = array2table(predLabelRTM);

cross_val_mdlRTM = crossval(DTMdl);

lossRTM = kfoldLoss(cross_val_mdlRTM)

resuberrorRTM = resubLoss(DTMdl)

%% 3.4 Plotting ROC curves for predicted data
% -- ROC Curves need to be generated for each class

lbls_test_arr = table2array(lbls_test_bal);

[Yfit,Sfit] = predict(DTMdl, data_test_bal);
[fpr,tpr] = perfcurve(lbls_test_arr, Sfit(:,1),'1');
[fpr1,tpr1] = perfcurve(lbls_test_arr,Sfit(:,2),'2');
[fpr2,tpr2] = perfcurve(lbls_test_arr,Sfit(:,3),'3');
[fpr3,tpr3] = perfcurve(lbls_test_arr,Sfit(:,4),'4');
[fpr4,tpr4] = perfcurve(lbls_test_arr,Sfit(:,5),'5');
[fpr5,tpr5] = perfcurve(lbls_test_arr,Sfit(:,6),'6');
[fpr6,tpr6] = perfcurve(lbls_test_arr,Sfit(:,7),'7');
[fpr7,tpr7] = perfcurve(lbls_test_arr,Sfit(:,8),'8');
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

%% 3.5 Plotting confusion matrix and calculating performance matrics

figure
cm = confusionchart(lbls_test_arr, Yfit,...
    'ColumnSummary','column-normalized',...
    'RowSummary','row-normalized');

conf_mat = confusionmat(lbls_test_arr, Yfit);

[accuracy, precision, recall, specificity, f1score] = conf_matrix_metrics(conf_mat);

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

             