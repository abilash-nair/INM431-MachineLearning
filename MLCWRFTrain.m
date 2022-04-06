%% 4.1 Loading and preprocessing data that is scaled and balanced (data2)
% -- SMOTE data should be better for predicting minority classes
% -- Data is already scaled and balanced in Python. The data is also split
% by 75/25% for train and test data

clc;
clear;

disp('Random Forest Training');

data_bal = readtable('data_train_scaled_smote.csv');
test_bal = readtable('data_test_scaled.csv');

data_train_bal = data_bal(:,1:end-1);
data_test_bal = test_bal(:,1:end-1);

lbls_train_bal = data_bal(:,end);
lbls_test_bal = test_bal(:,end);

isCategorical = [zeros(9,1);
                 ones(size(data_train_bal,2)-9,1)];

%% 4.2 Basic RF model - for reference only

rng(42);
tic
Mdl = TreeBagger(100,data_train_bal,lbls_train_bal,'OOBPrediction','On',...
    'OOBPredictorImportance','on','Method','classification',...
    'PredictorSelection','curvature',...
    'CategoricalPredictors',find(isCategorical == 1),...
    'NumPredictorsToSample','all','MinLeafSize',1)
model_time = toc;
view(Mdl.Trees{1},'Mode','graph')

%% 4.3 Basic RF Model - OOB loss and error

figure;
oobErrorBagged = oobError(Mdl);
plot(oobErrorBagged)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';

%% 4.4 Predictor Importance - show most important variables for prediction

pred_imp = Mdl.OOBPermutedPredictorDeltaError;

figure;
bar(pred_imp);
title('Feature Test');
ylabel('Predictor Importance Scale');
xlabel('Predictors');

% Any predictor above 1 (arbitrary) to find most valuable predictors
imp_pred_col = find(pred_imp>1.00);

figure;
bar(imp_pred_col);
title('Feature Test');
ylabel('Predictor Importance Scale');
xlabel('Predictors');

%% 4.5 Calculate error and plot error measures

% computes the misclassification probability for classification trees
err_train = error(Mdl,data_train_bal,lbls_train_bal);
figure
hold on
plot(err_train)
plot(oobErrorBagged)
hold off

% method excludes in-bag observations from computation of the out-of-bag error
Mdl.DefaultYfit = '';
figure
plot(oobError(Mdl));

% % monitor the fraction of observations in the training data that are in bag for all trees
% finbag = zeros(1,Mdl.NumTrees);
% for t=1:Mdl.NTrees
%     finbag(t) = sum(all(~Mdl.OOBIndices(:,1:t),2));
% end
% finbag = finbag / size(data_train_bal,1);
% figure
% plot(finbag);

% Margin for OOB
figure
plot(oobMeanMargin(Mdl));

%% 4.6 Plot ROC curves to evaluate model performance

% ROC
[Yfit,Sfit] = oobPredict(Mdl);
[fpr,tpr] = perfcurve(Mdl.Y,Sfit(:,1),'1');
[fpr1,tpr1] = perfcurve(Mdl.Y,Sfit(:,2),'2');
[fpr2,tpr2] = perfcurve(Mdl.Y,Sfit(:,3),'3');
[fpr3,tpr3] = perfcurve(Mdl.Y,Sfit(:,4),'4');
[fpr4,tpr4] = perfcurve(Mdl.Y,Sfit(:,5),'5');
[fpr5,tpr5] = perfcurve(Mdl.Y,Sfit(:,6),'6');
[fpr6,tpr6] = perfcurve(Mdl.Y,Sfit(:,7),'7');
[fpr7,tpr7] = perfcurve(Mdl.Y,Sfit(:,8),'8');
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

%% 4.7 Confusion matrix and prediction accuracy

predValues = predict(Mdl,data_test_bal);

predValues = str2double(predValues);

lbls_test_bal_cfm = table2array(lbls_test_bal);

figure
cm = confusionchart(lbls_test_bal_cfm, predValues,...
    'ColumnSummary','column-normalized',...
    'RowSummary','row-normalized');

%% 4.8 Creating confusionmat for analysing metrics and calling conf_matrix_metrics for getting performance parameters

conf_mat = confusionmat(lbls_test_bal_cfm, predValues);

[accuracy, precision, recall, specificity, f1score] = conf_matrix_metrics(conf_mat);

accuracy = accuracy';
precision = precision';
recall = recall';
specificity = specificity';
f1score = f1score';

%% 5.1 Hyperparameter Optimisation
% -- Grid search used for HPM tuning, with number of trees, number of
% parameters, and min leaf size used as paramters for tuning. Please note
% that the tuning will take in excess of 2 hours to finish. During HPM
% tuning, OOB error and other metrics (for majority class only) will be
% recorded for evaluation.

% hyp_num_trees = [1:12];
% hyp_num_trees = [5*hyp_num_trees];
% 
% min_leaf_size = [1:10];
% min_leaf_size = [10*min_leaf_size];
% 
% num_of_param = [1:10];
% num_of_param = [num_of_param*10];
% num_of_param(end+1) = 104;

hyp_num_trees = [5:10];
hyp_num_trees = [10*hyp_num_trees];

min_leaf_size = [1:5];
min_leaf_size = [5*min_leaf_size];

num_of_param = [1:6];
num_of_param = [num_of_param*5];

iter = 0;
total_iter = size(hyp_num_trees, 2) * size(min_leaf_size, 2) * size(num_of_param, 2);

model_hyp_time = 0;
tuning_data = [];

disp('Running grid search for random forest...');
disp('--------------');

for param = num_of_param
    for leaf = min_leaf_size
        for tree = hyp_num_trees
            iter = iter + 1;
            disp([iter, total_iter, param, leaf, tree, model_hyp_time]);
            tic            
            Mdl_Hyp = TreeBagger(tree,data_train_bal,lbls_train_bal,'OOBPrediction','On',...
                    'OOBPredictorImportance','on','Method','classification',...
                    'PredictorSelection','curvature',...
                    'CategoricalPredictors',find(isCategorical == 1),...
                    'NumPredictorsToSample',param,'MinLeafSize',leaf);
            model_hyp_time = toc;
            OOB_error_hyp = oobError(Mdl_Hyp);
            error_OOB = OOB_error_hyp(tree);
            pred_mdl_hyp = predict(Mdl_Hyp,data_test_bal);
            pred_mdl_hyp = str2double(pred_mdl_hyp);
            conf_mat_hyp = confusionmat(lbls_test_bal_cfm, pred_mdl_hyp);
            [accuracy_hyp, precision_hyp, recall_hyp, specificity_hyp, f1score_hyp] = conf_matrix_metrics(conf_mat_hyp);
            tuning_data = [tuning_data; model_hyp_time, error_OOB, param, leaf, tree,...
                accuracy_hyp(1), precision_hyp(1), recall_hyp(1), specificity_hyp(1), f1score_hyp(1);];
            disp(['----------------- ', num2str((iter / total_iter) * 100), '% Finished -----------------']);
            disp([newline]);
        end
    end
end

%% 5.2 Sort the hyperparameter metrices wrt OOBerror

[~,idx] = sort(tuning_data(:,2)); 
sorted_tuning_data = tuning_data(idx,:);   

%% 5.3 Save sorted tuning data for future reference

save('sorted_tuning_data.mat', 'sorted_tuning_data');

%% 5.4 Sort the hyperparameter metrices wrt accuracy

[~,idx_acc] = sort(tuning_data(:,6)); 
sorted_tuning_data_acc = tuning_data(idx_acc,:); 

%% 5.5 Save sorted tuning data wrt accuracy for future reference

save('sorted_tuning_data_acc.mat', 'sorted_tuning_data_acc');

%% 6.1 Using pre-saved tables for training RF model
% -- Loading and preprocessing data that is scaled and balanced (data2)
% -- SMOTE data should be better for predicting minority classes
% -- Data is already scaled and balanced in Python. The data is also split
% by 75/25% for train and test data

clc;
clear;

sorted_data = load('sorted_tuning_data.mat');
sorted_tuning_data = sorted_data.sorted_tuning_data;

data_bal = readtable('data_train_scaled_smote.csv');
test_bal = readtable('data_test_scaled.csv');

data_train_bal = data_bal(:,1:end-1);
data_test_bal = test_bal(:,1:end-1);

lbls_train_bal = data_bal(:,end);
lbls_test_bal = test_bal(:,end);

isCategorical = [zeros(9,1);
                 ones(size(data_train_bal,2)-9,1)];

%% 6.2 Use best parameters from grid search that has the lowest OOB loss 
%  to train best RF model

best_param = sorted_tuning_data(1,3);
best_min_leaf = sorted_tuning_data(1,4);
best_max_trees = sorted_tuning_data(1,5);

tic
RFMdlBest = TreeBagger(best_max_trees,data_train_bal,lbls_train_bal,'OOBPrediction','On',...
    'OOBPredictorImportance','on','Method','classification',...
    'PredictorSelection','curvature',...
    'CategoricalPredictors',find(isCategorical == 1),...
    'NumPredictorsToSample',best_param,'MinLeafSize',best_min_leaf);
best_model_train_time = toc;

view(RFMdlBest.Trees{1},'Mode','graph');

%% 6.3 Save best model for future use

save('RFMdlBest.mat', 'RFMdlBest');

%% 6.4 Plot OOB Loss

figure;
oobErrorBaggedBest = oobError(RFMdlBest);
plot(oobErrorBaggedBest)
title('OOB Error Plot');
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';

%% 6.5 Predictor importance - plotting feature performance

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

%% 6.6 Calculate error and plot error measures

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

% % monitor the fraction of observations in the training data that are in bag for all trees
% finbag_best = zeros(1,RFMdlBest.NumTrees);
% for t=1:RFMdlBest.NTrees
%     finbag_best(t) = sum(all(~RFMdlBest.OOBIndices(:,1:t),2));
% end
% finbag_best = finbag_best / size(data_train_bal,1);
% figure
% plot(finbag_best);

% Margin for OOB
figure
hold on
plot(oobMeanMargin(RFMdlBest));
title('OOB Margin Plot');
xlabel('Number of Trees')
ylabel('OOB Margin')
hold off

%% 6.7 ROC Curves

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

%% 6.8 Plotting confusion matrix for RF classifier

predValuesBest = predict(RFMdlBest,data_test_bal);

predValuesBest = str2double(predValuesBest);

figure
cm = confusionchart(lbls_test_bal_cfm, predValuesBest,...
    'ColumnSummary','column-normalized',...
    'RowSummary','row-normalized');

%% 6.9 Evaluating and measuring RF performance 

conf_mat_best = confusionmat(lbls_test_bal_cfm, predValuesBest); 

[accuracy, precision, recall, specificity, f1score] = conf_matrix_metrics(conf_mat_best);

accuracy = accuracy';
precision = precision';
recall = recall';
specificity = specificity';
f1score = f1score';

%% 6.10 Generate tables with all important metrics for DT and save

class_labels = (1:8);

best_metric_arr = [accuracy; precision; recall; specificity; f1score];

best_metric_table = array2table(best_metric_arr, 'VariableNames',...
    {'Class 1' 'Class 2' 'Class 3' 'Class 4' 'Class 5' 'Class 6' 'Class 7' 'Class 8'},...
    'RowNames',{'Accuracy' 'Precision' 'Recall' 'Specificity' 'F1Score'});

writetable(best_metric_table, 'RF_Best_Mdl_Metrics.xlsx', 'WriteRowNames', true);

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
