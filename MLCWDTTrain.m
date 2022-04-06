%% 1.1 Loading and preprocessing data that is scaled but not balanced (data1)
% -- For visualisation purposes and to get an idea of how the data looks like in DT
% -- IGNORE SECTION 1 - FOR INITIAL REFERENCE ONLY - PROCEED FROM SECTION 2

clc;
clear all;

data = readtable('data_mi_scaled.csv');

row_num = size(data, 1);

rng(42);

c = cvpartition(row_num, 'Holdout', 0.25);
data_train = data(training(c),1:end-1);
data_test = data(test(c),1:end-1);

lbls_train = data(training(c),end);
lbls_test = data(test(c),end);

isCategorical = [zeros(9,1);
                 ones(size(data_train,2)-9,1)];

%% 1.2 Full Data Model 

data_attr = data(:,1:end-1);
data_lbls = data(:,end);

rng(42);
data_mdl_full = fitctree(data_attr,data_lbls,'Reproducible',true,...
    'CategoricalPredictors',find(isCategorical == 1));
view(data_mdl_full, 'Mode', 'graph');

%% 1.3 Pruning through cvloss

[~,~,~,bestlevel] = cvLoss(data_mdl_full,'SubTrees','All','TreeSize','min')


%% 1.4 General default hyperparameter optimisation - min leaf, bayesian optimisation

rng(42);
MdlOpt = fitctree(data_attr,data_lbls,'OptimizeHyperparameters','auto','Reproducible',true)

%% 1.5 Advanced hyperparameter optimisation - all vars, gridsearch

rng(42);
MdlOptAdv_minleafsize = fitctree(data_attr,data_lbls,...
    'OptimizeHyperparameters','MinLeafSize','Reproducible',true,...
    'PredictorSelection','interaction-curvature','CategoricalPredictors',find(isCategorical == 1),...
    'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','Holdout',0.3,...
    'AcquisitionFunctionName','expected-improvement-plus',...
    'Verbose',2))

rng(42);
MdlOptAdv_maxsplitsize = fitctree(data_attr,data_lbls,...
    'OptimizeHyperparameters','MaxNumSplits','Reproducible',true,...
    'PredictorSelection','interaction-curvature','CategoricalPredictors',find(isCategorical == 1),...
    'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','Holdout',0.3,...
    'AcquisitionFunctionName','expected-improvement-plus',...
    'Verbose',2))

%% 1.6 Training decision tree model on 'data1'

rng(42);
MdlTrain = fitctree(data_train,lbls_train,'MinLeafSize',9,'Reproducible',true,...
    'PredictorSelection','curvature','CategoricalPredictors',find(isCategorical == 1));

view(MdlTrain, 'Mode', 'graph');

%% 1.7 Predict data and calculate loss
% -- Although loss error is low, the classification strength for minority
% classes is very poor. Model only looks at a few majority classes.

predLabel = predict(MdlTrain, data_test);

predLabel = array2table(predLabel);

[predLabel, lbls_test];

%predLoss = loss(MdlTrain, data_test, lbls_test);

cross_val_mdl = crossval(MdlTrain);

loss = kfoldLoss(cross_val_mdl)

resuberror = resubLoss(MdlTrain)

lbls_test_arr = table2array(lbls_test);

[Yfit,Sfit] = predict(MdlTrain, data_test);
[fpr,tpr] = perfcurve(lbls_test_arr, Sfit(:,1),'1');
figure
plot(fpr,tpr);
xlabel('False Positive Rate');
ylabel('True Positive Rate');

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


%% 2.1 Loading and preprocessing data that is scaled and balanced (data2)
% -- SMOTE data should be better for predicting minority classes
% -- Data is already scaled and balanced in Python. The data is also split
% by 75/25% for train and test data
% -- START FROM HERE!

clc;
clear;

data_bal = readtable('data_train_scaled_smote.csv');
test_bal = readtable('data_test_scaled.csv');

data_train_bal = data_bal(:,1:end-1);
data_test_bal = test_bal(:,1:end-1);

lbls_train_bal = data_bal(:,end);
lbls_test_bal = test_bal(:,end);

% Identifying variables that are categorical
isCategorical = [zeros(9,1);
                 ones(size(data_test_bal,2)-9,1)];

%% 2.2 Quickly building a DT for the training data

rng(42);
mdl_train = fitctree(data_train_bal,lbls_train_bal,'Reproducible',true,...
    'CategoricalPredictors',find(isCategorical == 1));
view(mdl_train, 'Mode', 'graph');

%% 2.3 Pruning to get an estimate of the depth - for reference only

[~,~,~,bestlevel] = cvLoss(mdl_train,'SubTrees','All','TreeSize','min')

%% 2.4 General default hyperparameter optimisation - min leaf, bayesian optimisation
% -- Using general default settings for hyperparameter optimisation to get
% an initial evaluation of parameters. Advanced tuning will be followed.
% Recording best estimated parameter values from tuning

% rng(42);
% 
% % Optimise for minleafsize (depth)
% tic
% GenMdlOpt_minleaf = fitctree(data_train_bal,lbls_train_bal,'OptimizeHyperparameters','auto','Reproducible',true,...
%     'PredictorSelection','interaction-curvature','CategoricalPredictors',find(isCategorical == 1))
% hyp_gen_leaf_time = toc;
% 
% % Optimise for number of splits
% tic
% GenMdlOpt_maxsplit = fitctree(data_train_bal,lbls_train_bal,'OptimizeHyperparameters','MaxNumSplits','Reproducible',true,...
%     'PredictorSelection','interaction-curvature','CategoricalPredictors',find(isCategorical == 1))
% hyp_gen_split_time = toc;

hyp_param = ["MaxNumSplits", "MinLeafSize"];

rng(42);

% Optimise for all eligible parameters
tic
GenMdlOpt = fitctree(data_train_bal,lbls_train_bal,'OptimizeHyperparameters',hyp_param,'Reproducible',true,...
    'PredictorSelection','interaction-curvature','CategoricalPredictors',find(isCategorical == 1))
hyp_gen_time = toc;

% Record best estimated values
gen_mdl_results = GenMdlOpt.HyperparameterOptimizationResults.XAtMinEstimatedObjective;

best_gen_hyp_minleafsize = table2array(gen_mdl_results(1,1));
best_gen_hyp_maxsplits = table2array(gen_mdl_results(1,2));


%% 2.5 Advanced hyperparameter optimisation - all vars, gridsearch
% -- Performing grid search for advanced tuning which can be used 
% for compariosn or better results

% rng(42);
% 
% % Optimising for depth
% tic
% AdvMdlOpt_minleafsize = fitctree(data_train_bal,lbls_train_bal,...
%     'OptimizeHyperparameters','MinLeafSize','Reproducible',true,...
%     'PredictorSelection','interaction-curvature',...
%     'CategoricalPredictors',find(isCategorical == 1),...
%     'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','Holdout',0.3,...
%     'AcquisitionFunctionName','expected-improvement-plus',...
%     'Verbose',2))
% hyp_adv_leaf_time = toc;

% rng(42);
% 
% % Optimising for splits
% tic
% AdvOptMdl_maxsplitsize = fitctree(data_train_bal,lbls_train_bal,...
%     'OptimizeHyperparameters','MaxNumSplits','Reproducible',true,...
%     'PredictorSelection','interaction-curvature',...
%     'CategoricalPredictors',find(isCategorical == 1),...
%     'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','Holdout',0.3,...
%     'AcquisitionFunctionName','expected-improvement-plus',...
%     'Verbose',2))
% hyp_adv_split_time = toc;

rng(42);

% Optimising for depth and splits
tic
AdvOptMdl = fitctree(data_train_bal,lbls_train_bal,...
    'OptimizeHyperparameters',hyp_param,'Reproducible',true,...
    'PredictorSelection','interaction-curvature',...
    'CategoricalPredictors',find(isCategorical == 1),...
    'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','Holdout',0.3,...
    'AcquisitionFunctionName','expected-improvement-plus',...
    'Verbose',2))
hyp_adv_time = toc; 

% Record best estimated values
adv_mdl_results = AdvOptMdl.HyperparameterOptimizationResults;

adv_mdl_results_arr = table2array(adv_mdl_results);

[~,idx] = sort(adv_mdl_results_arr(:,3)); 
adv_mdl_results_sorted = adv_mdl_results_arr(idx,:);  

best_adv_hyp_minleafsize = adv_mdl_results_sorted(1,1);
best_adv_hyp_maxsplits = adv_mdl_results_sorted(1,2);

%% 2.6 Training decision tree model using bayesian hpm optimization
% -- Evaluation of parameter suggest that the bayesian and grid search
% results concur with each other. Proceeding with the parameters obtained
% from bayesian tuning

rng(42);

tic
DTMdl = fitctree(data_train_bal,lbls_train_bal,'MinLeafSize',...
    best_gen_hyp_minleafsize,'MaxNumSplits',best_gen_hyp_maxsplits,...
    'Reproducible',true,...
    'PredictorSelection','interaction-curvature',...
    'CategoricalPredictors',find(isCategorical == 1));
best_model_time = toc;

view(DTMdl, 'Mode', 'graph');

%% 2.7 Save model for future use

save('DTMdl.mat', 'DTMdl');

%% 2.8 Predict data and calculate loss

% Provides array of predicted and actual labels for analysis
predLabelRTM = predict(DTMdl, data_test_bal);

predLabelRTM = array2table(predLabelRTM);

[predLabelRTM, lbls_test_bal]

% Qucik model generated intiially - crossval and resub loss

% cross_val_gen = crossval(mdl_train);
% 
% loss_gen = kfoldLoss(cross_val_gen)
% 
% resuberror_gen = resubLoss(mdl_train)

% Tuned model - Adv - crossval and resub loss

cross_val_mdlRTM = crossval(DTMdl);

lossRTM = kfoldLoss(cross_val_mdlRTM)

resuberrorRTM = resubLoss(DTMdl)

%% 2.9 Plotting ROC curves for predicted data
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

%% 2.10 Plotting confusion matrix and calculating performance matrics

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

%% 2.11 Generate tables with all important metrics for DT and save

class_labels = (1:8);

best_metric_arr = [accuracy; precision; recall; specificity; f1score];

best_metric_table = array2table(best_metric_arr, 'VariableNames',...
    {'Class 1' 'Class 2' 'Class 3' 'Class 4' 'Class 5' 'Class 6' 'Class 7' 'Class 8'},...
    'RowNames',{'Accuracy' 'Precision' 'Recall' 'Specificity' 'F1Score'});

writetable(best_metric_table, 'DT_Best_Mdl_Metrics.xlsx', 'WriteRowNames', true); 

loss_and_time = [best_gen_hyp_minleafsize; best_gen_hyp_maxsplits; lossRTM; resuberrorRTM; hyp_gen_time; best_model_time];

loss_and_time_table = array2table(loss_and_time, 'VariableNames',{'Results'},...
     'RowNames',{'MinLeaf' 'MaxSplit' 'KFoldLoss' 'ReSub Error' 'Tuning Time' 'Train Time'});
 
 writetable(loss_and_time_table, 'DT_Loss_Time.xlsx', 'WriteRowNames', true);

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



