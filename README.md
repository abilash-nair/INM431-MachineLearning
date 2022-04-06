# INM431-MachineLearning
Repo for Machine Learning module at City University

Instructions:

Datasets included in the zip file. Original data file name is data_mi.csv.

1. For EDA - Use MLCW - EDA.ipynb

Python is used for exploratory data analysis. A total of four files will be generated. The most important datasets are the ones which are scaled and balanced. Two datasets, one for training and the other for testing is provided. File names:

data_train_scaled_smote.csv

data_test_scaled.csv

Requirements:

pandas - for easy data processing through dataframes
matplotlib - for metrics 
numpy - for modifying, arranging, feeding and reshaping data
seaborn - for plotting histograms and boxplots
sklearn - for standard scaler to normalise variables
imblearn - for SMOTE, over and under sampling

MATLAB version: R2020a

2. For Decison Tree training: Use MLCWDTTrain.m

Execute sections from 2.1 to 2.11

3. For Random Forest Training with hyperparameter optimisation: Use MLCWRFTrain.m

Execute sections from 4.1 to 6.10. Will take 1 to 2 hours.

4. For Random Forest Training without hyperparameter optimisation: Use MLCWRFTrain.m

Execute sections from 6.1 to 6.10

5. For testing DT saved model: Use MLCWDTTest.m

6. For testing RF saved model: Use MLCWRFTest.m
