# Capstone Project for ML Ops course
Build an end-to-end machine learning pipeline project

## Input data

Data: Credit Card Churn Prediction

Description: A business manager of a consumer credit card bank is facing the problem of customer attrition. They want to analyze the data to find out the reason behind this and leverage the same to predict customers who are likely to drop off.


Source: https://www.kaggle.com/datasets/anwarsan/credit-card-bank-churn


## Implementation steps:

- [x] cleanup data
- [x] exploratory data analysis
- [x] train model
- [ ] ml pipeline for hiperparameter tuning
- [ ] model registry
- [ ] ML-serve server
- [ ] tests
- [ ] linters
- [ ] Makefile and CI
- [ ] logging and monitoring

## Data cleanup

- Removes rows with "Unknown" records, removes irellevant columns, lowercase column names, lowercase categoriacal values.

- Categories that can be ordered hiarachically are converted into ints, like "income" or "education level".

[prepareData.ipynb](prepareData.ipynb)

## Exploratory data analysis

- Checks correlations on numerical columns. 

- Checks count on categorical columns.

[exploratory_data_analysis.ipynb](exploratory_data_analysis.ipynb)

## Train Model

- Prepare a model using XGBoost. 

- Input data is split into 66%, 33%, 33% for training, validation and test data.

- Measures MAE, MSE, RMSE.

- Measures % of deviated predictions based on month threshold.

[model_train.ipynb](model_train.ipynb)