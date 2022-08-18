# Capstone Project for ML Ops course
Build an end-to-end machine learning pipeline project



# Project content

## Input data

Data: Credit Card Churn Prediction

Description: A business manager of a consumer credit card bank is facing the problem of customer attrition. They want to analyze the data to find out the reason behind this and leverage the same to predict customers who are likely to drop off.


Source: https://www.kaggle.com/datasets/anwarsan/credit-card-bank-churn


## Implementation steps:

- [x] cleanup data
- [x] exploratory data analysis
- [x] train model
- [x] ml pipeline for hiperparameter tuning
- [x] model registry
- [x] ML-serve API server
- [x] ML-serve Stream server (optional)
- [ ] tests
- [ ] linters
- [x] Makefile and CI/CD
- [x] deploy to cloud
- [x] logging and monitoring
- [x] batch reporting
- [x] docker and docker-compose everything
- [x] server to download reporting


To do reminders:
- [ ] deploy on filechange only. Maybe with tag.

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




## Other:

- Github Acction: add ssh keys from server: https://zellwk.com/blog/github-actions-deploy/