# Project for Machine Learning Operations (MLOps)

The challenge requested to build an fully automated end-to-end Machine Learning infrastructure, including reading data from a feature store, automated model creation with hyperoptimization tuning and automatically finding the most efficient model, storing models in a model registry, building a CI/CD pipeline for deploying the registered model into a production environment, serving the model using HTTP API or Streaming interfaces, implementing monitoring, writing tests and linters, creating regular automated reporting.

Note: this is a personal project to demonstrate an automated ML infrastructure. Security was not in the scope of this project. For deploying this infrastructure in a production environment please ensure proper credentials are set for each service and the infrastructure is not exposing unwanted endpoints to the public internet.

# Project content

### Input data

Data: Credit Card Churn Prediction

Description: A business manager of a consumer credit card bank is facing the problem of customer attrition. They want to analyze the data to find out the reason behind this and leverage the same to predict customers who are likely to drop off.


Source: https://www.kaggle.com/datasets/anwarsan/credit-card-bank-churn


### Implementation steps:

- [x] cleanup data
- [x] exploratory data analysis
- [x] train model
- [x] ml pipeline for hyperparameter tuning
- [x] model registry
- [x] ML-serve API server
- [x] ML-serve Stream server (optional)
- [x] tests (partial)
- [ ] linters
- [x] Makefile and CI/CD
- [x] deploy to cloud
- [x] logging and monitoring
- [x] batch reporting
- [x] docker and docker-compose everything
- [x] server to download reporting


To do reminders:
- [ ] deploy on filechange only. Maybe with version as tag.

### Data cleanup

- Removes rows with "Unknown" records, removes irellevant columns, lowercase column names, lowercase categoriacal values.

- Categories that can be ordered hiarachically are converted into ints, like "income" or "education level".

[prepareData.ipynb](prepareData.ipynb)

### Exploratory data analysis

- Checks correlations on numerical columns. 

- Checks count on categorical columns.

[exploratory_data_analysis.ipynb](exploratory_data_analysis.ipynb)

### Train Model

- Prepare a model using XGBoost. 

- Input data is split into 66%, 33%, 33% for training, validation and test data.

- Measures MAE, MSE, RMSE.

- Measures % of deviated predictions based on month threshold.

[model_train.ipynb](model_train.ipynb)


### Automated hyperoptimization tuning and model registry

System is using Prefect to orchestrate DAGs. Every few hours, Prefect Agent will start and read the training data from S3, it will build models using XGBoost by running hyperparameterization on the configurations, generating 50 models and calulating accuracy (rmse) for each of them. All 50 models are registered in the MLFlow model registry experiments. At the end of each run, the best model will be registered in MLFlow as ready for deployment.

Prefect UI: http://localhost:4200
MLFlow UI: http://localhost:5051



## Other:

- Github Acction: add ssh keys from server: https://zellwk.com/blog/github-actions-deploy/