# Project for Machine Learning Operations (MLOps)

The challenge requested to build an fully automated end-to-end Machine Learning infrastructure, including reading data from a feature store, automated model creation with hyperoptimization tuning and automatically finding the most efficient model, storing models in a model registry, building a CI/CD pipeline for deploying the registered model into a production environment, serving the model using HTTP API or Streaming interfaces, implementing monitoring, writing tests and linters, creating regular automated reporting.

Note: this is a personal project to demonstrate an automated ML infrastructure. Security was not in the scope of this project. For deploying this infrastructure in a production environment please ensure proper credentials are set for each service and the infrastructure is not exposing unwanted endpoints to the public internet.

# Project progress

### Input data

Data: Credit Card Churn Prediction

Description: A business manager of a consumer credit card bank is facing the problem of customer attrition. They want to analyze the data to find out the reason behind this and leverage the same to predict customers who are likely to drop off.


Source: https://www.kaggle.com/datasets/anwarsan/credit-card-bank-churn


### Implementation plan:

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
- [x] reporting server


To do reminders:
- [ ] deploy on filechange only. Maybe with version as tag.

### Data cleanup

- Removes rows with "Unknown" records, removes irellevant columns, lowercase column names, lowercase categoriacal values.

- Categories that can be ordered hiarachically are converted into ints, like "income" or "education level".

[prepareData.ipynb](model_preparation_analysis/prepareData.ipynb)

### Exploratory data analysis

- Checks correlations on numerical columns. 

- Checks count on categorical columns.

- [exploratory_data_analysis.ipynb](exploratory_data_analysis/exploratory_data_analysis.ipynb)
- [model_preparation_analysis](model_preparation_analysis)

### Train Model

- Prepare a model using XGBoost. 

- Input data is split into 66%, 33%, 33% for training, validation and test data.

- Measures MAE, MSE, RMSE.

- Measures % of deviated predictions based on month threshold.

- [model_train.ipynb](model_preparation_analysis/model_train.ipynb)

### Model Registry with MLFlow

- dockerized MLFlow: [Dockerfile-mlflow](mlflow/Dockerfile-mlflow)
- MLFlow UI: `http://localhost:5051`

### Automated hyperoptimization tuning

System is using Prefect to orchestrate DAGs. Every few hours, Prefect Agent will start and read the training data from S3, it will build models using XGBoost by running hyperparameterization on the configurations, generating 50 models and calculating accuracy (rmse) for each of them. All 50 models are registered in the MLFlow model registry experiments. At the end of each run, the best model will be registered in MLFlow as ready for deployment.

- model training Prefect flow: [model_train_flow.py](model_orchestration/model_train_flow.py)
- dockerized Prefect Server: [Dockerfile-prefect](model_orchestration/Dockerfile-prefect)
- dockerized Prefect Agent: [Dockerfile-prefect-agent](model_orchestration/Dockerfile-prefect-agent)

- Prefect UI: `http://localhost:4200`

### Model serving HTTP API and Stream

There are 2 ML service servers. One serving predictions using HTTP API build in Python with Flask. Second serving predictions using AWS Kinesis streams, both consuming and publishing results back.

- model serving using Python Flask HTTP API: [predict-api-server.py](server/predict-api-server.py)
- model serving using Python and AWS Kinesis Streams: [serve_kinesis.py](server/serve_kinesis.py)

- dockerized Flask API server: [Dockerfile-serve-api](server/Dockerfile-serve-api)
- dockerized AWS Kinesis server: [Dockerfile-serve-kinesis](server/Dockerfile-serve-kinesis)

### Simulation_business: sending data for realtime prediction

There are 2 Python scripts to simulate business requesting predictions from ML servers. One request data from HTTP API server and another one sending events to `predictions` Kinesis stream and receiving results to `results` Kinesis stream.

- sending data for prediction using HTTP API: [send_data-api.py](simulation_business/send_data-api.py)
- sending data for prediction using AWS Kinesis Streams: [serve_kinesis.py](simulation_business/serve_kinesis.py)

- dockerized sending data to HTTP API: [Dockerfile-send-data-api](simulation_business/Dockerfile-send-data-api)
- dockerized sending data to AWS Kinesis Streams: [Dockerfile-send-data-kinesis](simulation_business/Dockerfile-send-data-kinesis)

### Monitoring

There are 3 services for monitoring the model predictions is realtime:
- [Evidently AI](monitoring/evidently_service/) for calculating data drift. Evidently UI: 
- Prometheus for collecting monitoring data. Prometheus UI: 
- Grafana for Dashboards UI. Grafana UI: [http://localhost:3000](http://localhost:3000/d/U54hsxv7k/evidently-data-drift-dashboard?orgId=1&refresh=10s)


[Image]

### Reporting

There is a Prefect flow to generate reporting using Evidently: [create_report.py](reporting/create_report.py). This will generate reports every few hours save them in MongoDB and also generate static html pages with all data charts.

Report file example: ...

There is also an Nginx server to expose these html reports.

- Nginx server: [nginx](reporting/nginx/)
- Nginx address: `http://localhost:8888/`


### Deployment

All containers are put together in docker compose files for easy deployment of the entire infrastructure. Docker-compose if perfect for this project, for a more advanced production environment where each service is deployed in different VM, I recommend using more advance tools.

- Deployment: model training: [docker-compose-model-registry.yml](docker-compose-model-registry.yml)
- Deployment: model serving: [docker-compose-serve.yml](docker-compose-serve.yml)

All deployment commands are grouped using the Makefile for simplicity of use.
- [Makefile](Makefile)

The environment variables should be in `.env` file. The Makefile will use one of these: [.env.local](.env.local) or [.env.cloud](.env.cloud).

``` sh
$> make help

Commands:

run: make run_tests   to run tests locally
run: make reset_all   to delete all containers and cleanup volumes
run: make setup-model-registry env=local (or env=cloud)   to start model registry and training containers
run: make init_aws  to setup and initialize AWS services (uses localstack container)
run: make apply-model-train-flow   to apply the automated model training DAG 
run: make setup-model-serve env=local (or env=cloud)   to start the model serving containers
run: make apply-prediction-reporting   to apply the automated prediction reporting DAG
run: make stop-serve   to stop the model servers (http api and Stream)
run: make start-serve env=local   to start the model servers (http api and Stream)
```

### CI/CD in Cloud

The continuos deployment is done using Github actions. Once a change is made to the repo, the deployment pipeline is triggered. This will restart the model servers to load a new model from the MLFlow model registry. The deployed model is always specified in `.env.cloud` file under `RUN_ID` environment variable.

The pipeline will:
- run tests 
- ssh in the cloud virtual machine
- restart model-server-api and model-server-streams containers

### Deploying to cloud

Cloud deployment requirements:
-

Cloud deployment steps:
-

The entire infrastructure was deployed in the cloud using virtual machine provided by [Digital Ocean](https://www.digitalocean.com/).

[Image ssh & docker ps]


### Other useful links:

- Github Acction: add ssh keys from server: https://zellwk.com/blog/github-actions-deploy/


