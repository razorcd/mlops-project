# Project for Machine Learning Operations (MLOps)

The challenge requested to build an fully automated end-to-end Machine Learning infrastructure, including reading data from a feature store, automated model creation with hyperoptimization tuning and automatically finding the most efficient model, storing models in a model registry, building a CI/CD pipeline for deploying the registered model into a production environment, serving the model using HTTP API or Streaming interfaces, implementing monitoring, writing tests and linters, creating regular automated reporting.

Note: this is a personal project to demonstrate an automated ML infrastructure. Security was not in the scope of this project. For deploying this infrastructure in a production environment please ensure proper credentials are set for each service and the infrastructure is not exposing unwanted endpoints to the public internet.

# Final architecture:

<img width="1337" alt="image" src="https://user-images.githubusercontent.com/3721810/185756770-73bfea67-8455-4e51-9cbf-14e0ceba5909.png">

There is a Prefect orchestrator to run 2 flows on a scheduler basis. The `Hyperoptimization deployment flow` is executed by a `Prefect Agent` by pulling training data from the `AWS S3 feature store`, runs hyperoptimization ML model builds and it saved each model (around 50 models per run) in the MLFlow model registry. On each run it finds the most efficient model and it registeres it in MlFlow to be ready for deployment. 

An engineer must decide when and which models should be deployed. First copy the `RUN_ID` of the selected model for deployment and update `.env.local` (or `.env.cloud` for cloud) with the new `RUN_ID` field. 

Once the `RUN_ID` is updated, Github Actions triggers a new pipeline-run which will run tests and restart the 2 servers (http-api and kinesis-streams servers).

The `Business Simulation using AWS Kinesis Streams` simulates business regularly (every 60sec) sending events to Kinesis stream for prediction. `ML Model Serving Kinesis Stream service` is a ML serving server using Kinessis stream as input and output for running predictions is realtime.

The `Business Simulation using HTTP API` simulates business regularly (every 60sec) sending http requests. `ML Model Serving Flask HTTP API service` is a ML serving server using http APIs for running predictions in realtime. On each prediction request, input and prediction is saved in MongoDB for later processing and to `Evidently` for monitoring.

`Evidently` is calculating data drift to understand if the running predictions are degreding after a while.
`Prometheus` is storing monitoring data and `Grafana` is providing a Dashboard UI to monitor the prediction data drift in realtime.

The `Batch reporting flow` is running regularly (every 3 hours) on the MongoDB data to execute data drift reports. These reports are saved as `html` and the `File server - Nginx` gives access to the report files.


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



# Start infrastructure locally or cloud
To deploy in the cloud, the steps are similar except: use you cloud VM domain instead of localhost to access the UIs and replace `env=local` with `env=cloud`

- install `docker`, `docker compose`, `make`
- run `make reset_all` to ensure any existing containers are removed

- run `make setup-model-registry env=local`  to start model training infrastructure
- open `http://localhost:5051` to see MLFlow UI.
- run `make init_aws  to setup and initialize` to setup training data and streams in AWS
- run `make apply-model-train-flow` to apply model training script to the orchestrator. This will run trainings regularly.
- open `http://localhost:4200/#deployments`, it will show the `model_tuning_and_uploading` deployment scheduled. Start a `Quick Run` to not wait for the scheduler. This will run the model training pipeline and upload a bunch of models to MLFlow server and register the best model.

- from the MLFlow UI decide which model you want to deploy. Get the Run Id of the model and update `RUN_ID` in `.env.local` file (or `.env.cloud` for cloud deployment)
- run `make setup-model-serve env=local` to start prediction servers

- request a prediction using http API:
``` sh
$> curl -X POST -H 'Content-Type: application/json' http://127.0.0.1:9696/predict -d '{"customer_age":50,"gender":"M","dependent_count":2,"education_level":3,"marital_status":"married","income_category":2,"card_category":"blue","months_on_book":4,"total_relationship_count":3,"credit_limit":4000,"total_revolving_bal":2511}'

{"churn chance":0.5,"model_run_id":"70cc813fa2d64c598e3f5cd93ad674af"}
```

- run `make apply-prediction-reporting` to apply reporting script to the orchestrator. This will generate reports regularly.
- open ` `, it will show `evidently_data_reporting` deployment. This runs every 3 hours. The system needs to collect 3+ hours of predictions data first before generating any report. Running the reporting manually at this time will not generate reports yet.
- open `http://localhost:8888/` to see generated reports after 3+ hours.
- open `http://localhost:8085/metrics` to see prometheus data. 
- open `http://localhost:3000/dashboards` to see Grafana realtime monitoring dashboard of data drift


Optionally:
- publish to Kinesis. `data` is the request json payload base64 encoded
``` sh
aws kinesis put-record \
    --stream-name predictions --endpoint-url=http://localhost:4566 \
    --partition-key 1 \
    --data "ewogICAgICAiY3VzdG9tZXJfYWdlIjogMTAwLAogICAgICAiZ2VuZGVyIjogIkYiLAogICAgICAiZGVwZW5kZW50X2NvdW50IjogMiwKICAgICAgImVkdWNhdGlvbl9sZXZlbCI6IDIsCiAgICAgICJtYXJpdGFsX3N0YXR1cyI6ICJtYXJyaWVkIiwKICAgICAgImluY29tZV9jYXRlZ29yeSI6IDIsCiAgICAgICJjYXJkX2NhdGVnb3J5IjogImJsdWUiLAogICAgICAibW9udGhzX29uX2Jvb2siOiA2LAogICAgICAidG90YWxfcmVsYXRpb25zaGlwX2NvdW50IjogMywKICAgICAgImNyZWRpdF9saW1pdCI6IDQwMDAsCiAgICAgICJ0b3RhbF9yZXZvbHZpbmdfYmFsIjogMjUwMAogICAgfQ=="
``` 

- consume from Kinesis
``` sh
aws kinesis get-shard-iterator  --shard-id shardId-000000000000 --endpoint-url=http://localhost:4566 --shard-iterator-type TRIM_HORIZON --stream-name results --query 'ShardIterator'
# copy shard iterator without quotes

aws kinesis get-records --endpoint-url=http://localhost:4566 --shard-iterator {SHARD_ITERATOR_HERE}
# decode Data based64
```

- run `docker logs -t {container}` to see logs for now

All running containers:
<img width="1571" alt="image" src="https://user-images.githubusercontent.com/3721810/185755712-4591f7ba-a98e-4f7b-a8a1-231d33e8beea.png">


### Other useful links:

- Github Acction: add ssh keys from server: https://zellwk.com/blog/github-actions-deploy/


