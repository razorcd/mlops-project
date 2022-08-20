# Final capstone project:
Full end-to-end Machine Learning pipeline project

https://github.com/razorcd/mlops-training/tree/main/capstone-project



# MLOPS training based on MLOPS Zoomcamp course offered by DataTalks.Club
https://github.com/DataTalksClub/mlops-zoomcamp

## Notes:
- MLOPS maturity model: https://docs.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model

## MLFlow

- start: 
   - `pipenv shell`
   - `mlflow server --backend-store-uri sqlite:///mlflow.db --port 5051 --default-artifact-root file://$(pwd)/artifacts`
   - `mlflow ui --backend-store-uri sqlite:///mlflow.db -p 5052`

## Prefect setup
 - start prefect using Docker: 
    - `cd course3`
    - `in Dockerfile, edit 127.0.0.1 to server's public IP
    - `docker build -t prefect_test3 .`
    - `docker run --rm --name prefectTest3 -p 4200:4200 -p 8080:8080 prefect_test3`
 - enable TCP and UDP for port 4200 in firewall

- config local machine: `prefect config set PREFECT_API_URL="http://<external-ip>:4200/api"`
 - `prefect config view` -> `PREFECT_API_URL='http://127.0.0.1:4200/api' (from profile)`
 - open `<external-ip>:4200` in browser

 - create storage `prefect storage create`
 - get storage `prefect storage ls`
 
 
 - create prefect deployment `prefect deployment create prefect_deploy.py`
 - inspect deployment `prefect deployment inspect 'main/model_training'`
 - you still have to specify where to run. (Prefect server only schedules, does not execute runs) Create Agents and work queues in UI.
 - `prefect work-queue preview 3162c642-caca-45cc-bf1a-7a26599525c4`
 - start server in running server / local machine: `prefect agent start 3162c642-caca-45cc-bf1a-7a26599525c4`
