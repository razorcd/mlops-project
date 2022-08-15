#### Start MlFlow 
- locally:
```
SERVER and UI: mlflow server --backend-store-uri sqlite:///mlflow1.db --port 5051 --default-artifact-root file:///home/cristiandugacicu/projects/personal/mlops-zoomcamp/hw2/artifacts

ONLY UI: mlflow ui --backend-store-uri sqlite:///mlflow.db -p 5052
```    
- docker:
```
mkdir /tmp/mlops
mkdir /tmp/mlopsdb
mkdir /tmp/mlopsartifacts

docker run --rm --name mlflow5 -v /tmp/mlopsdb:/tmp/mlopsdb -v /tmp/mlopsartifacts:/tmp/mlopsartifacts -p 5051:5050 mlflow_4
OR
docker-compose up
```

### Build model:
```
python model_train.py --data_input ../input_clean/credit_card_churn_clean.csv --experiment exp1     
OR 
python model_train.py --data_input s3://capstone/ID1/credit_card_churn.csv --experiment exp1
```

### Start orchestration server:
```
docker run --rm --name prefectTest3 -p 4200:4200 -p 8080:8080 prefect_test3
```
deploy Prefect flow:
```
prefect deployment create model_train_flow.py
```

### Create prefect storage
```
prefect storage create
  Select a storage type to create: 1
  BASE PATH: /tmp/mlopsdb
```

### Start working agents
Prefect deployments only schedule runs. They don't execute.

Working agents are required to execute scheduled deployments:
- create working queue:
```
prefect work-queue create work_queue_1
prefect work-queue ls
prefect work-queue preview 32577ff0-f1bd-4da9-8b3a-ce8a89cab3ca
```

- start working agent:
```
prefect agent start work_queue_1
```



### AWS S3 setup:
- aws cli credentials setup:
```
$ aws configure
AWS Access Key ID [****************ID1]: ID1
AWS Secret Access Key [None]: None
Default region name [eu-west-1]: eu-west-1
Default output format [None]: 
```

- create S3 bucket:
```
aws s3 mb s3://capstone --endpoint-url=http://localhost:4566
```

- create list bucket:
```
aws s3 ls --endpoint-url=http://localhost:4566
```

- list S3 bucket files:
```
aws s3 ls --endpoint-url=http://localhost:4566 s3://capstone/ID1 --recursive --human-readable --summarize
```

-copy file to s3:
```
aws s3 cp ../input_clean/credit_card_churn_clean.csv --endpoint-url=http://localhost:4566 s3://capstone/ID1/credit_card_churn_2022-08-07.csv
```

# Quick setup

There are 2 environemnts: `local` and `cloud`, defined by .env.* files. This is required to specify when using `make`.

Model registry pipeline:
```
mkdir /tmp/mlopsdb
mkdir /tmp/mlopsartifacts  
mkdir /tmp/store   
mkdir /tmp/serve   
docker-compose -f docker-compose-model-registry.yml up --build --force-recreate
aws s3 mb s3://capstone --endpoint-url=http://localhost:4566  && aws s3 cp ../input_clean/credit_card_churn_clean.csv --endpoint-url=http://localhost:4566 s3://capstone/ID1/credit_card_churn_2022-08-07.csv
prefect deployment create model_train_flow.py
```

Serve:
```
docker-compose -f docker-compose-serve.yml  up --build --force-recreate
python send_data.py
```


### Start serve separately
```
docker build -t serve -f Dockerfile-serve .

docker run --rm --name=serve -p 9696:9696 --network=capstone-project_backend -v /tmp/serve:/tmp/serve -v /tmp/mlopsartifacts:/tmp/mlopsartifacts --env RUN_ID="541c7963880041319d17d2ee7a38003b" serve

curl -X POST -H 'Content-Type: application/json' localhost:9696/predict -d '{"customer_age":100,"gender":"F","dependent_count":2,"education_level":2,"marital_status":"married","income_category":2,"card_category":"blue","months_on_book":6,"total_relationship_count":3,"credit_limit":4000,"total_revolving_bal":2500}'
```

### Other
- convert `Pipfile` to `requirements.txt`
```
pipenv run pip freeze > requirements.txt
```