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

### AWS S3 setup:
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
aws s3 cp ../input_raw/credit_card_churn.csv --endpoint-url=http://localhost:4566 s3://capstone/ID1/credit_card_churn_2022-08-07.csv
```