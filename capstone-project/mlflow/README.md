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
```
