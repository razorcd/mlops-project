#### Start MlFlow
```
mlflow server --backend-store-uri sqlite:///mlflow1.db --port 5051 --default-artifact-root file:///home/cristiandugacicu/projects/personal/mlops-zoomcamp/hw2/artifacts
mlflow ui --backend-store-uri sqlite:///mlflow.db -p 5052
```    

### Build model:
```
python model_train.py --data_input ../input_clean/credit_card_churn_clean.csv --model_output model/
```
