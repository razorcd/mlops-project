# run: gunicorn --bind=0.0.0.0:9696 --chdir=server --log-level=debug  predict:app
# predict: curl -X POST -H 'Content-Type: application/json' localhost:9696/predict -d '{"customer_age":100,"gender":"F","dependent_count":2,"education_level":2,"marital_status":"married","income_category":2,"card_category":"blue","months_on_book":6,"total_relationship_count":3,"credit_limit":4000,"total_revolving_bal":2500}'

from ast import Mod
from multiprocessing.pool import RUN
import os
import logging
import pickle
import pandas as pd

import mlflow
from pandas import DataFrame
import xgboost as xgb
from flask import Flask, request, jsonify
from pymongo import MongoClient
import requests



class ModelService():
    def __init__(self, dv, model, model_version=None, callbacks=None):
        self.model = model
        self.dv = dv
        self.model_version = model_version
        self.callbacks = callbacks or []

    def prepare_features(_self, input):
        features = {
            'customer_age': input['customer_age'], 
            'gender': input['gender'], 
            'dependent_count': input['dependent_count'],
            'education_level': input['education_level'],
            'marital_status': input['marital_status'],
            'income_category': input['income_category'],
            'card_category': input['card_category'],
            'months_on_book': input['months_on_book'],
            'total_relationship_count': input['total_relationship_count'], 
            'credit_limit': float(input['credit_limit']), 
            'total_revolving_bal': input['total_revolving_bal']
        }
        return features


    def predict(_self, dicts):
        X = dv.transform(dicts)
        features = dv.get_feature_names()
        print(f'features={features}')
        dval = xgb.DMatrix(X, feature_names=features)
        y_pred = model.predict(dval)
        return y_pred[0]
        


EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTLY_SERVICE', 'http://127.0.0.1:8085')
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27018")
MLFLOW_ADDRESS = os.getenv("MLFLOW_ADDRESS", "http://127.0.0.1:5051")
# MLFLOW_ADDRESS = os.getenv("MLFLOW_ADDRESS", "http://mlflow_server:5050")
RUN_ID = os.getenv('RUN_ID', 'RUN_ID missing')

mlflow.set_tracking_uri(MLFLOW_ADDRESS)
mlflow.set_experiment("exp_flow_2")

app = Flask('duration-prediction')
# log = app.logger

log = logging.getLogger('gunicorn.error')
app.logger.handlers = log.handlers

from mlflow.tracking import MlflowClient
client = MlflowClient()


# mlflow_model_path = f'models:/best_model-exp_flow_2/Staging'
mlflow_model_path = f'runs:/{RUN_ID}/model'
# mlflow_dv_path = f'mlflow-artifacts:/best_model-exp   _flow_2/Staging'

if not os.path.exists('/tmp/serve'): os.mkdir('/tmp/serve')
model = mlflow.xgboost.load_model(mlflow_model_path)
client.download_artifacts(RUN_ID, "preprocesor", '/tmp/serve')

with open('/tmp/serve/preprocesor/preprocesor.b', 'rb') as f_in:
    dv = pickle.load(f_in)
modelService = ModelService(model, dv)

#DB
mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("prediction_service")
collection = db.get_collection("data")


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    log.info(f'Request payload: {request.get_data()}')

    input = request.get_json()
    log.info(f'Request json: {input}')

    features = modelService.prepare_features(input)
    pred = modelService.predict(features)

    result = {
        'churn chance': float(str(pred)),
        'model_run_id': RUN_ID
    }

    save_to_db(input, float(pred))
    send_to_evidently_service(input, float(pred))
    
    log.info(f'Response: {result}')
    return jsonify(result)


def save_to_db(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    collection.insert_one(rec)


def send_to_evidently_service(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/capstone", json=[rec])



if __name__ == "__main__":
    log.info('App starting ...')
    app.run(debug=True, host='0.0.0.0', port=9696)
    



# GET payload:
    # {
    #   'customer_age': 100,
    #   'gender': 'F',
    #   'dependent_count': 2,
    #   'education_level': 2,
    #   'marital_status': 'married',
    #   'income_category': 2,
    #   'card_category': 'blue',
    #   'months_on_book': 6,
    #   'total_relationship_count': 3,
    #   'credit_limit': 4000,
    #   'total_revolving_bal': 2500,
    # }