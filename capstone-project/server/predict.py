from ast import Mod
import os
import logging
import pickle
import pandas as pd

import mlflow
from pandas import DataFrame
import xgboost as xgb
from flask import Flask, request, jsonify


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
        

mlflow.set_tracking_uri("http://127.0.0.1:5051")
mlflow.set_experiment("green-taxi-duration")

app = Flask('duration-prediction')
# log = app.logger

log = logging.getLogger('gunicorn.error')
app.logger.handlers = log.handlers

from mlflow.tracking import MlflowClient
client = MlflowClient()

   
# with open('../model.bin', 'rb') as f_in:
#     dv, model = pickle.load(f_in)
# mlflow_model_path = f'models:/best_model-exp_flow_2/Staging'
mlflow_model_path = f'runs:/4a69b176cc654dd0a1852a7ce40c66b7/model'
mlflow_dv_path = f'mlflow-artifacts:/best_model-exp_flow_2/Staging'

model = mlflow.pyfunc.load_model(mlflow_model_path)
client.download_artifacts('4a69b176cc654dd0a1852a7ce40c66b7', "preprocesor", '.')

with open('preprocesor/preprocesor.b', 'rb') as f_in:
    dv = pickle.load(f_in)
modelService = ModelService(model, dv)




@app.route('/predict', methods=['POST'])
def predict_endpoint():
    log.info(f'Request payload: {request.get_data()}')

    ride = request.get_json()
    log.info(f'Request json: {ride}')

    features = modelService.prepare_features(ride)
    pred = modelService.predict(features)

    result = {
        'duration': float(str(pred)),
        'model_version': 'model.version'
    }

    log.info(f'Response: {result}')
    return jsonify(result)


if __name__ == "__main__":
    log.info('App starting ...')
    app.run(debug=True, host='0.0.0.0', port=9696)
    




# {
# 'customer_age': 100,
# 'gender': 'F',
# 'dependent_count': 2,
# 'education_level': 2,
# 'marital_status': 'married',
# 'income_category': 2,
# 'card_category': 'blue',
# 'months_on_book': 6,
# 'total_relationship_count': 3,
# 'credit_limit': 4000,
# 'total_revolving_bal': 2500,
# }