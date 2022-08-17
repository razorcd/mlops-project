# run: gunicorn --bind=0.0.0.0:9696 --chdir=server --log-level=debug  predict:app
# predict: curl -X POST -H 'Content-Type: application/json' localhost:9696/predict -d '{"customer_age":100,"gender":"F","dependent_count":2,"education_level":2,"marital_status":"married","income_category":2,"card_category":"blue","months_on_book":6,"total_relationship_count":3,"credit_limit":4000,"total_revolving_bal":2500}'

import os

from model_service import ModelService
from ast import Mod
from multiprocessing.pool import RUN
import logging
import pandas as pd
from datetime import datetime

from pandas import DataFrame
from flask import Flask, request, jsonify
from pymongo import MongoClient
import requests
from model_loader import ModelLoader


EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTLY_SERVICE', 'http://127.0.0.1:8085')
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27018")
MONGODB_DB = os.getenv("MONGODB_DB", "prediction_service")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "data")
RUN_ID = os.getenv('RUN_ID', 'No RUN_ID provided')

log = logging.getLogger('gunicorn.error')

def get_mongo_collection(database_name, collection_name):
    mongo_client = MongoClient(MONGODB_ADDRESS)
    db = mongo_client.get_database(database_name)
    return db.get_collection(collection_name)

mongo_collection = get_mongo_collection(MONGODB_DB, MONGODB_COLLECTION)
model, dv = ModelLoader().load_model_from_mlflow(RUN_ID)
model_service = ModelService(model, dv)

app = Flask('duration-prediction')
app.logger.handlers = log.handlers


# GET payload example:
    # {
    #   "customer_age": 100,
    #   "gender": "F"",
    #   "dependent_count": 2,
    #   "education_level": 2,
    #   "marital_status": "married",
    #   "income_category": 2,
    #   "card_category": "blue",
    #   "months_on_book": 6,
    #   "total_relationship_count": 3,
    #   "credit_limit": 4000,
    #   "total_revolving_bal": 2500
    # }
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    log.info(f'Request payload: {request.get_data()}')

    input = request.get_json()
    # log.info(f'Request json: {input}')

    features = model_service.prepare_features(input)
    pred = model_service.predict(features)

    result = {
        'churn chance': float(str(pred)),
        'model_run_id': RUN_ID
    }

    save_to_db(input, float(pred))
    send_to_evidently_service(input, float(pred))
    
    log.info(f'Response payload: {result}')
    return jsonify(result)


def save_to_db(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    rec['created_at'] = datetime.now()
    rec['model_run_id'] = RUN_ID
    mongo_collection.insert_one(rec)


def send_to_evidently_service(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/capstone", json=[rec])



if __name__ == "__main__":
    log.info('App starting ...')
    app.run(debug=True, host='0.0.0.0', port=9696)
    
