import os
import json
import base64
import pickle

import boto3

# import mlflow

def load_model(run_id):
    print(f'Loading model with run_id={run_id}')
    # logged_model = f's3://mlflow-models-alexey/1/{run_id}/artifacts/model'
    logged_model = 'lin_reg.bin'
    # logged_model = f'runs:/{run_id}/model'
    # model = mlflow.pyfunc.load_model(logged_model)

    with open(logged_model, 'rb') as f_in:
        (dv, model) = pickle.load(f_in)

    return dv, model

def base64_decode(encoded_data):
    print(f'decoding data: {encoded_data}')
    decoded_data = base64.b64decode(encoded_data).decode('utf-8')
    ride_event = json.loads(decoded_data)
    return ride_event


class ModelService():

    def __init__(self, dv, model, model_version=None, callbacks=None):
        self.model = model
        self.dv = dv
        self.model_version = model_version
        self.callbacks = callbacks or []

    def prepare_features(self, ride):
        features = {}
        features['PU_DO'] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
        features['trip_distance'] = ride['trip_distance']
        return features


    def predict(self, features):
        X = self.dv.transform(features)
        preds = self.model.predict(X)
        return float(preds[0])


    def lambda_handler(self,event):
        # print(json.dumps(event))

        predictions_events = []

        for record in event['Records']:
            encoded_data = record['kinesis']['data']
            ride_event = base64_decode(encoded_data)

            # print(ride_event)
            ride = ride_event['ride']
            ride_id = ride_event['ride_id']

            features = self.prepare_features(ride)
            prediction = self.predict(features)

            prediction_event = {
                'model': 'ride_duration_prediction_model',
                'version': self.model_version,
                'prediction': {
                    'ride_duration': prediction,
                    'ride_id': ride_id
                }
            }

            for callback in self.callbacks:
                print(f'Sending to callback: {callback}, prediction_event: {prediction_event}')
                callback(prediction_event)

            predictions_events.append(prediction_event)


        return {
            'predictions': predictions_events
        }

class KinesisCallback():
    def __init__(self, kinesis_client, prediction_stream_name):
        self.kinesis_client = kinesis_client
        self.prediction_stream_name = prediction_stream_name

    def put_record(self, prediction_event):
        ride_id = prediction_event['prediction']['ride_id']
        self.kinesis_client.put_record(
            StreamName = self.prediction_stream_name,
            Data = json.dumps(prediction_event),
            PartitionKey = str(ride_id)
        )


def create_kinesis_client():
    endpoint_url = os.getenv("KINESIS_ENDPOINT_URL")
    if endpoint_url is None:
        return boto3.client("kinesis")

    return boto3.client("kinesis", endpoint_url=endpoint_url)

def init(prediction_stream_name:str, run_id:str, test_run:bool=False):
    print(f'test_run={test_run}')
    dv, model = load_model(run_id)
    version = os.getenv("RUN_ID")


    callbacks = []
    if not test_run:
        print("Preparing Kinesis and sending record !!!")
        kinesis_client = create_kinesis_client()
        kinesis_callback = KinesisCallback(kinesis_client, prediction_stream_name)
        callbacks.append(kinesis_callback.put_record)

    model_service = ModelService(dv, model, model_version=version,callbacks=callbacks)

    return model_service
