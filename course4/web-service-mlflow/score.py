#!/usr/bin/env python
# coding: utf-8

# to run:
#   python score.py green_tripdata_ 2021 2 6e6e2893453049cf88231bc93b4e8e83

import os
import pickle
import uuid
import sys

import pandas as pd

# !pip install mlflow
import mlflow

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline




def generate_uuids(n):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids

def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['ride_id'] = generate_uuids(len(df))    
    return df


def prepare_dictionaries(df: pd.DataFrame):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts

def load_model(run_id):
    # logged_model = f's3://mlflow-models-alexey/1/{run_id}/artifacts/model'
    logged_model = f'runs:/{run_id}/model'
    model = mlflow.pyfunc.load_model(logged_model)
    return model


def apply_model(input_file, run_id,  output_file):
    print(f'Loading data from {input_file}')
    # !pip install pyarrow
    df = read_dataframe(input_file)

    dicts = prepare_dictionaries(df)

    print(f'Loading model with id {run_id}')
    model = load_model(run_id)

    print(f'applying the model...')
    y_pred = model.predict(dicts)

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['lpep_dropoff_datetime'] = df['lpep_dropoff_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id
    df_result.head(3)

    print(f'Writing data to {output_file}')
    df_result.to_parquet(output_file)

    return df_result


# !mkdir output
def run():
    taxi_type = sys.argv[1] #'green_tripdata_'
    year = int(sys.argv[2]) #2021
    month = int(sys.argv[3]) #2
    run_id = sys.argv[4]

    input_file = f'data/{taxi_type}{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}{year:04d}-{month:02d}.parquet'

    mlflow.set_tracking_uri("http://127.0.0.1:5051")
    mlflow.set_experiment("green-taxi-duration")

    apply_model(input_file, run_id,  output_file)


if __name__ == '__main__':
    run()

