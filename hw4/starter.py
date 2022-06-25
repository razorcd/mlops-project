#!/usr/bin/env python
# coding: utf-8

#use:
    # pipenv install
    # pipenv run python starter.py tripdata 2021 01

import os
import pickle
import uuid
import sys

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline


categorical = ['PUlocationID', 'DOlocationID']

with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


def read_data(filename, year, month):
    print(f'Reading data {filename}')
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    return df



# df = read_data('https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_????-??.parquet')
# for month in range(2,12):
#   new_df = read_data(f'data/fhv_tripdata_{year:04d}-{month:02d}.parquet')
#   df.append(df)
  

def predict(df):
    print('predicting')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print(f'Mean predicted duration: {y_pred.mean()}')
    return y_pred





def prepare_output(df, y_pred):
    print('preparing output')
    df_pred = pd.DataFrame()
    df_pred['ride_id'] = df['ride_id']
    df_pred['prediction'] = y_pred
    return df_pred


def write_data(df_pred, output_file):
    print(f'Writing data {output_file}')
    df_pred.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )





def run():
    taxi_type = sys.argv[1] #'tripdata'
    year = int(sys.argv[2]) #2021
    month = int(sys.argv[3]) #2
    # run_id = sys.argv[4]

    input_file = f'data/fhv_{taxi_type}_{year:04d}-{month:02d}.parquet'
    output_file = f'output/fhv_{taxi_type}_{year:04d}-{month:02d}.parquet'

    df = read_data(input_file, year, month)
    

    # mlflow.set_tracking_uri("http://127.0.0.1:5051")
    # mlflow.set_experiment("green-taxi-duration")

    y_pred = predict(df)
    df_pred = prepare_output(df, y_pred)
    write_data(df_pred, output_file)



if __name__ == '__main__':
    run()