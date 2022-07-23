#!/usr/bin/env python
# coding: utf-8
import os
import sys
import pickle
import pandas as pd

def get_input_path(year, month):
    default_input_pattern = 'data/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    # default_input_pattern = 'https://.../datasets/master/nyc-tlc/fhv/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    # default_output_pattern = 'output/output_{year:04d}_{month:02d}_predictions.parquet'
    default_output_pattern = 's3://nyc-duration/ID1/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def main(input_file, output_file):
    print(f'Input file: {input_file}')

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)


    categorical = ['PUlocationID', 'DOlocationID']

    rawdf = read_data(input_file)
    df = prepare_data(rawdf, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)


    print('predicted mean duration:', y_pred.mean())


    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    print(f'Output file: {output_file}')

    # dict_ecs = {anon': False, 'key':  'my_key', 'secret': 'my_secret', 'use_ssl': False,  'client_kwargs': {'endpoint_url': 'http://....'}}
    storage_options = {
        'client_kwargs': {'endpoint_url': 'http://localhost:4566'}
    }
    df_result.to_parquet(output_file, engine='pyarrow', index=False, storage_options=storage_options)


def read_data(filename):
    df = pd.read_parquet(filename)
    return df


def prepare_data(df, categorical):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    main(input_file, output_file)

