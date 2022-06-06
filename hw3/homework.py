import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import task, flow, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from prefect.context import TaskRunContext
from prefect.logging.loggers import *

from datetime import date, datetime
from dateutil.relativedelta import relativedelta


@task
def get_paths(d):
    if (d==None): dateobj = today = date.today()
    else: 
        date_format = '%Y-%m-%d'
        dateobj = datetime.strptime(d, date_format).date()
    
    train_date = dateobj - relativedelta(months=2)
    val_date = dateobj - relativedelta(months=1)

    train_date_str = train_date.strftime('%Y-%m')
    val_date_str = val_date.strftime('%Y-%m')

    train_path = f"./data/fhv_tripdata_{train_date_str}.parquet"
    get_run_logger().info(f"Train path: {train_path}")
    val_path = f"./data/fhv_tripdata_{val_date_str}.parquet"
    get_run_logger().info(f"Val path: {val_path}")
    return train_path, val_path

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        get_run_logger().info(f"The mean duration of training is {mean_duration}")
    else:
        get_run_logger().info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    get_run_logger().info(f"The shape of X_train is {X_train.shape}")
    get_run_logger().info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    get_run_logger().info(f"The MSE of training is: {mse}")

    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    get_run_logger().info(f"The MSE of validation is: {mse}")
    return



@flow(task_runner=SequentialTaskRunner()) #all tasks will run in order
def main(date="2021-08-15"):
    train_path, val_path = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

main()
