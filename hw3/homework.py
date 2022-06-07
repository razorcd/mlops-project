import pandas as pd
import pickle

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
def get_paths(dateobj):
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
def main(d="2021-08-15"):  #run_model-6559300c-0' - The MSE of validation is: 11.63703272181817
    if (d==None): dateobj = today = date.today()
    else: 
        date_format = '%Y-%m-%d'
        dateobj = datetime.strptime(d, date_format).date()
    
    train_path, val_path = get_paths(dateobj).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    files_date = dateobj.strftime('%Y-%m-%d')
    with open(f"models/model-{files_date}.bin", "wb") as f_out:
            pickle.dump(lr, f_out)
    with open(f"models/dv-{files_date}.b", "wb") as f_out:
            pickle.dump(dv, f_out) # 16K Jun  6 18:23 dv-2021-08-15.b
        
    run_model(df_val_processed, categorical, dv, lr)

# main()

#deploy scheduled runs:
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner


DeploymentSpec(
    flow=main,
    name="model_training",
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="UTC"
    ),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)  # 3 upcoming schedules

# prefect work-queue ls