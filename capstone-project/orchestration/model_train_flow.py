from prefect import task, flow, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from ast import arg
from ensurepip import version
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import mlflow
import pickle
from hyperopt import hp, space_eval
from hyperopt.pyll import scope
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import pyarrow

log = None

# 'client_kwargs': {'endpoint_url': 'http://localhost:4566'}
s3_storage_options = {
        'key': 'ID1',
        'secret' : 'None',
        'client_kwargs': {'endpoint_url': 'http://s3:4566', 'region_name': 'eu-west-1'}
    }

@task
def read_file(data_input):
    if (data_input.startswith("s3")):
        file = pd.read_csv(
            data_input,
            engine='pyarrow',
            storage_options=s3_storage_options
        )
        return file
    else: 
        return pd.read_csv(data_input)

@task
def split_dataFrame(df_clean):
    y_column = "active_customer"
    train_columns = ["customer_age", "gender", "dependent_count", "education_level", "marital_status", "income_category", "card_category", "months_on_book", "total_relationship_count", "credit_limit", "total_revolving_bal"]

    df_to_split = df_clean[train_columns+[y_column]]

    df_full_train, df_test = train_test_split(df_to_split, test_size=0.2, random_state=11)
    df_train,  df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

    df_full_train = df_full_train.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_full_train = df_full_train[y_column]
    y_train = df_train[y_column]
    y_val = df_val[y_column]
    y_test = df_test[y_column]

    del df_full_train[y_column]
    del df_train[y_column]
    del df_val[y_column]
    del df_test[y_column]

    # with pd.option_context('display.max_rows', 2, 'display.max_columns', None): 
    #     display(df_test)   

    # log.info(f"""
    #     df_to_split length: {len(df_to_split)}

    #     df_full_train length: {len(df_full_train)}
    #     df_train length: {len(df_train)}
    #     df_val length: {len(df_val)}
    #     df_test length: {len(df_test)}

    #     y_full_train length: {len(y_full_train)}
    #     y_train length: {len(y_train)}
    #     y_val length: {len(y_val)}
    #     y_test length: {len(y_test)}
    # """)

    return df_full_train, df_train, df_val, df_test, y_full_train, y_train, y_val, y_test

def train(dataFrame, y, xgb_params):
    # Hot Encoding
    dicts = dataFrame.to_dict(orient="records")
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(dicts)

    features = dv.get_feature_names_out()
    # log.info(features)
    dtrain = xgb.DMatrix(X, label=y, feature_names=features, enable_categorical=True)

    # train
    model = xgb.train(xgb_params, dtrain, num_boost_round=10)
    # log.info(model.feature_names)

    return dv, model

def predict(dataFrame, dv, model):
    dicts = dataFrame.to_dict(orient="records")
    X = dv.transform(dicts)
    features = dv.get_feature_names_out()
    dval = xgb.DMatrix(X, feature_names=features)
    y_pred = model.predict(dval)
    return y_pred, X

def get_rmse(y_val, y_pred_val):
    mae = metrics.mean_absolute_error(y_val, y_pred_val)
    mse = metrics.mean_squared_error(y_val, y_pred_val)
    rmse = np.sqrt(metrics.mean_squared_error(y_val, y_pred_val))

    # log.info("MAE for numerical linear:", mae)
    # log.info("MSE for numerical linear:", mse)
    # log.info("RMSE:", rmse)
    return mae, mse, rmse

@task
def set_mlflow(experiment):
    # mlflow.set_tracking_uri("sqlite:///db/mlflow.db")
    mlflow.set_tracking_uri("http://mlflow_server:5050")
    # mlflow.set_tracking_uri("http://localhost:5051")
    mlflow.set_experiment(experiment)


@task
def run_hyperoptimization(experiement, df_full_train, df_val, y_full_train, y_val):
    mlflow.xgboost.autolog()

    xgb_params_search_space = {
        'max_depth': scope.int(hp.choice('max_depth', [5, 10, 12, 13, 14, 20,30,40,50,100])),
        'eta': scope.int(hp.choice('eta', [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.6, 1, 2, 5, 10])),
        'min_child_weight': 1,
        'objective': 'reg:squarederror',
        'nthread': 8,
        'verbosity':0,
        "seed":42
    }

    def objective(params):
        with mlflow.start_run():
            active_mlflow_run_id = mlflow.active_run().info.run_id
            if (active_mlflow_run_id==None): raise ValueError("missing MLFlow active run.")
            # log.info(f'Training model. Active MLFlow run_id: {active_mlflow_run_id}')
            mlflow.set_tag("model", "xgboost")

            dv, model = train(df_full_train, y_full_train, params)

            preprocesor_path = f'./tmp/{active_mlflow_run_id}'
            os.mkdir(preprocesor_path)
            with open(f'{preprocesor_path}/preprocesor.b', "wb") as f_out:
                pickle.dump(dv, f_out)
            mlflow.log_artifact(f'{preprocesor_path}/preprocesor.b', artifact_path="preprocesor") #log dv as artifact
            os.remove(f'{preprocesor_path}/preprocesor.b')
            os.removedirs(preprocesor_path)

            y_pred_val, X_val = predict(df_val, dv, model)
            mae, mse, rmse = get_rmse(y_val, y_pred_val)
            mlflow.log_metric("mae", mae) 
            mlflow.log_metric("mse", mse) 
            mlflow.log_metric("rmse", rmse) 

        return {'loss':rmse, 'status':STATUS_OK}

    best_result = fmin(
        fn=objective,
        space=xgb_params_search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials()
    )

@task
def register_best_run(experiment_name):

    client = MlflowClient()

    # select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(experiment_name)
    # log.info(experiment)
    best_runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=50,
        order_by=["metrics.rmse ASC"]
    )
    
    # log.info(f'Models count: {len(best_runs)}')
    if (len(best_runs)==0): raise "No models found."
    # log.info(f'Top model found: {best_runs[0]}')

    # register the best model
    model_uri = f"runs:/{best_runs[0].info.run_id}/model"
    # log.info(f'Registering {model_uri}')
    mv = mlflow.register_model(model_uri=model_uri, name = f"best_model-{experiment_name}")
    # log.info(f"Registrered model {mv.name}, version: {mv.version}")
    # client.update_registered_model(
    #     name=mv.name,
    #     description=f"rmse={best_runs[0].data.metrics['rmse']}"
    # )
    # client.list_registered_models()

@flow(task_runner=SequentialTaskRunner())
def main():
    import subprocess
    subprocess.run('pwd')
    
    log = get_run_logger()

    experiment = "exp_flow_2"
    # data_input_path = "../input_clean/credit_card_churn_clean.csv"
    data_input_path = "s3://capstone/ID1/credit_card_churn_2022-08-07.csv"

    set_mlflow(experiment)
    df_clean = read_file(data_input_path)
    df_full_train, df_train, df_val, df_test, y_full_train, y_train, y_val, y_test = split_dataFrame(df_clean).result()
    run_hyperoptimization(experiment, df_full_train, df_val, y_full_train, y_val)
    register_best_run(experiment)

# main()    



#deploy scheduled runs:
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

DeploymentSpec(
    flow=main,
    name="model_tuning_and_uploading",
    schedule=IntervalSchedule(interval=timedelta(minutes=60)),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)