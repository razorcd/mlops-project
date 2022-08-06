#!/usr/bin/env python
# coding: utf-8

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


def split_dataFrame(df_to_split, y_column):
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

    print("df_to_split length: ", len(df_to_split))
    print()
    print("df_full_train length: ", len(df_full_train))
    print("df_train length: ", len(df_train))
    print("df_val length: ", len(df_val))
    print("df_test length: ", len(df_test))
    print()
    print("y_full_train length: ", len(y_full_train))
    print("y_train length: ", len(y_train))
    print("y_val length: ", len(y_val))
    print("y_test length: ", len(y_test))
    
    return df_full_train, df_train, df_val, df_test, y_full_train, y_train, y_val, y_test


def train(dataFrame, y, xgb_params):
    # Hot Encoding
    dicts = dataFrame.to_dict(orient="records")
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(dicts)
    features = dv.get_feature_names_out()
    dtrain = xgb.DMatrix(X, label=y, feature_names=features)

    # train
    model = xgb.train(xgb_params, dtrain, num_boost_round=10)
    return dv, model


def predict(dataFrame, dv, model):
    dicts = dataFrame.to_dict(orient="records")
    X = dv.transform(dicts)
    features = dv.get_feature_names()
    dval = xgb.DMatrix(X, feature_names=features)
    y_pred = model.predict(dval)
    return y_pred, X

def get_rmse(y_val, y_pred_val):
    # mae = metrics.mean_absolute_error(y_val, y_pred_val)
    # mse = metrics.mean_squared_error(y_val, y_pred_val)
    rmse = np.sqrt(metrics.mean_squared_error(y_val, y_pred_val))

    # print("MAE for numerical linear:", mae)
    # print("MSE for numerical linear:", mse)
    # print("RMSE:", rmse)
    return rmse

def set_mlflow(experiment):
    mlflow.set_tracking_uri("sq`lite:///mlflow.db")
    mlflow.set_tracking_uri("http://localhost:5051")
    mlflow.set_experiment(experiment)

def run(experiement, data_input):
    df_clean = pd.read_csv(data_input)

    y_column = "active_customer"
    train_columns = ["customer_age", "gender", "dependent_count", "education_level", "marital_status", "income_category", "card_category", "months_on_book", "total_relationship_count", "credit_limit", "total_revolving_bal"]

    df_full_train, df_train, df_val, df_test, y_full_train, y_train, y_val, y_test = split_dataFrame(df_clean[train_columns+[y_column]], y_column)

    mlflow.xgboost.autolog()

    # xgb_params_search_space = {
    #     'max_depth': scope.int(hp.choice('max_depth', [5, 10, 12, 13, 14, 20,30,40,50,100])),
    #     'eta': scope.int(hp.choice('eta', [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.6, 1, 2, 5, 10])),
    #     'min_child_weight': 1,
    #     'objective': 'reg:squarederror',
    #     'nthread': 8,
    #     'verbosity':0,
    #     "seed":42
    # }
    xgb_params_search_space = {
        'max_depth': scope.int(hp.choice('max_depth', [5, 10])),
        'eta': scope.int(hp.choice('eta', [0.001])),
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
            # print(f'Training model. Active MLFlow run_id: {active_mlflow_run_id}')
            mlflow.set_tag("model", "xgboost")

            dv, model = train(df_full_train, y_full_train, params)

            preprocesor_path = f'tmp/{active_mlflow_run_id}'
            os.mkdir(preprocesor_path)
            with open(f'{preprocesor_path}/preprocesor.b', "wb") as f_out:
                pickle.dump(dv, f_out)
            mlflow.log_artifact(f'{preprocesor_path}/preprocesor.b', artifact_path="preprocesor") #log dv as artifact
            os.remove(f'{preprocesor_path}/preprocesor.b')
            print(preprocesor_path)
            os.removedirs(preprocesor_path)
            print(os.listdir('.'))

            y_pred_val, X_val = predict(df_val, dv, model)
            rmse = get_rmse(y_val, y_pred_val)
            mlflow.log_metric("rmse", rmse) 

        return {'loss':rmse, 'status':STATUS_OK}

    best_result = fmin(
        fn=objective,
        space=xgb_params_search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials()
    )


def register_best_run(experiment_name):

    client = MlflowClient()

    # retrieve the top_n model runs and log the models to MLflow
    # experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    # runs = client.search_runs(
    #     experiment_ids=experiment.experiment_id,
    #     run_view_type=ViewType.ACTIVE_ONLY,
    #     max_results=log_top,
    #     order_by=["metrics.rmse ASC"]
    # )
    # for run in runs:
    #     train_and_log_model(data_path=data_path, params=run.data.params)

    # select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(experiment_name)
    print(experiment)
    best_runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=50,
        order_by=["metrics.rmse ASC"]
    )
    
    print(f'models count: {len(best_runs)}')
    if (len(best_runs)==0): raise "No models found."
    print(f'top models found: {best_runs[0]}')

    # register the best model
    model_uri = f"runs:/{best_runs[0].info.run_id}/model"
    print(f'Registering {model_uri}')
    mv = mlflow.register_model(model_uri=model_uri, name = f"best_model-{experiment_name}")
    print(f"Registrered model {mv.name}, version: {mv.version}")
    # client.update_registered_model(
    #     name=mv.name,
    #     description=f"rmse={best_runs[0].data.metrics['rmse']}"
    # )
    # client.list_registered_models()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_input",
        default="./input",
        help="the location where the input training data."
    )
    parser.add_argument(
        "--experiment",
        default="capstone",
        help="the MLFlow experiement."
    )
    args = parser.parse_args()

    # start servers:
    # mlflow server --backend-store-uri sqlite:///mlflow1.db --port 5051 --default-artifact-root file:///home/cristiandugacicu/projects/personal/mlops-zoomcamp/hw2/artifacts
    # mlflow ui --backend-store-uri sqlite:///mlflow.db -p 5052

    # mlflow.set_tracking_uri("sqlite:///mlflow.db")
    # mlflow.set_tracking_uri("http://localhost:5051")
    # mlflow.set_experiment("my-hw2")

    set_mlflow(args.experiment)
    run(args.experiment, args.data_input)
    register_best_run(args.experiment)
