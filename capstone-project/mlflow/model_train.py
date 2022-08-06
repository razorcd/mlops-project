#!/usr/bin/env python
# coding: utf-8

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


def train(dataFrame, y, max_depth, eta):
    # Hot Encoding
    dicts = dataFrame.to_dict(orient="records")
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(dicts)
    features = dv.get_feature_names()
    dtrain = xgb.DMatrix(X, label=y, feature_names=features)

    # train
    xgb_params = {
        'eta': eta,
        'max_depth': max_depth,
        'min_child_weight': 1,
        'objective': 'reg:squarederror',
        'nthread': 8,
        'seed':1,
        'verbosity':0
    }
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
    print("RMSE:", rmse)
    return rmse

def set_mlflow():
    mlflow.set_tracking_uri("sq`lite:///mlflow.db")
    mlflow.set_tracking_uri("http://localhost:5051")
    mlflow.set_experiment("capstone")

def run(data_input, model_output):
    df_clean = pd.read_csv(data_input)

    y_column = "active_customer"
    train_columns = ["customer_age", "gender", "dependent_count", "education_level", "marital_status", "income_category", "card_category", "months_on_book", "total_relationship_count", "credit_limit", "total_revolving_bal"]

    df_full_train, df_train, df_val, df_test, y_full_train, y_train, y_val, y_test = split_dataFrame(df_clean[train_columns+[y_column]], y_column)


    set_mlflow()

    with mlflow.start_run():

        mlflow.sklearn.autolog()
        mlflow.xgboost.autolog()
        mlflow.set_tag("model", "xgboost")
        dv, model = train(df_full_train, y_full_train, 6, 0.3)

        y_pred_val, X_val = predict(df_val, dv, model)

        rmse = get_rmse(y_val, y_pred_val)
        mlflow.log_metric("rmse", rmse) 

        with open("models/preprocesor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("models/preprocesor.b", artifact_path="preprocesor") #log model as artifact
        os.remove("models/preprocesor.b")


        # mlflow.log_artifact("models/preprocesor.b", artifact_path="preprocesor") #log model as artifact

        # mlflow.sklearn.log_model(booster, artifact_path="artifacts") #saves model, different way




    # pretty predict
    # df_pred_val_result = df_val.copy()
    # df_pred_val_result['y_pred'] = y_pred_val
    # df_pred_val_result['y_pred_bool'] = df_pred_val_result['y_pred']>0.5
    # df_pred_val_result['y_val'] = y_val
    # print(df_pred_val_result.head())
    # # df_pred_val_result[df_pred_val_result.y_pred_bool!=df_pred_val_result.y_val]



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_input",
        default="./input",
        help="the location where the input training data."
    )
    parser.add_argument(
        "--model_output",
        default="./model",
        help="the location where the output model."
    )
    args = parser.parse_args()

    # start servers:
    # mlflow server --backend-store-uri sqlite:///mlflow1.db --port 5051 --default-artifact-root file:///home/cristiandugacicu/projects/personal/mlops-zoomcamp/hw2/artifacts
    # mlflow ui --backend-store-uri sqlite:///mlflow.db -p 5052

    # mlflow.set_tracking_uri("sqlite:///mlflow.db")
    # mlflow.set_tracking_uri("http://localhost:5051")
    # mlflow.set_experiment("my-hw2")

    run(args.data_input, args.model_output)
