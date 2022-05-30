import argparse
import os
import pickle

import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run(data_path):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))


    with mlflow.start_run():

        mlflow.sklearn.autolog()

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_valid)

        rmse = mean_squared_error(y_valid, y_pred, squared=False)


        # mlflow.log_artifact("models/preprocesor.b", artifact_path="preprocesor") #log model as artifact

        # mlflow.sklearn.log_model(booster, artifact_path="artifacts") #saves model, different way


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./output",
        help="the location where the processed NYC taxi trip data was saved."
    )
    args = parser.parse_args()

    # start servers:
    # mlflow server --backend-store-uri sqlite:///mlflow1.db --port 5051 --default-artifact-root file:///home/cristiandugacicu/projects/personal/mlops-zoomcamp/hw2/artifacts
    # mlflow ui --backend-store-uri sqlite:///mlflow.db -p 5052

    # mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_tracking_uri("http://localhost:5051")
    mlflow.set_experiment("my-hw2")



    run(args.data_path)
