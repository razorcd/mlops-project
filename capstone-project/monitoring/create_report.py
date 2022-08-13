# pipenv run python create_report.py

import json
import os
import pickle
from datetime import datetime

import pandas
from prefect import flow, task
from pymongo import MongoClient
from bson.objectid import ObjectId
import pyarrow.parquet as pq
from prefect.task_runners import SequentialTaskRunner

from evidently import ColumnMapping

from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab,RegressionPerformanceTab

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection, RegressionPerformanceProfileSection
import mlflow
from mlflow.tracking import MlflowClient
import xgboost as xgb 

MLFLOW_ADDRESS = os.getenv("MLFLOW_ADDRESS", "http://127.0.0.1:5051")
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27018")
RUN_ID = os.getenv('RUN_ID', 'None')
REPORTS_FOLDER = "./reports"

#this data should come from data analysts after we know the results of the business
@task
def generate_fake_actual_values(filename):
    actual_values = []
    client = MongoClient(MONGODB_ADDRESS)
    collection = client.get_database("prediction_service").get_collection("data")
    for record in collection.find():
        v = {"id": str(record["_id"]), "actual_value": round(record["prediction"])}
        actual_values.append(v)
    return actual_values

@task
def upload_target(actual_values):
    client = MongoClient(MONGODB_ADDRESS)
    collection = client.get_database("prediction_service").get_collection("data")
    for actual_value in actual_values:
        collection.update_one({"_id": ObjectId(actual_value['id'])}, {"$set": {"target": float(actual_value['actual_value'])}})


@task
def load_reference_data(filename):
    mlflow.set_tracking_uri(MLFLOW_ADDRESS)
    mlflow.set_experiment("exp_flow_2")

    client = MlflowClient()
    mlflow_model_path = f'runs:/{RUN_ID}/model'
    # mlflow_model_path = f'models:/best_model-exp_flow_2/Staging'

    if not os.path.exists('/tmp/report_model'): os.mkdir('/tmp/report_model')
    model = mlflow.xgboost.load_model(mlflow_model_path)
    
    client.download_artifacts(RUN_ID, "preprocesor", '/tmp/report_model')
    with open('/tmp/report_model/preprocesor/preprocesor.b', 'rb') as f_in:
        dv = pickle.load(f_in)

    reference_data = pq.read_table(filename).to_pandas()

    # set column
    reference_data["target"] = reference_data["active_customer"].astype(float)
    # reference_data = reference_data.rename(columns={"active_customer": "target"})
    # reference_data = reference_data.drop(["active_customer"])
    features = ['customer_age', 'gender', 'dependent_count', 'education_level', 'marital_status', 'income_category', 'card_category', 'months_on_book', 'total_relationship_count', 'credit_limit', 'total_revolving_bal']
    x_pred = dv.transform(reference_data[features].to_dict(orient='records'))
    features = dv.get_feature_names()
    dpred = xgb.DMatrix(x_pred, feature_names=features)
    reference_data['prediction'] = model.predict(dpred)
    return reference_data


@task
def fetch_data():
    client = MongoClient(MONGODB_ADDRESS)
    data = client.get_database("prediction_service").get_collection("data").find({"target": {"$exists": True}})

    df = pandas.DataFrame(list(data))
    return df


@task
def run_evidently(ref_data, data):
    # ref_data.drop('ehail_fee', axis=1, inplace=True)
    # data.drop('ehail_fee', axis=1, inplace=True)  # drop empty column (until Evidently will work with it properly)
    profile = Profile(sections=[DataDriftProfileSection(), RegressionPerformanceProfileSection()])
    mapping = ColumnMapping(prediction="prediction", 
                            numerical_features=['customer_age', 'dependent_count', 'education_level', 'income_category', 'months_on_book', 'total_relationship_count', 'credit_limit', 'total_revolving_bal'],
                            categorical_features=['gender', 'marital_status', 'card_category'],
                            datetime_features=[])
    profile.calculate(ref_data, data, mapping)

    dashboard = Dashboard(tabs=[DataDriftTab(), RegressionPerformanceTab(verbose_level=0)])
    dashboard.calculate(ref_data, data, mapping)
    return json.loads(profile.json()), dashboard


@task
def save_report(result):
    client = MongoClient(MONGODB_ADDRESS)
    client.get_database("prediction_service").get_collection("report").insert_one(result[0])


@task
def save_html_report(result):
    if not os.path.exists('/tmp/reports'): os.mkdir('/tmp/reports')
    now = datetime.now() 
    date_time = now.strftime("%Y-%m-%d_%H-%M")
    result[1].save(f"{REPORTS_FOLDER}/data_report-{date_time}.html")


@flow(task_runner=SequentialTaskRunner())
def main():
    actual_values = generate_fake_actual_values("./evidently_service/datasets/credit_card_churn_clean.parquet")
    upload_target(actual_values)
    ref_data = load_reference_data("./evidently_service/datasets/credit_card_churn_clean.parquet")
    data = fetch_data()
    result = run_evidently(ref_data, data)
    save_report(result)
    save_html_report(result)

main()

#deploy scheduled runs:
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

DeploymentSpec(
    flow=main,
    name="evidently_data_reporting",
    schedule=IntervalSchedule(interval=timedelta(hours=24)),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)