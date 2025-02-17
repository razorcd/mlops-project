import json
import os
from datetime import datetime, timedelta
import re

import pandas as pd
from pendulum import date
from prefect import flow, task, get_run_logger
from pymongo import MongoClient
from prefect.task_runners import SequentialTaskRunner

from evidently import ColumnMapping

from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab,RegressionPerformanceTab

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection, RegressionPerformanceProfileSection

#deploy scheduled runs:
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner


REPORT_TIME_WINDOW_MINUTES = int(os.getenv("REPORT_TIME_WINDOW_MINUTES", 180))
# MLFLOW_ADDRESS = os.getenv("MLFLOW_ADDRESS", "http://127.0.0.1:5051")
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27018")
# RUN_ID = os.getenv('RUN_ID', 'None')
REPORTS_FOLDER = os.getenv('REPORTS_FOLDER', "./reports")

client = MongoClient(MONGODB_ADDRESS)
data_collection = client.get_database("prediction_service").get_collection("data")
report_collection = client.get_database("prediction_service").get_collection("report")



def load_mongo_data_between(collection_name, datetime_field, from_dt, to_dt):
    result = data_collection.find({datetime_field: {'$gte': from_dt, '$lt': to_dt}})
    return list(result)
    

#this data should come from data analysts after we know the results of the business, including ID of each prediction.
# source fake fields:  predictionID, actual_value
@task
def generate_fake_actual_target():
    actual_values = []
    from_dt = datetime.now() - timedelta(days=1)
    to_dt = datetime.now()

    data_collection.update_many(
        {"created_at": {'$gte': from_dt, '$lt': to_dt}},
        [{ "$set": { "target": { "$add": [ { "$round": ['$prediction', 0] }] } } }]
    )
    

# loads older data to use as referance
@task
def load_reference_data(log):
    # reference data from one REPORT_TIME_WINDOW_MINUTES before
    from_dt = datetime.now() - timedelta(minutes=REPORT_TIME_WINDOW_MINUTES*2)
    to_dt = datetime.now() - timedelta(minutes=REPORT_TIME_WINDOW_MINUTES)
    log.info(f'ref_data between {from_dt} and {to_dt}')
    reference_data_list = load_mongo_data_between('data', 'created_at', from_dt, to_dt)
    reference_data = pd.DataFrame(reference_data_list)
    return reference_data


@task
def fetch_recent_data(log):
    # data = client.get_database("prediction_service").get_collection("data").find({"target": {"$exists": True}})
    from_dt = datetime.now() - timedelta(minutes=REPORT_TIME_WINDOW_MINUTES)
    to_dt = datetime.now()
    log.info(f'recent_data between {from_dt} and {to_dt}')
    recent_data = load_mongo_data_between('data', 'created_at', from_dt, to_dt)

    return pd.DataFrame(recent_data)


@task
def run_evidently(ref_data, recent_data):
    profile = Profile(sections=[DataDriftProfileSection(), RegressionPerformanceProfileSection()])
    mapping = ColumnMapping(prediction="prediction", 
                            target="target",
                            numerical_features=['customer_age', 'dependent_count', 'education_level', 'income_category', 'months_on_book', 'total_relationship_count', 'credit_limit', 'total_revolving_bal'],
                            categorical_features=['gender', 'marital_status', 'card_category'],
                            datetime_features=[])
    profile.calculate(ref_data, recent_data, mapping)

    dashboard = Dashboard(tabs=[DataDriftTab(), RegressionPerformanceTab(verbose_level=0)])
    dashboard.calculate(ref_data, recent_data, mapping)
    return json.loads(profile.json()), dashboard


@task
def save_report(result):
    client = MongoClient(MONGODB_ADDRESS)
    report_collection.insert_one(result[0])


@task
def save_html_report(result):
    if not os.path.exists(REPORTS_FOLDER): os.mkdir(REPORTS_FOLDER)
    now = datetime.now() 
    date_time = now.strftime("%Y-%m-%d_%H-%M")
    result[1].save(f"{REPORTS_FOLDER}/data_report-{date_time}.html")


@flow(task_runner=SequentialTaskRunner())
def main():
    log = get_run_logger()
    generate_fake_actual_target()

    ref_data = load_reference_data(log).result()
    recent_data = fetch_recent_data(log).result()
    if (len(ref_data)>0 and len(recent_data)>0):
        log.info(f"Generating report for: ref_data size: {len(ref_data)}, recent_data size: {len(recent_data)}")
        result = run_evidently(ref_data, recent_data)
        save_report(result)
        save_html_report(result)
    else:
        log.warning(f"Ignoring generating report because: ref_data size: {len(ref_data)}, recent_data size: {len(recent_data)}")

# main()

DeploymentSpec(
    flow=main,
    name="evidently_data_reporting",
    schedule=IntervalSchedule(interval=timedelta(minutes=REPORT_TIME_WINDOW_MINUTES)),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)