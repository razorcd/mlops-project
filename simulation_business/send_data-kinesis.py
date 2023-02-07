import os
import json
import boto3
from time import sleep
import logging
import pyarrow.csv as pcsv

RESULTS_STREAM_NAME = os.getenv('RESULTS_STREAM_NAME', 'results')
KINESIS_ADDRESS = os.getenv('KINESIS_ADDRESS', 'http://127.0.0.1:4566')

log = logging.getLogger(__name__)
# logging.basicConfig()
# logging.root.setLevel(logging.NOTSET)
logging.basicConfig(level=logging.INFO)


kinesis_client = boto3.client('kinesis', 
                              endpoint_url=KINESIS_ADDRESS,
                              region_name='eu-west-1')

def publish_result(partition_key, event):
    kinesis_client.put_record(
        StreamName=RESULTS_STREAM_NAME,
        Data=json.dumps(event),
        PartitionKey=str(partition_key)
    )

log.info("Starting Kinesis publisher.")

table = pcsv.read_csv("input_clean/credit_card_churn_clean.csv")
data = table.to_pylist()


while True:

  for row in data:
      payload = {
        'customer_age': row['customer_age'],
        'gender': row['gender'],
        'dependent_count': row['dependent_count'],
        'education_level': row['education_level'],
        'marital_status': row['marital_status'],
        'income_category': row['income_category'],
        'card_category': row['card_category'],
        'months_on_book': row['months_on_book'],
        'total_relationship_count': row['total_relationship_count'],
        'credit_limit': row['credit_limit'],
        'total_revolving_bal': row['total_revolving_bal'],
      }

      print(f'Publishing: {payload}')
      publish_result("pred1", payload)
      sleep(60) #sec
