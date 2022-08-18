from http import server
import os
import json
import boto3
from time import sleep
import logging

from model_loader import ModelLoader
from model_service import ModelService

PREDICTIONS_STREAM_NAME = os.getenv('PREDICTIONS_STREAM_NAME', 'predictions')
RESULTS_STREAM_NAME = os.getenv('RESULTS_STREAM_NAME', 'results')
KINESIS_ADDRESS = os.getenv('KINESIS_ADDRESS', 'http://127.0.0.1:4566')
RUN_ID = os.getenv('RUN_ID', "c68164ce869e4c90a7c93752436e9bc7")

log = logging.getLogger(__name__)
# logging.basicConfig()
# logging.root.setLevel(logging.NOTSET)
logging.basicConfig(level=logging.INFO)


model, dv = ModelLoader().load_model_from_mlflow(RUN_ID)
model_service = ModelService(model, dv)


kinesis_client = boto3.client('kinesis', 
                              endpoint_url=KINESIS_ADDRESS,
                              region_name='eu-west-1')

response = kinesis_client.describe_stream(StreamName=PREDICTIONS_STREAM_NAME)
my_shard_id = response['StreamDescription']['Shards'][0]['ShardId']
shard_iterator = kinesis_client.get_shard_iterator(StreamName=PREDICTIONS_STREAM_NAME,
                                                   ShardId=my_shard_id,
                                                   ShardIteratorType='LATEST')
my_shard_iterator = shard_iterator['ShardIterator']

record_response = kinesis_client.get_records(ShardIterator=my_shard_iterator, Limit=2)


def publish_result(partition_key, result):
    kinesis_client.put_record(
        StreamName=RESULTS_STREAM_NAME,
        Data=json.dumps(result),
        PartitionKey=str(partition_key)
    )

log.info("Starting Kinesis listener.")

# aws kinesis put-record \          
#     --stream-name predictions --endpoint-url=http://localhost:4566 \
#     --partition-key 1 \
#     --data "ewogICAgICAiY3VzdG9tZXJfYWdlIjogMTAwLAogICAgICAiZ2VuZGVyIjogIkYiLAogICAgICAiZGVwZW5kZW50X2NvdW50IjogMiwKICAgICAgImVkdWNhdGlvbl9sZXZlbCI6IDIsCiAgICAgICJtYXJpdGFsX3N0YXR1cyI6ICJtYXJyaWVkIiwKICAgICAgImluY29tZV9jYXRlZ29yeSI6IDIsCiAgICAgICJjYXJkX2NhdGVnb3J5IjogImJsdWUiLAogICAgICAibW9udGhzX29uX2Jvb2siOiA2LAogICAgICAidG90YWxfcmVsYXRpb25zaGlwX2NvdW50IjogMywKICAgICAgImNyZWRpdF9saW1pdCI6IDQwMDAsCiAgICAgICJ0b3RhbF9yZXZvbHZpbmdfYmFsIjogMjUwMAogICAgfQ=="
while 'NextShardIterator' in record_response:
    record_response = kinesis_client.get_records(ShardIterator=record_response['NextShardIterator'], Limit=2)
    records = record_response['Records']

    for record in records:
        log.info(f"Received event: {record['Data']}")
        input = json.loads(record['Data'])

        features = model_service.prepare_features(input)
        pred = model_service.predict(features)

        result = {
            'churn chance': float(str(pred)),
            'model_run_id': RUN_ID
        }
        log.info(f"Response: {result}")
        publish_result("partition1", result)

    sleep(5)

