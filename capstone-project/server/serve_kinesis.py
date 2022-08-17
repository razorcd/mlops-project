from http import server
import os
import json
import boto3
from time import sleep

from model_loader import ModelLoader
from model_service import ModelService

PREDICTIONS_STREAM_NAME = os.getenv('PREDICTIONS_STREAM_NAME', 'predictions')
RUN_ID = os.getenv('RUN_ID', "c68164ce869e4c90a7c93752436e9bc7")

model, dv = ModelLoader().load_model_from_mlflow(RUN_ID)
model_service = ModelService(model, dv)


kinesis_client = boto3.client('kinesis', 
                              endpoint_url='http://127.0.0.1:4566',
                              region_name='eu-west-1')

response = kinesis_client.describe_stream(StreamName=PREDICTIONS_STREAM_NAME)
my_shard_id = response['StreamDescription']['Shards'][0]['ShardId']
shard_iterator = kinesis_client.get_shard_iterator(StreamName=PREDICTIONS_STREAM_NAME,
                                                   ShardId=my_shard_id,
                                                   ShardIteratorType='LATEST')
my_shard_iterator = shard_iterator['ShardIterator']

record_response = kinesis_client.get_records(ShardIterator=my_shard_iterator, Limit=2)



# aws kinesis put-record \          
#     --stream-name predictions --endpoint-url=http://localhost:4566 \
#     --partition-key 1 \
#     --data "ewogICAgICAiY3VzdG9tZXJfYWdlIjogMTAwLAogICAgICAiZ2VuZGVyIjogIkYiLAogICAgICAiZGVwZW5kZW50X2NvdW50IjogMiwKICAgICAgImVkdWNhdGlvbl9sZXZlbCI6IDIsCiAgICAgICJtYXJpdGFsX3N0YXR1cyI6ICJtYXJyaWVkIiwKICAgICAgImluY29tZV9jYXRlZ29yeSI6IDIsCiAgICAgICJjYXJkX2NhdGVnb3J5IjogImJsdWUiLAogICAgICAibW9udGhzX29uX2Jvb2siOiA2LAogICAgICAidG90YWxfcmVsYXRpb25zaGlwX2NvdW50IjogMywKICAgICAgImNyZWRpdF9saW1pdCI6IDQwMDAsCiAgICAgICJ0b3RhbF9yZXZvbHZpbmdfYmFsIjogMjUwMAogICAgfQ=="
while 'NextShardIterator' in record_response:
    record_response = kinesis_client.get_records(ShardIterator=record_response['NextShardIterator'], Limit=2)
    records = record_response['Records']

    for record in records:
        print(record['Data'])
        input = json.loads(record['Data'])


        features = model_service.prepare_features(input)
        pred = model_service.predict(features)

        result = {
            'churn chance': float(str(pred)),
            'model_run_id': RUN_ID
        }

        print(result)

    sleep(5)


# kinesis_client.put_record(
#     StreamName=PREDICTIONS_STREAM_NAME,
#     Data=json.dumps(prediction_event),
#     PartitionKey=str(ride_id)
# )
