
##Commands:

Run `pytest tests` for unit tests.

Run `run.sh` for full integration test.

```
docker build -t stream-model-duration:v2 .
```

```
docker run -it --rm \                     
    -p 8080:8080 \
    -e PREDICTIONS_STREAM_NAME="ride_predictions" \
    -e RUN_ID="e1efc53e9bd149078b0c12aeaa6365df" \
    -e TEST_RUN="True" \
    -e AWS_DEFAULT_REGION="eu-west-1" \
    stream-model-duration:v2

```


```
python test_docker.py
```

```
pipenv run pytest tests/

//or 

pipenv shell
pytest tests/
```


```
docker-copose up
aws --endpoint-url=http://localhost:4566 kinesis create-stream --stream-name ride-predictions --shard-count 1
aws --endpoint-url=http://localhost:4566 kinesis list-streams
aws kinesis get-shard-iterator \
        --shard-id shardId-000000000000 \
        --shard-iterator-type TRIM_HORIZON \
        --stream-name ride_predictions \
        --query 'ShardIterator' \
      --endpoint-url=http://localhost:4566
```