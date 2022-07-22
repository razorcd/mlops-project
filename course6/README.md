
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