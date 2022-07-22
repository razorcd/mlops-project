import model


def test_base64_decode():
    base64_input = "ewogICAgICAgICJyaWRlIjogewogICAgICAgICAgICAiUFVMb2NhdGlvbklEIjogMTMwLAogICAgICAgICAgICAiRE9Mb2NhdGlvbklEIjogMjA1LAogICAgICAgICAgICAidHJpcF9kaXN0YW5jZSI6IDMuNjYKICAgICAgICB9LCAKICAgICAgICAicmlkZV9pZCI6IDI1NgogICAgfQ=="
    actual_result = model.base64_decode(base64_input)
    expected_result = {
        "ride": {
            "PULocationID": 130,
            "DOLocationID": 205,
            "trip_distance": 3.66
        }, 
        "ride_id": 256
    }

    assert actual_result == expected_result


def test_prepare_features():
    model_service = model.ModelService(None, None)

    ride = {
        "PULocationID": 130,
        "DOLocationID": 205,
        "trip_distance": 3.66
    }
    actual_features = model_service.prepare_features(ride)

    expected_features = {
        "PU_DO": "130_205",
        "trip_distance": 3.66
    }

    assert actual_features == expected_features


class ModelMock:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return [self.value] * len(X)


class DvMock:
    def transform(self, features):
        return features

def test_predict():
    modelMock = ModelMock(10.0)
    dvMock = DvMock()
    model_service = model.ModelService(dvMock, modelMock)

    features = {
        "PU_DO": "130_205",
        "trip_distance": 3.66
    }

    actual_prediction = model_service.predict(features)

    
    assert actual_prediction == 10.0


def test_lambda_handler():
    modelMock = ModelMock(10.0)
    dvMock = DvMock()
    model_version = "v123"
    model_service = model.ModelService(dvMock, modelMock, model_version)

    event = {
        "Records": [
            {
                "kinesis": {
                    "data": "ewogICAgICAgICJyaWRlIjogewogICAgICAgICAgICAiUFVMb2NhdGlvbklEIjogMTMwLAogICAgICAgICAgICAiRE9Mb2NhdGlvbklEIjogMjA1LAogICAgICAgICAgICAidHJpcF9kaXN0YW5jZSI6IDMuNjYKICAgICAgICB9LCAKICAgICAgICAicmlkZV9pZCI6IDI1NgogICAgfQ=="
                }
            }
        ]
    }


    actual_prediction = model_service.lambda_handler(event)
    
    assert actual_prediction == {
        'predictions': [{
                'model': 'ride_duration_prediction_model',
                'version': model_version,
                'prediction': {
                    'ride_duration': 10.0,
                    'ride_id': 256
                }
            }]
    }