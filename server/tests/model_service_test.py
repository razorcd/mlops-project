from model_service import ModelService
import numpy as np

class ModelMock:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return [self.value]


class DvMock:
    def transform(_self, dicts):
        return np.random.rand(5, 2)
    def get_feature_names_out(_self):
        return ["f0", "f1"]    

def test_predict():
    modelMock = ModelMock([22.5])
    dvMock = DvMock()
    # model_service = model.ModelService(dvMock, modelMock)

    features = {
      "customer_age": 100,
      "gender": "F",
      "dependent_count": 2,
      "education_level": 2,
      "marital_status": "married",
      "income_category": 2,
      "card_category": "blue",
      "months_on_book": 6,
      "total_relationship_count": 3,
      "credit_limit": 4000,
      "total_revolving_bal": 2500
    }

    actual_prediction = ModelService(modelMock, dvMock).predict([features])


    assert actual_prediction == [22.5]

def test_prepare_features():
    input = {
      "customer_age": 100,
      "gender": "F",
      "dependent_count": 2,
      "education_level": 2,
      "marital_status": "married",
      "income_category": 2,
      "card_category": "blue",
      "months_on_book": 6,
      "total_relationship_count": 3,
      "credit_limit": 4000,
      "total_revolving_bal": 2500
    }

    expected_output = {
      "customer_age": 100,
      "gender": "F",
      "dependent_count": 2,
      "education_level": 2,
      "marital_status": "married",
      "income_category": 2,
      "card_category": "blue",
      "months_on_book": 6,
      "total_relationship_count": 3,
      "credit_limit": 4000,
      "total_revolving_bal": 2500
    }
    
    actual_output = ModelService(None, None).prepare_features(input)

    assert actual_output == expected_output
    assert type(actual_output["customer_age"]) == int
    assert type(actual_output["gender"]) == str
    assert type(actual_output["dependent_count"]) == int
    assert type(actual_output["education_level"]) == int
    assert type(actual_output["marital_status"]) == str
    assert type(actual_output["income_category"]) == int
    assert type(actual_output["card_category"]) == str
    assert type(actual_output["months_on_book"]) == int
    assert type(actual_output["total_relationship_count"]) == int
    assert type(actual_output["credit_limit"]) == float
    assert type(actual_output["total_revolving_bal"]) == int
