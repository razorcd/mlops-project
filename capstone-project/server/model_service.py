import xgboost as xgb

class ModelService():
    def __init__(self, model, dv, model_version=None, callbacks=None):
        self.model = model
        self.dv = dv
        self.model_version = model_version
        self.callbacks = callbacks or []

    def prepare_features(_self, input):
        features = {
            'customer_age': input['customer_age'], 
            'gender': input['gender'], 
            'dependent_count': input['dependent_count'],
            'education_level': input['education_level'],
            'marital_status': input['marital_status'],
            'income_category': input['income_category'],
            'card_category': input['card_category'],
            'months_on_book': input['months_on_book'],
            'total_relationship_count': input['total_relationship_count'], 
            'credit_limit': float(input['credit_limit']), 
            'total_revolving_bal': input['total_revolving_bal']
        }
        return features

    def predict(_self, dicts):
        X = _self.dv.transform(dicts)
        features = _self.dv.get_feature_names_out()
        # print(f'features={features}')
        dval = xgb.DMatrix(X, feature_names=features)
        y_pred = _self.model.predict(dval)
        return y_pred[0]